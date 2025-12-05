from __future__ import annotations
from typing import List, Tuple
from .data_models import Individual, EvaluatedIndividual, ParamLayout
from .cache import MetricsCache
from .objectives import memory_bytes, latency_ms_lut, combine_into_fitness
from .lut import LatencyLUT
from .config import latency_config
from .bnb_quantize import BnbQuantizeBackend
import logging
import time
import torch
import multiprocessing as mp
import multiprocessing.queues as mp_queues
import time

"""
Main wrapper for evaluating individuals:
- memory_bytes from param_counts + bits
- latency from LUT (or 0 if disabled)
- validation loss from backend (BnB quantized model on MATH-500, or mock fallback)
"""

def get_devices():
    """Get available CUDA devices, or CPU if CUDA is not available"""
    if torch.cuda.is_available():
        return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    else:
        return [torch.device("cpu")]

logger = logging.getLogger(__name__)
class Evaluator:
    def __init__(
        self,
        model_name: str,
        layout: ParamLayout,
        lut: LatencyLUT,
        default_fp_bits: int,
        measure_latency: latency_config,
    ):
        self.layout = layout
        self.lut = lut
        self.default_fp_bits = default_fp_bits
        self.measure_latency = measure_latency

        self.quant_backend: BnbQuantizeBackend | None = None
        self._backend_error: Exception | None = None

        try:
            logger.info("Initializing BnbQuantizeBackend...")
            self.quant_backend = BnbQuantizeBackend(model_name, layout)
            logger.info("BnbQuantizeBackend initialized successfully.")
        except Exception as e:
            logger.exception(
                "Failed to initialize BnbQuantizeBackend; falling back to mock loss."
            )
            self.quant_backend = None
            self._backend_error = e


    def _bytes_total(self, bits: Tuple[int, ...]) -> int:
        return memory_bytes(self.layout.param_counts, bits, overhead=0)

    def _latency(self, bits: Tuple[int, ...]) -> float:
        bytes_total = self._bytes_total(bits)

        op_list = []
        for idx, unit_name in enumerate(self.layout.units):
            if self.layout.excluded_mask[idx]:
                continue
            bit = bits[idx]
            for op in self.layout.unit_ops[unit_name]:
                op_list.append((op.op_type, op.m, op.n, op.k, bit))

        if self.measure_latency == latency_config.LOOKUP:
            # Pure LUT path; no updates
            return latency_ms_lut(self.lut, op_list, bytes_total)

        if self.measure_latency == latency_config.REAL:
            total_us = 0.0
            for op_type, m, n, k, bitw in op_list:
                us = self.lut.lookup(op_type, m, n, k, bitw)
                if us is None:
                    # New op-shape+bit: measure and record
                    us = self.measure_gemm_latency_us(op_type, m, n, k, bitw)
                    self.lut.record(op_type, m, n, k, bitw, us)
                total_us += us

            bw_alpha = float(self.lut.overheads.get("bandwidth_alpha", 0.0))
            total_us += bw_alpha * (bytes_total / 1e6)

            return total_us / 1000.0  # ms

        return 0.0

    def backend_eval(self, bits: Tuple[int, ...], val_tokens: int, device: torch.device) -> float:
        """
        Delegate to the quantization backend if available; otherwise use a mock loss.
        """
        if self.quant_backend is None:
            logger.warning(
                f"Quantization backend not available (error: {self._backend_error}); using mock loss."
            )
            # Return a mock loss value (e.g., 2.0) when backend is unavailable
            return 2.0
        
        logger.debug(
            f"backend_eval(real): val_tokens={val_tokens}, calling BnbQuantizeBackend.eval_bits..."
        )
        self.quant_backend.set_device(device)
        return self.quant_backend.eval_bits(bits, val_tokens)

    def evaluate_one(
        self,
        ind: Individual,
        cache: MetricsCache,
        val_tokens: int,
        device: torch.device,
    ) -> EvaluatedIndividual:
        cached_individual = cache.get(ind.id)
        if cached_individual:
            logger.debug(f"evaluate_one: cache hit for id={ind.id}")
            return EvaluatedIndividual(ind, cached_individual, aux={"cached": True})

        logger.debug(f"evaluate_one: cache miss for id={ind.id}")

        memory = self._bytes_total(ind.bits)
        latency = self._latency(ind.bits)
        loss = self.backend_eval(ind.bits, val_tokens, device)

        fit = combine_into_fitness(loss, memory, latency)
        cache.put(ind.id, ind.bits, fit)
        return EvaluatedIndividual(ind, fit, aux={"cached": False})

    def _evaluate_many_worker(self, pop_queue: mp.Queue, res_queue: mp.Queue, device: torch.device, val_tokens: int):
        """Worker function that evaluates individuals from the queue"""
        # Re-initialize logger in worker process
        worker_logger = logging.getLogger(__name__)
        
        while True:
            try:
                idx, ind = pop_queue.get(timeout=1.0)  # Use timeout to allow checking if queue is done
            except mp_queues.Empty:
                # Queue is empty, check if we should continue
                if pop_queue.empty():
                    break
                continue
            except Exception as e:
                worker_logger.error(f"Worker {mp.current_process().name} error getting from queue: {e}")
                break
            
            try:
                worker_logger.debug(
                    f"Worker {mp.current_process().name} evaluating id={ind.id}, seed_tag={ind.seed_tag}"
                )
                memory = self._bytes_total(ind.bits)
                latency = self._latency(ind.bits)
                loss = self.backend_eval(ind.bits, val_tokens, device)
                fit = combine_into_fitness(loss, memory, latency)
                res = EvaluatedIndividual(ind, fit, aux={"cached": False, "error": None})
                res_queue.put((idx, res))
                worker_logger.debug(f"Worker {mp.current_process().name} completed evaluation for {ind.id}")
            except Exception as e:
                worker_logger.error(
                    f"Worker {mp.current_process().name} failed to evaluate individual {ind.id}: {e}",
                    exc_info=True
                )
                # Return a fallback evaluation with high loss to indicate failure
                try:
                    memory = self._bytes_total(ind.bits)
                    latency = self._latency(ind.bits)
                except Exception as e2:
                    worker_logger.error(f"Worker failed to compute memory/latency: {e2}")
                    memory = 0
                    latency = 0.0
                fallback_loss = 10.0  # High loss to indicate evaluation failure
                fit = combine_into_fitness(fallback_loss, memory, latency)
                res = EvaluatedIndividual(ind, fit, aux={"cached": False, "error": str(e)})
                res_queue.put((idx, res))

    def evaluate_many(
        self,
        population: List[Individual],
        cache: MetricsCache,
        val_tokens: int = 4096,
    ) -> List[EvaluatedIndividual]:
        logger.info(
            f"Evaluating {len(population)} individuals with val_tokens={val_tokens}"
        )

        # results: List[EvaluatedIndividual] = []
        results: List[EvaluatedIndividual | None] = [None] * len(population)
        population_queue = mp.Queue()
        items_to_evaluate = 0  # Track how many items we put in the queue
        for i, ind in enumerate(population):
            cached_individual = cache.get(ind.id)
            if cached_individual:
                logger.debug(f"evaluate_many: cache hit for id={ind.id}")
                results[i] = EvaluatedIndividual(ind, cached_individual, aux={"cached": True})
            else:
                logger.debug(f"evaluate_many: cache miss for id={ind.id}")
                # Only add to queue if not cached
                population_queue.put((i, ind))
                items_to_evaluate += 1

        results_queue: mp.Queue = mp.Queue()
        
        # If backend is None, evaluate in main process (multiprocessing doesn't work well without backend)
        if self.quant_backend is None:
            logger.info(f"Backend is None, evaluating {items_to_evaluate} individuals in main process (no multiprocessing)")
            device = get_devices()[0] if get_devices() else torch.device("cpu")
            
            # Process all items from queue - use the counter we tracked
            processed = 0
            while processed < items_to_evaluate:
                try:
                    idx, ind = population_queue.get(timeout=1.0)
                    processed += 1
                except mp_queues.Empty:
                    # If queue is empty but we haven't processed all items, something went wrong
                    logger.warning(f"Queue empty but only processed {processed}/{items_to_evaluate} items")
                    break
                except Exception as e:
                    logger.error(f"Error getting from queue: {e}")
                    break
                
                try:
                    logger.debug(f"Evaluating individual {ind.id} (index {idx})")
                    memory = self._bytes_total(ind.bits)
                    latency = self._latency(ind.bits)
                    loss = self.backend_eval(ind.bits, val_tokens, device)
                    fit = combine_into_fitness(loss, memory, latency)
                    res = EvaluatedIndividual(ind, fit, aux={"cached": False, "error": None})
                    results_queue.put((idx, res))
                    logger.debug(f"Completed evaluation for individual {ind.id}")
                except Exception as e:
                    logger.error(f"Failed to evaluate individual {ind.id}: {e}", exc_info=True)
                    try:
                        memory = self._bytes_total(ind.bits)
                        latency = self._latency(ind.bits)
                    except Exception as e2:
                        logger.error(f"Failed to compute memory/latency: {e2}")
                        memory = 0
                        latency = 0.0
                    fallback_loss = 10.0
                    fit = combine_into_fitness(fallback_loss, memory, latency)
                    res = EvaluatedIndividual(ind, fit, aux={"cached": False, "error": str(e)})
                    results_queue.put((idx, res))
            
            logger.info(f"Processed {processed}/{items_to_evaluate} individuals in main process")
        else:
            # Get devices in the main process (workers will get their own devices)
            available_devices = get_devices()
            if len(available_devices) == 0:
                available_devices = [torch.device("cpu")]
            
            logger.info(f"Creating {len(available_devices)} worker processes for {population_queue.qsize()} individuals")
            
            # If no workers needed (all cached), skip worker creation
            if population_queue.empty():
                logger.info("All individuals are cached, skipping worker processes")
            else:
                workers = [
                    mp.Process(target=self._evaluate_many_worker, args=(population_queue, results_queue, device, val_tokens))
                    for device in available_devices
                ]

                for w in workers:
                    w.start()
                    logger.debug(f"Started worker process {w.name} (PID: {w.pid})")

                # Wait for all workers to finish
                for w in workers:
                    w.join(timeout=300)  # 5 minute timeout per worker
                    if w.is_alive():
                        logger.error(f"Worker {w.name} did not finish in time, terminating")
                        w.terminate()
                        w.join()
                    logger.debug(f"Worker {w.name} finished with exit code {w.exitcode}")

        logger.info("Finished evaluating population.")
        
        # Collect all results from the queue
        # Note: qsize() may not be reliable, so we use a timeout-based approach
        results_collected = 0
        max_attempts = len(population) * 2  # Safety limit
        attempts = 0
        
        while attempts < max_attempts:
            try:
                idx, res = results_queue.get(timeout=0.1)
                results[idx] = res
                cache.put(res.ind.id, res.ind.bits, res.fit)
                results_collected += 1
                logger.debug(f"Collected result for individual at index {idx} (id={res.ind.id})")
            except mp_queues.Empty:
                # No more results available
                break
            except Exception as e:
                logger.error(f"Error getting result from queue: {e}")
                break
            attempts += 1
        
        logger.info(f"Collected {results_collected} results from queue")

        # Check for any missing results and create fallback evaluations
        for i in range(len(results)):
            if results[i] is None:
                logger.warning(
                    f"Missing evaluation result for individual at index {i} (id={population[i].id}). "
                    f"Creating fallback evaluation."
                )
                ind = population[i]
                # Create a fallback evaluation with high loss
                memory = self._bytes_total(ind.bits)
                latency = self._latency(ind.bits)
                fallback_loss = 10.0  # High loss to indicate evaluation failure
                fit = combine_into_fitness(fallback_loss, memory, latency)
                results[i] = EvaluatedIndividual(
                    ind, fit, aux={"cached": False, "error": "Worker did not return result"}
                )
                # Don't cache failed evaluations

        return results

    @staticmethod
    def measure_gemm_latency_us(op_type: str, m: int, n: int, k: int, bitw: int, iters: int = 20) -> float:
        """
        for building the lut table in real time
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            _ = a @ b
        torch.cuda.synchronize() if device.type == "cuda" else None

        # Timed runs
        start = time.time()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.time()

        avg_s = (end - start) / iters
        return avg_s * 1e6
