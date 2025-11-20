from __future__ import annotations
from typing import List, Tuple
from .data_models import Individual, EvaluatedIndividual, ParamLayout
from .cache import MetricsCache
from .objectives import memory_bytes, latency_ms_lut, combine_into_fitness
from .lut import LatencyLUT
from .config import latency_config
from .bnb_quantize import BnbQuantizeBackend
import logging

"""
Main wrapper for evaluating individuals:
- memory_bytes from param_counts + bits
- latency from LUT (or 0 if disabled)
- validation loss from backend (BnB quantized model on MATH-500, or mock fallback)
"""

logger = logging.getLogger(__name__)
class Evaluator:
    def __init__(
        self,
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
            self.quant_backend = BnbQuantizeBackend(layout)
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

    def backend_eval(self, bits: Tuple[int, ...], val_tokens: int) -> float:
        """
        Delegate to the quantization backend if available; otherwise use a mock loss.
        """
        logger.debug(
            f"backend_eval(real): val_tokens={val_tokens}, calling BnbQuantizeBackend.eval_bits..."
        )
        return self.quant_backend.eval_bits(bits, val_tokens)
        

    def evaluate_one(
        self,
        ind: Individual,
        cache: MetricsCache,
        val_tokens: int,
    ) -> EvaluatedIndividual:
        cached_individual = cache.get(ind.id)
        if cached_individual:
            logger.debug(f"evaluate_one: cache hit for id={ind.id}")
            return EvaluatedIndividual(ind, cached_individual, aux={"cached": True})

        logger.debug(f"evaluate_one: cache miss for id={ind.id}")

        memory = self._bytes_total(ind.bits)
        latency = self._latency(ind.bits)
        loss = self.backend_eval(ind.bits, val_tokens)

        fit = combine_into_fitness(loss, memory, latency)
        cache.put(ind.id, ind.bits, fit)
        return EvaluatedIndividual(ind, fit, aux={"cached": False})

    def evaluate_many(
        self,
        population: List[Individual],
        cache: MetricsCache,
        val_tokens: int,
    ) -> List[EvaluatedIndividual]:
        logger.info(
            f"Evaluating {len(population)} individuals with val_tokens={val_tokens}"
        )

        results: List[EvaluatedIndividual] = []
        for idx, ind in enumerate(population):
            logger.debug(
                f"  -> [{idx+1}/{len(population)}] Evaluating id={ind.id}, seed_tag={ind.seed_tag}"
            )
            res = self.evaluate_one(ind, cache, val_tokens)
            results.append(res)

        logger.info("Finished evaluating population.")
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
