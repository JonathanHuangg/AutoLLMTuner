from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Iterable
from .data_models import Individual, EvaluatedIndividual, Fitness, ParamLayout
from .cache import MetricsCache
from .objectives import memory_bytes, latency_ms_lut, mock_val_loss, combine_into_fitness
from .lut import LatencyLUT
from .config import latency_config

"""
Is the main wrapper function for calling the model. Can do a single indivudal or all of the individuals
"""
class Evaluator:
    def __init__(self, layout: ParamLayout, lut: LatencyLUT, default_bits: int, measure_latency: latency_config):
        self.layout = layout
        self.lut = lut 
        self.default_bits = default_bits
        self.measure_latency = measure_latency
    
    def _bytes_total(self, bits: Tuple[int, ...]):
        return memory_bytes(self.layout.param_counts, bits, overhead=0)

    def _latency(self, bits: Tuple[int, ...]) -> float:
        # loop over shapes here. Need to first get a list of all of the shapes. 
        return None 
    
    def backend_eval(self, bits: Tuple[int, ...], val_tokens: int) -> float:
        #TODO by Darin, quantize + inference goes here
        return None

    def evaluate_one(self, ind: Individual, cache: MetricsCache, val_tokens: int) -> EvaluatedIndividual:
        cached_indivdual = cache.get(ind.id)
        if cached_indivdual:
            return EvaluatedIndividual(ind, cached_indivdual, aux={"cached": True})
        memory = self._bytes_total(ind.bits)
        latency = self._latency(ind.bits)
        loss = self.backend_eval(ind.bits, val_tokens)
        fit = combine_into_fitness(loss, memory, latency)
        cache.put(ind.id, ind.bits, fit)
        return EvaluatedIndividual(ind, fit, aux={"cached": False})

    def evaluate_many(self, population: List[Individual], cache: MetricsCache, val_tokens: int, threads: int) -> List[EvaluatedIndividual]:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(self.evaluate_one, ind, cache, val_tokens) for ind in population]
            return [f.result() for f in as_completed(futs)]
        



