from __future__ import annotations
from typing import List, Tuple
import random
from .data_models import Fitness
from .lut import LatencyLUT

"""
Computes the Fitness object for a given indivdual
(validation_loss, memory, latency)
"""

"""
Estimate the total memory footprint.

param_counts: number of parameters in each layer
bits: quantization bandwith for each layer (len(param_counts) should equal len(bits))
overhead: constant overhead like quantization scales, or just i/o

returns: memory of a single layer in bytes
"""
def memory_bytes(param_counts: List[int], bits: Tuple[int, ...], overhead: int = 0) -> int:
    assert len(param_counts) == len(bits)
    total = 0
    for p, b in zip(param_counts, bits):
        # p * b = number of bits for entire layer
        # (n + 7) // 8 converts from bits to bytes
        total += (p * b + 7) // 8
        total += overhead
    return int(total)


def latency_ms_lut(lut: LatencyLUT, op_shapes: List[Tuple[str, int, int, int, int]], bytes_total: int) -> float:
    total_us = 0.0
    for op_type, m, n, k, bitw in ops:
        us = lut.lookup(op_type, m, n, k, bitw)
        if us is None:
            raise KeyError(f"No LUT entry for {(op_type, m, n, k, bitw)}")
        total_us += us 
    
    # calculate bandwidth
    bw_alpha = lut.overheads.get("bandwidth_alpha", 0.0) if hasattr(lut, "overheads") else 0.0
    total_us += bw_alpha * bytes_total / 1e6
    return total_us / 1000 # return ms 



def mock_val_loss(bits: Tuple[int, ...], base: float = 3.0) -> float:
    """Placeholder val loss: penalize low-bit allocations and add slight noise for diversity."""
    avg_bits = sum(bits) / len(bits)
    penalty = max(0.0, (8 - avg_bits) * 0.05)
    noise = random.uniform(-0.02, 0.02)
    return base + penalty + noise


def create_fitness(validation_loss: float, mem_bytes: int, latency_ms: float) -> Fitness:
    return Fitness(val_loss=validation_loss, mem_bytes=mem_bytes, latency_ms=latency_ms)


def combine_into_fitness(validation_loss: float, mem_bytes: int, latency_ms: float) -> Fitness:
    return create_fitness(validation_loss, mem_bytes, latency_ms)
