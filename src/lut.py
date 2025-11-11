from __future__ import annotations
from typing import Dict, Any, Tuple
import yaml

"""
Hardware latency estimator. 
Approximates how a long will take given a list of quantized operations
Does not run the model
"""

class LatencyLUT: 
    def __init__(self, path: str):
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        # key = (op_type, m, n, k, bitw), val = float
        self.lut: Dict[Tuple[str, int, int, int, int], float] = {}
        for row in raw.get("latency_lut", []):
            key = tuple(row["key"]) # from hardware.yaml: - key: ["matmul", 4096, 4096, 4096, 4]
            op, m, n, k, bit_width = key
            self.lut[(op, int(m), int(n), int(k), int(bit_width))] = float(row["us"]) / 1000.0  # convert us to ms
        
        # add overhead costs. I/O operations like loading weights form memory is based on bandwidth
        # estimate correction with alpha * bytes_total / 1e9
        overhead = raw.get("overheads", {}) or {}

        # bandwidth alpha is just a correction term. 
        self.bandwidth_alpha = float(overhead.get("bandwidth_alpha"), 0.0)

    def estimate(self, ops: list[tuple[str, int, int, int, int]], bytes_total: int) -> float:
        time = 0
        for op in ops:
            time += self.lut.get(op, 0.0)
        time += self.bandwidth_alpha * (bytes_total / 1e9)
        return time