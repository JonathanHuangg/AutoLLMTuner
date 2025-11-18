from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import yaml

"""
Hardware latency estimator. 
Approximates how a long will take given a list of quantized operations
Does not run the model
"""

class LatencyLUT:
    def __init__(self, path: str):
        self.path = Path(path)
        self._table = {}  # (op_type, m, n, k, bitw) -> us

        if self.path.exists():
            with self.path.open("r") as f:
                data = yaml.safe_load(f)
            for entry in data.get("latency_lut", []):
                op_type, m, n, k, bitw = entry["key"]
                us = float(entry["us"])
                self._table[(op_type, m, n, k, bitw)] = us

            # store any extra overhead fields if needed
            self._overheads = data.get("overheads", {})
        else:
            self._overheads = {}

    def lookup(self, op_type: str, m: int, n: int, k: int, bitw: int) -> float | None:
        return self._table.get((op_type, m, n, k, bitw), None)

    def record(self, op_type: str, m: int, n: int, k: int, bitw: int, us: float, alpha=0.1):
        """
        from chat: use the exponential moving average because it smooths well. 
        """
        key = (op_type, m, n, k, bitw)
        old = self._table.get(key)
        if old is None:
            self._table[key] = us
        else:
            self._table[key] = old * (1 - alpha) + us * alpha

    def dump(self, path: str | None = None):
        """
        Write the LUT back to YAML.
        """
        out_path = Path(path) if path is not None else self.path
        entries = []
        for (op_type, m, n, k, bitw), us in sorted(self._table.items()):
            entries.append({
                "key": [op_type, int(m), int(n), int(k), int(bitw)],
                "us": float(us),
            })
        data = {
            "latency_lut": entries,
            "overheads": self._overheads,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)