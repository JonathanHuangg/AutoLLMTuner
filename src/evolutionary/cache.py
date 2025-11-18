from __future__ import annotations
from typing import Optional, Tuple
import sqlite3
import json
from .data_models import Fitness

"""
Uses sqlite to save previously ran individuals so you are not
recomputing the same thing
"""
class MetricsCache:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS evals(
            id TEXT PRIMARY KEY,
            bits TEXT,
            val_loss REAL,
            mem_bytes INTEGER,
            latency REAL
        )"""
        )
        self.connection.commit()
    
    def get(self, cid: str) -> Optional[Fitness]:
        cur = self.connection.execute("SELECT val_loss, mem_bytes, latency FROM evals WHERE id=?", (cid,))
        row = cur.fetchone()
        if not row: return None
        return Fitness(val_loss=row[0], mem_bytes=row[1], latency_ms=row[2])

    def put(self, cid: str, bits: Tuple[int, ...], fit: Fitness) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO evals(id, bits, val_loss, mem_bytes, latency) VALUES(?,?,?,?,?)",
            (cid, json.dumps(bits), fit.val_loss, fit.mem_bytes, fit.latency_ms)
        )
        self.connection.commit()