from __future__ import annotations
from typing import List, Dict, Any
import json
import os
from .data_models import EvaluatedIndividual
from .core_evolutionary import dominates

"""
given list of evaluated individuals, compute the pareto front
if no other individual dominates it, it is dominant 
this is for global use acrossed epochs
"""
def pareto_archive(evals: List[EvaluatedIndividual]) -> List[EvaluatedIndividual]:
    # Elites are re-evaluated in later generations, so the same individual
    # can appear more than once; the archive is a set of configurations.
    seen: set = set()
    unique: List[EvaluatedIndividual] = []
    for e in evals:
        if e.ind.id not in seen:
            seen.add(e.ind.id)
            unique.append(e)
    out: List[EvaluatedIndividual] = []
    for e in unique:
        if not any(dominates(other.fit, e.fit) for other in unique if other is not e):
            out.append(e)
    return out

"""
Creates the json file to save it to
"""
def save_frontier(path: str, archive: List[EvaluatedIndividual]) -> None:
    rows = [{
        "id": e.ind.id,
        "bits": e.ind.bits,
        "val_loss": e.fit.val_loss,
        "mem_bytes": e.fit.mem_bytes,
        "latency_ms": e.fit.latency_ms,
        "seed_tag": e.ind.seed_tag
    } for e in archive]

    rows.sort(key=lambda x: (x["val_loss"], x["mem_bytes"], x["latency_ms"]))
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

def load_frontier(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def slice_frontier(frontier: List[Dict[str, Any]], max_mem: int | None, max_lat: float | None):
    cand = []
    for row in frontier:
        if max_mem is not None and row["mem_bytes"] > max_mem: continue
        if max_lat is not None and row["latency_ms"] > max_lat: continue
        cand.append(row)
    if not cand: 
        return None 
    cand.sort(key=lambda r: r["val_loss"])
    return cand[0]
