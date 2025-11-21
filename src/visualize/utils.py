from pathlib import Path
import pandas as pd
import json, glob
import os
import sqlite3
from typing import List

def ensure_dir(path) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _normalize_record(rec):
    """
    Take a single result record dict and add derived metrics.
    """
    rec = rec.copy()  # don't mutate original
    val_loss = float(rec["val_loss"])
    mem_bytes = int(rec["mem_bytes"])

    # Normalize metrics for visualization
    rec["val_loss"] = val_loss  # Cross Entropy Loss (lower is better)
    rec["memory_gb"] = mem_bytes / 1e9

    return rec

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all SQLITE result files and compute derived metrics."""
    records = []
    
    # frontier json
    for file in glob.glob(f"{results_dir}/*.json"):
        with open(file, "r") as f:
            content = json.load(f)

        if isinstance(content, list):
            for rec in content:
                records.append(_normalize_record(rec))
    
    # sqlite
    sqlite_path = os.path.join(results_dir, "metrics.sqlite")
    if os.path.exists(sqlite_path):
        print(f"[INFO] Loading SQLite metrics from {sqlite_path}")

        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()

        tables = [t[0] for t in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        for t in tables:
            rows = cur.execute(f"SELECT * FROM {t}").fetchall()

            col_names = [desc[0] for desc in cur.description]

            for row in rows:
                row_dict = dict(zip(col_names, row))
                records.append(_normalize_record(row_dict))

        conn.close()
    else:
        print("[WARN] No metrics.sqlite found.")


    if not records:
        raise FileNotFoundError(f"No usable JSON results found in {results_dir}")
    print(f"[INFO] Loaded {len(records)} result records.")
    return pd.DataFrame(records)

def dominates(a, b, objectives):
    better_or_equal = True
    strictly_better = False
    for col, sense in objectives.items():
        if sense == "min":
            if a[col] > b[col]:
                better_or_equal = False
                break
            if a[col] < b[col]:
                strictly_better = True
        else:  # 'max'
            if a[col] < b[col]:
                better_or_equal = False
                break
            if a[col] > b[col]:
                strictly_better = True
    return better_or_equal and strictly_better

def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, objective_x: str = "max", objective_y: str = "max") -> pd.DataFrame:
    """2D Pareto frontier: e.g., minimize latency and val_loss."""
    objectives = {
        x_col: objective_x,
        y_col: objective_y
    }
    data = df.reset_index(drop=True)
    is_dominated = [False] * len(data)

    for i in range(len(data)):
        if is_dominated[i]:
            continue

        for j in range(len(data)):
            if i == j or is_dominated[j]:
                continue

            a = data.iloc[i]
            b = data.iloc[j]

            # If j dominates i → discard i
            if dominates(b, a, objectives):
                is_dominated[i] = True
                break

            # If i dominates j → discard j
            if dominates(a, b, objectives):
                is_dominated[j] = True

    return data[[not d for d in is_dominated]]

def compute_pareto_frontier_multi(df: pd.DataFrame, objectives: dict[str, str]) -> pd.DataFrame:
    """Multi-objective Pareto frontier for 3+ objectives."""
    data = df.reset_index(drop=True)
    is_dominated = [False] * len(data)

    for i in range(len(data)):
        if is_dominated[i]:
            continue
        for j in range(len(data)):
            if i == j or is_dominated[j]:
                continue
            if dominates(data.iloc[i], data.iloc[j], objectives):
                is_dominated[j] = True
            elif dominates(data.iloc[j], data.iloc[i], objectives):
                is_dominated[i] = True
                break

    return data[[not d for d in is_dominated]]


def slice_pareto(df: pd.DataFrame, metrics_for_pareto: List[str] = ("val_loss", "latency_ms", "memory_gb"), latency_min: float | None = None, latency_max: float | None = None, memory_min: float | None = None, 
                 memory_max: float | None = None, val_loss_min: float | None = None, val_loss_max: float | None = None, n_best: int = 3) -> pd.DataFrame:
    """Filter paterto frontier on constraints, return n best configurations"""

    # If three metrics inputted, use multi pareto function
    if len(metrics_for_pareto) == 3:
        frontier = compute_pareto_frontier_multi(
            df,
            objectives={m: "min" for m in metrics_for_pareto}
        )
    # If two metrics inputted, use 2D pareto function
    elif len(metrics_for_pareto) == 2:
        # deterministic order
        m1, m2 = sorted(metrics_for_pareto)
        frontier = compute_pareto_frontier(
            df,
            x_col=metrics_for_pareto[0],
            y_col=metrics_for_pareto[1],
            objective_x="min",
            objective_y="min"
        )
    # Otherwise, raise error
    else:
        raise ValueError("metrics_for_pareto must contain exactly 2 or 3 metrics.")
    
    # Apply constraints
    if latency_min is not None:
        frontier = frontier[frontier["latency_ms"] >= latency_min]

    if latency_max is not None:
        frontier = frontier[frontier["latency_ms"] <= latency_max]

    if memory_min is not None:
        frontier = frontier[frontier["memory_gb"] >= memory_min]

    if memory_max is not None:
        frontier = frontier[frontier["memory_gb"] <= memory_max]

    if val_loss_min is not None:
        frontier = frontier[frontier["val_loss"] >= val_loss_min]

    if val_loss_max is not None:
        frontier = frontier[frontier["val_loss"] <= val_loss_max]
    
    # If no configurations meet the constraints, proivde a message and return
    if frontier.empty:
        print("[WARN] No configurations meet the specified constraints.")
        return frontier
    
    # Sort in preferred order of val_loss -> latency -> memory

    sort_cols = [m for m in ["val_loss", "latency_ms", "memory_gb"] if m in frontier.columns]
    frontier = frontier.sort_values(by=sort_cols, ascending=True)

    return frontier.head(n_best)

