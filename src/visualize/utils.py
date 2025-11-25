from pathlib import Path
import pandas as pd
import json, glob
import os
import sqlite3

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

def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, objective_y: str = "max") -> pd.DataFrame:
    """2D Pareto frontier: e.g., minimize latency and val_loss."""
    df_sorted = df.sort_values(by=x_col, ascending=True)
    pareto = []
    best_y = float("inf") if objective_y == "min" else -float("inf")

    for _, row in df_sorted.iterrows():
        if objective_y == "min":
            if row[y_col] < best_y:
                pareto.append(row)
                best_y = row[y_col]
        else:
            if row[y_col] > best_y:
                pareto.append(row)
                best_y = row[y_col]

    return pd.DataFrame(pareto)


def compute_pareto_frontier_multi(df: pd.DataFrame, objectives: dict[str, str]) -> pd.DataFrame:
    """Multi-objective Pareto frontier for 3+ objectives."""
    data = df.reset_index(drop=True)
    is_dominated = [False] * len(data)

    def dominates(a, b):
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

    for i in range(len(data)):
        if is_dominated[i]:
            continue
        for j in range(len(data)):
            if i == j or is_dominated[j]:
                continue
            if dominates(data.iloc[i], data.iloc[j]):
                is_dominated[j] = True
            elif dominates(data.iloc[j], data.iloc[i]):
                is_dominated[i] = True
                break

    return data[[not d for d in is_dominated]]
