from pathlib import Path
import pandas as pd
import json, glob

def ensure_dir(path) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all JSON result files and compute derived metrics."""
    records = []
    for file in glob.glob(f"{results_dir}/*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            data["accuracy"] = 1.0 / data["val_loss"]
            data["memory_gb"] = data["mem_bytes"] / 1e9
            records.append(data)

    if not records:
        raise FileNotFoundError(f"No JSON results found in {results_dir}")

    df = pd.DataFrame(records)
    return df

def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """2D Pareto frontier: e.g., minimize latency, maximize accuracy."""
    df_sorted = df.sort_values(by=x_col, ascending=True)
    pareto = []
    best_y = -float("inf")

    for _, row in df_sorted.iterrows():
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
