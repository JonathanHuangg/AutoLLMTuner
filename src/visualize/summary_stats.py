"""
summary_stats.py

Compute aggregate statistics + PNG plots + verbose JSON describing
everything useful for interpreting evolutionary quantization runs.

New features added:
  - Frontier-only strict per-layer bit statistics
  - Accuracy buckets (high/medium/low) with per-layer and global bit stats
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir, compute_pareto_frontier_multi


# ================================================================
# Objectives: only use columns that actually exist
# ================================================================

DEFAULT_OBJECTIVES: Dict[str, str] = {
    "val_loss": "min",
    "latency_ms": "min",
    "memory_gb": "min",
}


# ================================================================
# Numeric summarization with NaN safety
# ================================================================

def summarize_objectives(df: pd.DataFrame,
                         cols: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:

    if df.empty:
        return {}

    if cols is None:
        candidate_cols = ["latency_ms", "memory_gb", "val_loss"]
        cols = [c for c in candidate_cols if c in df.columns]

    summary = {}

    for col in cols:
        if col not in df.columns:
            continue

        x = df[col].to_numpy(dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() == 0:
            continue

        xv = x[mask]

        summary[col] = {
            "count": float(xv.size),
            "min": float(np.nanmin(xv)),
            "p25": float(np.nanquantile(xv, 0.25)),
            "median": float(np.nanmedian(xv)),
            "p75": float(np.nanquantile(xv, 0.75)),
            "max": float(np.nanmax(xv)),
            "mean": float(np.nanmean(xv)),
            "std": float(np.nanstd(xv)),
        }
    return summary


# ================================================================
# Bit parsing utilities (robust to strings from SQLite)
# ================================================================

def _parse_bits_field(b) -> np.ndarray:
    if isinstance(b, np.ndarray):
        return b.astype(int)
    if isinstance(b, (list, tuple)):
        return np.asarray(b, dtype=int)
    if isinstance(b, str):
        s = b.strip()
        if not s:
            return np.asarray([], dtype=int)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                s_clean = s.strip("[](){}")
                parts = [p.strip() for p in s_clean.split(",") if p.strip()]
                try:
                    return np.asarray([int(p) for p in parts], dtype=int)
                except Exception:
                    return np.asarray([], dtype=int)
        if isinstance(obj, (list, tuple, np.ndarray)):
            try:
                return np.asarray(obj, dtype=int)
            except Exception:
                return np.asarray([], dtype=int)
        return np.asarray([], dtype=int)

    return np.asarray([], dtype=int)


def flatten_bits(df: pd.DataFrame) -> np.ndarray:
    if "bits" not in df.columns:
        return np.asarray([], dtype=int)

    all_bits = []
    for b in df["bits"]:
        arr = _parse_bits_field(b)
        if arr.size > 0:
            all_bits.extend(arr.tolist())
    return np.asarray(all_bits, dtype=int)


def bit_histogram(df: pd.DataFrame) -> Optional[Dict]:
    flat = flatten_bits(df)
    if flat.size == 0:
        return None

    unique, counts = np.unique(flat, return_counts=True)
    total = float(flat.size)
    return {
        "counts": {int(u): int(c) for u, c in zip(unique, counts)},
        "fractions": {int(u): float(c) / total for u, c in zip(unique, counts)},
    }


# ================================================================
# Per-layer statistics (dominant bit-vector length only)
# ================================================================

def per_layer_bit_stats(df: pd.DataFrame) -> List[Dict]:
    if "bits" not in df.columns:
        return []

    parsed = []
    for b in df["bits"]:
        arr = _parse_bits_field(b)
        if arr.size > 0:
            parsed.append(arr)

    if not parsed:
        return []

    # choose most common bitvector length
    length_counts = {}
    for arr in parsed:
        length_counts[arr.size] = length_counts.get(arr.size, 0) + 1

    target_len, _ = max(length_counts.items(), key=lambda kv: kv[1])
    filtered = [a for a in parsed if a.size == target_len]
    if not filtered:
        return []

    mat = np.stack(filtered, axis=0)  # (N, L)
    n_layers = mat.shape[1]

    out = []
    for layer_idx in range(n_layers):
        col = mat[:, layer_idx]
        vals, counts = np.unique(col, return_counts=True)
        out.append({
            "layer": int(layer_idx),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "hist": {int(v): int(c) for v, c in zip(vals, counts)}
        })
    return out


# ================================================================
# Accuracy bucket utilities
# ================================================================

def get_accuracy_buckets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create logical buckets by val_loss.
    """
    if "val_loss" not in df.columns:
        return {"all": df}

    v = df["val_loss"].astype(float)

    high = df[v <= 0.53]
    medium = df[(v > 0.53) & (v <= 0.60)]
    low = df[v > 0.60]

    return {
        "high": high,
        "medium": medium,
        "low": low,
    }


def bucket_bit_layer_stats(bucket_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute:
      - global bit histogram
      - per-layer bit stats
    """
    return {
        "size": len(bucket_df),
        "bits": bit_histogram(bucket_df),
        "per_layer": per_layer_bit_stats(bucket_df)
    }


# ================================================================
# Frontier and main summary
# ================================================================

def compute_run_summary(df: pd.DataFrame,
                        objectives: Optional[Dict[str, str]] = None) -> Dict:

    if objectives is None:
        objectives = {c: s for c, s in DEFAULT_OBJECTIVES.items()
                      if c in df.columns}

    df_front = compute_pareto_frontier_multi(df, objectives)

    obj_all = summarize_objectives(df)
    obj_front = summarize_objectives(df_front)

    bits_all = bit_histogram(df)
    bits_front = bit_histogram(df_front)

    per_layer_all = per_layer_bit_stats(df)
    per_layer_front = per_layer_bit_stats(df_front)

    # NEW: accuracy buckets
    buckets = get_accuracy_buckets(df)
    bucket_stats = {
        name: bucket_bit_layer_stats(bdf)
        for name, bdf in buckets.items()
    }

    return {
        "columns": list(df.columns),
        "objectives_used": objectives,
        "n_all": float(len(df)),
        "n_frontier": float(len(df_front)),
        "frontier_ratio": float(len(df_front)) / float(len(df)) if len(df) > 0 else 0.0,

        "objectives_all": obj_all,
        "objectives_frontier": obj_front,

        "bits_all": bits_all,
        "bits_frontier": bits_front,
        "per_layer_all": per_layer_all,

        # NEW strict frontier-only per-layer stats
        "per_layer_frontier_strict": per_layer_front,

        # NEW Accuracy buckets
        "accuracy_buckets": bucket_stats,

        "frontier_df": df_front,
    }


# ================================================================
# Plotting (unchanged)
# ================================================================

def _format_stats_table_data(obj_all, obj_front):
    metrics_set = set(obj_all.keys()) | set(obj_front.keys())
    preferred = ["latency_ms", "memory_gb", "val_loss"]
    ordered = [m for m in preferred if m in metrics_set] + sorted(metrics_set - set(preferred))
    columns = ["metric","all_min","all_med","all_mean","all_max",
               "front_min","front_med","front_mean","front_max"]
    rows = []

    fmt = lambda v: "" if v is None or np.isnan(v) else f"{float(v):.4g}"

    for m in ordered:
        a = obj_all.get(m,{})
        f = obj_front.get(m,{})
        rows.append([
            m,
            fmt(a.get("min")), fmt(a.get("median")), fmt(a.get("mean")), fmt(a.get("max")),
            fmt(f.get("min")), fmt(f.get("median")), fmt(f.get("mean")), fmt(f.get("max")),
        ])
    return columns, rows


def _plot_objective_summary_table(obj_all, obj_front, n_all, n_front, ratio, out_path):
    _, rows = _format_stats_table_data(obj_all, obj_front)
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.3*n_rows))
    ax.axis("off")
    ax.text(0.01,0.98,f"All points: {int(n_all)}  |  Frontier points: {int(n_front)}  |  Frontier ratio: {ratio:.3f}",
            transform=ax.transAxes, va="top")

    columns, rows = _format_stats_table_data(obj_all, obj_front)
    table = ax.table(cellText=[columns]+rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.2)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_bit_histogram(bits_all, bits_front, out_path):
    if bits_all is None or bits_front is None:
        return
    bits = sorted(set(bits_all["fractions"].keys()) |
                  set(bits_front["fractions"].keys()))
    all_f = [bits_all["fractions"].get(b,0) for b in bits]
    fr_f  = [bits_front["fractions"].get(b,0) for b in bits]

    x = np.arange(len(bits))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x-0.35/2, all_f, 0.35, label="All")
    ax.bar(x+0.35/2, fr_f, 0.35, label="Frontier")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bits])
    ax.set_title("Bit-width usage: All vs Frontier")
    ax.legend()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_layer_mean_bits(per_all, per_front, out_path):
    if not per_all or not per_front:
        return
    layers = sorted(set([d["layer"] for d in per_all]) &
                    set([d["layer"] for d in per_front]))

    m_all = {d["layer"]: d["mean"] for d in per_all}
    m_fr  = {d["layer"]: d["mean"] for d in per_front}

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(layers, [m_all[l] for l in layers], marker="o", label="All")
    ax.plot(layers, [m_fr[l] for l in layers],  marker="o", label="Frontier")
    ax.set_title("Mean Bit Width per Layer")
    ax.grid(True, linestyle="--")
    ax.legend()

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ================================================================
# JSON serializer
# ================================================================

def _to_serializable(x):
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_to_serializable(v) for v in x]
    return x


# ================================================================
# Main public entry point
# ================================================================

def plot_summary_statistics(df: pd.DataFrame,
                            save_dir: str | Path,
                            objectives: Optional[Dict[str,str]] = None):

    save_dir = Path(save_dir)
    ensure_dir(save_dir)

    summary = compute_run_summary(df, objectives)

    _plot_objective_summary_table(
        summary["objectives_all"],
        summary["objectives_frontier"],
        summary["n_all"],
        summary["n_frontier"],
        summary["frontier_ratio"],
        save_dir / "summary_objectives.png"
    )

    _plot_bit_histogram(summary["bits_all"], summary["bits_frontier"],
                        save_dir / "summary_bit_hist.png")

    _plot_layer_mean_bits(summary["per_layer_all"],
                          summary["per_layer_frontier_strict"],
                          save_dir / "summary_layer_mean_bits.png")

    # verbose JSON
    frontier_records = summary["frontier_df"].to_dict(orient="records")

    json_out = {
        k: _to_serializable(v)
        for k, v in summary.items()
        if k != "frontier_df"
    }
    json_out["frontier_points"] = _to_serializable(frontier_records)

    with open(save_dir / "summary_stats.json", "w") as f:
        json.dump(json_out, f, indent=2)
