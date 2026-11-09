"""Visualization loader: dedupes individuals seen in both the frontier
JSONs and SQLite, normalizes the two storage formats, and drops failed
evaluations. Also checks the plot pipeline produces real figures."""

import json
import sqlite3

import pytest

from visualize.all_plots import generate_all_plots
from visualize.summary_stats import compute_run_summary
from visualize.utils import load_results


def _make_run_dir(tmp_path):
    """A tiny run dir: 3 individuals in SQLite (bits as JSON strings,
    latency under the `latency` column) plus a frontier JSON that repeats
    one of them, plus one failed evaluation (loss 10.0)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    con = sqlite3.connect(run_dir / "metrics.sqlite")
    con.execute(
        "CREATE TABLE evals(id TEXT PRIMARY KEY, bits TEXT, val_loss REAL,"
        " mem_bytes INTEGER, latency REAL)"
    )
    rows = [
        ("aaa", json.dumps([4, 4]), 0.50, 1_000_000_000, 10.0),
        ("bbb", json.dumps([8, 8]), 0.40, 2_000_000_000, 12.0),
        ("ccc", json.dumps([16, 16]), 0.30, 4_000_000_000, 14.0),
        ("bad", json.dumps([4, 4]), 10.0, 1_000_000_000, 10.0),
    ]
    con.executemany("INSERT INTO evals VALUES(?,?,?,?,?)", rows)
    con.commit()
    con.close()

    (run_dir / "frontier_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "ccc",
                    "bits": [16, 16],
                    "val_loss": 0.30,
                    "mem_bytes": 4_000_000_000,
                    "latency_ms": 14.0,
                    "seed_tag": "x",
                }
            ]
        )
    )
    return run_dir


def test_load_results_dedupes_and_normalizes(tmp_path):
    df = load_results(str(_make_run_dir(tmp_path)))
    # 3 unique successful individuals: "ccc" counted once despite being in
    # both stores, "bad" dropped as a failed evaluation.
    assert len(df) == 3
    assert set(df["id"]) == {"aaa", "bbb", "ccc"}
    # sqlite rows: bits parsed from JSON strings, latency renamed
    assert df.set_index("id").loc["aaa", "bits"] == [4, 4]
    assert df.set_index("id").loc["aaa", "latency_ms"] == 10.0
    assert df.set_index("id").loc["aaa", "memory_gb"] == 1.0


def test_load_results_raises_on_empty_dir(tmp_path):
    import pytest

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_results(str(empty))


def test_summary_and_plots_run_on_small_input(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    df = load_results(str(run_dir))
    summary = compute_run_summary(df)
    assert summary["n_all"] == 3
    assert summary["n_frontier"] == 3  # all three trade off against each other
    assert summary["objectives_all"]["val_loss"]["mean"] == pytest.approx(0.4)

    out = tmp_path / "plots"
    generate_all_plots(str(run_dir), str(out))
    for name in (
        "pareto_accuracy_vs_latency.png",
        "pareto_loss_vs_latency.png",
        "pareto_accuracy_latency_memory.png",
        "bit_allocation_heatmap.png",
        "all_non_frontier_accuracy_vs_latency.png",
        "summary_objectives.png",
        "summary_bit_hist.png",
        "summary_layer_mean_bits.png",
        "summary_stats.json",
    ):
        assert (out / name).exists(), f"missing {name}"
        assert (out / name).stat().st_size > 0
    saved = json.loads((out / "summary_stats.json").read_text())
    assert saved["n_all"] == 3
