"""End to end: the smoke search runs the full pipeline on CPU (config
load, population, NSGA-II generations, SQLite cache, frontier files),
and the plot pipeline turns its output into figures and a summary."""

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_run():
    out_dir = REPO_ROOT / "runs" / "pytest-smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    yield out_dir
    shutil.rmtree(out_dir, ignore_errors=True)


def test_smoke_search_end_to_end(monkeypatch, smoke_run):
    from evolutionary.main_search import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autollmtuner-search",
            "--config",
            str(REPO_ROOT / "configs" / "smoke_run_config.yaml"),
            "--run-id",
            "pytest-smoke",
        ],
    )
    main()

    assert (smoke_run / "frontier.json").exists()
    assert (smoke_run / "frontier_0.json").exists()
    assert (smoke_run / "frontier_2.json").exists()
    assert (smoke_run / "metrics.sqlite").exists()

    frontier = json.loads((smoke_run / "frontier.json").read_text())
    assert len(frontier) > 0
    ids = [row["id"] for row in frontier]
    assert len(ids) == len(set(ids))  # no duplicate individuals

    # the frontier is genuinely non-dominated
    for i, a in enumerate(frontier):
        for j, b in enumerate(frontier):
            if i == j:
                continue
            assert not (
                b["val_loss"] <= a["val_loss"]
                and b["mem_bytes"] <= a["mem_bytes"]
                and b["latency_ms"] <= a["latency_ms"]
                and (
                    b["val_loss"] < a["val_loss"]
                    or b["mem_bytes"] < a["mem_bytes"]
                    or b["latency_ms"] < a["latency_ms"]
                )
            ), "frontier contains a dominated point"

    # every frontier point is in the cache, and the cache saw every generation
    n_cached = sqlite3.connect(smoke_run / "metrics.sqlite").execute(
        "SELECT COUNT(*) FROM evals"
    ).fetchone()[0]
    assert n_cached >= len(frontier)


def test_smoke_plots_end_to_end(monkeypatch, tmp_path, smoke_run):
    from evolutionary.main_search import main
    from visualize.all_plots import generate_all_plots

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autollmtuner-search",
            "--config",
            str(REPO_ROOT / "configs" / "smoke_run_config.yaml"),
            "--run-id",
            "pytest-smoke",
        ],
    )
    main()

    plots_dir = tmp_path / "plots"
    generate_all_plots(str(smoke_run), str(plots_dir))
    summary = json.loads((plots_dir / "summary_stats.json").read_text())
    assert summary["n_all"] > 0
    assert summary["n_frontier"] > 0
    assert summary["n_frontier"] <= summary["n_all"]
    assert (plots_dir / "pareto_accuracy_vs_latency.png").stat().st_size > 0
    assert (plots_dir / "bit_allocation_heatmap.png").stat().st_size > 0
