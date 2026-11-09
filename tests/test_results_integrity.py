"""Guards the numbers the README reports for the two committed Qwen runs.

These summaries are the analyzed outputs of runs whose raw data was never
committed, so they cannot be re-derived from this repo. What this test does
verify: the files are internally consistent (counts, ratios, bit fractions,
frontier non-domination), and the exact figures quoted in the README still
match them. If a summary file changes, the README must change with it.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _load(run):
    return json.loads((RESULTS / run / "summary_stats.json").read_text())


def _assert_frontier_sound(summary):
    pts = summary["frontier_points"]
    assert len(pts) == int(summary["n_frontier"])
    # no frontier point is dominated by another on (val_loss, memory)
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i != j:
                assert not (
                    b["val_loss"] <= a["val_loss"]
                    and b["mem_bytes"] <= a["mem_bytes"]
                    and (b["val_loss"] < a["val_loss"] or b["mem_bytes"] < a["mem_bytes"])
                )
    for section in ("bits_all", "bits_frontier"):
        assert sum(summary[section]["fractions"].values()) == pytest.approx(1.0)
    assert summary["frontier_ratio"] == pytest.approx(
        summary["n_frontier"] / summary["n_all"]
    )


def test_correct_answers_run_matches_readme():
    """Generate-and-grade run (val_loss = 1 - accuracy on 50 MATH-500
    problems). See README for the protocol note."""
    d = _load("correct_answers")
    assert int(d["n_all"]) == 175
    assert int(d["n_frontier"]) == 37
    # accuracy = 1 - val_loss
    assert 1 - d["objectives_all"]["val_loss"]["mean"] == pytest.approx(0.7697, abs=1e-4)
    assert 1 - d["objectives_frontier"]["val_loss"]["mean"] == pytest.approx(
        0.7989, abs=1e-4
    )
    assert d["objectives_all"]["val_loss"]["min"] == pytest.approx(0.06)  # best: 94%
    assert d["objectives_all"]["memory_gb"]["mean"] == pytest.approx(6.076, abs=1e-3)
    assert d["objectives_frontier"]["memory_gb"]["mean"] == pytest.approx(5.130, abs=1e-3)
    assert d["objectives_all"]["memory_gb"]["min"] == pytest.approx(4.833, abs=1e-3)
    assert d["objectives_all"]["memory_gb"]["max"] == pytest.approx(12.493, abs=1e-3)
    assert d["objectives_all"]["latency_ms"]["mean"] == pytest.approx(8.514, abs=1e-3)
    assert d["bits_frontier"]["fractions"]["4"] == pytest.approx(0.675, abs=1e-3)
    _assert_frontier_sound(d)


def test_overnight_run_matches_readme():
    """Cross-entropy-loss run. val_loss here is CE loss, NOT 1 - accuracy,
    so the README reports it as loss, not accuracy."""
    d = _load("overnight_everything")
    assert int(d["n_all"]) == 646
    assert int(d["n_frontier"]) == 81
    assert d["objectives_all"]["val_loss"]["mean"] == pytest.approx(0.5793, abs=1e-4)
    assert d["objectives_frontier"]["val_loss"]["mean"] == pytest.approx(0.5360, abs=1e-4)
    assert d["objectives_all"]["memory_gb"]["mean"] == pytest.approx(7.079, abs=1e-3)
    assert d["objectives_frontier"]["memory_gb"]["mean"] == pytest.approx(8.323, abs=1e-3)
    assert d["objectives_all"]["latency_ms"]["mean"] == pytest.approx(8.871, abs=1e-3)
    _assert_frontier_sound(d)


def test_mistral_v003_was_regenerated_with_fixed_pipeline():
    """The Mistral run is the one run whose raw data is in the repo
    (examples/mistral-evoq-v003). Its summary must match a fresh analysis
    of that data: 93 unique individuals, 11-point frontier."""
    from visualize.summary_stats import compute_run_summary
    from visualize.utils import load_results

    df = load_results(str(REPO_ROOT / "examples" / "mistral-evoq-v003"))
    fresh = compute_run_summary(df)
    committed = _load("mistral-v003")
    assert int(committed["n_all"]) == int(fresh["n_all"]) == 93
    assert int(committed["n_frontier"]) == int(fresh["n_frontier"]) == 11
    assert committed["objectives_frontier"]["val_loss"]["mean"] == pytest.approx(
        fresh["objectives_frontier"]["val_loss"]["mean"]
    )
    _assert_frontier_sound(committed)
