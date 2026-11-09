"""Objectives, LUT, cache, frontier persistence, and the mock evaluator."""

import json
import sqlite3

import pytest
import yaml

from evolutionary.cache import MetricsCache
from evolutionary.config import LatencyConfig
from evolutionary.data_models import EvaluatedIndividual, Fitness, Individual, ParamLayout
from evolutionary.evaluator import Evaluator, surrogate_mock_loss
from evolutionary.frontier import load_frontier, pareto_archive, save_frontier, slice_frontier
from evolutionary.lut import LatencyLUT
from evolutionary.objectives import latency_ms_lut, memory_bytes
from evolutionary.utils import create_individual_id


def test_memory_bytes_counts_bits_per_parameter():
    # 8 params at 4 bits = 4 bytes; 8 params at 16 bits = 16 bytes
    assert memory_bytes([8, 8], (4, 16)) == 20
    assert memory_bytes([1_000_000], (8,)) == 1_000_000


def test_latency_ms_lut_sums_entries_plus_bandwidth():
    lut = LatencyLUT("/nonexistent.yaml")
    lut.record("matmul", 128, 64, 64, 4, 100.0)
    lut.record("matmul", 128, 64, 64, 8, 200.0)
    ops = [("matmul", 128, 64, 64, 4), ("matmul", 128, 64, 64, 8)]
    assert latency_ms_lut(lut, ops, bytes_total=0) == pytest.approx(0.3)
    # bandwidth term: alpha * bytes/1e6 us, added on top
    lut._overheads = {"bandwidth_alpha": 1.0}
    assert latency_ms_lut(lut, ops, bytes_total=1_000_000) == pytest.approx(0.301)


def test_latency_ms_lut_raises_on_missing_entry():
    lut = LatencyLUT("/nonexistent.yaml")
    with pytest.raises(KeyError):
        latency_ms_lut(lut, [("matmul", 1, 2, 3, 4)], bytes_total=0)


def test_lut_dump_and_reload_round_trips(tmp_path):
    path = tmp_path / "lut.yaml"
    lut = LatencyLUT(str(path))
    lut.record("matmul", 128, 64, 64, 4, 123.5)
    lut.dump()
    reloaded = LatencyLUT(str(path))
    assert reloaded.lookup("matmul", 128, 64, 64, 4) == pytest.approx(123.5)
    assert reloaded.lookup("matmul", 128, 64, 64, 8) is None


def test_cache_put_get_round_trips(tmp_path):
    db = str(tmp_path / "metrics.sqlite")
    cache = MetricsCache(db)
    assert cache.get("abc") is None
    cache.put("abc", (4, 8), Fitness(0.5, 1234, 2.5))
    got = cache.get("abc")
    assert got == Fitness(0.5, 1234, 2.5)
    # same key, new value overwrites
    cache.put("abc", (4, 8), Fitness(0.4, 1234, 2.5))
    assert cache.get("abc").val_loss == pytest.approx(0.4)


def _ev(bits, loss, mem, lat):
    ind = Individual(bits=tuple(bits), seed_tag="t", id=create_individual_id(tuple(bits)))
    return EvaluatedIndividual(ind, Fitness(loss, mem, lat), aux={})


def test_pareto_archive_dedupes_repeated_individuals():
    a = _ev((4,), 0.1, 100, 5.0)
    b = _ev((8,), 0.2, 50, 4.0)
    archive = pareto_archive([a, b, a, b])
    assert len(archive) == 2
    assert {e.ind.id for e in archive} == {a.ind.id, b.ind.id}


def test_pareto_archive_keeps_only_non_dominated():
    a = _ev((4,), 0.1, 100, 5.0)
    b = _ev((8,), 0.2, 50, 4.0)
    c = _ev((16,), 0.3, 200, 6.0)  # dominated by both
    assert [e.ind.id for e in pareto_archive([a, b, c])] == [a.ind.id, b.ind.id]


def test_frontier_save_load_slice_round_trips(tmp_path):
    path = tmp_path / "frontier.json"
    a = _ev((4,), 0.1, 100, 5.0)
    b = _ev((8,), 0.2, 50, 4.0)
    save_frontier(str(path), [a, b])
    rows = load_frontier(str(path))
    assert len(rows) == 2
    assert rows[0]["val_loss"] == pytest.approx(0.1)
    # cheapest config under a 60-byte memory cap is b
    assert slice_frontier(rows, max_mem=60, max_lat=None)["mem_bytes"] == 50
    assert slice_frontier(rows, max_mem=10, max_lat=None) is None


def test_surrogate_mock_loss_is_deterministic_and_rewards_precision():
    all16 = (16,) * 8
    all4 = (4,) * 8
    assert surrogate_mock_loss(all16) == surrogate_mock_loss(all16)
    assert surrogate_mock_loss(all16) < surrogate_mock_loss(all4)
    assert 0.3 < surrogate_mock_loss(all16) < 1.0


def test_mock_evaluator_evaluates_and_caches_without_torch(tmp_path):
    """The mock path exercises the real evaluation flow (memory, latency,
    cache) with no model and no torch import. If the evaluator imported
    torch at module level, this test module would fail to import at all
    in a torch-less environment."""
    layout = ParamLayout(
        units=["a", "b"],
        param_counts=[1000, 2000],
        excluded_mask=[False, False],
        unit_ops={"a": [], "b": []},
    )
    lut = LatencyLUT(str(tmp_path / "lut.yaml"))
    ev = Evaluator(
        "fake-model", layout, lut, 16, LatencyConfig.NONE, mock_backend=True
    )
    cache = MetricsCache(str(tmp_path / "m.sqlite"))
    ind = Individual(bits=(4, 16), seed_tag="t", id=create_individual_id((4, 16)))

    first = ev.evaluate_one(ind, cache, val_tokens=512, device=None)
    assert first.fit.val_loss == pytest.approx(surrogate_mock_loss((4, 16)))
    assert first.fit.mem_bytes == memory_bytes([1000, 2000], (4, 16))
    assert first.fit.latency_ms == 0.0  # NONE latency mode
    assert first.aux == {"cached": False}

    second = ev.evaluate_one(ind, cache, val_tokens=512, device=None)
    assert second.aux == {"cached": True}
    assert second.fit == first.fit

    many = ev.evaluate_many(
        [
            ind,
            Individual(bits=(16, 16), seed_tag="t", id=create_individual_id((16, 16))),
        ],
        cache,
        val_tokens=512,
    )
    assert len(many) == 2
    assert all(m.fit.val_loss > 0 for m in many)
