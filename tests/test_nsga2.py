"""NSGA-II operators: dominance, non-dominated sorting, crowding distance,
mutation, crossover, and tournament selection."""

import random

from evolutionary.core_evolutionary import (
    crossover,
    dominates,
    get_crowding_distances,
    mutate,
    partition_pareto_fronts,
    tournament_pick,
)
from evolutionary.data_models import EvaluatedIndividual, Fitness, Individual
from evolutionary.utils import BITS_CHOICES, create_individual_id


def _ev(bits, loss, mem, lat):
    ind = Individual(bits=tuple(bits), seed_tag="test", id=create_individual_id(tuple(bits)))
    return EvaluatedIndividual(ind, Fitness(loss, mem, lat), aux={})


def test_dominates_requires_all_no_worse_and_one_strictly_better():
    a = Fitness(0.2, 100, 5.0)
    assert dominates(a, Fitness(0.3, 200, 6.0))
    assert dominates(a, Fitness(0.2, 200, 6.0))  # strictly better on two
    assert not dominates(a, Fitness(0.2, 100, 5.0))  # identical: no strict win
    assert not dominates(a, Fitness(0.1, 200, 6.0))  # worse on loss
    assert not dominates(a, Fitness(0.2, 90, 6.0))  # worse on memory


def test_partition_pareto_fronts_peels_layers():
    # (loss, mem, lat): first two are non-dominated, third is dominated by both
    evals = [
        _ev((4,), 0.1, 100, 5.0),
        _ev((8,), 0.2, 50, 4.0),
        _ev((16,), 0.3, 200, 6.0),
    ]
    fronts = partition_pareto_fronts(evals)
    assert len(fronts) == 2
    assert len(fronts[0]) == 2
    assert len(fronts[1]) == 1


def test_crowding_distance_marks_extremes_infinite():
    evals = [
        _ev((4,), 0.1, 100, 5.0),
        _ev((8,), 0.2, 80, 4.0),
        _ev((16,), 0.3, 60, 3.0),
    ]
    dists = get_crowding_distances(evals)
    assert dists[0] == float("inf")  # best loss
    assert dists[2] == float("inf")  # best memory/latency
    assert 0 < dists[1] < float("inf")


def test_mutate_only_uses_valid_bit_widths_and_preserves_length():
    random.seed(0)
    bits = (4, 8, 16, 4, 8)
    for _ in range(50):
        out = mutate(bits, 0.5)
        assert len(out) == len(bits)
        assert all(b in BITS_CHOICES for b in out)


def test_mutate_with_zero_rate_is_identity():
    bits = (4, 8, 16, 4)
    assert mutate(bits, 0.0) == bits


def test_crossover_is_single_point_and_preserves_length():
    random.seed(1)
    a = (4,) * 6
    b = (16,) * 6
    for _ in range(50):
        out = crossover(a, b)
        assert len(out) == 6
        # single cut: some 4s then only 16s
        assert out == tuple(sorted(out))


def test_tournament_pick_prefers_better_front():
    evals = [
        _ev((4,), 0.1, 100, 5.0),  # front 0
        _ev((8,), 0.2, 50, 4.0),  # front 0
        _ev((16,), 0.9, 900, 9.0),  # dominated -> front 1
    ]
    fronts = partition_pareto_fronts(evals)
    random.seed(0)
    picks = [tournament_pick(evals, fronts) for _ in range(50)]
    assert all(p.bits != (16,) for p in picks)
