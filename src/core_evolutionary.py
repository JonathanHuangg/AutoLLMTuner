from __future__ import annotations
from typing import List, Tuple, Dict, Any
import random
from .data_models import Individual, EvaluatedIndividual, Fitness
from .utils import BITS_CHOICES, config_id_for_bits

"""
a dominates B if it is equal or better in all objectives
and strictly dominates in one
"""
def dominates(a: Fitness, b: Fitness):
    all = (a.val_loss <= b.val_loss and a.mem_bytes <= b.mem_bytes and a.latency_ms <= b.latency_ms)
    at_least_one = (a.val_loss < b.val_loss or a.mem_bytes < b.mem_bytes or a.latency_ms < b.latency_ms)
    return all and at_least_one

"""
Creates partitions/layers

- Partition 0 -> the best
- Partition 1 -> less good,
- etc

This algorithm starts by looping over tracking dominate relationships and holds a count of individuals that dominate it
if the num_domnated_by == 0, it is the pareto. 
This can be peeled back pareto layer by layer to return the partitions
"""
def partition_pareto_fronts(evals: List[EvaluatedIndividual]) -> List[List[EvaluatedIndividual]]:
    dominate_indices = {i: [] for i in range(len(evals))} # hashmap that tracks the ith individual and the ones it dominates
    num_domminated_by = [0] * len(evals) # how many individuals dominate candidate i
    fronts = List[List[int]]

    # o(n^2) search to compare every pair to get dominate list
    for first_candidate_idx in range(len(evals)):
        for sec_candidate_idx in range(len(evals)):
            if dominates(evals[first_candidate_idx].fit, evals[sec_candidate_idx].fit):
                dominate_indices[first_candidate_idx].append(sec_candidate_idx)
            elif dominates(evals[sec_candidate_idx].fit, evals[first_candidate_idx].fit):
                num_domminated_by[first_candidate_idx] += 1
        
        # if nothing dominates this idx, it is in the 0th pareto frontier
        if num_domminated_by[first_candidate_idx] == 0:
            fronts[0].append(first_candidate_idx)
    
    # dominate_indices is filled and we can now peel back the layers
    final_ret = List[List[EvaluatedIndividual]]
    i = 0
    while (i < len(fronts) and fronts[i]):
        final_ret.append([evals[idx] for idx in fronts[i]]) 
        next_pareto: List[int] = []
        for first_candidate_idx in fronts[i]:
            for sec_candidate_idx in dominate_indices[first_candidate_idx]:
                num_domminated_by[sec_candidate_idx] -= 1 # subtract the items that dominate in the layer before
                if num_domminated_by[sec_candidate_idx] == 0:
                    next_pareto.append(sec_candidate_idx)
        i += 1
        fronts.append(next_pareto)

    return final_ret

"""
Crowding distance is how far apart an individual is away from other indiviudal. A measurement of uniqueness
High number -> sparse region (good for diversity)
Low number -> around other items (less unique)
"""
def get_crowding_distances(front: List[EvaluatedIndividual]) -> list[float]:
    length = len(front)
    if length == 0:
        return []
    distances = [0.0] * length 

    def get_details(getter):
        return [getter(entry) for entry in front]

    def normalize(values):
        low, high = min(values), max(values)
        if low == high:
            return [0.0] * len(values)
        return [(v - low) / (high - low) for v in values]

    getters = [
        lambda e: e.fit.val_loss,
        lambda e: e.fit.mem_bytes,
        lambda e: e.fit.latency_ms
    ] 
    for getter in getters:
        values = get_details(getter)
        sorted_indices_in_front = sorted(range(length), key=lambda x: values[x])
        normalized = normalize(values[i] for i in sorted_indices_in_front)

        # the best and worst have infinite because they are the extremes
        distances[sorted_indices_in_front[0]] = distances[sorted_indices_in_front[-1]] = float('inf')

        # look at neighbors, look how far apart they are
        # this runs through len(getters) times (3) weighing each the same
        for other_points in range(1, length - 1):
            distances[sorted_indices_in_front[other_points]] += (normalized[other_points + 1] - normalized[other_points - 1])
    
    return distances

def mutate(bits: Tuple[int, ...], mutation_chance: float, choices: Tuple[int, ...] = BITS_CHOICES) -> Tuple[int, ...]:
    out = list(bits)
    for i in range(len(out)):
        if random.random() < mutation_chance: 

            # hardcoded, 30% for a global mutation, 70% local mutation
            # global -> jump to random bandwidth
            # local -> move up/down to neighbor bandwidth
            if random.random() < 0.7:
                cand = sorted(choices + (out[i],)) # create valid bitwidths
                j = cand.index(out[i])
                j_new = max(0, min(len(cand) - 1, j + random.choice([-1, 1]))) # step up or down
                out[i] = min(choices, key=lambda c: abs(c - cand[j_new])) # pick the nearest valid choice
            else:
                out[i] = random.choice(choices)

    return tuple(out)

def crossover(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    assert len(a) == len(b)
    if (len(a)) == 1:
        return a 

    cut_idx = random.randint(1, len(a) - 1)
    return tuple(a[:cut_idx] + b[cut_idx:])

def build_individual(bits: Tuple[int, ...], seed_tag: str, group_version: str = "v1") -> Individual:
    return Individual(bits=bits, seed_tag=seed_tag, id=config_id_for_bits(bits, group_version))

def tournament_pick(evals: List[EvaluatedIndividual], fronts: List[List[EvaluatedIndividual]]) -> Individual:
    a, b = random.sample(evals, 2) # pick 2 at random
    def rank_of(x: EvaluatedIndividual): # is it in front 0, front 1, etc
        for idx, front in enumerate(fronts, start=1):
            if x in front:
                return idx 
        return 10**9
    
    rank_a, rank_b = rank_of(a), rank_of(b)

    # rank 0 is better than rank 1
    if rank_a < rank_b:
        return a.ind 
    elif rank_a > rank_b:
        return b.ind 
    
    # use val_loss as a tie breaker
    return a.ind if a.fit.val_loss < b.fit.val_loss else b.ind
