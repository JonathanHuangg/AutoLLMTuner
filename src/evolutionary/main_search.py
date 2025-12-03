from __future__ import annotations
import argparse
import os
import json
import random
from typing import List, Tuple, Dict

from tqdm import tqdm  # optional; currently unused but handy for progress bars

from .config import GlobalConfig, load_config
from .utils import set_seed, BITS_CHOICES, create_individual_id
from .data_models import Individual, EvaluatedIndividual, ParamLayout, OpSpec
from .group_ties import GroupTies
from .lut import LatencyLUT
from .cache import MetricsCache
from .frontier import pareto_archive, save_frontier
from .evaluator import Evaluator
from .logging_utils import setup_logger
from .core_evolutionary import (
    partition_pareto_fronts,
    get_crowding_distances,
    mutate,
    crossover,
    build_individual,
    tournament_pick,
)
import yaml
import multiprocessing as mp

# Build a ParamLayout from YAML
def _load_units_and_params(config: GlobalConfig) -> ParamLayout:
    # 1. Units list from quant_units.yaml
    with open(config.units.spec_file, "r") as f:
        units_yaml = yaml.safe_load(f)

    unit_entries = units_yaml["units"]            # if it's list of {name: ...}
    unit_names = [u["name"] for u in unit_entries]

    # 2. Param counts
    if config.model.param_counts_file:
        with open(config.model.param_counts_file, "r") as f:
            pc = json.load(f)
        param_counts = [int(pc.get(n, 100_000_000)) for n in unit_names]
    else:
        param_counts = [100_000_000 for _ in unit_names]

    # 3. GroupTies / excluded mask (if GroupTies uses units spec)
    ties = GroupTies(unit_names, config.units.spec_file)
    excluded_mask = ties.excluded_mask

    # 4. Per-unit op shapes from layer_spec_file (ops spec)
    with open(config.units.layer_spec_file, "r") as f:
        layer_spec_yaml = yaml.safe_load(f)

    unit_ops: Dict[str, List[OpSpec]] = {}
    for u in unit_names:
        entry = layer_spec_yaml["units"].get(u)
        if entry is None:
            raise KeyError(f"Unit '{u}' not found in layer spec {config.units.layer_spec_file}")
        ops_from_yaml = entry.get("ops", [])

        ops: List[OpSpec] = []
        for o in ops_from_yaml:
            ops.append(OpSpec(
                op_type=o["op_type"],
                m=int(o["m"]),
                n=int(o["n"]),
                k=int(o["k"]),
            ))
        unit_ops[u] = ops

    return ParamLayout(
        units=unit_names,
        param_counts=param_counts,
        excluded_mask=excluded_mask,
        unit_ops=unit_ops,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    mp.set_start_method("spawn")

    config = load_config(args.config)
    out_directory = os.path.join(config.io.out_dir, args.run_id)
    os.makedirs(out_directory, exist_ok=True)
    logger = setup_logger()

    set_seed(config.run.seed)

    layout = _load_units_and_params(config)

    ties = GroupTies(layout.units, config.units.spec_file)
    lut = LatencyLUT(config.hardware.lut_file)
    cache = MetricsCache(os.path.join(out_directory, "metrics.sqlite"))
    evaluator = Evaluator(
        config.model.name,
        layout,
        lut,
        default_fp_bits=config.model.default_fp_bits,
        measure_latency=config.evaluation.measure_latency,
    )

    num_layout_units = len(layout.units)

    # Helper to update an Individual and ensure it's group-ties-consistent.
    def update_individual(bits: Tuple[int, ...], tag: str) -> Individual:
        b = ties.repair(bits)
        return Individual(
            bits=b,
            seed_tag=tag,
            id=create_individual_id(b, group_version="v1"),
        )

    population: List[Individual] = []

    # Seed with a base 8-bit precision across all units as an anchor point.
    population.append(
        update_individual(tuple(8 for _ in range(num_layout_units)), "uniform8")
    )

    # Randomly seed the rest of the initial population.
    while len(population) < config.run.population_size:
        bits = tuple(random.choice(BITS_CHOICES) for _ in range(num_layout_units))
        population.append(update_individual(bits, "randmix"))

    best_archive: List[EvaluatedIndividual] = []

    for epoch in tqdm(range(config.run.generations), desc="Generations", unit="gen"):        # In the YAML, val_tokens_schedule is [cheap_tokens, expensive_tokens].
        logger.info(f"[Epoch {epoch}] Starting evaluation phase")

        # # First half of generations uses the cheaper eval, second half uses the larger.
        # mid_point = config.run.generations // 2
        # if epoch < mid_point:
        #     validation_tokens = config.evaluation.val_tokens_schedule[0]
        # else:
        #     validation_tokens = config.evaluation.val_tokens_schedule[1]

        # logger.debug(f"[Epoch {epoch}] validation_tokens = {validation_tokens}")

        logger.debug(f"[Epoch {epoch}] Evaluating {len(population)} individuals...")
        evaluated = evaluator.evaluate_many(population, cache)
        logger.debug(f"[Epoch {epoch}] Evaluation complete")

        # Update archive
        logger.debug(f"[Epoch {epoch}] Updating Pareto archive...")
        merged = best_archive + evaluated
        best_archive = pareto_archive(merged)
        logger.debug(f"[Epoch {epoch}] Archive size now {len(best_archive)}")

        logger.debug(f"[Epoch {epoch}] Partitioning fronts...")
        # NSGA-II style selection: non-dominated fronts + crowding distance
        fronts = partition_pareto_fronts(evaluated)
        logger.debug(f"[Epoch {epoch}] {len(fronts)} fronts found")

        # Keep elites
        logger.debug(f"[Epoch {epoch}] Selecting elites...")
        keep = max(1, int(config.run.elitism_frac * config.run.population_size))
        logger.debug(f"[Epoch {epoch}] Elite target count = {keep}")

        elite: List[Individual] = []
        for front in fronts:
            cd = get_crowding_distances(front)
            order = sorted(
                range(len(front)),
                key=lambda i: (-float(cd[i]), front[i].fit.val_loss),
            )
            for idx in order:
                elite.append(front[idx].ind)
                if len(elite) >= keep:
                    break
            if len(elite) >= keep:
                break
        logger.debug(f"[Epoch {epoch}] Elite selection complete")

        logger.debug(f"[Epoch {epoch}] Breeding next population...")
        # Breed next population
        next_population: List[Individual] = elite[:]
        while len(next_population) < config.run.population_size:
            p1 = tournament_pick(evaluated, fronts)
            p2 = tournament_pick(evaluated, fronts)

            child_bits = p1.bits
            if random.random() < config.run.crossover_probability:
                child_bits = crossover(p1.bits, p2.bits)

            # Mutation + repair
            child_bits = mutate(child_bits, config.run.mutation_rate)
            child_bits = ties.repair(child_bits)
            next_population.append(
                Individual(
                    bits=child_bits,
                    seed_tag="xover_mut",
                    id=create_individual_id(child_bits),
                )
            )

        population = next_population
        logger.debug(f"[Epoch {epoch}] Next population size = {len(next_population)}")

        frontier_path = os.path.join(out_directory, f"frontier_{epoch}.json")
        save_frontier(frontier_path, best_archive)
        logger.debug(f"[Epoch {epoch}] Frontier saved to {frontier_path}")

        best_loss = min(e.fit.val_loss for e in best_archive)
        logger.info(f"[Epoch {epoch}] Archive={len(best_archive)} | best_loss={best_loss:.4f}")

    logger.info(
        f"Done. Frontier at {os.path.join(out_directory, 'frontier.json')}"
    )


if __name__ == "__main__":
    main()
