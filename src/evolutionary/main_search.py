from __future__ import annotations
import argparse, os, json, random
from typing import List, Tuple
from tqdm import tqdm
from config import GlobalConfig, load_config
from utils import set_seed, BITS_CHOICES, create_individual_id
from data_models import Individual, EvaluatedIndividual, ParamLayout
from group_ties import GroupTies
from lut import LatencyLUT
from cache import MetricsCache
from frontier import pareto_archive, save_frontier
from evaluator import Evaluator
from logging_utils import setup_logger
from core_evolutionary import partition_pareto_fronts, get_crowding_distances, mutate, crossover, build_individual, tournament_pick
import yaml

# build a ParamLayout from the yAML 
def _load_units_and_params(config: GlobalConfig, units_path: str) -> ParamLayout:
    # unit names
    with open(units_path, "r") as f:
        units_yaml = yaml.safe_load(f)
    unit_names: List[str] = units_yaml["units"]

    # param_counts
    if config.model.param_counts_file:
        with open(config.model.param_counts_file, "r") as f:
            pc = json.load(f)
        param_counts = [int(pc.get(n, 100_000_000)) for n in unit_names]
    else:
        param_counts = [100_000_000 for _ in unit_names]
    
    ties = GroupTies(unit_names, units_path)
    excluded_mask = ties.excluded_mask

    # layer_spec
    layer_spec_path = config.units.spec_file
    with open(layer_spec_path, "r") as f:
        layer_spec_yaml = yaml.safe_load(f)
     
    units_ops = Dict[str, List[OpSpec]] = {}
    for u in unit_names:
        ops_from_yaml = layer_spec_yaml["units"][u]["ops"]
        ops: List[OpSpec] = []
        for o in ops_from_yaml:
            ops.append(OpSpec(
                op_type=o["op_type"],    # or "op_type" depending on your schema
                m=int(o["m"]),
                n=int(o["n"]),
                k=int(o["k"]),
            ))
        unit_ops[u] = ops
    return ParamLayout(
        units=unit_names, 
        param_counts=param_counts, 
        excluded_mask=excluded_mask
        unit_ops=unit_ops
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    out_directory = os.path.join(config.io.out_dir, args.run_id)
    os.makedirs(out_directory, exist_ok=True)
    logger = setup_logger()

    set_seed(config.run.seed)
    layout = _load_units_and_params(config, config.units.spec_file)
    ties = GroupTies(layout.units, config.units.spec_file)
    lut = LatencyLUT(config.hardware.lut_file)
    cache = MetricsCache(os.path.join(out_directory, "metrics.sqlite"))
    evaluator = Evaluator(layout, lut, default_fp_bits=config.model.default_fp_bits,
                          measure_latency=config.evaluation.measure_latency)

    num_layout_units = len(layout.units)

    # create the initial population below

    # updates the individual to make sure it is valid
    def update_individual(bits, tag):
        b = ties.repair(bits)
        return Individual(bits=b, seed_tag=tag, id=create_individual_id(b, group_version="v1"))
    population: List[Individual] = []

    # seed with a base 8-bit precision across. Anchor point
    population.append(update_individual(tuple(8 for _ in range(num_layout_units)), "uniform8"))

    # randomly seed the other individuals
    while len(population) < config.run.population_size:
        bits = tuple(random.choice(BITS_CHOICES) for _ in range(num_layout_units))
        population.append(update_individual(bits, "randmix"))
    
    best_archive: List[EvaluatedIndividual] = []

    for epoch in range(config.run.generations):

        # in the yaml, it is 2 values. Start and end. First half is cheap evolution, second is more expensive. 
        validation_tokens = config.evaluation.val_tokens_schedule[0] if epoch < config.run.generations // 2 else config.evaluation.val_tokens_schedule[1]
        logger.info(f"Gen {epoch+1}/{config.run.generations} | pop={len(population)} | tokens={validation_tokens}")
        evaluated = evaluator.evaluate_many(population, cache, validation_tokens, threads=config.run.threads)

        # update archive
        merged = best_archive + evaluated
        tmp = pareto_archive(merged) # do this later
        best_archive = tmp

        fronts = partition_pareto_fronts(evaluated)

        # keep elites
        keep = max(1, int(config.run.elitism_frac * config.run.population_size))
        elite: List[Individual] = []
        for front in fronts:
            cd = get_crowding_distances(front)
            order = sorted(range(len(front)), key=lambda i: (-float(cd[i]), front[i].fit.val_loss))
            for idx in order:
                elite.append(front[idx].ind)
                if (len(elite)) >= keep:
                    break 
            if len(elite) >= keep:
                break
        
        # breed next population
        next_population: List[Individual] = elite[:]
        while len(next_population) < config.run.population_size:
            p1, p2 = tournament_pick(evaluated, fronts), tournament_pick(evaluated, fronts)
            child_bits = p1.bits 
            if random.random() < config.run.crossover_probability:
                child_bits = crossover(p1.bits, p2.bits)
            
            # do mutation
            child_bits = mutate(child_bits, config.run.mutation_rate)
            child_bits = ties.repair(child_bits)
            next_population.append(Individual(bits=child_bits, seed_tag="xover_mut", id=create_individual_id(child_bits)))
        
        population = next_population
        frontier_path = os.path.join(out_directory, "frontier.json")
        save_frontier(frontier_path, best_archive)

        best_loss = min(e.fit.val_loss for e in best_archive)
        logger.info(f"Archive size={len(best_archive)} | best loss={best_loss:.4f}")
    
    logger.info(f"Done. Frontier at {os.path.join(out_directory, 'frontier.json')}")

if __name__ == "__main__":
    main()



    



