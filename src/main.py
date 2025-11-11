import random
import yaml
from pathlib import Path
from statistics import mean

from src.utils import create_individual_id, BITS_CHOICES, set_seed
from src.data_models import Individual, Fitness, EvaluatedIndividual
from src.group_ties import GroupTies
from src.eval import EvalConfig, eval_perplexity_latency_mem

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

def load_yaml(name: str):
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)

layer_spec = load_yaml("layer_spec.yaml")
run_cfg    = load_yaml("run_config.yaml")

# Default fallback hyperparameters
pop_size = run_cfg.get("population_size", 6)
generations = run_cfg.get("generations", 1)
elite_ratio = run_cfg.get("elite_ratio", 0.2)
mutation_rate = run_cfg.get("mutation_rate", 0.3)
random_seed = run_cfg.get("seed", 42)
set_seed(random_seed)

# -------------------------------------------------------------------
# Build full layer name list
unit_names = []
for unit in layer_spec["units"]:
    base = unit["name"]
    for layer in unit["layers"]:
        unit_names.append(f"{base}.{layer}")

gt = GroupTies(unit_names, CONFIG_DIR / "layer_spec.yaml")
num_layers = len(unit_names)

print(f"\nQuantization units: {num_layers}")
print(f"Bits choices: {BITS_CHOICES}")

# -------------------------------------------------------------------
# Evolutionary operators
def random_individual():
    bits = tuple(random.choice(BITS_CHOICES) for _ in range(num_layers))
    return Individual(bits=bits, seed_tag="rand", id=create_individual_id(bits))

def mutate(bits):
    bits = list(bits)
    for i in range(num_layers):
        if random.random() < mutation_rate:
            bits[i] = random.choice(BITS_CHOICES)
    return tuple(bits)

def evaluate(ind: Individual) -> EvaluatedIndividual:
    # TODO: later: apply per-layer quantization according to ind.bits
    cfg = EvalConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_samples=128,     # keep it quick
        max_seq_len=128,
        batch_size=8,
        fp16=True,
        device="cuda"
    )
    metrics = eval_perplexity_latency_mem(cfg)
    fit = Fitness(
        val_loss=metrics["val_loss"],
        mem_bytes=int(metrics["vram_bytes"]),
        latency=metrics["lat_ms_per_tok"]
    )
    return EvaluatedIndividual(ind=ind, fit=fit, aux=metrics)

# -------------------------------------------------------------------
# Initialize population
population = [random_individual() for _ in range(pop_size)]
evaluated = [evaluate(ind) for ind in population]

print("\nInitial population:")
for ev in evaluated:
    print(f"{ev.ind.id[:6]} | avg_bits={mean(ev.ind.bits):.1f} | "
          f"loss={ev.fit.val_loss:.3f} | mem={ev.fit.mem_bytes/1e3:.1f}kB | "
          f"lat={ev.fit.latency:.2f}ms")

# -------------------------------------------------------------------
# Selection (elitism)
evaluated.sort(key=lambda e: e.fit.val_loss)
elite_count = max(1, int(pop_size * elite_ratio))
elites = evaluated[:elite_count]

# -------------------------------------------------------------------
# Mutation to form next generation
next_gen = []
for e in elites:
    next_bits = mutate(e.ind.bits)
    next_gen.append(Individual(bits=next_bits, seed_tag="mut", id=create_individual_id(next_bits)))

print("\nNext generation preview:")
for ind in next_gen:
    print(f"{ind.id[:6]} | avg_bits={mean(ind.bits):.1f}")

print("\n✅ 1-generation evolutionary cycle complete.")
