import yaml
from pathlib import Path

from src.group_ties import GroupTies
from src.data_models import Individual, Fitness
from src.utils import create_individual_id, BITS_CHOICES

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

def load_yaml(name: str):
    path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)

layer_spec = load_yaml("layer_spec.yaml")
quant_units = load_yaml("quant_units.yaml")
run_config  = load_yaml("run_config.yaml")

print("✅ Configs loaded:")
for name, cfg in [("layer_spec", layer_spec), ("quant_units", quant_units), ("run_config", run_config)]:
    print(f"• {name}: {list(cfg.keys())}")

# --- Build the layer names ---------------------------------------------------
unit_names = []
for unit in layer_spec["units"]:
    base = unit["name"]
    for layer in unit["layers"]:
        unit_names.append(f"{base}.{layer}")

gt = GroupTies(unit_names, CONFIG_DIR / "layer_spec.yaml")
print("\nParsed tie groups:", gt.tie_groups)
print("Excluded mask:", gt.excluded_mask)

# --- Dummy individual / fitness ---------------------------------------------
bits = tuple(4 for _ in unit_names)
ind_id = create_individual_id(bits)
ind = Individual(bits=bits, seed_tag="rand", id=ind_id)
fit = Fitness(val_loss=0.0, mem_bytes=0, latency=0.0)
print(f"\nCreated dummy individual {ind.id} with {len(bits)} layers.")

print("\n✅ Sanity check complete — your repo structure is working.")
