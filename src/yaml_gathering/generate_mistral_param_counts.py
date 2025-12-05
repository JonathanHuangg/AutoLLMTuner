#!/usr/bin/env python
"""
Generate parameter counts for Mistral-7B-Instruct units.
Architecture:
- hidden_size:         4096
- intermediate_size:  14336
No experts, no MoE.
"""

from pathlib import Path
import json
import yaml
from transformers import AutoConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

UNITS_PATH = CONFIGS_DIR / "mistral_units.yaml"
OUT_PARAM_COUNTS_PATH = CONFIGS_DIR / "mistral_param_counts.json"


def count_params_for_unit(unit: str, hidden: int, intermediate: int) -> int:
    """Return parameter count for each projection (weights only)."""

    # ---- Attention projections ----
    if any(x in unit for x in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]):
        # 4096 × 4096
        return hidden * hidden

    # ---- MLP projections ----
    if ".mlp.gate_proj" in unit or ".mlp.up_proj" in unit:
        # 14336 × 4096
        return intermediate * hidden

    if ".mlp.down_proj" in unit:
        # 4096 × 14336
        return hidden * intermediate

    raise ValueError(f"Unknown unit type: {unit}")


def main():
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Reading units from {UNITS_PATH}")
    print(f"Writing counts to  {OUT_PARAM_COUNTS_PATH}")

    # Load unit names
    with UNITS_PATH.open("r") as f:
        units_yaml = yaml.safe_load(f)

    unit_paths = [u["name"] for u in units_yaml["units"]]
    print(f"Found {len(unit_paths)} units")

    # Load architecture config from HF (lightweight, does not load weights)
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    hidden = cfg.hidden_size          # 4096
    intermediate = cfg.intermediate_size  # 14336

    print(f"Mistral hidden={hidden}, intermediate={intermediate}")

    param_counts = {}

    for unit in unit_paths:

        # Add "model." prefix for consistency with layer_spec.yaml
        module_path = f"model.{unit}"

        count = count_params_for_unit(unit, hidden, intermediate)
        param_counts[module_path] = count

    # Write JSON
    OUT_PARAM_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PARAM_COUNTS_PATH.open("w") as f:
        json.dump(param_counts, f, indent=2)

    total_params = sum(param_counts.values())

    print(f"Wrote {OUT_PARAM_COUNTS_PATH}")
    print(f"Total quantizable parameters = {total_params:,} ({total_params/1e9:.2f}B)")


if __name__ == "__main__":
    main()
