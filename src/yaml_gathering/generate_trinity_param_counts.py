#!/usr/bin/env python
"""
Generate parameter counts for Trinity-Mini units.
Trinity-Mini architecture:
- hidden_size: 2048
- intermediate_size: 6144
- expert_intermediate_size: 1024
- num_experts: 128
"""

from pathlib import Path
import json
import yaml

# Trinity-Mini architecture specs
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
EXPERT_INTERMEDIATE_SIZE = 1024
NUM_EXPERTS = 128

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRINITY_UNITS_PATH = CONFIGS_DIR / "trinity_units.yaml"
OUT_PARAM_COUNTS_PATH = CONFIGS_DIR / "trinity_param_counts.json"


def count_params_for_unit(unit: str) -> int:
    """Calculate parameter count for a unit (weight only, no bias)"""
    # Attention projections (all are hidden_size x hidden_size)
    if any(proj in unit for proj in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]):
        return HIDDEN_SIZE * HIDDEN_SIZE

    # MoE gate (router): hidden_size -> num_experts
    if ".block_sparse_moe.gate" in unit:
        return HIDDEN_SIZE * NUM_EXPERTS

    # Shared expert projections
    if ".shared_experts.gate_proj" in unit or ".shared_experts.up_proj" in unit:
        return HIDDEN_SIZE * INTERMEDIATE_SIZE
    if ".shared_experts.down_proj" in unit:
        return INTERMEDIATE_SIZE * HIDDEN_SIZE

    # Individual expert projections
    if ".block_sparse_moe.experts." in unit:
        if ".gate_proj" in unit or ".up_proj" in unit:
            return HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE
        if ".down_proj" in unit:
            return EXPERT_INTERMEDIATE_SIZE * HIDDEN_SIZE

    raise ValueError(f"Unknown unit type: {unit}")


def main():
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Reading units from {TRINITY_UNITS_PATH}")
    print(f"Writing counts to  {OUT_PARAM_COUNTS_PATH}")

    # Load units
    with TRINITY_UNITS_PATH.open("r") as f:
        units_yaml = yaml.safe_load(f)

    unit_paths = [u["name"] for u in units_yaml["units"]]
    print(f"Computing parameter counts for {len(unit_paths)} units...")

    param_counts = {}
    for unit in unit_paths:
        # Use full path with 'model.' prefix
        if not unit.startswith("model."):
            module_path = f"model.{unit}"
        else:
            module_path = unit

        count = count_params_for_unit(unit)
        param_counts[module_path] = count

    OUT_PARAM_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PARAM_COUNTS_PATH.open("w") as f:
        json.dump(param_counts, f, indent=2)

    total_params = sum(param_counts.values())
    print(f"Wrote parameter counts to {OUT_PARAM_COUNTS_PATH}")
    print(f"Total quantizable parameters: {total_params:,} ({total_params / 1e9:.2f}B)")


if __name__ == "__main__":
    main()
