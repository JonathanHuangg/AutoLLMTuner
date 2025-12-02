#!/usr/bin/env python
"""
Generate Trinity-Mini layer specifications with shapes.
Trinity-Mini architecture:
- 32 layers
- hidden_size: 2048
- intermediate_size: 6144 (shared experts)
- expert_intermediate_size: 1024 (individual experts)
- num_experts: 128
"""

from pathlib import Path
import yaml

M_CANONICAL = 128

# Trinity-Mini architecture specs
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
EXPERT_INTERMEDIATE_SIZE = 1024
NUM_EXPERTS = 128
NUM_LAYERS = 32

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRINITY_UNITS_PATH = CONFIGS_DIR / "trinity_units.yaml"
OUT_LAYER_SPEC_PATH = CONFIGS_DIR / "trinity_layer_spec.yaml"


def infer_shape_for_unit(unit: str) -> tuple[int, int]:
    """Infer (out_features, in_features) for a unit"""
    # Attention projections
    if ".self_attn.q_proj" in unit:
        return HIDDEN_SIZE, HIDDEN_SIZE
    if ".self_attn.k_proj" in unit:
        return HIDDEN_SIZE, HIDDEN_SIZE
    if ".self_attn.v_proj" in unit:
        return HIDDEN_SIZE, HIDDEN_SIZE
    if ".self_attn.o_proj" in unit:
        return HIDDEN_SIZE, HIDDEN_SIZE

    # MoE gate (router)
    if ".block_sparse_moe.gate" in unit:
        return NUM_EXPERTS, HIDDEN_SIZE

    # Shared expert projections
    if ".block_sparse_moe.shared_experts.gate_proj" in unit:
        return INTERMEDIATE_SIZE, HIDDEN_SIZE
    if ".block_sparse_moe.shared_experts.up_proj" in unit:
        return INTERMEDIATE_SIZE, HIDDEN_SIZE
    if ".block_sparse_moe.shared_experts.down_proj" in unit:
        return HIDDEN_SIZE, INTERMEDIATE_SIZE

    # Individual expert projections (not included in this simplified version)
    if ".block_sparse_moe.experts." in unit:
        if ".gate_proj" in unit:
            return EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE
        if ".up_proj" in unit:
            return EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE
        if ".down_proj" in unit:
            return HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE

    raise ValueError(f"Unknown unit type: {unit}")


def main():
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Reading units from {TRINITY_UNITS_PATH}")
    print(f"Writing spec to    {OUT_LAYER_SPEC_PATH}")

    # Load units
    with TRINITY_UNITS_PATH.open("r") as f:
        units_yaml = yaml.safe_load(f)

    unit_paths = [u["name"] for u in units_yaml["units"]]

    print(f"Building layer spec for {len(unit_paths)} units...")

    layer_spec = {"units": {}}

    for unit in unit_paths:
        # Ensure full path is prefixed with 'model.'
        if not unit.startswith("model."):
            module_path = f"model.{unit}"
        else:
            module_path = unit

        out_features, in_features = infer_shape_for_unit(unit)

        layer_spec["units"][module_path] = {
            "ops": [
                {
                    "op_type": "matmul",
                    "m": int(M_CANONICAL),
                    "n": int(out_features),
                    "k": int(in_features),
                }
            ]
        }

    OUT_LAYER_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LAYER_SPEC_PATH.open("w") as f:
        yaml.safe_dump(layer_spec, f, sort_keys=False)

    print(f"Wrote layer spec to {OUT_LAYER_SPEC_PATH}")
    print(f"Generated specs for {len(layer_spec['units'])} units")


if __name__ == "__main__":
    main()
