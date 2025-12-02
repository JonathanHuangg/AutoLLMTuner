#!/usr/bin/env python

import os
from pathlib import Path

import yaml
from transformers import AutoConfig

# Optional: point caches to scratch (only small config.json downloaded)
os.environ["HF_HOME"] = "/storage/ice1/2/3/jhuang691/scratch/hfcache"
os.environ["HF_DATASETS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfdatasets"
os.environ["TRANSFORMERS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfmodels"

MODEL_NAME = "arcee-ai/Trinity-Mini"
M_CANONICAL = 128  # matches your hardware.yaml keys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # yaml_gathering -> src -> AutoLLMTuner

CONFIGS_DIR = PROJECT_ROOT / "configs"
TRINITY_UNITS_PATH = CONFIGS_DIR / "trinity_units.yaml"
OUT_LAYER_SPEC_PATH = CONFIGS_DIR / "trinity_layer_spec.yaml"


def infer_shape_for_unit(
    unit: str,
    hidden_size: int,
    intermediate_size: int,
    expert_intermediate_size: int,
    num_experts: int,
) -> tuple[int, int]:
    """
    Infer (out_features, in_features) for a given logical unit name
    for Trinity-Mini's MoE architecture.
    """
    # Attention projections (standard)
    if ".self_attn.q_proj" in unit:
        return hidden_size, hidden_size
    if ".self_attn.k_proj" in unit:
        return hidden_size, hidden_size
    if ".self_attn.v_proj" in unit:
        return hidden_size, hidden_size
    if ".self_attn.o_proj" in unit:
        return hidden_size, hidden_size

    # MoE gate (router) - maps hidden to num_experts
    if ".block_sparse_moe.gate" in unit:
        return num_experts, hidden_size

    # Shared expert projections
    if ".block_sparse_moe.shared_experts.gate_proj" in unit:
        return intermediate_size, hidden_size
    if ".block_sparse_moe.shared_experts.up_proj" in unit:
        return intermediate_size, hidden_size
    if ".block_sparse_moe.shared_experts.down_proj" in unit:
        return hidden_size, intermediate_size

    # Individual expert projections
    if ".block_sparse_moe.experts." in unit:
        if ".gate_proj" in unit:
            return expert_intermediate_size, hidden_size
        if ".up_proj" in unit:
            return expert_intermediate_size, hidden_size
        if ".down_proj" in unit:
            return hidden_size, expert_intermediate_size

    raise ValueError(f"Don't know how to infer shape for unit '{unit}'")


def main():
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Reading units from {TRINITY_UNITS_PATH}")
    print(f"Writing spec to    {OUT_LAYER_SPEC_PATH}")

    # 1) Load Trinity units produced by make_trinity_unit_names.py
    with TRINITY_UNITS_PATH.open("r") as f:
        units_yaml = yaml.safe_load(f)

    if isinstance(units_yaml["units"][0], str):
        unit_paths = units_yaml["units"]
    else:
        unit_paths = [u["name"] for u in units_yaml["units"]]

    # 2) Load only the config (tiny), not the full model weights
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hidden_size = cfg.hidden_size
    intermediate_size = getattr(cfg, "intermediate_size", hidden_size * 4)
    expert_intermediate_size = getattr(cfg, "moe_intermediate_size", 1024)
    num_experts = getattr(cfg, "num_local_experts", 128)

    print(f"hidden_size={hidden_size}")
    print(f"intermediate_size={intermediate_size}")
    print(f"expert_intermediate_size={expert_intermediate_size}")
    print(f"num_experts={num_experts}")
    print(f"Building layer spec for {len(unit_paths)} units...")

    layer_spec = {"units": {}}

    for unit in unit_paths:
        # Ensure full path is prefixed with 'model.' if needed
        if not unit.startswith("model."):
            module_path = f"model.{unit}"
        else:
            module_path = unit

        out_features, in_features = infer_shape_for_unit(
            unit, hidden_size, intermediate_size, expert_intermediate_size, num_experts
        )

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

        print(
            f"{module_path}: matmul(m={M_CANONICAL}, n={out_features}, k={in_features})"
        )

    OUT_LAYER_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LAYER_SPEC_PATH.open("w") as f:
        yaml.safe_dump(layer_spec, f, sort_keys=False)

    print(f"Wrote layer spec to {OUT_LAYER_SPEC_PATH}")


if __name__ == "__main__":
    main()
