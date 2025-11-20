#!/usr/bin/env python

import os
from pathlib import Path

import yaml
from transformers import AutoConfig

# Optional: point caches to scratch (only small config.json downloaded)
os.environ["HF_HOME"] = "/storage/ice1/2/3/jhuang691/scratch/hfcache"
os.environ["HF_DATASETS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfdatasets"
os.environ["TRANSFORMERS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfmodels"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
M_CANONICAL = 128  # matches your hardware.yaml keys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # yaml_gathering -> src -> AutoLLMTuner

CONFIGS_DIR = PROJECT_ROOT / "configs"
QUANT_UNITS_PATH = CONFIGS_DIR / "quant_units.yaml"
OUT_LAYER_SPEC_PATH = CONFIGS_DIR / "qwen2.5_layer_spec.yaml"


def infer_shape_for_unit(unit: str, hidden_size: int, intermediate_size: int) -> tuple[int, int]:
    """
    Infer (out_features, in_features) for a given logical unit name
    like 'layers.0.self_attn.q_proj' using Qwen2.5 conventions.
    """
    if ".self_attn.q_proj" in unit:
        return hidden_size, hidden_size
    if ".self_attn.k_proj" in unit:
        return hidden_size, hidden_size
    if ".self_attn.v_proj" in unit:
        return hidden_size, hidden_size
    if ".self_attn.o_proj" in unit:
        return hidden_size, hidden_size

    if ".mlp.gate_proj" in unit:
        return intermediate_size, hidden_size
    if ".mlp.up_proj" in unit:
        return intermediate_size, hidden_size
    if ".mlp.down_proj" in unit:
        return hidden_size, intermediate_size

    raise ValueError(f"Don't know how to infer shape for unit '{unit}'")


def main():
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Reading units from {QUANT_UNITS_PATH}")
    print(f"Writing spec to    {OUT_LAYER_SPEC_PATH}")

    # 1) Load quant units produced by unit_names.py
    with QUANT_UNITS_PATH.open("r") as f:
        units_yaml = yaml.safe_load(f)

    if isinstance(units_yaml["units"][0], str):
        unit_paths = units_yaml["units"]
    else:
        unit_paths = [u["name"] for u in units_yaml["units"]]

    # 2) Load only the config (tiny), not the full model weights
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hidden_size = cfg.hidden_size
    intermediate_size = getattr(cfg, "intermediate_size", hidden_size * 4)

    print(f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    print(f"Building layer spec for {len(unit_paths)} units...")

    layer_spec = {"units": {}}

    for unit in unit_paths:
        # Ensure full path is prefixed with 'model.' if needed
        if not unit.startswith("model."):
            module_path = f"model.{unit}"
        else:
            module_path = unit

        out_features, in_features = infer_shape_for_unit(
            unit, hidden_size, intermediate_size
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
