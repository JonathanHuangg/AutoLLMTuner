#!/usr/bin/env python
"""
Generate layer_spec.yaml for Mistral-7B-Instruct.

Architecture:
- hidden_size:        4096
- intermediate_size: 14336
- attention proj:    (4096, 4096)
- mlp gate/up:       (14336, 4096)
- mlp down:          (4096, 14336)
"""

from pathlib import Path
import yaml
from transformers import AutoConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
M_CANONICAL = 128  # matches your hardware.yaml

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

UNITS_PATH = CONFIGS_DIR / "mistral_units.yaml"
OUT_LAYER_SPEC_PATH = CONFIGS_DIR / "mistral_layer_spec.yaml"


def infer_shape(unit: str, hidden_size: int, inter_size: int):
    """Return (out_features, in_features)"""

    # Attention
    if ".self_attn.q_proj" in unit: return hidden_size, hidden_size
    if ".self_attn.k_proj" in unit: return hidden_size, hidden_size
    if ".self_attn.v_proj" in unit: return hidden_size, hidden_size
    if ".self_attn.o_proj" in unit: return hidden_size, hidden_size

    # MLP
    if ".mlp.gate_proj" in unit: return inter_size, hidden_size
    if ".mlp.up_proj" in unit:   return inter_size, hidden_size
    if ".mlp.down_proj" in unit: return hidden_size, inter_size

    raise ValueError(f"Unknown unit: {unit}")


def main():
    print(f"Reading unit list from {UNITS_PATH}")

    with UNITS_PATH.open("r") as f:
        units_yaml = yaml.safe_load(f)

    unit_paths = [u["name"] for u in units_yaml["units"]]

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    hidden = cfg.hidden_size           # 4096
    inter = cfg.intermediate_size      # 14336
    layers = cfg.num_hidden_layers     # 32

    print(f"Mistral config: hidden={hidden}, intermediate={inter}, layers={layers}")
    print(f"Building layer spec for {len(unit_paths)} units...")

    spec = {"units": {}}

    for unit in unit_paths:

        module_path = f"model.{unit}"

        out_f, in_f = infer_shape(unit, hidden, inter)

        spec["units"][module_path] = {
            "ops": [
                {
                    "op_type": "matmul",
                    "m": M_CANONICAL,
                    "n": int(out_f),
                    "k": int(in_f),
                }
            ]
        }

        print(f"{module_path}: matmul(m={M_CANONICAL}, n={out_f}, k={in_f})")

    OUT_LAYER_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LAYER_SPEC_PATH.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    print(f"\nWrote layer spec to {OUT_LAYER_SPEC_PATH}")
    print(f"Total units: {len(spec['units'])}")


if __name__ == "__main__":
    main()
