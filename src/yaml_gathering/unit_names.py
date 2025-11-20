#!/usr/bin/env python

import os
from pathlib import Path

import yaml
from transformers import AutoConfig

# Optional: keep cache on scratch, but this script only downloads tiny config.json
os.environ["HF_HOME"] = "/storage/ice1/2/3/jhuang691/scratch/hfcache"
os.environ["HF_DATASETS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfdatasets"
os.environ["TRANSFORMERS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfmodels"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # yaml_gathering -> src -> AutoLLMTuner
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUT_UNITS_PATH = CONFIGS_DIR / "quant_units.yaml"


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Writing quant units to {OUT_UNITS_PATH}")

    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers

    unit_names: list[str] = []

    for i in range(num_layers):
        prefix = f"layers.{i}"
        # Attention projections
        unit_names.extend(
            [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
            ]
        )
        # MLP projections
        unit_names.extend(
            [
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
            ]
        )

    print(f"Generated {len(unit_names)} quantizable units")

    config = {
        "units": [{"name": n} for n in unit_names],
        "groups": [
            [
                "layers.*.self_attn.q_proj",
                "layers.*.self_attn.k_proj",
                "layers.*.self_attn.v_proj",
            ],
            [
                "layers.*.mlp.gate_proj",
                "layers.*.mlp.up_proj",
            ],
        ],
        "exclude": [
            "embed_tokens",
            "lm_head",
            "layers.*.input_layernorm",
            "layers.*.post_attention_layernorm",
        ],
        "choices": [4, 8, 16],
    }

    OUT_UNITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_UNITS_PATH.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote YAML to {OUT_UNITS_PATH}")


if __name__ == "__main__":
    main()
