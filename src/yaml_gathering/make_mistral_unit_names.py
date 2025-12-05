#!/usr/bin/env python

"""
Generate mistral_units.yaml for Mistral-7B.

Correct naming must match HF parameters:
    model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}
    model.layers.N.mlp.{gate_proj,up_proj,down_proj}

Your previous version used:
    layers.N.self_attn.q_proj
which DOES NOT match HF and causes KeyErrors in evolutionary search.
"""

import yaml
from pathlib import Path
from transformers import AutoConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUT_PATH = CONFIGS_DIR / "mistral_units.yaml"


def main():
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = cfg.num_hidden_layers  # (expected = 32 for Mistral-7B)

    units = []

    for i in range(num_layers):

        prefix = f"model.layers.{i}"

        # Attention projections
        units.extend([
            {"name": f"{prefix}.self_attn.q_proj"},
            {"name": f"{prefix}.self_attn.k_proj"},
            {"name": f"{prefix}.self_attn.v_proj"},
            {"name": f"{prefix}.self_attn.o_proj"},
        ])

        # MLP projections
        units.extend([
            {"name": f"{prefix}.mlp.gate_proj"},
            {"name": f"{prefix}.mlp.up_proj"},
            {"name": f"{prefix}.mlp.down_proj"},
        ])

    config = {
        "units": units,

        "groups": [
            [
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.k_proj",
                "model.layers.*.self_attn.v_proj",
            ],
            [
                "model.layers.*.mlp.gate_proj",
                "model.layers.*.mlp.up_proj",
                "model.layers.*.mlp.down_proj",
            ]
        ],

        "exclude": [
            "model.embed_tokens",
            "model.lm_head",
            "model.layers.*.input_layernorm",
            "model.layers.*.post_attention_layernorm",
        ],

        "choices": [4, 8, 16],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote {OUT_PATH} with {len(units)} units")


if __name__ == "__main__":
    main()
