#!/usr/bin/env python

import os
from pathlib import Path

import yaml
from transformers import AutoConfig

# Optional: keep cache on scratch, but this script only downloads tiny config.json
os.environ["HF_HOME"] = "/storage/ice1/2/3/jhuang691/scratch/hfcache"
os.environ["HF_DATASETS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfdatasets"
os.environ["TRANSFORMERS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfmodels"

MODEL_NAME = "arcee-ai/Trinity-Mini"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # yaml_gathering -> src -> AutoLLMTuner
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUT_UNITS_PATH = CONFIGS_DIR / "trinity_units.yaml"


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Writing Trinity quant units to {OUT_UNITS_PATH}")

    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers
    num_experts = getattr(cfg, "num_local_experts", 0)

    print(f"Model: {MODEL_NAME}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of experts per layer: {num_experts}")

    unit_names: list[str] = []

    for i in range(num_layers):
        prefix = f"layers.{i}"
        # Attention projections (standard for all layers)
        unit_names.extend(
            [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
            ]
        )

        # MoE block experts
        # Trinity-Mini uses block_sparse_moe structure
        if num_experts > 0:
            # Gate (router)
            unit_names.append(f"{prefix}.block_sparse_moe.gate")

            # Shared expert if exists
            unit_names.extend(
                [
                    f"{prefix}.block_sparse_moe.shared_experts.gate_proj",
                    f"{prefix}.block_sparse_moe.shared_experts.up_proj",
                    f"{prefix}.block_sparse_moe.shared_experts.down_proj",
                ]
            )

            # Individual experts
            for expert_idx in range(num_experts):
                unit_names.extend(
                    [
                        f"{prefix}.block_sparse_moe.experts.{expert_idx}.gate_proj",
                        f"{prefix}.block_sparse_moe.experts.{expert_idx}.up_proj",
                        f"{prefix}.block_sparse_moe.experts.{expert_idx}.down_proj",
                    ]
                )

    print(f"Generated {len(unit_names)} quantizable units")

    config = {
        "units": [{"name": n} for n in unit_names],
        "groups": [
            # Group QKV projections together
            [
                "layers.*.self_attn.q_proj",
                "layers.*.self_attn.k_proj",
                "layers.*.self_attn.v_proj",
            ],
            # Group shared expert projections
            [
                "layers.*.block_sparse_moe.shared_experts.gate_proj",
                "layers.*.block_sparse_moe.shared_experts.up_proj",
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
