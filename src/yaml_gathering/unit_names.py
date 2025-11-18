import os

os.environ["HF_HOME"] = "/storage/ice1/2/3/jhuang691/scratch/hfcache"
os.environ["HF_DATASETS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfdatasets"
os.environ["TRANSFORMERS_CACHE"] = "/storage/ice1/2/3/jhuang691/scratch/hfmodels"

from transformers import AutoModelForCausalLM
import yaml
import torch
import os


model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",      # or torch.float16
    device_map="auto"        # or omit if you only have CPU
)

state_keys = list(model.state_dict().keys())

def logical_name(k: str) -> str:
    # strip "model." prefix if present
    if k.startswith("model."):
        k = k[len("model."):]
    # drop .weight / .bias
    for suffix in (".weight", ".bias"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
    return k

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading {model_name} just to inspect param names...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",   # or remove if you only want CPU
    )

    state_keys = list(model.state_dict().keys())

    interesting = [
        k for k in state_keys
        if (".self_attn." in k or ".mlp." in k) and k.endswith(".weight")
    ]

    unit_names = sorted({logical_name(k) for k in interesting})
    print(f"Found {len(unit_names)} quantizable units.")

    # If you still want to see them:
    for n in unit_names[:20]:
        print("  ", n)
    if len(unit_names) > 20:
        print("  ...")

    # Build config dict for YAML
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

        "choices": [2, 4, 8, 16],
    }

    # IMPORTANT: write directly into the configs dir you mentioned
    output_path = "/home/hice1/jhuang691/scratch/AutoLLMTuner/configs/quant_units.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote YAML to {output_path}")

if __name__ == "__main__":
    main()