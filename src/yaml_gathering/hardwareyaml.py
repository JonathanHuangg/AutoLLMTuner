#!/usr/bin/env python
import math
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# How many timing iterations
WARMUP_ITERS = 10
BENCH_ITERS = 50

# Bitwidths we care about
BITWIDTHS = [2, 3, 4, 6, 8, 16]


def collect_linear_shapes(seq_len: int = 128):
    """
    Load Qwen2.5-7B on CPU, run a dummy forward pass and collect
    (m, n, k) shapes for all nn.Linear layers.

    We interpret a Linear as:  (m, k) @ (k, n) -> (m, n)
    where:
      - m = batch * seq_len
      - k = in_features
      - n = out_features
    """
    print(f"Loading {MODEL_NAME} on CPU just to inspect architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,   # dtype doesn't really matter for shapes
        device_map=None,            # force CPU
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    shapes = set()

    def make_hook(mod_name, linear_module):
        def hook(module, inputs, output):
            # inputs[0] is typically (..., in_features)
            x = inputs[0]
            # ensure it's at least 2D
            x_shape = x.shape
            if x.dim() < 2:
                return

            # Flatten everything except the last dim
            m = math.prod(x_shape[:-1])
            k = linear_module.in_features
            n = linear_module.out_features

            shapes.add((m, n, k))
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    # Build a dummy input with fixed seq_len to get representative shapes
    dummy = tokenizer(
        "Dummy prompt for shape capture.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    with torch.no_grad():
        _ = model(**dummy)

    for h in hooks:
        h.remove()

    shapes = sorted(shapes)
    print("Collected unique Linear shapes (m, n, k):")
    for m, n, k in shapes:
        print(f"  (m={m}, n={n}, k={k})")

    return shapes


def bench_matmul(m: int, n: int, k: int, dtype=torch.float16) -> float:
    """
    Benchmark a single matmul with shape (m, k) @ (k, n) on CUDA.
    Returns average latency in microseconds.
    """
    device = torch.device("cuda")

    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)

    # Warmup
    for _ in range(WARMUP_ITERS):
        c = a @ b
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c = a @ b
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))  # ms

    avg_ms = sum(times_ms) / len(times_ms)
    return avg_ms * 1000.0  # microseconds


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on a GPU node (salloc ... --gres=gpu:1).")

    print("Using CUDA device:", torch.cuda.get_device_name(0))

    # 1) Get representative (m, n, k) shapes from Qwen2.5
    shapes = collect_linear_shapes(seq_len=128)

    # Optionally, you might want to prune shapes that are very small or duplicates
    # In practice Qwen usually has just a few distinct (in_features, out_features).
    # We'll just use everything we collected.

    # 2) Benchmark each shape once in fp16 and reuse for each bitwidth
    latency_lut_entries = []

    for (m, n, k) in shapes:
        print(f"\nBenchmarking matmul with (m={m}, n={n}, k={k}) ...")
        base_us = bench_matmul(m, n, k, dtype=torch.float16)
        print(f"  avg latency (fp16) ≈ {base_us:.2f} µs")

        # For now, store the same measured latency for all bitwidths.
        # If you later have real kernels for 2/3/4/6-bit, you can replace
        # this with actual measurements or a more principled scaling.
        for bitw in BITWIDTHS:
            entry = {
                "key": ["matmul", int(m), int(n), int(k), int(bitw)],
                "us": float(base_us),
            }
            latency_lut_entries.append(entry)

    hardware = {
        "latency_lut": latency_lut_entries,
        "overheads": {
            # You can tune these later when you model bandwidth / memory overheads.
            "bandwidth_alpha": 0.1,
        },
    }

    out_path = Path("configs/hardware.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        yaml.safe_dump(hardware, f, sort_keys=False)

    print(f"\nWrote {out_path} with {len(latency_lut_entries)} LUT entries.")
    print("Done.")


if __name__ == "__main__":
    main()
