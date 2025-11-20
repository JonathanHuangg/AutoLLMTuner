#!/usr/bin/env python
import math
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# How many timing iterations
WARMUP_ITERS = 10
BENCH_ITERS = 50

# Bitwidths we care about
BITWIDTHS = [2, 3, 4, 6, 8, 16]

# Resolve project root: .../AutoLLMTuner
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # yaml_gathering -> src -> AutoLLMTuner
CONFIGS_DIR = PROJECT_ROOT / "configs"


def collect_linear_shapes(seq_len: int = 128, batch_size: int = 1):
    """
    Collect representative (m, n, k) shapes for Qwen2.5-7B *without*
    loading model weights.

    We interpret a Linear as:  (m, k) @ (k, n) -> (m, n)
    where:
      - m = batch_size * seq_len
      - k, n depend on hidden_size / intermediate_size and the projection type.
    """
    print(f"Loading config for {MODEL_NAME} (no weights)...")
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    hidden_size = cfg.hidden_size
    intermediate_size = getattr(cfg, "intermediate_size", hidden_size * 4)

    print(f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    m = batch_size * seq_len

    shapes = set()

    # Attention projections: q, k, v, o: hidden -> hidden
    # (m, k=hidden, n=hidden)
    shapes.add((m, hidden_size, hidden_size))

    # MLP gate/up: hidden -> intermediate
    # (m, k=hidden, n=intermediate)
    shapes.add((m, intermediate_size, hidden_size))   # treat as (m, k, n) for matmul(m,k)@(k,n)

    # MLP down: intermediate -> hidden
    # (m, k=intermediate, n=hidden)
    shapes.add((m, hidden_size, intermediate_size))

    # You can extend this with other ops (embed, lm_head) if you care.

    shapes = sorted(shapes)
    print("Representative Linear shapes (m, n, k) we'll benchmark:")
    for m_, n_, k_ in shapes:
        print(f"  (m={m_}, n={n_}, k={k_})")

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
        _ = a @ b
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = a @ b
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))  # ms

    avg_ms = sum(times_ms) / len(times_ms)
    return avg_ms * 1000.0  # microseconds


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this on a GPU node (srun/salloc ... --gres=gpu:1)."
        )

    print("Using CUDA device:", torch.cuda.get_device_name(0))
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Configs dir:  {CONFIGS_DIR}")

    # 1) Get representative (m, n, k) shapes from Qwen2.5 config
    shapes = collect_linear_shapes(seq_len=128, batch_size=1)

    # 2) Benchmark each shape once in fp16 and reuse for each bitwidth
    latency_lut_entries = []

    for (m, n, k) in shapes:
        print(f"\nBenchmarking matmul with (m={m}, n={n}, k={k}) ...")
        base_us = bench_matmul(m, n, k, dtype=torch.float16)
        print(f"  avg latency (fp16) ≈ {base_us:.2f} µs")

        # For now, store the same measured latency for all bitwidths.
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

    out_path = CONFIGS_DIR / "hardware.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        yaml.safe_dump(hardware, f, sort_keys=False)

    print(f"\nWrote {out_path} with {len(latency_lut_entries)} LUT entries.")
    print("Done.")


if __name__ == "__main__":
    main()