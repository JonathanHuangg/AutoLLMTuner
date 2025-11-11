# src/eval.py
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Tuple, Optional
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class EvalConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    split: str = "validation"
    n_samples: int = 256          # keep tiny for smoke test
    max_seq_len: int = 128
    batch_size: int = 8
    fp16: bool = True
    device: str = "cuda"

def _prep_wikitext(cfg: EvalConfig, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")[cfg.split]
    texts = [r["text"] for r in ds if r["text"].strip()][:cfg.n_samples]
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=cfg.max_seq_len, return_tensors="pt"
    )
    return enc

def load_model(cfg: EvalConfig):
    dtype = torch.float16 if cfg.fp16 else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, type=dtype, device_map="auto"
    )
    return tok, model

@torch.inference_mode()
def eval_perplexity_latency_mem(cfg: EvalConfig) -> Dict[str, float]:
    tok, model = load_model(cfg)
    enc = _prep_wikitext(cfg, tok)
    input_ids = enc["input_ids"].to(cfg.device)
    attn = enc["attention_mask"].to(cfg.device)

    # VRAM baseline & timing
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Measure loss (teacher forcing)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    losses = []
    t0 = perf_counter()
    for i in range(0, input_ids.size(0), cfg.batch_size):
        x = input_ids[i:i+cfg.batch_size]
        m = attn[i:i+cfg.batch_size]
        out = model(input_ids=x, attention_mask=m, labels=x)
        losses.append(out.loss.detach().float().item())
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_s = perf_counter() - t0

    # Latency per token (very rough): total time / (tokens processed)
    tokens_proc = int(attn.sum().item())
    lat_ms_per_tok = 1000.0 * total_s / max(tokens_proc, 1)

    # Perplexity
    import math
    loss_mean = sum(losses) / max(len(losses), 1)
    ppl = math.exp(loss_mean)

    # VRAM
    mem_bytes = 0
    if torch.cuda.is_available():
        mem_bytes = torch.cuda.max_memory_allocated()

    return {
        "val_loss": float(loss_mean),
        "perplexity": float(ppl),
        "lat_ms_per_tok": float(lat_ms_per_tok),
        "vram_bytes": float(mem_bytes),
        "tokens_proc": float(tokens_proc),
        "elapsed_s": float(total_s),
    }
