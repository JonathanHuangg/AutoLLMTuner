from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import polars as pl  # kept in case you use it elsewhere
from .data_models import ParamLayout

logger = logging.getLogger(__name__)
# MODEL_NAME = "arcee-ai/Trinity-Mini"


class BnbQuantizeBackend:
    """
    Use bitsandbytes to quantize selected linear layers of an HF CausalLM
    (e.g. Mistral, Qwen2.5, Trinity) to 4 or 8 bits based on a per-unit bits vector,
    without deepcopying the whole model per evaluation.

    Evaluation is done via *cross-entropy loss* on MATH-500 with teacher forcing:
      - Prompt:  "Problem: ... Answer: <ground-truth-answer>"
      - We compute CE over the full token sequence.
      - Returned metric is the *average CE loss* (lower is better).
    """

    def __init__(self, model_name: str, layout: ParamLayout, device: str = "cuda:0") -> None:
        self.model_name = model_name
        self.layout = layout
        self.device = device

        logger.info(f"BnbQuantizeBackend: using MODEL_NAME={self.model_name}")
        logger.info("BnbQuantizeBackend: loading tokenizer...")
        # Tokenizer / base model on CPU
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        logger.info(
            "BnbQuantizeBackend: loading base model on CPU with torch_dtype=float16..."
        )
        # Keep a single base model; we will mutate its Linear modules in-place
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        self.model.eval()

        # Cache original FP16 weights for each quantizable unit so we can
        # rebuild either an FP16 Linear or a BnB-quantized Linear on demand.
        logger.info("BnbQuantizeBackend: caching FP16 weights for quantizable units...")

        self.fp16_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        for unit_name in self.layout.units:
            parent, child_name, lin = self._resolve_linear(self.model, unit_name)
            if not isinstance(lin, nn.Linear):
                raise TypeError(f"Unit {unit_name} is {type(lin)}, expected nn.Linear")
            self.fp16_weights[unit_name] = {
                "weight": lin.weight.detach().cpu().clone(),
                "bias": None if lin.bias is None else lin.bias.detach().cpu().clone(),
            }

        # Move working model to device; fp16_weights stay as CPU master copy.
        logger.info(f"BnbQuantizeBackend: moving model to device={self.device}...")
        self.model.to(self.device)
        self.model.eval()

        # Load MATH-500 from HF (or point this to a local path if you prefer)
        logger.info("BnbQuantizeBackend: loading MATH-500 dataset...")
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

        # Keep only the first 50 problems to save evaluation time
        self.math_rows = ds.select(range(50))
        logger.info(f"BnbQuantizeBackend: loaded {len(self.math_rows)} MATH problems.")

    def _resolve_linear(
        self, model: nn.Module, module_path: str
    ) -> tuple[nn.Module, str, nn.Linear]:
        """
        Resolve a dotted module path like "model.layers.0.self_attn.q_proj"
        to (parent_module, child_name, child_linear_module).

        This includes some alias handling to cope with different architectures
        (e.g. Trinity's AfmoeDecoderLayer, MoE blocks, etc.).
        """
        parts = module_path.split(".")
        parent = model

        # If it doesn't start with "model", but the top-level model has a .model attribute,
        # try going inside that first (common for HF architectures).
        if parts[0] != "model" and hasattr(model, "model"):
            inner = getattr(model, "model")
            if hasattr(inner, parts[0]):
                parent = inner

        # Mapping for some known alias structures
        attribute_aliases = {
            "block_sparse_moe": [
                "mlp",
                "moe",
                "afmoe",
                "block_sparse_moe",
                "experts",
                "ffn",
            ],
            "shared_experts": [
                "shared_experts",
                "shared",
                "experts",
                "shared_mlp",
            ],
        }

        for part in parts[:-1]:
            if part.isdigit():
                try:
                    parent = parent[int(part)]
                except (KeyError, IndexError, TypeError) as e:
                    raise AttributeError(
                        f"While resolving '{module_path}': "
                        f"failed to index component '{part}' on {parent.__class__.__name__}"
                    ) from e
            else:
                if hasattr(parent, part):
                    parent = getattr(parent, part)
                elif part in attribute_aliases:
                    found = False
                    for alias in attribute_aliases[part]:
                        if hasattr(parent, alias):
                            parent = getattr(parent, alias)
                            found = True
                            break
                    if not found:
                        all_attrs = dir(parent)
                        available_attrs = []
                        for name in all_attrs:
                            if name.startswith("_"):
                                continue
                            try:
                                attr = getattr(parent, name)
                                if isinstance(attr, nn.Module) or not callable(attr):
                                    available_attrs.append(name)
                            except Exception:
                                pass
                        logger.error(
                            f"Available attributes on {parent.__class__.__name__}: {available_attrs}"
                        )
                        raise AttributeError(
                            f"While resolving '{module_path}': "
                            f"failed at component '{part}' on {parent.__class__.__name__} "
                            f"(tried: {', '.join(attribute_aliases[part])}). "
                            f"Available attributes: {available_attrs}"
                        )
                else:
                    raise AttributeError(
                        f"While resolving '{module_path}': "
                        f"failed at component '{part}' on {parent.__class__.__name__}"
                    )

        child_name = parts[-1]

        # Possible aliases for final attribute names (MoE routers etc.)
        child_aliases = {
            "gate": [
                "router",
                "gate",
                "gate_proj",
                "gate_layer",
                "gating_network",
                "expert_weights",
            ]
        }

        if hasattr(parent, child_name):
            child = getattr(parent, child_name)
        elif child_name in child_aliases:
            found = False
            for alias in child_aliases[child_name]:
                if hasattr(parent, alias):
                    child = getattr(parent, alias)
                    child_name = alias
                    found = True
                    break
            if not found:
                all_attrs = dir(parent)
                available_attrs = []
                for name in all_attrs:
                    if name.startswith("_"):
                        continue
                    try:
                        attr = getattr(parent, name)
                        if isinstance(attr, nn.Module) or not callable(attr):
                            available_attrs.append(name)
                    except Exception:
                        pass
                logger.error(
                    f"Available attributes on {parent.__class__.__name__}: {available_attrs}"
                )
                raise AttributeError(
                    f"While resolving '{module_path}': "
                    f"could not find final attribute '{child_name}' on {parent.__class__.__name__} "
                    f"(tried: {', '.join(child_aliases[child_name])}). "
                    f"Available attributes: {available_attrs}"
                )
        else:
            try:
                child = getattr(parent, child_name)
            except AttributeError as e:
                all_attrs = dir(parent)
                available_attrs = []
                for name in all_attrs:
                    if name.startswith("_"):
                        continue
                    try:
                        attr = getattr(parent, name)
                        if isinstance(attr, nn.Module) or not callable(attr):
                            available_attrs.append(name)
                    except Exception:
                        pass
                logger.error(
                    f"Available attributes on {parent.__class__.__name__}: {available_attrs}"
                )
                raise AttributeError(
                    f"While resolving '{module_path}': "
                    f"could not find final attribute '{child_name}' on {parent.__class__.__name__}. "
                    f"Available attributes: {available_attrs}"
                ) from e

        if not isinstance(child, nn.Linear):
            raise TypeError(
                f"Unit '{module_path}' is {type(child)}, expected nn.Linear"
            )
        return parent, child_name, child

    def _build_original_linear(self, unit_name: str) -> nn.Linear:
        """
        Rebuild a plain FP16 nn.Linear from the cached master weights.
        """
        info = self.fp16_weights[unit_name]
        w = info["weight"]
        b = info["bias"]

        out_features, in_features = w.shape
        lin = nn.Linear(in_features, out_features, bias=b is not None)
        with torch.no_grad():
            lin.weight.copy_(w)
            if b is not None:
                lin.bias.copy_(b)

        lin.to(self.device, dtype=torch.float16)
        return lin

    def _build_bnb_linear(self, unit_name: str, bitw: int) -> nn.Module:
        if bitw not in (4, 8):
            return self._build_original_linear(unit_name)

        info = self.fp16_weights[unit_name]
        w = info["weight"].to(dtype=torch.float16, device="cpu")
        b = info["bias"]
        if b is not None:
            b = b.to(dtype=torch.float16, device="cpu")

        out_features, in_features = w.shape
        bias_flag = b is not None

        fp16_lin = nn.Linear(in_features, out_features, bias=bias_flag)
        with torch.no_grad():
            fp16_lin.weight.copy_(w)
            if b is not None:
                fp16_lin.bias.copy_(b)
        fp16_lin = fp16_lin.to("cpu", dtype=torch.float16)

        if bitw == 8:
            q_lin = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias_flag,
                has_fp16_weights=False,
            )
        else:
            q_lin = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias_flag,
                compute_dtype=torch.float16,
                quant_type="nf4",
            )

        q_lin.load_state_dict(fp16_lin.state_dict())
        q_lin = q_lin.to(self.device)
        return q_lin

    def set_device(self, device: torch.device) -> None:
        """
        Set the device for the model.
        """
        if device != self.device:
            logger.info(f"BnbQuantizeBackend: moving model to device={device}...")
            self.model.to(device)
            self.device = device

    def build_quantized_model(self, bits: Tuple[int, ...]) -> nn.Module:
        """
        Given a tuple of bit-widths (one per layout unit), rebuilds the
        corresponding Linear modules in-place on `self.model`.

        - bit = 4 or 8 -> BnB quantized linear
        - any other bit value -> restore original FP16 linear
        """
        if len(bits) != len(self.layout.units):
            raise ValueError(
                f"bits len {len(bits)} != layout.units len {len(self.layout.units)}"
            )

        for idx, unit_name in enumerate(self.layout.units):
            if getattr(self.layout, "excluded_mask", None) is not None:
                if self.layout.excluded_mask[idx]:
                    continue

            bitw = bits[idx]
            parent, child_name, current = self._resolve_linear(self.model, unit_name)

            if bitw in (4, 8):
                if bitw == 8 and isinstance(current, bnb.nn.Linear8bitLt):
                    continue
                if bitw == 4 and isinstance(current, bnb.nn.Linear4bit):
                    continue

                new_mod = self._build_bnb_linear(unit_name, bitw)
                setattr(parent, child_name, new_mod)
            else:
                if isinstance(current, nn.Linear) and not isinstance(
                    current, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
                ):
                    continue
                new_mod = self._build_original_linear(unit_name)
                setattr(parent, child_name, new_mod)

        return self.model

    # cross entropy evaluation
    def eval_bits(self, bits: Tuple[int, ...], max_tokens: int) -> float:
        logger.info(
            f"eval_bits: starting evaluation for bits (len={len(bits)}), max_tokens={max_tokens}"
        )
        self.build_quantized_model(bits)
        self.model.eval()

        losses: List[float] = []

        for idx, row in enumerate(self.math_rows):
            problem = row["problem"]
            answer = row["answer"]

            text = (
                f"<s> Problem: {problem}\n"
                f"Answer: {answer}\n"
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
            )

            # Optionally truncate to at most `max_tokens` tokens (take last segment)
            if max_tokens is not None and inputs["input_ids"].size(1) > max_tokens:
                for k in inputs.keys():
                    inputs[k] = inputs[k][:, -max_tokens:]

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (batch, seq, vocab)
                # Standard LM CE: shift logits and labels by one
                logits_shifted = logits[:, :-1, :].contiguous()
                labels = inputs["input_ids"][:, 1:].contiguous()

                loss = F.cross_entropy(
                    logits_shifted.view(-1, logits_shifted.size(-1)),
                    labels.view(-1),
                    reduction="mean",
                )

            losses.append(loss.item())

        mean_loss = float(sum(losses) / len(losses))
        logger.info(
            f"eval_bits: completed evaluation for bits (len={len(bits)}), "
            f"mean CE loss={mean_loss:.4f}"
        )

        return mean_loss
