from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
import polars as pl
from .data_models import ParamLayout
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


class BnbQuantizeBackend:
    """
    Use bitsandbytes to quantize selected linear layers of Qwen2.5
    to 4 or 8 bits based on a per-unit bits vector, without deepcopying
    the whole model per evaluation.
    """

    def __init__(self, layout: ParamLayout, device: str = "cuda:0") -> None:
        self.layout = layout
        self.device = device
        # Cap for MATH-500 sequences
        self.max_seq_len = 2048

        logger.info(f"BnbQuantizeBackend: using MODEL_NAME={MODEL_NAME}")
        logger.info(f"BnbQuantizeBackend: loading tokenizer...")
        # Tokenizer / base model on CPU
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        logger.info(
            f"BnbQuantizeBackend: loading base model on CPU with torch_dtype=float16..."
        )
        # Keep a single base model; we will mutate its Linear modules in-place
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
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
        logger.info("BnbQuantizeBackend: caching FP16 weights for quantizable units...")

        self.math_df = pl.read_ndjson(
            "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl"
        )
        self.math_rows: List[Dict[str, str]] = (
            self.math_df.head(50).to_dicts()
        )
        logger.info(
            f"BnbQuantizeBackend: loaded MATH-500 with {len(self.math_rows)} rows."
        )

    def _resolve_linear(
        self, model: nn.Module, module_path: str
    ) -> tuple[nn.Module, str, nn.Linear]:
        """

        parse "model.layers.0.self_attn.q_proj".
        """
        parts = module_path.split(".")
        parent = model

        # it doesn't start with model.model because it was a relative path
        if parts[0] != "model" and hasattr(model, "model"):
            inner = getattr(model, "model")
            # Check if the inner model has the first part; use it as the starting point
            if hasattr(inner, parts[0]):
                parent = inner

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
                try:
                    parent = getattr(parent, part)
                except AttributeError as e:
                    raise AttributeError(
                        f"While resolving '{module_path}': "
                        f"failed at component '{part}' on {parent.__class__.__name__}"
                    ) from e

        child_name = parts[-1]
        try:
            child = getattr(parent, child_name)
        except AttributeError as e:
            raise AttributeError(
                f"While resolving '{module_path}': "
                f"could not find final attribute '{child_name}' on {parent.__class__.__name__}"
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
        """
        Build a bitsandbytes Linear4bit / Linear8bitLt layer for a unit.
        """
        if bitw not in (4, 8):
            return self._build_original_linear(unit_name)

        # rebuild the linear layer
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
        # load the state from fp16 then quantize
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
        build the whole model
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

    # For accuracy using MATH-500: compute token-averaged cross-entropy loss.
    def eval_bits(self, bits: Tuple[int, ...], max_tokens: int) -> float:
        """
        Quantize according to `bits`, then evaluate average cross-entropy loss
        on MATH-500, truncated after `max_tokens` target tokens.
        """
        logger.info(
            f"eval_bits: starting evaluation for bits (len={len(bits)}), max_tokens={max_tokens}"
        )
        self.build_quantized_model(bits)
        self.model.eval()

        checker = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map=self.device,
        )
        checker_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        total_correct = 0

        for idx, row in enumerate(self.math_rows):
            problem = row["problem"]
            answer = row["answer"]

            inputs = self.tokenizer(self.tokenizer.apply_chat_template([
                {
                    "role": "system",
                    "content": "You are a math teacher. Solve the following problem step by step.",
                },
                {
                    "role": "user",
                    "content": problem,
                },
            ], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_answer = output_text.split("assistant\n")[-1].strip()

            inputs = checker_tokenizer(checker_tokenizer.apply_chat_template([
                {
                    "role": "system",
                    "content": "You are a math grader who grades students' answers to math questions. You compare the student's answer to the correct answer and give a score of 1 if they match and 0 if they do not. Please only respond with the score. Ignore minor formatting differences such as including variable names (i.e. '1' is acceptable for 'x=1' and vice-versa), units (i.e. '1' is acceptable for '1 meter' and vice-versa), dollar signs (i.e. '1' is acceptable for '$1' and vice-versa), etc.",
                },
                {
                    "role": "user",
                    "content": f"Question: {problem}\nCorrect Answer: {answer}\nStudent's Answer: {output_answer}"
                },
            ], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(self.device)
            checker_output = checker.generate(**inputs, max_new_tokens=10)
            checker_text = checker_tokenizer.decode(checker_output[0], skip_special_tokens=True)
            try:
                score = int(checker_text.split("assistant\n")[-1].strip())
                total_correct += score
            except ValueError:
                logger.warning(
                    f"eval_bits: could not parse score from checker output: '{checker_text}'"
                )

        accuracy = total_correct / len(self.math_rows)
        logger.info(
            f"eval_bits: completed evaluation for bits (len={len(bits)}), accuracy={accuracy:.4f}"
        )

        return 1.0 - accuracy  # return 1 - accuracy as "loss"
