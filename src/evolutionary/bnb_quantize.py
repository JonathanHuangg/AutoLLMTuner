from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from data_models import ParamLayout

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

class BnbQuantizeBackend:
    """
    use bnb to allow the quantization of each layer to 4 or 8 bits
    """

    def __init__(self, layout: ParamLayout, device: str="cuda:0") -> None:
        self.layout = layout 
        self.device = device 

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )

        self.base_model.eval()

        # add in math500 later
        # Tiny built-in "validation set" so backend_eval is fully concrete.
        # You can later replace this with a real dataset.
        self._val_prompts: List[str] = [
            "The capital of France is",
            "In a right triangle with legs 3 and 4, the hypotenuse is",
            "The derivative of x squared is",
            "Qwen is a large language model that",
        ]

    def _resolve_linear(self, model: nn.Module, module_path: str) -> tuple[nn.Module, str, nn.Linear]:
        """
        given module path, get (parent_module, child_name, child_linear_module)

        its based on the structure of how the models work, i don't really know myself but it works
        """
        parts = module.path.split(".")
        parent = model 

        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        
        child_name = parts[-1]
        child = getattr(parent, child_name)

        if not isinstance(child, nn.Linear):
            raise TypeError(
                f"Unit '{module_path}' is {type(child)}, expected nn.Linear"
            )

        return parent, child_name, child

    def _wrap_with_bnb(self, parent:nn.Module, child_name: str, old: nn.Linear, bitw: int) -> None:
        # bnb can only do 4 and 8 bit quantization
        if bitw not in (4, 8):
            return 
        
        in_features = old.in_features
        out_features = old.out_features

        bias_flag = old.bias is not None 

        # create the new layer
        if bitw == 8:
            # LLM.int8() inference mode
            new_layer = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias_flag,
                has_fp16_weights=False,  # pure int8 inference
                threshold=6.0,
            )
        else:  # bitw == 4
            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias_flag,
                compute_dtype=torch.float16,
                quant_type="nf4",
        )

        # add the weights + flag
        with torch.no_grad():
            new_layer.weight.copy_(old.weight.data)
            if bias_flag and old.bias is not None:
                new_layer.bias.copy_(old.bias.data)
        
        setattr(parent, child_name, new_layer)

    def build_quantized_model(self, bits: Tuple[int, ...]) -> nn.Module:
        """
        main function that builds a fully quantized model
        """

        if (len(bits) != len(self.layout.units)):
            raise ValueError(
                f"bits len {len(bits)} != layout.units len {len(self.layout.units)}"
            )
        
        # deep copy the base model
        model = copy.deepcopy(self.base_model)

        for idx, unit_name in enumerate(self.layout.units):
            if (getattr(self.layout, "excluded_mask", None) is not None):
                if self.layout.excluded_mask[idx]:
                    continue 
            bitw = bits[idx]
            module_path = unit_name 

            parent, child_name, old_lin = self._resolve_linear(model, module_path)
            self._wrap_with_bnb(parent, child_name, old_lin, bitw)
        
        model.to(self.device)
        model.eval()
        return model
