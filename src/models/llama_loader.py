import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaModelLoader:
    def __init__(self, run_config_path: str, layer_spec_path: str):
        # Load both YAMLs
        self.run_cfg = yaml.safe_load(open(run_config_path))
        self.layer_spec = yaml.safe_load(open(layer_spec_path))
        self.model = None
        self.tokenizer = None
        self.layer_metadata = {}

    def load_model(self):
        """Load the base (pre-quantized) LLaMA model"""
        model_cfg = self.run_cfg["model"]
        model_path = model_cfg["name_or_path"]
        dtype = torch.float16 if self.run_cfg["hardware"]["torch_dtype"] == "float16" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.model, self.tokenizer

    def extract_layer_metadata(self):
        """Extract layer names and parameter counts based on layer_spec.yaml"""
        if self.model is None:
            raise RuntimeError("Load the model first using load_model().")

        for unit in self.layer_spec["units"]:
            name = unit["name"]
            layer_list = unit["layers"]
            total_params = 0
            for lname in layer_list:
                # Access submodule safely
                submodule = dict(self.model.named_modules()).get(lname, None)
                if submodule:
                    total_params += sum(p.numel() for p in submodule.parameters())
            self.layer_metadata[name] = {
                "layers": layer_list,
                "param_count": total_params
            }
        return self.layer_metadata
