"""
Trinity-Mini-specific inference utilities for chat-based evaluation
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional


class TrinityInference:    
    def __init__(self, model_name: str = "arcee-ai/Trinity-Mini"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load(self, torch_dtype=torch.float16, device_map="auto"):
        """Load tokenizer and model"""
        print(f"Loading Trinity model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model, self.tokenizer
    
    def format_prompt(self, system_prompt: str, user_message: str) -> str:
        """Format message using Trinity's chat template"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate(self, 
                 system_prompt: str, 
                 user_message: str, 
                 max_new_tokens: int = 1024,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """Generate response to a prompt"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Format the prompt
        text = self.format_prompt(system_prompt, user_message)
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def get_model_info(self) -> Dict:
        """Get model statistics"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        total_size_mb = total_params * 2 / (1024 ** 2)  # Assuming float16
        
        num_layers = len(self.model.model.layers) if hasattr(self.model.model, 'layers') else 0
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "total_size_mb": total_size_mb,
            "num_layers": num_layers
        }
