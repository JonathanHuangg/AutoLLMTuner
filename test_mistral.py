import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

device = "cuda"

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
print("Loading vanilla model...")
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# one example from MATH-500
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
row = ds[0]
problem = row["problem"]
answer = row["answer"]

prompt = f"<s> Problem: {problem}\nAnswer: {answer}\n"
inputs = tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits[:, :-1, :]
    labels = inputs["input_ids"][:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

print("\n=== VANILLA MODEL CE LOSS ===")
print(loss.item())
