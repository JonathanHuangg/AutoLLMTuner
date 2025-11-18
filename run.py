import json
import sys
import time
from pathlib import Path

import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-7B-Instruct"

def get_model(config: list[bool]):
    skip_modules = [f".{i}" for i, q in enumerate(config) if not q]
    quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quant,
        trust_remote_code=True,
        device_map="auto",
    )
    return tokenizer, model

def run_sample(tokenizer, model, sample: str):
    device = model.device
    text = tokenizer.apply_chat_template([
        {
            "role": "system",
            "content": "You are a math teacher who helps students with their math questions. You give the answer with solution steps. Please enclose your final answer in <answer></answer> tags.",
        },
        {
            "role": "user",
            "content": sample,
        }
    ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=4000)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <config.json> <output_dir>")
        return

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = get_model(config)

    with open(output_dir / "memory", "w") as f:
        f.write(str(model.get_memory_footprint()))

    dataset = pl.read_parquet("math50.parquet")

    for i, row in enumerate(tqdm(dataset.to_dicts())):
        sample = row["problem"]
        start_time = time.time()
        output = run_sample(tokenizer, model, sample)
        end_time = time.time()

        with open(output_dir / f"{i:05d}.json", "w") as f:
            json.dump({
                "unique_id": row["unique_id"],
                "problem": sample,
                "solution": row["solution"],
                "answer": row["answer"],
                "output": output,
                "time": end_time - start_time,
            }, f, indent=4)


if __name__ == "__main__":
    main()
