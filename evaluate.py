import json
import sys
from functools import cache
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"


@cache
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        device_map="auto",
    )
    return model, tokenizer

def evaluate_output(question, answer, output):
    model, tokenizer = get_model()

    text = tokenizer.apply_chat_template([
        {
            "role": "system",
            "content": "You are a math grader who grades students' answers to math questions. You compare the student's answer to the correct answer and give a score of 1 if they match and 0 if they do not. Please only respond with the score. Ignore minor formatting differences such as including variable names (i.e. '1' is acceptable for 'x=1' and vice-versa), units (i.e. '1' is acceptable for '1 meter' and vice-versa), dollar signs (i.e. '1' is acceptable for '$1' and vice-versa), etc.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nCorrect Answer: {answer}\nStudent's Answer: {output}"
        },
    ], tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return int(result.split("assistant\n")[-1].strip())


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <out_dir>")
        return

    out_dir = Path(sys.argv[1])
    total = 0
    for p in sorted(out_dir.glob("*.json")):
        with open(p, "r") as f:
            data = json.load(f)

        question = data["problem"]
        answer = data["answer"]
        student_answer = data["output"].split("assistant\n")[-1].strip()

        score = evaluate_output(question, answer, student_answer)
        total += score
        print(f"{p.name}: {score}")

    print(f"Total Score: {total}")

if __name__ == "__main__":
    main()
