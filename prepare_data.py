from datasets import load_dataset
import json

dataset = load_dataset("4DR1455/finance_questions")

with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        instruction = item["instruction"]
        input_text = item["input"]
        output = item["output"]

        prompt = f"{instruction}\n{input_text}" if input_text else instruction

        f.write(json.dumps({
            "prompt": prompt,
            "response": output
        }, ensure_ascii=False) + "\n")
