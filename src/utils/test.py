
import json



# LLM = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

with open("data/llm8_test_dataset.json", "r") as f:
    test_dataset = json.load(f)

print(len(test_dataset))