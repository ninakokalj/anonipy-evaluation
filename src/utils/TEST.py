import json
import ollama
from ollama import chat
import pandas as pd


with open("data/evaluation/E3-JSI/test_dataset.json", "r") as f:
        data = json.load(f)

eng_data = [d for d in data if d["language"] == "Italian"]

with open("data/training/test_dataset/test_it.json", "w") as f:
    json.dump(eng_data, f, indent=4, ensure_ascii=False)

