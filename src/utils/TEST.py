import json
import ollama
from ollama import chat
from pydantic import BaseModel

def remove_duplicates(lst):
    unique_tuples = set(tuple(sublist) for sublist in lst)
    result = sorted([list(t) for t in unique_tuples], key=lambda x: (x[0], x[1]))
    return result

with open("data/NEW_test.json", "r") as f:
    train_dataset = json.load(f)

for d in train_dataset:
    ner = d["ner"]
    d["ner"] = remove_duplicates(ner)

with open("data/NEW_test.json", "w") as f:
    json.dump(train_dataset, f, indent=4, ensure_ascii=False) 

