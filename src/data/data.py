import json

from datasets import load_dataset

from data_helpers import convert_to_gliner, split_dataset


data = load_dataset("E3-JSI/synthetic-multi-pii-ner-v1", split="train")

data = convert_to_gliner(data)

train_dataset, test_dataset = split_dataset(data)

# 70 : 10
train_dataset, validation_dataset = split_dataset(train_dataset, train=0.9)

# Save both datasets as JSON
with open("data/train_dataset.json", "w") as f:
    json.dump(train_dataset, f, indent=4, ensure_ascii=False)  


with open("data/test_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=4, ensure_ascii=False)


with open("data/validation_dataset.json", "w") as f:
    json.dump(validation_dataset, f, indent=4, ensure_ascii=False)