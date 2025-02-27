import json

from datasets import load_dataset

from data_helpers import convert_to_gliner, split_dataset, remove_overlapping_entities


data = load_dataset("E3-JSI/synthetic-multi-pii-ner-v1", split="train")

data = convert_to_gliner(data)

train_dataset, test_dataset = split_dataset(data)
remove_overlapping_entities(test_dataset)

# 70 : 10
train_dataset, validation_dataset = split_dataset(train_dataset, train=0.9)

# Save all datasets as JSON
with open("data/E3-JSI/train_dataset.json", "w") as f:
    json.dump(train_dataset, f, indent=4, ensure_ascii=False)  


with open("data/E3-JSI/test_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=4, ensure_ascii=False)


with open("data/E3-JSI/validation_dataset.json", "w") as f:
    json.dump(validation_dataset, f, indent=4, ensure_ascii=False)