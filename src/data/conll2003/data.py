import json

from datasets import load_dataset

from data_helpers import convert_to_gliner


train_data = load_dataset("eriktks/conll2003", trust_remote_code=True, split="train")
test_data = load_dataset("eriktks/conll2003", trust_remote_code=True, split="test")
validation_data = load_dataset("eriktks/conll2003", trust_remote_code=True, split="validation")

train_data = convert_to_gliner(train_data) 
test_data = convert_to_gliner(test_data)
validation_data = convert_to_gliner(validation_data)

with open("data/conll2003/train_dataset.json", "w") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False) 

with open("data/conll2003/test_dataset.json", "w") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False) 

with open("data/conll2003/validation_dataset.json", "w") as f:
    json.dump(validation_data, f, indent=4, ensure_ascii=False) 