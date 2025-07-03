import json

from datasets import load_dataset

from data_helpers import convert_to_gliner, split_dataset, remove_overlapping_entities

# load data
data = load_dataset("E3-JSI/synthetic-multi-pii-ner-v1", split="train")

# convert data into GLiNER compatible format
data = convert_to_gliner(data)

# split data 80 : 20 for testing
train_dataset, test_dataset = split_dataset(data)
# additionaly split the 80 into 90 : 10 for validation
train_dataset, validation_dataset = split_dataset(train_dataset, train=0.9)

# remove overlapping entities because it's impossible to handle them when modifying the test dataset
remove_overlapping_entities(test_dataset)

# save data as JSON
with open("data/E3-JSI/train_dataset.json", "w") as f:
    json.dump(train_dataset, f, indent=4, ensure_ascii=False)  
with open("data/E3-JSI/test_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=4, ensure_ascii=False)
with open("data/E3-JSI/validation_dataset.json", "w") as f:
    json.dump(validation_dataset, f, indent=4, ensure_ascii=False)