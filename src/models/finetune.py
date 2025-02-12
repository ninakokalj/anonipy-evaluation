import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from finetune_helpers import train_gliner_model



with open("data/train_dataset.json", "r") as f:
    train_dataset = json.load(f)

with open("data/validation_dataset.json", "r") as f:
    test_dataset = json.load(f)


train_gliner_model(train_dataset, test_dataset)




