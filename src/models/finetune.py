import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from finetune_helpers import train_gliner_model



with open("data/conll2003/train_dataset.json", "r") as f:
    train_dataset = json.load(f)

with open("data/conll2003/validation_dataset.json", "r") as f:
    validation_dataset = json.load(f)


train_gliner_model(train_dataset, validation_dataset)




