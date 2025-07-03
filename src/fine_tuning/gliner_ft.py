import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from src.fine_tuning.gliner_helpers import train_gliner_model


# load data
with open("data/conll2003/train_dataset.json", "r") as f:
    train_dataset = json.load(f)

with open("data/conll2003/validation_dataset.json", "r") as f:
    validation_dataset = json.load(f)

# fine-tune model
train_gliner_model(train_dataset, validation_dataset)
