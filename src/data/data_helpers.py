import json

import random
from collections import defaultdict


#===============================================	
# DATASET HELPER FUNCTIONS
#===============================================


def convert_to_gliner(dataset: list) -> list:
    """Converts the dataset into a list of objects suitable to train the GLiNER model"""

    return [
        {
            "text": sample["text"],  
            "language": sample["language"],
            "domain": sample["domain"],
            "tokenized_text": sample.pop("gliner_tokenized_text"),  # Rename to 'tokenized_text'
            "ner": json.loads(sample.pop("gliner_entities", "[]"))  # Rename to 'ner' and parse JSON
        }
        for sample in dataset
        if "text" in sample and "gliner_tokenized_text" in sample and "gliner_entities" in sample
    ]


def split_dataset(data: list, train: int = 0.8) -> tuple:
    """Splits the dataset into train and test sets based on the specified ratio."""

    grouped_data = defaultdict(list)

    for d in data:
        grouped_data[(d["language"], d["domain"])].append(d)

    train_dataset = []
    test_dataset = []

    for (language, domain), records in grouped_data.items():
        random.shuffle(records)  
        split_point = int(len(records) * train) 
        train_dataset.extend(records[:split_point])  # Add to train dataset
        test_dataset.extend(records[split_point:])  # Add to test dataset

    return train_dataset, test_dataset




