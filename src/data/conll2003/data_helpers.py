import re


#===============================================	
# CoNLL-2003 DATASET HELPER FUNCTIONS
#===============================================


# IOB2 tagging scheme
label_dict = {0: "O", 1: "B-PER" , 2: "I-PER" , 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
tags_dict = {"PER": "person", "ORG": "organization", "LOC": "location", "MISC": "miscellaneous"} 

def convert_to_gliner(dataset) -> list:
    """Transforms the dataset into a structured JSON format expected by the GLiNER model."""

    return [
        {
            "text": create_text(sample["tokens"]),
            "language": "English",
            "tokenized_text": sample.pop("tokens"),  # Rename to 'tokenized_text'
            "ner": ner_entities  # Rename to 'ner' and create ner entities
        }
        for sample in dataset
        if "tokens" in sample and "ner_tags" in sample
        if (ner_entities := create_ner(sample["ner_tags"])) # Filter out samples without entities
    ]

def create_text(tokens: list) -> str:
    """Converts a list of tokens into a string."""

    text = " ".join(tokens)
    text = re.sub(r"\s([.,!?;:'])", r"\1", text)
    return text

def create_ner(ner_tags: list) -> list:
    """Uses IOB2 tagging scheme to construct entities.

    Returns:
        A list of entities in the format [[start, end, entity_type], ...]
    """

    entities = []
    start, end, label = None, None, None

    for idx, tag in enumerate(ner_tags):
        tag_label = label_dict[tag]
        if tag_label.startswith("B-"):
            if start is not None:
                entities.append([start, end, tags_dict[label]])
                start, end, label = None, None, None
            start, end = idx, idx
            label = tag_label.split("-")[1]
        elif tag_label.startswith("I-"):
            end = idx

    if start is not None:
                entities.append([start, end, tags_dict[label]])

    return entities
    
        
    
        