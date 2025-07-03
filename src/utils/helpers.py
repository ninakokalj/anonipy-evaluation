import csv
import json
from typing import List

from anonipy.definitions import Entity
from token_splitter import SpaCyTokenSplitter
from utils.llm_label_generator import LLMLabelGenerator
from src.utils.ollama_label_generator import OllamaLabelGenerator


def get_true_entities(ner_ents: List[List], tokenized_text: List[str]) -> List[dict]:
  """Returns a list of true entities for a given example."""

  example_true_ents = []
  for ent in ner_ents:
    type = ent[2]
    start = ent[0]
    end = ent[1]
    text = " ".join(tokenized_text[start:end + 1])
    example_true_ents.append({"text": text, "label": type, "start_index": start, "end_index": end})

  return example_true_ents


def get_labels(entities: List[List]) -> List[str]:
  """Returns a list of labels for a given example."""
  
  return list({ent[2] for ent in entities})


def generate_entities(true_ents: List[dict]) -> List[Entity]:
  """Returns a list of Entity objects for a given example."""

  entities = []
  for ent in true_ents:
    entities.append(
      Entity(
        text=ent["text"], 
        label=ent["label"],
        start_index=ent["start_index"],
        end_index=ent["end_index"],
        ))
  return entities


def generate_LLM_labels(test_dataset_path: str, new_dataset_output_path: str, llm_generator: object, use_entity_attrs: bool = False):
    """Saves a new dataset with LLM generated entities."""
    
    # Dataset to modify
    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    # File of replacements
    with open("data/replacements.csv", 'w', newline='') as csvfile:
      fieldnames = ['label', 'original_text', 'generated_text']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
      writer.writeheader()
      counter = 0

      # Attributes to modify: 'text', 'tokenized_text', 'ner'
      for data in test_dataset:
          
          # Get entities to replace
          true_entities = get_true_entities(data["ner"], data["tokenized_text"])
          # Turn those entities into Entity objects
          entities = generate_entities(true_entities)

          new_entities = []  # list of new entities
          replacements = {}  # dictionary of replacements

          for ent in entities:
              
              # If the entity has already been replaced under different label
              if ((ent.start_index, ent.end_index) in replacements):
                generated_text = replacements[(ent.start_index, ent.end_index)]
              
              else:
                if use_entity_attrs:
                  entity_attrs = data["language"]
                else:
                  entity_attrs = ""

                # LLM from the Transformers library
                if isinstance(llm_generator, LLMLabelGenerator):
                  generated_text = llm_generator.generate(entity=ent, add_entity_attrs = entity_attrs, temperature=0.7)
                # LLM from the Ollama library
                elif isinstance(llm_generator, OllamaLabelGenerator):
                  generated_text = llm_generator.generate(entity=ent, add_entity_attrs = entity_attrs, structured_output = True)
                else:
                  raise ValueError("Invalid LLM generator")
                
                # Fixes problems with parsing
                if generated_text.endswith("."):
                  generated_text = generated_text[:-1]
                if (generated_text == ""):
                  generated_text = " "

                writer.writerow({'label': ent.label, 'original_text': ent.text, 'generated_text': generated_text})
                replacements[(ent.start_index, ent.end_index)] = generated_text

                # Replace the first occurrence of the original entity in text
                data["text"] = data["text"].replace(ent.text, generated_text, 1)
              
              new_entities.append({"text": generated_text, "label": ent.label}) # for 'tokenized_text' & 'ner'
              
            
          # Replace original entities in attributes 'tokenized_text' & 'ner'
          updated_data = create_new_gliner_example(data, new_entities)
          # Update the dataset
          data.update(updated_data)
          counter += 1
          print(counter)  
    
      # Save the new modified dataset
      with open(new_dataset_output_path, "w") as f:
        json.dump(test_dataset, f, indent=4, ensure_ascii=False)


def find_sub_list(sub_list: list, main_list: list) -> list:
    """Returns the indices of the sub_list in the main_list. The sub_list is tokenized entity text."""

    sub_list_indices = []
    sll = len(sub_list)
    for idx in (i for i, e in enumerate(main_list) if e == sub_list[0]):
        if main_list[idx : idx + sll] == sub_list:  
            sub_list_indices.append((idx, idx + sll - 1))

    return sub_list_indices


def remove_duplicates(lst: list) -> list:
    """Removes duplicates from a list of lists."""

    unique_tuples = set(tuple(sublist) for sublist in lst)
    result = sorted([list(t) for t in unique_tuples], key=lambda x: (x[0], x[1]))
    return result


LANGUAGE_MAPPING = {
    "Dutch": ("nl", "Dutch"),
    "Slovene": ("sl", "Slovenian"),
    "Italian": ("it", "Italian"),
    "Greek": ("el", "Greek"),
    "French": ("fr", "French"),
    "English": ("en", "English"),
    "German": ("de", "German"),
}

def create_new_gliner_example(example: dict, new_entities: list) -> dict:
    """
    Updates attributes 'ner' and 'tokenized_text' in a GLiNER-compatible example.

    This function tokenizes the input text and identifies the indices of the 
    specified entities within the tokenized text. It then updates the Named 
    Entity Recognition (NER) labels for the example with these entities.

    Args:
        example (dict): A dictionary containing the original text and its 
            language. Example format: 
            {
                "text": "Nina Kokalj",
                "language": "English",
                "tokenized_text": ["Elon", "Musk"], -> this column has to be corrected
                "ner": [[0, 1, "PERSON"]] -> this column has to be corrected
            }
        new_entities (list): A list of new entities to be annotated in the text. 
            Each entity is a dictionary with "text" and "label" keys. Example format:
            [
                {"text": "Nina Kokalj", "label": "PERSON"}
            ]

    Returns:
        dict: A dictionary containing the original text, language, updated tokenized text,
        and updated NER labels. The NER labels are deduplicated and sorted by 
        their start and end indices.
    """

    tokenizer = SpaCyTokenSplitter(
        lang=LANGUAGE_MAPPING[example["language"]]
    )

    ttext = [t[0] for t in tokenizer(example["text"])]
    ner = []
    for entity in new_entities: # new_entities = [{"text": text, "label": label}]
        entity_text = [t[0] for t in tokenizer(entity["text"])]   
        entity_indices = find_sub_list(entity_text, ttext) 
        for entity_index in entity_indices:
            ner.append([entity_index[0], entity_index[1], entity["label"]])

    return {
        "text": example["text"],
        "language": example["language"],
        # "domain": example["domain"],
        "tokenized_text": ttext,
        "ner": remove_duplicates(ner),
    }