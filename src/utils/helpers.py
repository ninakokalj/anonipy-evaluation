import json

from typing import List

from token_splitter import create_new_gliner_example
from anonipy.definitions import Entity

def get_true_ents(ner_ents: List[List], tokenized_text: List[str]) -> List[dict]:
  """Returns a list of true entities for a given example."""

  example_true_ents = []
  for ent in ner_ents:
    type = ent[2]
    start = ent[0]
    end = ent[1]
    text = " ".join(tokenized_text[start:end + 1])
    example_true_ents.append({"text": text, "label": type})
  return example_true_ents


def get_labels(entities: List[List]) -> List[str]:
   """Returns a list of labels for a given example."""
   labels = list({ent[2] for ent in entities})
   return labels


def generate_entities(text: str, true_ents: List[dict]) -> List[Entity]:
  """Returns a list of Entity objects for a given example."""

  used_indices = set()

  entities = []
  for ent in true_ents:
    start, end = find_indices(text, ent["text"], used_indices)
    entities.append(
      Entity(
        text=ent["text"], 
        label=ent["label"],
        start_index=start,
        end_index=end,
        ))
  return entities
  

def find_indices(text: str, entity_text: str, used_indices: set) -> tuple:
    """Returns the start and end indices of a given entity in a given text."""

    entity_length = len(entity_text)
    start_index = -1

    for i in range(len(text) - entity_length + 1):
        if (
            text[i:i + entity_length] == entity_text
            and i not in used_indices
        ):
            start_index = i
            for j in range(i, i + entity_length):
                used_indices.add(j)
            break


    end_index = start_index + entity_length - 1

    return start_index, end_index


def generate_LLM_labels(test_dataset_path: str, new_dataset_output_path: str, llm_generator: object):
    """Saves a new dataset with LLM generated entities."""

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    # zamenjat morm: text, tokenized_text, ner
    for data in test_dataset:
        
        true_ents = get_true_ents(data["ner"], data["tokenized_text"])  # dobim pravilne entitete, ki jih mora najdit ner
        entities = generate_entities(data["text"], true_ents)   # iz pravilnih entitet generiram pravilne Entity objekte
        
        for ent in entities:
            generated_text = llm_generator.generate(ent, temperature=0.7)   # zgeneriram nov text za entity
            data["text"] = data["text"].replace(ent.text, generated_text)   # zamenjam v textu
            
            # zamenjam v tokenized_text
            generated_words = generated_text.split(" ")
            original_ner_entity = [ner for ner in data["ner"] if ner[2] == ent.label][0] 
            
            start_index, end_index, label = original_ner_entity
            data["tokenized_text"] = data["tokenized_text"][:start_index] + generated_words + data["tokenized_text"][end_index + 1:]
            
            # zamenjam v ner
            index = data["ner"].index(original_ner_entity)
            data["ner"][index] = [start_index, start_index + len(generated_words) - 1, generated_text]
    
    with open(new_dataset_output_path, "w") as f:
      json.dump(test_dataset, f, indent=4)


# zamenjam v textu
# naredim nove pravilne entitije (text, label)
# naredim tokenized text
# iz tega ner 

def generate_LLM_labels(test_dataset_path: str, new_dataset_output_path: str, llm_generator: object):
    """Saves a new dataset with LLM generated entities."""

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    # zamenjat morm: text, tokenized_text, ner
    for data in test_dataset:
        
        true_ents = get_true_ents(data["ner"], data["tokenized_text"])  # dobim pravilne entitete, ki jih mora najdit ner
        entities = generate_entities(data["text"], true_ents)   # iz pravilnih entitet generiram pravilne Entity objekte
        new_entities = []

        for ent in entities:
            generated_text = llm_generator.generate(ent, temperature=0.7)   # zgeneriram nov text za entity
            if (generated_text == ""):
              print("empty")
              generated_text = " "
            new_entities.append({"text": generated_text, "label": ent.label}) # list of new entites, label stays the same, text is replaced
            data["text"] = data["text"].replace(ent.text, generated_text)   # zamenjam v textu
          
        # zamenjam v 'tokenized_text' in 'ner'
        updated_data = create_new_gliner_example(data, new_entities)
        data.update(updated_data)
            
            
    with open(new_dataset_output_path, "w") as f:
      json.dump(test_dataset, f, indent=4)