import json

from typing import List

from token_splitter import create_new_gliner_example
from anonipy.definitions import Entity, Replacement

def get_true_ents(ner_ents: List[List], tokenized_text: List[str]) -> List[dict]:
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
   labels = list({ent[2] for ent in entities})
   return labels


def generate_entities(true_ents: List[dict]) -> List[Entity]:
  """Returns a list of Entity objects for a given example."""

  used_indices = set()

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



# zamenjam v textu
# naredim nove pravilne entitije (text, label)
# naredim tokenized text
# iz tega ner 
   
def generate_LLM_labels(test_dataset_path: str, new_dataset_output_path: str, llm_generator: object, use_entity_attrs: bool = False):
    """Saves a new dataset with LLM generated entities."""

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    repl = {}
    counter = 0
    # zamenjat morm: text, tokenized_text, ner
    for data in test_dataset:
        
        true_ents = get_true_ents(data["ner"], data["tokenized_text"])  # dobim pravilne entitete, ki jih mora najdit ner
        entities = generate_entities(true_ents)   # iz pravilnih entitet generiram pravilne Entity objekte
        new_entities = []
  
        replacements = {}  # Store generated replacements

        for ent in entities:
            if ((ent.start_index, ent.end_index) in replacements):
              generated_text = replacements[(ent.start_index, ent.end_index)]
            else:
              if use_entity_attrs:
                generated_text = llm_generator.generate(entity=ent, add_entity_attrs = data["language"], temperature=0.7)
              else:
                generated_text = llm_generator.generate(entity=ent, temperature=0.7)   # zgeneriram nov text za entity
              
              if generated_text.endswith("."):
                generated_text = generated_text[:-1]
              if (generated_text == ""):
                generated_text = " "

              repl[ent.text] = generated_text
              replacements[(ent.start_index, ent.end_index)] = generated_text
              data["text"] = data["text"].replace(ent.text, generated_text, 1)   # zamenjam v textu prvo ponovitev
            
            new_entities.append({"text": generated_text, "label": ent.label}) 
            
          
        # zamenjam v 'tokenized_text' in 'ner'
        updated_data = create_new_gliner_example(data, new_entities)
        data.update(updated_data)

        counter += 1
        print(counter)  
    
    with open("data/replacements.json", "w") as f:
      json.dump(repl, f, indent=4, ensure_ascii=False)

    with open(new_dataset_output_path, "w") as f:
      json.dump(test_dataset, f, indent=4, ensure_ascii=False)