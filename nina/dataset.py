from datasets import load_dataset
import json
import random
import numpy as np
from anonipy.anonymize.generators.llm_label_generator import LLMLabelGenerator

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

# python -m evaluation.dataset


   
dataset = load_dataset("E3-JSI/synthetic-multi-pii-ner-v1", split="train")

# za gliner
def convert_to_gliner(dataset):
  """Converts the dataset into a list of objects suitable to train the GLiNER model"""
  dataset = dataset.to_pandas()
  dataset = dataset[["gliner_tokenized_text", "gliner_entities"]]
  dataset = dataset.rename(columns={"gliner_tokenized_text": "tokenized_text", "gliner_entities": "ner"})
  dataset["ner"] = dataset["ner"].apply(lambda x: json.loads(x))
  return dataset.to_dict(orient="records") 

# convert the dataset to GLiNER compatible format
data = convert_to_gliner(dataset)

random.seed(42)
random.shuffle(data)

train_dataset = data[:int(len(data)*0.8)]
test_dataset = data[int(len(data)*0.8):]

# balanced za jezike
# balanced za domene
# subsete in poshufflas 

# medical term -> common folder -> evaluate ner

"""
1) razdelim dataset 80% train, 20% test
2) NER extractor naučim na train
3) evaluation na test
Za vsak llm:
4) llm na testne podatke
5) NER testiram + evaluation

"""
def train_model(train_dataset, test_dataset):

  device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  model = GLiNER.from_pretrained("urchade/gliner_small")
  data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

  num_steps = 500
  batch_size = 8
  data_size = len(train_dataset)
  num_batches = data_size // batch_size
  num_epochs = max(1, num_steps // num_batches)

  training_args = TrainingArguments(
      output_dir="models",
      learning_rate=5e-6,
      weight_decay=0.01,
      others_lr=1e-5,
      others_weight_decay=0.01,
      lr_scheduler_type="linear", #cosine
      warmup_ratio=0.1,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      focal_loss_alpha=0.75,
      focal_loss_gamma=2,
      num_train_epochs=num_epochs,
      evaluation_strategy="steps",
      save_steps = 100,
      save_total_limit=10,
      dataloader_num_workers = 0,
      use_cpu = False,
      report_to="none",
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      tokenizer=model.data_processor.transformer_tokenizer,
      data_collator=data_collator,
  )

  trainer.train()

  trained_model = GLiNER.from_pretrained("models/checkpoint-100", load_tokenizer=True)
  return trained_model


def evaluate(trained_model, test_dataset):
  true_ents = []
  pred_ents = []
# Perform entity prediction
  for example in test_dataset:
    text = " ".join(example["tokenized_text"])
    # print(example)
    # Labels for entity prediction
    labels = list({ent[2] for ent in example["ner"]})
    pred_ents.append(trained_model.predict_entities(text, labels, threshold=0.5))
    # print(pred_ents)
    # True entities
    example_true_ents = []
    for ent in example["ner"]:
      type = ent[2]
      start = ent[0]
      end = ent[1]
      text = " ".join(example["tokenized_text"][start:end + 1])

      example_true_ents.append({"text": text, "label": type})

    true_ents.append(example_true_ents)
    # print(true_ents)
  
  p, r, f1 = evaluate_ner_performance(true_ents, pred_ents)
  return p, r, f1


"""
trained_model = train_model(train_dataset, test_dataset)
p, r, f1 = evaluate(trained_model, test_dataset)
print("precission: {}, recall: {}, f1: {}".format(p, r, f1))
"""
trained_model = train_model(train_dataset, test_dataset)
p, r, f1 = evaluate(trained_model, test_dataset)
print("precission: {}, recall: {}, f1: {}".format(p, r, f1))

# precission: 0.722887323943662, recall: 0.6117401668653158, f1: 0.6626856036152358

"""
LLMs

meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
HuggingFaceTB/SmolLM2-135M-Instruct
HuggingFaceTB/SmolLM2-360M-Instruct
HuggingFaceTB/SmolLM2-135M-Instruct
mistralai/Mistral-7B-Instruct-v0.3
microsoft/Phi-3.5-mini-instruct
Qwen/Qwen2.5-7B-Instruct
"""
