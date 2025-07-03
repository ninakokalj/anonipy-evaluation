import csv
import json
import random

import numpy as np
import pandas as pd
from pydantic import BaseModel
from ollama import chat
from typing import Union, List, Type


#===============================================	
# SYNTHETIC DATA GENERATION HELPER FUNCTIONS
#===============================================

# classes for Ollama's structured output
class Organizations(BaseModel):
    organizations: list[str]

class Usernames(BaseModel):
    usernames: list[str]

class Addresses(BaseModel):
    addresses: list[str]


def _create_message(label: str, language: str, num_cases: int) -> list:
    return [
                {
                    "role": "system",
                    "content": f"You are a helpful AI assistant for generating {label}.",
                },
                {
                    "role": "user",
                    "content": f"Generate me {num_cases} different {language} {label}. Respond only with the {label}.",
                }
            ]
    
def generate_data(label: str, language: str, num_cases: int, output_class: Type[BaseModel], structured_output: bool = False, model_name: str = "qwen2.5:14b") -> str:
    """Generates synthetic data with a language model."""
    if structured_output:
        response = chat(
            messages=_create_message(label, language, num_cases),
            model=model_name,
            format=output_class.model_json_schema()
        )
        data = output_class.model_validate_json(response.message.content).addresses
        np.savetxt(f"data/training/{label}_{language}.csv", data, delimiter=',', fmt='%s')
    else:   
        response = chat(
                messages=_create_message(label, language, num_cases),
                model=model_name
            ) 
        data = response.message.content
    return data


def shuffle(first_data, second_data, output_file):
    """Shuffles two datasets and removes duplicates resulting from the merge.
        Saves the result to a CSV file."""

    lines = set()

    for file_path in [first_data, second_data]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines.update(line.strip() for line in f if line.strip())

    shuffled = list(lines)
    np.random.shuffle(shuffled)

    np.savetxt(output_file, shuffled, delimiter=',', fmt='%s')


def csv_2_json(language: str, label: Union[str, List[str]], csv_path: str, json_path: str):
    """Generates a JSON file with instruction-output pairs from a CSV file of entities for a given language and label."""

    df = pd.read_csv(csv_path, header=None)

    def choose_label():
        return label if isinstance(label, str) else random.choice(label)
        
    data = [
            {"instruction": f"What is a random {language} {choose_label()} replacement for {df[0][i]}?", 
                "output": df[0][random.choice([j for j in range(0, len(df)) if j != i])]
            } 
            for i in range(0, len(df))
            ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# used for putting same format dates in the same instruction-output pair
def formats_2_json(language: str, labels: list, csv_path: str, json_path: str, num_formats: int, num_cases: int):

    df = pd.read_csv(csv_path, header=None)
    dates = []

    for i in range(0, num_formats):

        x = i * num_cases
        a = num_cases - 1
        y = i * num_cases + a

        data = [
                {"instruction": f"What is a random {language} {random.choice(labels)} replacement for {df[0][i]}?", 
                    "output": df[0][random.choice([j for j in range(x, y) if j != i])]
                } 
                for i in range(x, y+1)
                ]
        dates.extend(data)

    with open(json_path, "w") as f:
        json.dump(dates, f, indent=4, ensure_ascii=False) 


labels = ["person", "address", "emails", "orgs", "dates", "passport", "tax", "user"]
def merge_data(language: str, num_cases: int):
    """Shuffles and saves all the data for a given language in the same file."""
    all_data = []
    for file_path in [f"data/training/data_{language}/{label}.json" for label in labels]:
        with open(file_path, "r") as f:
            data = random.shuffle(json.load(f))
            all_data.extend(data[:num_cases])

    random.shuffle(all_data)
    with open(f"data/training/data_{language}/DATASET_{num_cases}.json", "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


def merge_all_languages(num_cases: int):
    """Shuffles and saves data for all languages in the same file."""
    language = ["ENGLISH", "SLOVENE", "GREEK", "ITALIAN", "FRENCH", "GERMAN", "DUTCH"]
    all_data = []

    for lang in language:
        with open(f"data/training/data_{lang}/DATASET_{num_cases}.json", "r") as f:
            data = json.load(f)
            all_data.extend(data)

    random.shuffle(data)
    with open(f"data/training/data_ALL/DATASET_{num_cases}.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


#===============================================	
# ALL ENTITIES DATASET FUNCTIONS
#===============================================


# used to create a dataset with all the generated synthetic data
# attributes of the new dataset are: Entity Language, Entity Type, Entity Text
LANGUAGES = {"en": "English", "sl": "Slovenian", "el": "Greek", "it": "Italian", "fr": "French", "de": "German", "nl": "Dutch"}
LABELS = {"names": "Name", "lastn": "Last Name", "address": "Address", "orgs": "Organization/Company", "emails": "Email", "user": "Username"}
def create_dataset(filepath: str):
    data_all_languages = ["en", "de", "nl", "sl", "it", "fr","el"]
    labels = ["names", "lastn", "address", "orgs", "emails", "user"]
    results = []
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Entity Language", "Entity Type", "Entity Text"])
         
        for lang in data_all_languages:
            language = LANGUAGES[lang]
            for label in labels:
                csv_path = f"data/training/helpers/{lang}/{label}.csv"
                label = LABELS[label]
                df = pd.read_csv(csv_path, header=None)
                results.extend([(language, label, df[0][i])  for i in range(0, len(df))])

        writer.writerows(results)


# used to get statistics of the generated synthetic dataset
def statistics(filepath: str):

    data_all_languages = ["en", "de", "nl", "sl", "it", "fr","el"]
    labels = ["names", "lastn", "address", "orgs", "emails", "user"]
    results = []

    df = pd.read_csv("data/training/ALL_data.csv", header=None)

    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Language", "Name", "Last Name", "Address","Organization/Company", "Email", "Username"])
         
        for lang in data_all_languages:
            language = LANGUAGES[lang]
            res = [language]
            for label in labels:
                label = LABELS[label]
                cases = [df[2][i] for i in range(0, len(df)) if df[1][i] == label and df[0][i] == language]
                res.append(len(cases))
        
            results.append(res)

        writer.writerows(results)
