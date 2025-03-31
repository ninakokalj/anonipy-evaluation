from ollama import chat
from pydantic import BaseModel
import numpy as np
import re
import json
import pandas as pd
import random

class Organizations(BaseModel):
    organizations: list[str]

class Usernames(BaseModel):
    usernames: list[str]

class Addresses(BaseModel):
    addresses: list[str]


messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for generating addresses.",
            },
            {
                "role": "user",
                "content": f"Generate me 50 different addresses. Respond only with the addresses.",
            }
        ]

mess = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for generating organizaton names.",
            },
            {
                "role": "user",
                "content": f"Respond with 20 different real organization names in English. Respond only with the names.",
            }
        ]


# personalizirano organizacijam
def generate_data(messages: list, structured_output: bool = False):

    if structured_output:
        response = chat(
            messages=messages,
            model="qwen2.5:14b",
            format=Addresses.model_json_schema()
        )
        data = Addresses.model_validate_json(response.message.content).addresses
        np.savetxt("data/training/address_eng.csv", data, delimiter=',', fmt='%s')
    else:   
        response = chat(
                messages=messages,
                model="qwen2.5:14b"
            ) 
        data = response.message.content
    return data



# očisti gpt odgovore - odstrani jim zaporedno številko
def format_gpt_orgs(file_path: str):

    orgs = set()

    f = open(file_path, "r")
    for line in f:
        cleaned_content = re.sub(r"^\d+\.\s*", "", line, flags=re.DOTALL)
        orgs.add(cleaned_content.strip())

    np.savetxt("data/training/gpt.csv", list(orgs), delimiter=',', fmt='%s')



# dva dataseta premeša in izloči ponovitve
def shuffle(first_data, second_data, output_file):
    mn = set()
    f = open(first_data, "r")
    for line in f:
        mn.add(line.strip())

    f = open(second_data, "r")
    for line in f:
        mn.add(line.strip())

    np.savetxt(output_file, list(mn), delimiter=',', fmt='%s')



# iz csv fajla naredi json prompte
def csv_2_json(language, label, csv_path, json_path):

    df = pd.read_csv(csv_path, header=None)

    data = [
            {"instruction": f"What is a random {language} {label} replacement for {df[0][i]}?", 
                "output": df[0][random.choice([j for j in range(0, len(df)) if j != i])]
            } 
            for i in range(0, len(df))
            ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False) 

def names_2_json(language, csv_path, json_path):

    df = pd.read_csv(csv_path, header=None)
    labels = ["osebno ime", "ime"]

    data = [
            {"instruction": f"What is a random {language} {random.choice(labels)} replacement for {df[0][i]}?", 
                "output": df[0][random.choice([j for j in range(0, len(df)) if j != i])]
            } 
            for i in range(0, len(df))
            ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False) 

def dates_2_json(language, label, csv_path, json_path):

    df = pd.read_csv(csv_path, header=None)
    dates = []
    for i in range(0, 8):
        x = i * 60
        y = i * 60 + 59
        data = [
                {"instruction": f"What is a random {language} {label} replacement for {df[0][i]}?", 
                    "output": df[0][random.choice([j for j in range(x, y) if j != i])]
                } 
                for i in range(x, y)
                ]
        dates.extend(data)

    with open(json_path, "w") as f:
        json.dump(dates, f, indent=4, ensure_ascii=False) 

def adds_2_json(language, label, csv_path, json_path):
    f = open(csv_path, "r")
    df = [line.strip() for line in f]
    dates = []

    for i in range(0, 3):
        x = i * 100
        y = i * 100 + 100
        data = [
                {"instruction": f"What is a random {language} {label} replacement for {df[i]}?", 
                    "output": df[random.choice([j for j in range(x, y) if j != i])]
                } 
                for i in range(x, y)
                ]
        dates.extend(data)

    with open(json_path, "w") as f:
        json.dump(dates, f, indent=4, ensure_ascii=False) 


#shuffle("data/training/person_slo.csv", "data/training/person2_slo.csv", "data/training/person_slo.csv")

csv_2_json("Slovene", "osebno ime", "data/training/slo/person_slo.csv", "data/training/slo/person_slo.json")
# names_2_json("Slovene", "data/training/slo/imena_wiki.csv", "data/training/slo/imena.json")
# generate_data(messages, structured_output=True)

# adds_2_json("English", "address", "data/training/address_eng.txt", "data/training/address_eng.json")
# format_gpt_orgs("data/training/gpt.txt")

def shuffle_all():

    with open("data/training/eng/person_eng.json", "r") as f:
        person = json.load(f)
    all = person
    with open("data/training/eng/address_eng.json", "r") as f:
        address = json.load(f)
    all.extend(address)
    with open("data/training/eng/emails_eng.json", "r") as f:
        emails = json.load(f)
    all.extend(emails)
    with open("data/training/eng/orgs_eng.json", "r") as f:
        orgs = json.load(f)
    all.extend(orgs)
    with open("data/training/eng/dates_eng.json", "r") as f:
        dates = json.load(f)
    all.extend(dates)
    with open("data/training/eng/pass_eng.json", "r") as f:
        passport = json.load(f)
    all.extend(passport)
    with open("data/training/eng/tax_eng.json", "r") as f:
        tax = json.load(f)
    all.extend(tax)
    with open("data/training/eng/user_eng.json", "r") as f:
        user = json.load(f)
    all.extend(user)

    random.shuffle(all)
    with open("data/training/eng/mix_eng.json", "w") as f:
        json.dump(all, f, indent=4, ensure_ascii=False)


# shuffle_all()