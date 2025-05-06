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
                "content": f"Respond with 20 different real organization names in GERMAN. Respond only with the names.",
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
def csv_2_json(language: str, label: str, csv_path: str, json_path: str):

    df = pd.read_csv(csv_path, header=None)

    data = [
            {"instruction": f"What is a random {language} {label} replacement for {df[0][i]}?", 
                "output": df[0][random.choice([j for j in range(0, len(df)) if j != i])]
            } 
            for i in range(0, len(df))
            ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False) 


# ko sestavljaš naslove: df[0] je ulica, df[1] je pošta
def address_2_json(language: str, label: str, csv_path: str, json_path: str):

    df = pd.read_csv(csv_path, header=None)

    data = []
    for i in range(0, len(df)):
        j = random.choice([j for j in range(0, len(df)) if j != i])
        data.append(
            {"instruction": f"What is a random {language} {label} replacement for {df[0][i]},{df[1][i]}?", 
                "output": f"{df[0][j]},{df[1][j]}"  
            } 
        )

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# labele podaš kot seznam in se naključno izbirajo
def labels_2_json(language: str, labels: list, csv_path: str, json_path: str):

    df = pd.read_csv(csv_path, header=None)

    data = [
            {"instruction": f"What is a random {language} {random.choice(labels)} replacement for {df[0][i]}?", 
                "output": df[0][random.choice([j for j in range(0, len(df)) if j != i])]
            } 
            for i in range(0, len(df))
            ]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False) 


# da so isti formati skupaj
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


# združi vse json fajle
def shuffle_all():

    with open("data/training/data_GERMAN/person.json", "r") as f:
        person = json.load(f)
    all = person
    with open("data/training/data_GERMAN/address.json", "r") as f:
        address = json.load(f)
    all.extend(address)
    with open("data/training/data_GERMAN/emails.json", "r") as f:
        emails = json.load(f)
    all.extend(emails)
    with open("data/training/data_GERMAN/orgs.json", "r") as f:
        orgs = json.load(f)
    all.extend(orgs)
    with open("data/training/data_GERMAN/dates.json", "r") as f:
        dates = json.load(f)
    all.extend(dates)
    with open("data/training/data_GERMAN/passport.json", "r") as f:
        passport = json.load(f)
    all.extend(passport)
    with open("data/training/data_GERMAN/tax.json", "r") as f:
        tax = json.load(f)
    all.extend(tax)
    with open("data/training/data_GERMAN/user.json", "r") as f:
        user = json.load(f)
    all.extend(user)

    random.shuffle(all)
    with open("data/training/data_GERMAN/DATASET_de.json", "w") as f:
        json.dump(all, f, indent=4, ensure_ascii=False)



# skrajšam dataset na x primerov za labelo
def shorten_dataset(num_cases: int):

    with open("data/training/data_GERMAN/person.json", "r") as f:
        person = json.load(f)
    random.shuffle(person)
    all = person[:num_cases]
    with open("data/training/data_GERMAN/address.json", "r") as f:
        address = json.load(f)
    random.shuffle(address)
    all.extend(address[:num_cases])
    with open("data/training/data_GERMAN/emails.json", "r") as f:
        emails = json.load(f)
    random.shuffle(emails)
    all.extend(emails[:num_cases])
    with open("data/training/data_GERMAN/orgs.json", "r") as f:
        orgs = json.load(f)
    random.shuffle(orgs)
    all.extend(orgs[:num_cases])
    with open("data/training/data_GERMAN/dates.json", "r") as f:
        dates = json.load(f)
    random.shuffle(dates)
    all.extend(dates[:num_cases])
    with open("data/training/data_GERMAN/passport.json", "r") as f:
        passport = json.load(f)
    random.shuffle(passport)
    all.extend(passport[:num_cases])
    with open("data/training/data_GERMAN/tax.json", "r") as f:
        tax = json.load(f)
    random.shuffle(tax)
    all.extend(tax[:num_cases])
    with open("data/training/data_GERMAN/user.json", "r") as f:
        user = json.load(f)
    random.shuffle(user)
    all.extend(user[:num_cases])

    random.shuffle(all)
    with open(f"data/training/data_GERMAN/DATASET_de_{num_cases}.json", "w") as f:
        json.dump(all, f, indent=4, ensure_ascii=False)


#formats_2_json("French", ["date", "date", "date de naissance"], "data/training/helpers/fr/dates.csv", "data/training/data_FRENCH/dates.json", 5, 80)
#shuffle("data/training/helpers/fr/names.csv", "data/training/helpers/fr/nam.csv", "data/training/helpers/fr/names.csv")
#address_2_json("GERMAN", "adresse", "data/training/helpers/de/address.csv", "data/training/data_GERMAN/address1.json")
#csv_2_json("Greek", "ΑΦΜ", "data/training/helpers/el/tax.csv", "data/training/data_GREEK/tax.json")
#labels_2_json("GERMAN", ["ime", "osebno ime", "oseba"], "data/training/helpers/sl/person.csv", "data/training/data_GERMAN/person.json")
#generate_data(messages, structured_output=True)
#shuffle_all()
#shorten_dataset(200)
