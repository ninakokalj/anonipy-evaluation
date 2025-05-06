import random
import numpy as np
import pandas as pd
import re

# AK1234567
# A12345678
def generate_passport_nums(csv_path: str):
    nums = set()

    while len(nums) < 400:

        passport_num = chr(random.randint(65, 90))
        if random.randint(1, 2) == 2:
            passport_num += chr(random.randint(65, 90))

        while len(passport_num) < 9:
            passport_num += str(random.randint(0, 9))

        nums.add(passport_num)

    np.savetxt(csv_path, list(nums), delimiter=',', fmt='%s')


# 920-43-3453
def generate_tax_ids(csv_path: str):

    ids = set()
    while len(ids) < 400:
        first = random.randint(100, 999)
        second = random.randint(10, 99)
        third = random.randint(1000, 9999)
        ids.add(f"{first}-{second}-{third}")
    
    np.savetxt(csv_path, list(ids), delimiter=',', fmt='%s')


# 1 80 04 76 123 456 78
def generate_fr_tax_ids(csv_path: str):

    ids = set()
    while len(ids) < 400:
        ids.add(f"{random.randint(0, 9)} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(10, 99)}")
    
    np.savetxt(csv_path, list(ids), delimiter=',', fmt='%s')


# 123456789
def generate_slo_tax_ids(csv_path: str):

    ids = set()
    while len(ids) < 400:
        first = ""
        for _ in range(9):
            first += str(random.randint(0, 9))
        ids.add(first)
    
    np.savetxt(csv_path, list(ids), delimiter=',', fmt='%s')


# RSSMRA80A01F205X
# SSSNNN YYMDD HZZZZ
def generate_it_tax_ids(csv_path: str):

    people = pd.read_csv("data/training/helpers/it/person.csv", header=None)
    ids = set()

    for i in range(0, len(people)):

        name = extract_codice_fiscale_part(people[0][i])
        date = generate_date_code()
        H = random.choice(LETTERS)
        checksum = str(random.randint(100, 999)) + random.choice(LETTERS)
        ids.add(f"{name}{date}{H}{checksum}")
    
    np.savetxt(csv_path, list(ids), delimiter=',', fmt='%s')


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
MONTH_CODES = {
    1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "H",
    7: "L", 8: "M", 9: "P", 10: "R", 11: "S", 12: "T"
}

def extract_codice_fiscale_part(name_surname: str) -> str:
    name_surname = name_surname.strip().upper()
    try:
        name, surname = name_surname.split(" ", 1)
    except ValueError:
        return "XXXXXX"  # fallback if name format is wrong

    def extract_three_letters(s: str, is_name=False) -> str:
        consonants = re.findall(r'[BCDFGHJKLMNPQRSTVWXYZ]', s)
        vowels = re.findall(r'[AEIOU]', s)

        if is_name and len(consonants) >= 4:
            # For names with 4+ consonants, use 1st, 3rd, 4th
            code = consonants[0] + consonants[2] + consonants[3]
        else:
            code = ''.join(consonants + vowels)[:3]
        
        return code.ljust(3, 'X')

    code_surname = extract_three_letters(surname)
    code_name = extract_three_letters(name, is_name=True)

    return code_surname + code_name

def generate_date_code() -> str:
    day = random.randint(1, 30)
    if random.choice(["M", "F"]) == "F":
        day += 40

    DD = f"{day:02d}"
    M = MONTH_CODES.get(random.randint(1, 12), "X")
    YY = str(random.randint(1940, 2023))[-2:]

    return f"{DD}{M}{YY}"



#generate_passport_nums("data/training/helpers/passport_nums.csv")
#generate_tax_ids("data/training/helpers/eng/tax.csv")
generate_fr_tax_ids("data/training/helpers/fr/tax.csv")



