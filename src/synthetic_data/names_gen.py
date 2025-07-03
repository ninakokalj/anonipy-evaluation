import random

import numpy as np
import pandas as pd


#===============================================	
# NAMES, EMAILS, USERNAMES GENERATION FUNCTIONS
#===============================================


def capitalize_names(input_path: str, output_path: str):
    """Capitalizes names from a CSV file and saves them to a new CSV file."""

    df = pd.read_csv(input_path, header=None)
    names = set()

    for i in range(0, len(df)):
        lastn = df[0][i]
        names.add(lastn.capitalize())

    np.savetxt(output_path, list(names), delimiter=',', fmt='%s')


def mix_first_last(names_path: str, surnames_path: str, person_path: str):
    """Mixes first names and last names to create a dataset of people."""

    names = set()
    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)

    for i in range(0, len(first)):
        if i < len(last):
            j = i
        else:
            j = random.choice([i for i in range(len(last))])
        name = first[0][i] + " " + last[0][j]
        names.add(name)

    np.savetxt(person_path, list(names), delimiter=',', fmt='%s')


def random_names(names_path: str, surnames_path: str, person_path: str):
    """Randomly mixes first names and last names to create a dataset of people."""
    names = set()
    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)

    for x in range(72):
        i = random.choice([i for i in range(len(first))])
        j = random.choice([i for i in range(len(last))])
        name = first[0][i] + " " + last[0][j]
        names.add(name)

    np.savetxt(person_path, list(names), delimiter=',', fmt='%s')


def generate_emails(names_path: str, surnames_path: str, emails_path: str):
    """Generates emails from names and surnames."""

    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)
    emails = set()

    nums = [1, 2, 2, 2, 2]
    endings = ["@outlook.com", "@live.nl", "@ziggo.nl", "@xs4all.nl", "@kpnmail.nl", "@gmail.com"]

    for _ in range(350):
        i = random.choice([i for i in range(len(first))])
        j = random.choice([i for i in range(len(last))])
        mail = first[0][i].lower() + "." + last[0][j].lower()

        if random.choice(nums) == 1:
            mail += str(random.choice([i for i in range(0, 100)]))

        mail += random.choice(endings)
        emails.add(mail)

    np.savetxt(emails_path, list(emails), delimiter=',', fmt='%s')


def usernames_from_names(names_path: str, surnames_path: str, user_path: str):
    """Generates usernames from names and surnames."""

    names = set()
    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)

    for x in range(350):
        ime = first[0][random.randint(0, len(first)-1)]
        priimek = last[0][random.randint(0, len(last)-1)]
        format = random.randint(0, 2)

        if format == 0:
            username = f"@{ime.lower()}_{priimek.lower()}"
        elif format == 2:
            username = f"@{ime}.{priimek}{random.randint(0, 100)}"
        else:
            username = f"@{ime}{priimek}_{random.randint(0, 100)}"
            
        names.add(username)
        print(username)
    
    np.savetxt(user_path, list(names), delimiter=',', fmt='%s')


# used to convert latin characters to greek
def latin_to_greek(text):
    translit_dict = {
        'a': 'α', 'b': 'β', 'c': 'κ', 'd': 'δ', 'e': 'ε',
        'f': 'φ', 'g': 'γ', 'h': 'η', 'i': 'ι', 'j': 'ξ',
        'k': 'κ', 'l': 'λ', 'm': 'μ', 'n': 'ν', 'o': 'ο',
        'p': 'π', 'q': 'θ', 'r': 'ρ', 's': 'σ', 't': 'τ',
        'u': 'υ', 'v': 'β', 'w': 'ω', 'x': 'χ', 'y': 'ψ',
        'z': 'ζ',
        'A': 'Α', 'B': 'Β', 'C': 'Κ', 'D': 'Δ', 'E': 'Ε',
        'F': 'Φ', 'G': 'Γ', 'H': 'Η', 'I': 'Ι', 'J': 'Ξ',
        'K': 'Κ', 'L': 'Λ', 'M': 'Μ', 'N': 'Ν', 'O': 'Ο',
        'P': 'Π', 'Q': 'Θ', 'R': 'Ρ', 'S': 'Σ', 'T': 'Τ',
        'U': 'Υ', 'V': 'Β', 'W': 'Ω', 'X': 'Χ', 'Y': 'Ψ',
        'Z': 'Ζ'
    }

    return ''.join(translit_dict.get(char, char) for char in text)

def to_greek_alphabet(csv_path: str, output_path: str):
    """Converts a CSV file in latin alphabet to Greek alphabet."""

    words = []
    df = pd.read_csv(csv_path, header=None, delimiter=';')

    for i in range(0, len(df)):
        words.append(latin_to_greek(df[0][i]))

    np.savetxt(output_path, list(words), delimiter=',', fmt='%s')