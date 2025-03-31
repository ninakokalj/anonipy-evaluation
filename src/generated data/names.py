import pandas as pd
import numpy as np
import random
import csv
from bs4 import BeautifulSoup
import requests


def generate_emails():

    df = pd.read_csv("data/training/adjusted-name-combinations-list.csv", header=None)
    emails = []

    for i in range(0, len(df)):
        endings = ["@yahoo.com", "@gmail.com", "@hotmail.com"]
        combo = df[4][i]
        combo = combo.replace(" ", ".")

        combo += random.choice(endings)
        emails.append(combo)

    np.savetxt("data/training/emails.csv", emails, delimiter=',', fmt='%s')


def get_names():

    df = pd.read_csv("data/training/lastn.csv", header=None)
    names = set()

    for i in range(0, len(df)):
        lastn = df[0][i]
        names.add(lastn.capitalize())

    np.savetxt("data/training/lastn2.csv", list(names), delimiter=',', fmt='%s')


def getHTMLdocu(url: str):
    response = requests.get(url)
    return response.text


def parse_website():
    html_docu = getHTMLdocu("https://sl.wikipedia.org/wiki/Seznam_najpogostej%C5%A1ih_priimkov_v_Sloveniji")
    soup = BeautifulSoup(html_docu, 'html.parser')

    data = set()
    table = soup.select_one("#mw-content-text > div.mw-content-ltr.mw-parser-output > table")
    for anchor in table.find_all('a'):
        # anchors = tr.find_all('a')
        txt = anchor.get_text(strip=True) 
        # print(txt)
        data.add(txt)

    np.savetxt("data/training/priimki_wiki.csv", list(data), delimiter=',', fmt='%s')


def mix_first_last():
    names = set()
    first = pd.read_csv("data/training/slo/imena_wiki.csv", header=None)
    last = pd.read_csv("data/training/slo/priimki_wiki.csv", header=None)

    for i in range(0, len(first)):
        j = random.choice([i for i in range(len(last))])
        name = first[0][i] + " " + last[0][j]
        names.add(name)

    np.savetxt("data/training/person_slo.csv", list(names), delimiter=',', fmt='%s')


def random_names():
    names = set()
    first = pd.read_csv("data/training/slo/imena_wiki.csv", header=None)
    last = pd.read_csv("data/training/slo/priimki_wiki.csv", header=None)

    for x in range(100):
        i = random.choice([i for i in range(len(first))])
        j = random.choice([i for i in range(len(last))])
        name = first[0][i] + " " + last[0][j]
        names.add(name)

    np.savetxt("data/training/person2_slo.csv", list(names), delimiter=',', fmt='%s')


random_names()