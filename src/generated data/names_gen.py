import pandas as pd
import numpy as np
import random
from bs4 import BeautifulSoup
import requests
import re

# web scrapers
def getHTMLdocu(url: str):
    response = requests.get(url)
    return response.text

def parse_website():
    html_docu = getHTMLdocu("https://parade.com/living/greek-last-names")
    soup = BeautifulSoup(html_docu, 'html.parser')
    print(soup)
    data = set()
    div = soup.select_one("#phxdetail-1 > article > div > div.m-detail--contents.l-content-well > section > div.l-grid--content-body")
    pattern = re.compile(r"^\d-")

    for h3 in div.find_all('h3'):
        h3_id = h3.get("id")
        if h3_id and pattern.match(h3_id):
            text = h3.get_text(strip=True)
            print(text)
            data.add(text)

    np.savetxt("data/training/helpers/el/lastn.csv", list(data), delimiter=',', fmt='%s')


# za italijanske naslove
def parse_it_website():
    url = "https://www.paginebianche.it/prefissi-telefonici/veneto/pd/padova.htm"
    html_docu = getHTMLdocu(url)
    soup = BeautifulSoup(html_docu, 'html.parser')
    
    data = []
    div1 = soup.select_one("body > main > div:nth-child(4) > div.row.box-codice-istat-regione__column-cont.mt-30 > div.col-xl-8 > div.box-topic-ricerche > ul > div:nth-child(1) > div")
    div2 = soup.select_one("body > main > div:nth-child(4) > div.row.box-codice-istat-regione__column-cont.mt-30 > div.col-xl-8 > div.box-topic-ricerche > ul > div:nth-child(2)")
    
    for list in div1.find_all('li'):
        link_tag = list.select_one('a')
        if link_tag and link_tag.has_attr("href"):
            href = link_tag["href"]
            full_url = requests.compat.urljoin(url, href)  # Makes it absolute if it's relative
            
            new_response = requests.get(full_url)
            new_soup = BeautifulSoup(new_response.text, "html.parser")
            span = new_soup.select_one('#headerScheda > div.header-scheda__col-1.col-lg-8 > div > div:nth-child(1) > div > div > div > span')
            if span:
                spans = span.find_all('span')
                addy = f"{spans[0].get_text(strip=True)} {spans[1].get_text(strip=True)} {spans[2].get_text(strip=True)}"
                data.append(addy)

    for list in div2.find_all('li'): 
        link_tag = list.select_one('a')
        if link_tag and link_tag.has_attr("href"):
            href = link_tag["href"]
            full_url = requests.compat.urljoin(url, href)  # Makes it absolute if it's relative
            
            new_response = requests.get(full_url)
            new_soup = BeautifulSoup(new_response.text, "html.parser")
            span = new_soup.select_one('#headerScheda > div.header-scheda__col-1.col-lg-8 > div > div:nth-child(1) > div > div > div > span')
            if span:
                spans = span.find_all('span')
                addy = f"{spans[0].get_text(strip=True)} {spans[1].get_text(strip=True)} {spans[2].get_text(strip=True)}"
                data.append(addy)
    
    print(data[0])
    np.savetxt("data/training/helpers/it/addy8.csv", data, delimiter=',', fmt='%s')


# za slovenske naslove iz bizi.si
def parse_website_naslov():
    html_docu = getHTMLdocu("https://www.bizi.si/TSMEDIA/L/lesna-dejavnost-2560/")
    soup = BeautifulSoup(html_docu, 'html.parser')
    
    data = set()
    div = soup.select_one("#divResults > div.col-12.b-table-body")
    
    for div in div.find_all('div', class_='row b-table-row'):   
        podatki = div.find_all('div')
        ulica = podatki[2].find_all('a')[0].get_text(strip=True) 
        mesto = podatki[3].get_text(strip=True) 
        naslov = ulica + ", " + mesto
        print(naslov)
        data.add(naslov) 

    np.savetxt("data/training/helpers/slo/naslovi4.csv", list(data), delimiter=',', fmt='%s')


# kapitalizira imena / priimke
def capitalize_names():

    df = pd.read_csv("data/training/lastn.csv", header=None)
    names = set()

    for i in range(0, len(df)):
        lastn = df[0][i]
        names.add(lastn.capitalize())

    np.savetxt("data/training/lastn2.csv", list(names), delimiter=',', fmt='%s')


# naredi dataset oseb iz imen in priimkov
def mix_first_last(names_path: str, surnames_path: str, person_path: str):
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


# osebe naključno sestavlja 
def random_names(names_path: str, surnames_path: str, person_path: str):
    names = set()
    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)

    for x in range(150):
        i = random.choice([i for i in range(len(first))])
        j = random.choice([i for i in range(len(last))])
        name = first[0][i] + " " + last[0][j]
        names.add(name)

    np.savetxt(person_path, list(names), delimiter=',', fmt='%s')


# iz imen in priimkov naredi emaile
def generate_emails(names_path: str, surnames_path: str, emails_path: str):

    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)
    emails = set()

    nums = [1, 2, 2, 2, 2]
    endings = ["@yahoo.gr", "@gmail.com", "@mail.gr", "@outlook.com", "@hotmail.com"]

    for _ in range(4):
        i = random.choice([i for i in range(len(first))])
        j = random.choice([i for i in range(len(last))])
        mail = first[0][i].lower() + "." + last[0][j].lower()

        if random.choice(nums) == 1:
            mail += str(random.choice([i for i in range(0, 100)]))

        mail += random.choice(endings)
        emails.add(mail)

    np.savetxt(emails_path, list(emails), delimiter=',', fmt='%s')


# iz imen in priimkov naredi username dataset
def usernames_from_names(names_path: str, surnames_path: str, user_path: str):
    names = set()
    first = pd.read_csv(names_path, header=None)
    last = pd.read_csv(surnames_path, header=None)

    for x in range(300):
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

def get_addresses(csv_path: str):
    addresses = set()
    df = pd.read_csv(csv_path, header=None, delimiter=';')

    for x in range (0, 50):
        idx = random.randint(1, len(df)-1)
        address = f"{df[0][idx]} {random.randint(1, 20)}, {df[2][idx]} {df[1][idx]}"
        addresses.add(address)

    np.savetxt("data/training/helpers/de/address2.txt", list(addresses), delimiter=',', fmt='%s')

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
    words = []
    df = pd.read_csv(csv_path, header=None, delimiter=';')

    for i in range(0, len(df)):
        words.append(latin_to_greek(df[0][i]))

    np.savetxt(output_path, list(words), delimiter=',', fmt='%s')




#parse_website()
#generate_emails("data/training/helpers/el/names_normal.csv", "data/training/helpers/el/lastn_normal.csv", "data/training/helpers/el/emails_gr.csv")
mix_first_last("data/training/helpers/fr/names.csv", "data/training/helpers/fr/lastn.csv", "data/training/helpers/fr/person.csv")
# random_names("data/training/helpers/el/names.csv", "data/training/helpers/el/lastn.csv", "data/training/helpers/el/per2.csv")
#usernames_from_names("data/training/helpers/el/names_normal.csv", "data/training/helpers/el/lastn_normal.csv", "data/training/helpers/el/user.csv")
#get_addresses("data/training/helpers/de/hohenlohekreis.csv")
#parse_it_website()
#to_greek_alphabet("data/training/helpers/el/lastn_normal.csv", "data/training/helpers/el/lastn.csv")