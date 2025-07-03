import numpy as np
import requests
from bs4 import BeautifulSoup


#===============================================	
# WEB SCRAPING FUNCTIONS
#===============================================

def getHTMLdocu(url: str):
    response = requests.get(url)
    return response.text

# used to web scrape
def parse_website():
    html_docu = getHTMLdocu("https://www.fakexy.com/gr-fake-address-generator-attica")
    soup = BeautifulSoup(html_docu, 'html.parser')
    print(soup.prettify())
    data = set()
    table = soup.select_one("#cmkt > div.table-container.shadow > table > tbody")
    
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) >= 5:
            company = tds[2].select_one('.company-name')
            text = company.get_text(strip=True)
            print(text)
            data.add(text)

    np.savetxt("data/training/helpers/fr/comp.csv", list(data), delimiter=',', fmt='%s')


# used to web scrape Italian addresses
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


# used to web scrape Slovenian addresses
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