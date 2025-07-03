"""
Language labels
en - English
sl - Slovenian
de - German
it - Italian
nl - Dutch
fr - French
el - Greek
"""

LANGUAGE_LABELS = {
    "en": [
        "person name",
        "person",
        "name",
        "date",
        "address",
        "organization",
        "company",
        "email",
        "username",
        "passport number",
        "tax identification number"
    ],
    "sl": [
        "ime",
        "osebno ime", 
        "datum",
        "datum rojstva", 
        "naslov", 
        "organizacija", 
        "podjetje", 
        "elektronski naslov", 
        "e-poštni naslov", 
        "email", 
        "uporabniško ime", 
        "številka dokumenta", 
        "davčna številka"
    ],
    "de": [
        "name", 
        "person", 
        "datum", 
        "geburtsdatum", 
        "adresse", 
        "unternehmen", 
        "organisation", 
        "email", 
        "username", 
        "dokumentnummer", 
        "reisepassnummer", 
        "steuernummer"
    ],
    "it": [
        "nome", 
        "persona", 
        "data",
        "data di nascita", 
        "indirizzo", 
        "organizzazione", 
        "azienda", 
        "email", 
        "username",  
        "maniglia social media", 
        "numero di passaporto", 
        "numero passaporto", 
        "codice fiscale"
    ],
    "nl": [
        "naam", 
        "persoon", 
        "datum", 
        "geboortedatum", 
        "adres", 
        "organisatie", 
        "bedrijf", 
        "email",
        "e-mailadres",
        "emailadres", 
        "gebruikersnaam", 
        "social media handle", 
        "paspoortnummer", 
        "belastingidentificatienummer"
    ],
    "fr": [
        "nom",
        "personne",
        "nom complet", 
        "date",
        "date de naissance", 
        "adresse", 
        "organisation",
        "entreprise",
        "nom d'entreprise", 
        "adresse email",
        "email", 
        "nom d'utilisateur", 
        "numéro de passeport", 
        "numéro de sécurité sociale"
    ],
    "el": [
        "όνομα",
        "προσωπικό όνομα",
        "πρόσωπο", 
        "ημερομηνία",
        "ημερομηνία γέννησης", 
        "διεύθυνση", 
        "εταιρεία", 
        "οργανισμός",
        "ηλεκτρονικό ταχυδρομείο", 
        "κοινωνικά μέσα", 
        "αριθμός διαβατηρίου", 
        "ΑΦΜ"
    ]
}

ALL_LABELS = sorted(set(label for labels in LANGUAGE_LABELS.values() for label in labels))