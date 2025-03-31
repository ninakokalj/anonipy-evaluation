import numpy as np
import random

import re
from datetime import date

import dateparser
from babel.dates import format_datetime, format_datetime



POSSIBLE_FORMATS = [
    "yyyy-MM-dd",
    "dd-MM-yyyy",
    "yyyy/MM/dd",
    "dd/MM/yyyy",
    "dd.MM.yyyy",
    "d.M.yyyy",
    "d MMMM yyyy",
    "d. MMMM yyyy",
]
days_31 = [1, 3, 5, 7, 8, 10, 12]
days_30 = [4, 6, 9, 11]

def generate_dates(count, lang):
    all_formats = []
    for FMT in POSSIBLE_FORMATS:
        dates = set()
        for i in range(count):
            year = random.choice([i for i in range(1950, 2026)])
            month = random.choice([i for i in range(1, 13)])

            if month in days_31:
                day = random.choice([i for i in range(1, 32)])
            elif month in days_30:
                day = random.choice([i for i in range(1, 31)])
            else:
                day = random.choice([i for i in range(1, 29)])

            rand_date = date(year, month, day)
            formatted_date = format_datetime(
                rand_date, format=FMT, locale=lang
            )
            dates.add(formatted_date)
        all_formats.extend(list(dates))
        
    np.savetxt("data/training/dates_eng.csv", all_formats, delimiter=',', fmt='%s')

    
              
generate_dates(60, "en")


