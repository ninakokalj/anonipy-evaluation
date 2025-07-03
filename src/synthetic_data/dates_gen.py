import random
from datetime import date

import numpy as np
from babel.dates import format_datetime


#===============================================	
# RANDOM DATES GENERATION FUNCTIONS
#===============================================


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


def generate_dates(count, lang, csv_path):
    """Generate random dates of various formats and save them to a CSV file."""

    all_formats = []
    for FMT in POSSIBLE_FORMATS:
        dates = set()

        for _ in range(count):
            year = random.randint(1950, 2025)
            month = random.randint(1, 12)

            if month in days_31:
                day = random.randint(1, 31)
            elif month in days_30:
                day = random.randint(1, 30)
            else:
                day = random.randint(1, 28)

            rand_date = date(year, month, day)
            formatted_date = format_datetime(
                rand_date, format=FMT, locale=lang
            )
            dates.add(formatted_date)

        all_formats.extend(list(dates))
        
    np.savetxt(csv_path, all_formats, delimiter=',', fmt='%s')
