import json
from difflib import SequenceMatcher

import pandas as pd

from utils.constants import LANGUAGE_LABELS, ALL_LABELS


def analyze_replacements(csv_path: str, output_file: str, labels: list | None = None) -> tuple:
    """Analyzes a CSV file with replacements and writes the results to a JSON file.

    The CSV file should have the columns: label, original, replacement.
    The function calculates the average length of the replacements and the average similarity between the original and replacement strings.
    The results are written to the specified output file.

    Args:
    - csv_path (str): The path to the CSV file.
    - output_file (str): The path to the output JSON file.
    - labels (list | None): A list of labels to consider. If None, all labels are considered.

    Returns:
    - tuple: A tuple containing the average length and average similarity.
    """
    
    df = pd.read_csv(csv_path, header=None)

    avg_len = 0
    avg_sim = 0
    count = 0
    for i in range(0, len(df)):
        if labels and df[0][i] not in labels:
            continue
        original, replacement = df[1][i], df[2][i]
        avg_len += len(replacement)
        avg_sim += SequenceMatcher(None, original, replacement).ratio()
        count += 1

    avg_len /= count
    avg_sim /= count
 
    results = {"Average length": avg_len, "Average similarity": avg_sim}
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return avg_len, avg_sim
