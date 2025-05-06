import json
from difflib import SequenceMatcher
import pandas as pd

from constants import ENGLISH_LABELS

def analyze_replacements(csv_path: str, output_file: str, labels: list | None = None):

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



analyze_replacements("data/replacements.csv", "results/ANALYSIS.json", ENGLISH_LABELS)
