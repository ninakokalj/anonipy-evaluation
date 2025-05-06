import json
from difflib import SequenceMatcher
import pandas as pd

from constants import FRENCH_LABELS

def analyze_replacements(csv_path: str, output_file: str, labels: list | None = None):

    df = pd.read_csv(csv_path, header=None)

    avg_len = 0
    count = 0
    for i in range(0, len(df)):
        if labels and df[0][i] not in labels:
            continue
        original, replacement = df[1][i], df[2][i]
        avg_len += len(original)
        count += 1

    avg_len /= count
    results = {"Average length": avg_len}
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return avg_len



analyze_replacements("data/training/results/Llama-3.2-1B/french/nenatreniran_repl.csv", "results/ANALYSIS.json", FRENCH_LABELS)
