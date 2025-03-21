import json
from difflib import SequenceMatcher


def analyze_replacements(repl_file: str, output_file: str):
    with open(repl_file, "r") as f:
        replacements = json.load(f)

    avg_len = 0
    avg_sim = 0
    for original, replacement in replacements.items():
        avg_len += len(replacement)
        avg_sim += SequenceMatcher(None, original, replacement).ratio()

    avg_len /= len(replacements)
    avg_sim /= len(replacements)

    results = {"Average length": avg_len, "Average similarity": avg_sim}
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return avg_len, avg_sim



analyze_replacements("results/ollama/conll2003/replacements/phi4_SO.json", "results/ANALYSIS.json")
