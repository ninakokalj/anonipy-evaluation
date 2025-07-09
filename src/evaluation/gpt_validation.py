import csv
import openai
import pandas as pd

from utils.constants import ALL_LABELS, LANGUAGE_LABELS


# Set OpenAI API key
#openai.api_key = ''

def check_with_gpt(csv_path: str, labels: list, language: str):
    """Checks the validity of entities in a CSV file using OpenAI's GPT-3 model."""

    df = pd.read_csv(csv_path, header=None, delimiter=',')
    results = []
    batch = []

    # Iterate through the CSV file
    for i in range(len(df)):
        entity_type = df.iloc[i, 0]
        generated_text = df.iloc[i, 2]

        if entity_type in labels:
            batch.append((entity_type, generated_text))

            # Batch of 15 entities at a time
            if len(batch) == 15:
                results.extend(_send_to_gpt(batch, language))
                batch = []

    # Send the remaining batch
    if batch:
        results.extend(_send_to_gpt(batch, language))

    _save_results_to_csv(results)


def _send_to_gpt(batch: list, language: str) -> list:
    results = []
    combined = ""
    
    # Format the batch
    for j, (etype, etext) in enumerate(batch, start=1):
        combined += f"{j}. Entity type: {etype} | Entity: {etext}\n"

    prompt = (
        "You are a strict validator. For each line below, respond with 1 if the entity is valid in " + language + " for the given type, or 0 if not.\n"
        "Only return a list of 0s and 1s in order, separated by new lines.\n\n" + combined
    )

    # Send the batch
    response = openai.chat.completions.create(
        model="o3-mini",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Parse the response
    output_lines = [line.strip() for line in response.choices[0].message.content.strip().splitlines()]
    if len(output_lines) != len(batch):
        print("Warning: Mismatch in number of outputs. Check GPT response:")
    for j, result in enumerate(output_lines):
        etype, etext = batch[j]
        results.append((etype, etext, result))
        print(f"{etype} | {etext} => {result}")

    return results


def _save_results_to_csv(results: list, filepath: str = "data/gpt_validation_results.csv"):
    correct = sum(1 for _, _, r in results if r == '1')
    incorrect = len(results) - correct

    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Entity Type", "Generated Text", "Is Valid (1/0)"])
        writer.writerows(results)
        writer.writerow([]) 
        writer.writerow(["Summary", f"Correct: {correct}", f"Incorrect: {incorrect}"])
    
    print(f"Results saved to {filepath}:)")