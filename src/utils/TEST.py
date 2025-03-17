import json

with open("results/E3-JSI/replacements/llm3_repl.json", "r") as f:
        replacements = json.load(f)


avg_len = 0
for original, replacement in replacements.items():
    avg_len += len(original)

avg_len /= len(replacements)

print(avg_len)