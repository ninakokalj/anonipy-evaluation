
import json



with open("data/test_dataset.json", "r") as f:
    test_dataset = json.load(f)


for data in test_dataset:
    ner = sorted(data["ner"], key=lambda x: (x[0], -(x[1] - x[0])))
    filtered_ner = []
    
    for entity in ner:
        start, end, _ = entity
        if not any(
            (e_start <= start < e_end or e_start < end <= e_end)
            and (e_start != start or e_end != end) for e_start, e_end, _ in filtered_ner
        ):
            filtered_ner.append(entity)
    
    data["ner"] = filtered_ner

with open("data/test_dataset_filtered.json", "w") as f:
    json.dump(test_dataset, f, indent=4, ensure_ascii=False)