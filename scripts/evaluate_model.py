import json
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict

import torch
from gliner import GLiNER

from src.utils.evaluate import evaluate
from src.utils.constants import LANGUAGE_LABELS, ALL_LABELS


# used to evaluate a GLiNER model on the dataset that was modified with an LLM
def main(args):
    """
    Evaluate a GLiNER model on a test dataset.
    Args:
        - args.data_test_file: The path to the test dataset
        - args.model_path: The path to the GLiNER model
        - args.output_file: The path where the evaluation results will be saved
        - args.threshold: The threshold for the prediction (default is 0.5)
        - args.use_cpu: Whether to use CPU or GPU (default is to use GPU)
        - args.per_language_eval: Whether to evaluate the model per language
    """

    if not Path(args.data_test_file).exists():
        raise FileNotFoundError(f"Test data file {args.data_test_file} does not exist")
    
    with open(args.data_test_file, "r") as f:
        test_dataset = json.load(f)

    # load the device (GPU or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )

    model = GLiNER.from_pretrained(args.model_path, load_tokenizer=True)
    model.to(device)

    if args.per_language_eval:
        # evaluate the model per language
        language_datasets = defaultdict(list)
        for d in test_dataset:
            language_datasets[d["language"]].append(d)

        performances = []

        all_performances = evaluate(model, test_dataset, args.threshold)
        all_performances["language"] = "All"
        performances.append(all_performances)

        for language, lang_data in language_datasets.items():
            lang_performances = evaluate(model, lang_data, args.threshold)
            lang_performances["language"] = language
            performances.append(lang_performances)

    else:
        performances = evaluate(model, test_dataset, args.threshold, selected_labels=ALL_LABELS) # default selected_labels = None 
    
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(performances, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_test_file", 
        type=str, 
        required=True,
        help="path to the test data file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="path to the pre-trained model"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="path to the output file"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="threshold for the prediction (default is 0.5)"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="whether to use CPU (default is GPU)"
    )
    parser.add_argument(
        "--per_language_eval",
        action="store_true",
        help="whether to evaluate the model per language (default is to not)"
    )
    args = parser.parse_args()
    main(args)

# Example:
# python scripts/evaluate_model.py --data_test_file data/NEW_test.json --model_path models/E3-JSI/checkpoint-265 --output_file results/RES.json

"""
  File "/home/ninak/PREPARE-anonipy-evaluation/scripts/evaluate_model.py", line 9, in <module>
    from ..src.utils.evaluate import evaluate
ImportError: attempted relative import with no known parent package
"""