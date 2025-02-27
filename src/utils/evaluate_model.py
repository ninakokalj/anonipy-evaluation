import json
from argparse import ArgumentParser
from pathlib import Path
import torch
from collections import defaultdict

from gliner import GLiNERConfig, GLiNER

from evaluate import evaluate


def main(args):

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

        for language, lang_data in language_datasets.items():
            lang_performances = evaluate(model, lang_data, args.threshold)
            lang_performances["language"] = language
            performances.append(lang_performances)
    else:
        performances = evaluate(model, test_dataset, args.threshold)
    
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(performances, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_test_file", 
        type=str, 
        help="path to the test data file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="path to the pre-trained model"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
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



# python src/utils/evaluate_model.py --data_test_file data/E3-JSI/test_dataset.json --model_path models/checkpoint-200 --output_file results/ckp_200.json



