import json
from argparse import ArgumentParser
from pathlib import Path
import torch

from gliner import GLiNERConfig, GLiNER

from ..src.utils.evaluate import evaluate


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
    args = parser.parse_args()
    main(args)



# args.model_path = "models/checkpoint-100"
# args.data_test_file = "data/test_dataset.json"
# python scripts/evaluate_model.py --data_test_file data/test_dataset.json --model_path models/checkpoint-100 --output_file results/results.json
