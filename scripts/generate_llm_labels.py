from argparse import ArgumentParser

from ..src.utils.helpers import generate_LLM_labels
from ..src.utils.llm_label_generator import LLMLabelGenerator


# used to create replacements for the entitites in the test dataset using an LLM
def main(args):
    """
    This script uses the LLMLabelGenerator to generate new entities based on the entities in the test dataset.
    The new dataset with the generated entities is saved to the path given in args.new_data_output_file.

    Args:
        - args.data_test_file: The path to the test dataset
        - args.llm: The path to the llm
        - args.adapter: The path to the adapter
        - args.new_data_output_file: The path where the new dataset will be saved
        - args.use_cpu: Whether to use CPU or GPU (default is GPU)
        - args.use_quant: Whether to quantize the model (default is to not)

    """

    llm_generator = LLMLabelGenerator(model_name = args.llm, adapter_name = args.adapter, use_gpu = not args.use_cpu, use_quant = args.use_quant)

    # saves the new dataset with LLM generated entities
    generate_LLM_labels(args.data_test_file, args.new_data_output_file, llm_generator, use_entity_attrs = True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_test_file", 
        type=str, 
        required=True,
        help="path to the test data file"
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        required=True,
        help="path to the llm"
    )
    parser.add_argument(
        "--adapter", 
        type=str, 
        help="path to the adapter"
    )
    parser.add_argument(
        "--new_data_output_file", 
        type=str, 
        required=True,
        help="path to the new dataset file"    
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="whether to use CPU (default is GPU)"
    )
    parser.add_argument(
        "--use_quant",
        action="store_true",
        help="whether to quantize the model (default is to not)"
    )
    args = parser.parse_args()
    main(args)

# Example:
# python src/utils/generate_llm_labels.py --data_test_file data/evaluation/E3-JSI/test_dataset.json --llm meta-llama/Llama-3.2-1B-Instruct --new_data_output_file data/NEW_test.json