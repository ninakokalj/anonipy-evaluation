from argparse import ArgumentParser

import ollama

from src.utils.helpers import generate_LLM_labels
from src.utils.ollama_label_generator import OllamaLabelGenerator


# used to create replacements for the entitites in the test dataset using an LLM from Ollama
def main(args):
    """
    This script uses the ollamaLabelGenerator to generate new entities based on the entities in the test dataset.
    The new dataset with the generated entities is saved to the path given in args.new_data_output_file.

    Args:
        - args.data_test_file: The path to the test dataset
        - args.llm: The name of the llm
        - args.new_data_output_file: The path where the new dataset will be saved

    """

    #ollama.pull(model_name)
    llm_generator = OllamaLabelGenerator(model_name = args.llm)

    # saves the new dataset with LLM generated entities
    generate_LLM_labels(args.data_test_file, args.new_data_output_file, llm_generator, use_entity_attrs=False)

    llm_generator.print_logs()
    #ollama.delete(model_name)

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
        "--new_data_output_file", 
        type=str, 
        required=True,
        help="path to the new dataset file"    
    )
    args = parser.parse_args()
    main(args)

# Example:
# python src/utils/ollama_llm_labels.py --data_test_file data/E3-JSI/test_dataset.json --llm deepseek-r1:8b --new_data_output_file data/NEW_test.json