from argparse import ArgumentParser

from ollama_interface import OllamaInterface

from helpers import generate_LLM_labels



def main(args):

    llm_generator = OllamaInterface(model_name = args.llm)

    # saves the new dataset with LLM generated entities
    generate_LLM_labels(args.data_test_file, args.new_data_output_file, llm_generator, use_entity_attrs=False)

    llm_generator.print_logs()
    # llm_generator.clean()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_test_file", 
        type=str, 
        help="path to the test data file"
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        help="path to the llm"
    )
    parser.add_argument(
        "--new_data_output_file", 
        type=str, 
        help="path to the new dataset file"    
    )
    args = parser.parse_args()
    main(args)


# python src/utils/ollama_llm_labels.py --data_test_file data/E3-JSI/test_dataset.json --llm deepseek-r1:8b --new_data_output_file data/NEW_test.json

# python src/utils/ollama_llm_labels.py --data_test_file data/conll2003/test_dataset.json --llm deepseek-r1:8b --new_data_output_file data/NEW_test.json