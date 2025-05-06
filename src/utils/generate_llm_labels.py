from argparse import ArgumentParser

from llm_label_generator import LLMLabelGenerator
#from gams import LLMLabelGenerator

from helpers import generate_LLM_labels



def main(args):

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


# python src/utils/generate_llm_labels.py --data_test_file data/training/test_dataset/test_en.json --llm HuggingFaceTB/SmolLM2-360M-Instruct --new_data_output_file data/NEW_test.json

# python src/utils/generate_llm_labels.py --data_test_file data/training/test_dataset/test_it.json --llm models/360M/italian_300/checkpoint-720 --new_data_output_file data/NEW_test.json

# python src/utils/generate_llm_labels.py --data_test_file data/training/test_dataset/test_en.json --llm meta-llama/Llama-3.2-1B-Instruct --adapter models/Llama-3.2-1B/english_400/checkpoint-960 --new_data_output_file data/NEW_test.json