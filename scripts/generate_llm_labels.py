from argparse import ArgumentParser

from anonipy.anonymize.generators import LLMLabelGenerator

from ..src.utils.helpers import generate_LLM_labels



def main(args):

    llm_generator = LLMLabelGenerator(model_name = args.llm, use_gpu = not args.use_cpu, use_quant = args.use_quant)

    # saves the new dataset with LLM generated entities
    generate_LLM_labels(args.data_test_file, args.new_data_output_file, llm_generator, use_entity_attrs = False)



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
