import argparse
from config.config_generator import ConfigGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model-name", type=str, help="The abbr name of the model to be evaluated", required=True)
parser.add_argument("-PATH", "--path-to-model", type=str, help="The path of the model to be evaluated", required=True)
parser.add_argument("--max-out-len", type=int, default=4096, help="The maximum output length of the model to be evaluated")
args = parser.parse_args()

ConfigGenerator().generate_eval_config(
    model_name = args.model_name,
    path_to_eval_model = args.path_to_model,
    max_out_len = args.max_out_len
)