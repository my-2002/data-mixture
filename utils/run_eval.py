import argparse
from config.config_generator import ConfigGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", type=str, help="The name of the model to be evaluated", required=True)
parser.add_argument("-PATH", "--path_to_model", type=str, help="The name of the model to be evaluated", required=True)
args = parser.parse_args()

ConfigGenerator().generate_eval_config(
    model_name = args.model_name,
    path_to_eval_model = args.path_to_eval_model
)