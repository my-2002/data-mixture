import argparse
from config.config_generator import ConfigGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model-name", type=str, help="The abbr name of the model to be evaluated", required=True)
parser.add_argument("-PATH", "--path-to-model", type=str, help="The path of the model to be evaluated", required=True)
parser.add_argument("--max-out-len", type=int, default=4096, help="The maximum output length of the model to be evaluated")
parser.add_argument("--batch-size", type=int, default=8, help="The batch size for evaluation")
parser.add_argument("--gpu-cnt", type=int, default=4, help="The number of GPUs to use for evaluation")
args = parser.parse_args()

ConfigGenerator().generate_eval_config(
    model_name = args.model_name,
    path_to_eval_model = args.path_to_model,
    max_out_len = args.max_out_len,
    batch_size = args.batch_size,
    gpu_cnt = args.gpu_cnt
)