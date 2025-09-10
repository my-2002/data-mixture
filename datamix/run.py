import argparse
import json
from config.config_generator import ConfigGenerator

mixing_strategy_list = [] #TODO
with open("./config/config_info.json", 'r', encoding='utf-8') as file:
    configs = json.load(file)

parser = argparse.ArgumentParser()
#parser.add_argument("-FT", "--finetuning-type", type=str, help="The type of finetuning method", choices=["lora", "fft"])
parser.add_argument("-MS", "--mixing-strategy", type=str, help="The strategy of data mixing", choices=mixing_strategy_list, required=True)
parser.add_argument("-CN", "--config-number", type=int, help="The config number of data mixing", choices=list(configs), required=True)
#parser.add_argument("-PBM", "--path-to-base-model", type=str, help="The path to the base model")
#parser.add_argument("-CL", "--cutoff-len", type=int, help="The cutoff length of the model")
#parser.add_argument("-E", "--epoch", type=float, help="The number of epochs for training")
#parser.add_argument("-BS", "--batch-size", type=int, help="The per device batch size for training")
#parser.add_argument("-GA", "--grad-accum", type=int, help="The gradient accumulation steps for training")
#parser.add_argument("-LR", "--learning-rate", type=float, help="The learning rate for training")
#parser.add_argument("-WR", "--warmup-ratio", type=float, help="The warmup ratio for training")
args = parser.parse_args()

config = configs[args.config_number]
default_config = configs["default"]
ConfigGenerator().generate(
    finetuning_type = config["finetuning_type"],
    mixing_strategy = args.mixing_strategy,
    config_number = args.config_number,
    path_to_base_model = config["path_to_base_model"],
    cutoff_len = config["cutoff_len"] if "cutoff_len" in config else default_config["cutoff_len"],
    epoch = config["epoch"] if "epoch" in config else default_config["epoch"],
    batch_size = config["batch_size"] if "batch_size" in config else default_config["batch_size"],
    grad_accum = config["grad_accum"] if "grad_accum" in config else default_config["grad_accum"],
    learning_rate = config["learning_rate"] if "learning_rate" in config else default_config["learning_rate"],
    warmup_ratio = config["warmup_ratio"] if "warmup_ratio" in config else default_config["warmup_ratio"]
)

