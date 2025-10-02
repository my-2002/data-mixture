import argparse
import json
from config.config_generator import ConfigGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", type=str, help="The path of the model to be finetuned", required=True)
parser.add_argument("-D", "--dataset", type=str, help="The path of the dataset to be used", required=True)
parser.add_argument("-FT", "--finetuning-type", type=str, help="The type of finetuning method", choices=["lora", "fft"], required=True)
#parser.add_argument("-PBM", "--path-to-base-model", type=str, help="The path to the base model")
#parser.add_argument("-CL", "--cutoff-len", type=int, help="The cutoff length of the model")
#parser.add_argument("-E", "--epoch", type=float, help="The number of epochs for training")
parser.add_argument("--batch-size", type=int, default=1, help="The per device batch size for training")
#parser.add_argument("-GA", "--grad-accum", type=int, help="The gradient accumulation steps for training")
#parser.add_argument("-LR", "--learning-rate", type=float, help="The learning rate for training")
#parser.add_argument("-WR", "--warmup-ratio", type=float, help="The warmup ratio for training")
args = parser.parse_args()

#config = configs[args.config_number]
#default_config = configs["default"]
ConfigGenerator().generate_train_config(
    finetuning_type = args.finetuning_type,
    dataset = args.dataset,
    model = args.model,
    #path_to_base_model = config["path_to_base_model"],
    #cutoff_len = config["cutoff_len"] if "cutoff_len" in config else default_config["cutoff_len"],
    #epoch = config["epoch"] if "epoch" in config else default_config["epoch"],
    batch_size = args.batch_size
    #grad_accum = config["grad_accum"] if "grad_accum" in config else default_config["grad_accum"],
    #learning_rate = config["learning_rate"] if "learning_rate" in config else default_config["learning_rate"],
    #warmup_ratio = config["warmup_ratio"] if "warmup_ratio" in config else default_config["warmup_ratio"]
)