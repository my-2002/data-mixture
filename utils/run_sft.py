import argparse
import json
from config.config_generator import ConfigGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", type=str, help="The path of the model to be finetuned", required=True)
parser.add_argument("-D", "--dataset", type=str, help="The path of the dataset to be used", required=True)
parser.add_argument("-FT", "--finetuning-type", type=str, help="The type of finetuning method", choices=["lora", "fft"], required=True)
#parser.add_argument("-PBM", "--path-to-base-model", type=str, help="The path to the base model")
parser.add_argument("--cutoff-len", type=int, default=4096, help="The cutoff length of the model")
parser.add_argument("--logging-steps", type=int, default=20, help="The logging steps during training")
parser.add_argument("--save-steps", type=int, default=2000, help="The save steps during training")
parser.add_argument("--batch-size", type=int, default=1, help="The per device batch size for training")
parser.add_argument("--grad-accum", type=int, default=8, help="The gradient accumulation steps for training")
parser.add_argument("--learning-rate", type=float, default=5.0e-6, help="The learning rate for training")
parser.add_argument("--epochs", type=float, default=3.0, help="The number of epochs for training")
parser.add_argument("--warmup-ratio", type=float, default=0.03, help="The warmup ratio for training")
args = parser.parse_args()

#config = configs[args.config_number]
#default_config = configs["default"]
ConfigGenerator().generate_train_config(
    finetuning_type = args.finetuning_type,
    dataset = args.dataset,
    model = args.model,
    #path_to_base_model = config["path_to_base_model"],
    cutoff_len = args.cutoff_len,
    logging_steps = args.logging_steps,
    save_steps = args.save_steps,
    batch_size = args.batch_size,
    grad_accum = args.grad_accum,
    learning_rate = args.learning_rate,
    epochs = args.epochs,
    warmup_ratio = args.warmup_ratio
)