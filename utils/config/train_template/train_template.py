import json
import numpy as np
import random
import logging
import sys
import os
import subprocess
from safetensors.torch import load_file, save_file
from huggingface_hub import split_torch_state_dict_into_shards

# Configure the logger
if os.path.exists('../logs/full_name.log'):
    os.remove('../logs/full_name.log')
logging.basicConfig(filename='../logs/full_name.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1.Training phase
logging.info("Training phase started")

train_config="../datamix/config/full_name/train.yaml"

return_code = subprocess.run(['llamafactory-cli', 'train', train_config], cwd='../../LLaMA-Factory').returncode
if return_code == 0:
    logging.info("model was trained successfully")
else:
    logging.info("failed to train model.")
    sys.exit()

logging.info("Training phase completed")

# 2.Merging phase
logging.info("Merging phase started")

merge_config="../datamix/config/full_name/merge.yaml"

return_code = subprocess.run(['llamafactory-cli', 'export', merge_config], cwd='../../LLaMA-Factory').returncode
if return_code == 0:
    logging.info("model was merged successfully")
else:
    logging.info("failed to merge model.")
    sys.exit()

logging.info("Merging phase completed")

# 3.Sharding phase
logging.info("Sharding phase started")

single_file = "../sft_results/full_name/merge/model.safetensors"
state_dict = load_file(single_file)

max_shard_size = "3GB"
state_dict_split = split_torch_state_dict_into_shards(
    state_dict,
    max_shard_size=max_shard_size,
    filename_pattern="model-{suffix}.safetensors"
)

for shard_name, tensor_names in state_dict_split.filename_to_tensors.items():
    shard = {k: state_dict[k] for k in tensor_names}
    save_file(shard, f"../sft_results/full_name/merge/{shard_name}")

index = {
    "metadata": state_dict_split.metadata,
    "weight_map": state_dict_split.tensor_to_filename,
}
with open("../sft_results/full_name/merge/model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=4)
os.remove(single_file)

logging.info("Sharding phase completed")