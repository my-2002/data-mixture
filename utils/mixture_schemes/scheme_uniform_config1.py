# In this scheme, we sample the data from the data source evenly
import json
import numpy as np
import random
import logging
import sys
import os
import subprocess
from safetensors.torch import load_file, save_file
from huggingface_hub import split_torch_state_dict_into_shards

# 设置环境变量
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0,2,4'

# 配置日志记录器
if os.path.exists('../logs/uniform_config1.log'):
    os.remove('../logs/uniform_config1.log')
logging.basicConfig(filename='../logs/uniform_config1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1.混合阶段
logging.info("Mixing phase started")

data_source_num = 3
data_source_paths =['../source_data/alpaca/alpaca_data.json',
             '../source_data/dart-math-hard/dart-math-hard.json',
             '../source_data/opc-sft-stage2/opc-sft-stage2.json']
train_size = 50000
random.seed(114514)
logging.info(f"Mixing ratio is [0.333 0.333 0.333 ]")

output_dataset = []
for data_path in data_source_paths:
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    output_dataset.extend(dataset[ : train_size//data_source_num])

with open('../data/uniform_config1/train_data.json', 'w') as file:
    json.dump(output_dataset, file, indent = 4)

logging.info("Mixing phase completed")

# 2.训练阶段
logging.info("Training phase started")

train_config="../utils/config/uniform_config1/train.yaml"

return_code = subprocess.run(['llamafactory-cli', 'train', train_config], cwd='../../LLaMA-Factory', env=env).returncode
if return_code == 0:
    logging.info("model was trained successfully")
else:
    logging.info("failed to train model.")
    sys.exit()

logging.info("Training phase completed")

# 3.模型合并阶段
logging.info("Merging phase started")

merge_config="../utils/config/uniform_config1/merge.yaml"

return_code = subprocess.run(['llamafactory-cli', 'export', merge_config], cwd='../../LLaMA-Factory', env=env).returncode
if return_code == 0:
    logging.info("model was merged successfully")
else:
    logging.info("failed to merge model.")
    sys.exit()

logging.info("Merging phase completed")

# 4.模型分片阶段
logging.info("Sharding phase started")

single_file = "../sft_results/uniform_config1/merge/model.safetensors"
state_dict = load_file(single_file)

max_shard_size = "3GB"
state_dict_split = split_torch_state_dict_into_shards(
    state_dict,
    max_shard_size=max_shard_size,
    filename_pattern="../sft_results/uniform_config1/merge/model-{suffix}.safetensors"
)

for shard_name, tensor_names in state_dict_split.filename_to_tensors.items():
    shard = {k: state_dict[k] for k in tensor_names}
    save_file(shard, shard_name)
index_file = "../sft_results/uniform_config1/merge/model.safetensors.index.json"
state_dict_split.save_index(index_file)

logging.info("Sharding phase completed")