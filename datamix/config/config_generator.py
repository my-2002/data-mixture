import yaml
import json
from pathlib import Path

class ConfigGenerator:
    def __init__(self):
        pass

    def generate(
        self, 
        finetuning_type : str,
        mixing_strategy : str,
        config_number : int,
        path_to_base_model : str,
        cutoff_len : int,
        epoch : float,
        batch_size : int,
        grad_accum : int,
        learning_rate : float,
        warmup_ratio : float
    ):
        if finetuning_type not in ["lora", "fft"]:
            raise ValueError("Unsupported finetuning type")
        
        folder_path = Path(f"config/{mixing_strategy}_config{config_number}")
        if not folder_path.exists():
            folder_path.mkdir(exist_ok=True)

        if finetuning_type == "lora":
            train_config = yaml.safe_load(open("config/train_template/train_lora.yaml"))
            merge_config = yaml.safe_load(open("config/train_template/merge_lora.yaml"))
        else:
            train_config = yaml.safe_load(open("config/train_template/train_fft.yaml"))
            merge_config = yaml.safe_load(open("config/train_template/merge_fft.yaml"))

        train_config['model_name_or_path'] = path_to_base_model
        train_config['dataset'] = f"datamix_{mixing_strategy}_config{config_number}"
        train_config['cutoff_len'] = cutoff_len
        train_config['output_dir'] = f"../datamix/sft_results/{mixing_strategy}_config{config_number}/train"
        train_config['per_device_train_batch_size'] = batch_size
        train_config['gradient_accumulation_steps'] = grad_accum
        train_config['learning_rate'] = learning_rate
        train_config['num_train_epochs'] = epoch
        train_config['warmup_ratio'] = warmup_ratio
        with open(f"config/{mixing_strategy}_config{config_number}/train.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)

        if finetuning_type == "lora":
            merge_config['model_name_or_path'] = path_to_base_model
            merge_config['adapter_name_or_path'] = f"../datamix/sft_results/{mixing_strategy}_config{config_number}/train"
        else:
            merge_config['model_name_or_path'] = f"../datamix/sft_results/{mixing_strategy}_config{config_number}/train"
        merge_config['export_dir'] = f"../datamix/sft_results/{mixing_strategy}_config{config_number}/merge"
        with open(f"config/{mixing_strategy}_config{config_number}/merge.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(merge_config, f, sort_keys=False, default_flow_style=False)

        with open(f"../LLaMA-Factory/data/dataset_info.json", 'r', encoding='utf-8') as file:
            dataset_info = json.load(file)

        dataset_info[f'datamix_{mixing_strategy}_config{config_number}'] = {
            "file_name": f"../../datamix/data/{mixing_strategy}_config{config_number}/train_data.json"
        }

        with open(f"../LLaMA-Factory/data/dataset_info.json", 'w', encoding='utf-8') as file:
            json.dump(dataset_info, file, indent=2)