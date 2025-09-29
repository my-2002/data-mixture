import yaml
import json
import os
from pathlib import Path

class ConfigGenerator:
    def __init__(self):
        pass

    def generate_train_config(
        self, 
        finetuning_type : str,
        dataset : str,
        model : str
        #path_to_base_model : str,
        #cutoff_len : int,
        #epoch : float,
        #batch_size : int,
        #grad_accum : int,
        #learning_rate : float,
        #warmup_ratio : float
    ):
        if finetuning_type not in ["lora", "fft"]:
            raise ValueError("Unsupported finetuning type")
        
        model_name = os.path.normpath(model)
        model_name = os.path.basename(model_name)

        dataset_name = os.path.basename(dataset)
        dataset_name, _ = os.path.splitext(dataset_name)

        full_name = f"{dataset_name}_{model_name}_{finetuning_type}"

        folder_path = Path(f"config/{full_name}")
        folder_path.mkdir(exist_ok=True)

        # 1. Generate train and merge configurations for LLaMA-Factory
        if finetuning_type == "lora":
            train_config = yaml.safe_load(open("config/train_template/train_lora.yaml"))
            merge_config = yaml.safe_load(open("config/train_template/merge_lora.yaml"))
        else:
            train_config = yaml.safe_load(open("config/train_template/train_fft.yaml"))
            merge_config = yaml.safe_load(open("config/train_template/merge_fft.yaml"))

        train_config['model_name_or_path'] = model
        train_config['dataset'] = dataset_name
        #train_config['cutoff_len'] = cutoff_len
        train_config['output_dir'] = f"../utils/sft_results/{full_name}/train"
        #train_config['per_device_train_batch_size'] = batch_size
        #train_config['gradient_accumulation_steps'] = grad_accum
        #train_config['learning_rate'] = learning_rate
        #train_config['num_train_epochs'] = epoch
        #train_config['warmup_ratio'] = warmup_ratio
        with open(f"config/{full_name}/train.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)

        if finetuning_type == "lora":
            merge_config['model_name_or_path'] = model
            merge_config['adapter_name_or_path'] = f"../utils/sft_results/{full_name}/train"
        else:
            merge_config['model_name_or_path'] = f"../utils/sft_results/{full_name}/train"
        merge_config['export_dir'] = f"../utils/sft_results/{full_name}/merge"
        with open(f"config/{full_name}/merge.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(merge_config, f, sort_keys=False, default_flow_style=False)

        # 2. Generate dataset_info in LLaMA-Factory
        with open(f"../LLaMA-Factory/data/dataset_info.json", 'r', encoding='utf-8') as file:
            dataset_info = json.load(file)
        dataset_info[dataset_name] = {
            "file_name": dataset
        }
        with open(f"../LLaMA-Factory/data/dataset_info.json", 'w', encoding='utf-8') as file:
            json.dump(dataset_info, file, indent=2)

        # 3. Generate the py file of the training process
        with open("config/train_template/train_template.py", 'r', encoding='utf-8') as f:
            template_content = f.read()
        new_content = template_content.replace("full_name", full_name)
        with open(f"training_scripts/{full_name}.py", 'w', encoding='utf-8') as f:
            f.write(new_content)

        # 4. Generate the script file for training
        with open("config/train_template/run_template.sh", 'r', encoding='utf-8') as f:
            template_content = f.read()
        new_content = template_content.replace("full_name", full_name)
        with open(f"training_scripts/run_{full_name}.sh", 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"The full name of this training task is: {full_name}")

    def generate_eval_config(
            self,
            model_name : str = None,
            path_to_eval_model : str = None
    ):
        if path_to_eval_model is None or model_name is None:
            raise ValueError("Provided infomation is insufficient")
        
        folder_path = Path(f"../opencompass/examples/data_mixture")
        folder_path.mkdir(exist_ok=True)
        folder_path = Path(f"../opencompass/opencompass/configs/models/data_mixture")
        folder_path.mkdir(exist_ok=True)
        folder_path = Path(f"../opencompass/scripts")
        folder_path.mkdir(exist_ok=True)

        with open("config/eval_template/model_template.py", 'r', encoding='utf-8') as f:
            content = f.read()
        modified_content = content.replace("{path_to_eval_model}", path_to_eval_model)
        modified_content = modified_content.replace("{model_name}", model_name)
        with open(f"../opencompass/opencompass/configs/models/data_mixture/model_{model_name}.py", 'w', encoding='utf-8') as f:
            f.write(modified_content)

        with open("config/eval_template/eval_template.py", 'r', encoding='utf-8') as f:
            content = f.read()
        modified_content = content.replace("{model_name}", model_name)
        with open(f"../opencompass/examples/data_mixture/eval_{model_name}.py", 'w', encoding='utf-8') as f:
            f.write(modified_content)

        with open("config/eval_template/script_template.sh", 'r', encoding='utf-8') as f:
            content = f.read()
        modified_content = content.replace("{model_name}", model_name)
        with open(f"../opencompass/scripts/{model_name}.sh", 'w', encoding='utf-8') as f:
            f.write(modified_content)