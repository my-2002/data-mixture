

class ConfigGenerator:
    def __init__(self):
        pass

    def generate(
        self, 
        finetuning_type : str,
        mixing_strategy : str,
        config_number : int,
        path_to_base_model = "/data/models/Qwen2.5-7B-Instruct",
        cutoff_len = 2048,
        epoch = 3,
        batch_size = 1,
        grad_accum = 8,
        learning_rate = None,
        warmup_ratio = None
    ):
        if finetuning_type not in ["lora", "fft"]:
            raise ValueError("Unsupported finetuning type")
        
        train_dataset_name = f"datamix_{mixing_strategy}_config{config_number}"
        path_to_trained_model = f"../datamix/sft_results/{mixing_strategy}_config{config_number}/train"
        path_to_merged_model = f"../datamix/sft_results/{mixing_strategy}_config{config_number}/merge"

        if finetuning_type == "lora":
            self.generate_lora_config(
                train_dataset_name = train_dataset_name,
                path_to_trained_model = path_to_trained_model,
                path_to_merged_model = path_to_merged_model, 
                path_to_base_model = path_to_base_model, 
                cutoff_len = cutoff_len, 
                epoch = epoch, 
                batch_size = batch_size, 
                grad_accum = grad_accum, 
                learning_rate = learning_rate, 
                warmup_ratio = warmup_ratio
            )
        else:
            self.generate_fft_config(
                train_dataset_name = train_dataset_name,
                path_to_trained_model = path_to_trained_model,
                path_to_merged_model = path_to_merged_model, 
                path_to_base_model = path_to_base_model, 
                cutoff_len = cutoff_len, 
                epoch = epoch, 
                batch_size = batch_size, 
                grad_accum = grad_accum, 
                learning_rate = learning_rate, 
                warmup_ratio = warmup_ratio
            )
        

    def generate_lora_config(
        self, 
        train_dataset_name : str,
        path_to_trained_model : str,
        path_to_merged_model : str,
        path_to_base_model : str,
        cutoff_len : int,
        epoch : float,
        batch_size : int,
        grad_accum : int,
        learning_rate = 5e-5,
        warmup_ratio = 0.1
    ):
        pass

    def generate_fft_config(
        self, 
        train_dataset_name : str,
        path_to_trained_model : str,
        path_to_merged_model : str,
        path_to_base_model : str,
        cutoff_len : int,
        epoch : float,
        batch_size : int,
        grad_accum = int,
        learning_rate = 5e-6,
        warmup_ratio = 0.03
    ):
        pass