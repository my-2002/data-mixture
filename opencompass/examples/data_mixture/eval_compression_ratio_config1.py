from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCBCodeGeneration_dataset
    from opencompass.configs.models.data_mixture.model_compression_ratio_config1 import models as compression_ratio_model
    
work_dir = 'outputs/compression_ratio_config1'

models = [*compression_ratio_model]
datasets = [LCBCodeGeneration_dataset]