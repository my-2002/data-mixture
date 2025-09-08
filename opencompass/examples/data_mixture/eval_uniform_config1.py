from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCBCodeGeneration_dataset
    from opencompass.configs.models.data_mixture.model_uniform_config1 import models as uniform_model
    
work_dir = 'outputs/uniform_config1'

models = [*uniform_model]
datasets = [LCBCodeGeneration_dataset]