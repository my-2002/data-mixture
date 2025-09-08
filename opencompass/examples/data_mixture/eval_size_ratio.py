from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCBCodeGeneration_dataset
    from opencompass.configs.models.data_mixture.model_size_ratio import models as size_ratio_model
    
work_dir = 'outputs/size_ratio'

models = [*size_ratio_model]
datasets = [LCBCodeGeneration_dataset]