from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets
    from opencompass.configs.models.data_mixture.model_{model_name} import models 
    
work_dir = 'outputs/{model_name}'

datasets = LCB_datasets