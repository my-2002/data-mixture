from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2.5-7b-hf',
        path='../datamix/sft_results/uniform_config1/merge/',
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
    )
]