from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2.5-7b-hf',
        path='../datamix/sft_results/size_ratio/merge/',
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
    )
]