from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='deepseek-r1-distill-qwen-1.5b-hf',
        path="{path_to_eval_model}",
        max_out_len=32768,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
    )
]