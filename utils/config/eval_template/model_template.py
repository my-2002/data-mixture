from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='{model_name}',
        path="{path_to_eval_model}",
        max_out_len={max_out_len},
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]