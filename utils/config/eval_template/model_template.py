from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='{model_name}',
        path="{path_to_eval_model}",
        max_out_len={max_out_len},
        batch_size={batch_size},
        run_cfg=dict(num_gpus={gpu_cnt}),
    )
]