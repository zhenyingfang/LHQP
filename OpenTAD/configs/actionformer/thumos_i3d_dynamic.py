import os
from pathlib import Path

_base_ = ["./thumos_i3d.py"]

repo_root = Path(os.environ.get("PTAL_REPO_ROOT", "/root/autodl-tmp/code/ptal_v2")).resolve()
run_root = repo_root / "runs/dynamic_self_train/thumos/beta1_gamma2_v1"
initial_annotation = repo_root / "runs/q_base/thumos/actionformer_annotations/thumos_14_q_base.json"

dataset = dict(
    train=dict(ann_file=str(initial_annotation)),
    val=dict(ann_file=str(initial_annotation)),
    test=dict(ann_file=str(initial_annotation)),
)
evaluation = dict(ground_truth_filename=str(initial_annotation))

scheduler = dict(
    type="LinearWarmupMultiStepLR",
    warmup_epoch=5,
    milestones=[30],
    gamma=0.1,
    max_epoch=35,
)
workflow = dict(
    end_epoch=35,
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=30,
)

dynamic_pseudo = dict(
    enabled=True,
    start_epoch=6,
    stop_epoch=27,
    interval=2,
    pipeline_config=str(repo_root / "configs/acl_tal.yaml"),
    initial_pseudo=str(repo_root / "runs/q_base/thumos/q_base_pseudo_labels.json"),
    output_dir=str(run_root / "dynamic_updates"),
    top_k=10,
    beta=1.0,
    gamma=2.0,
    min_agreement=1.00,
    min_agreement_end=0.00,
    update_fraction=1.00,
    update_momentum=0.90,
    updater_options=dict(
        boundary_min_tiou=0.5,
        alpha_start=0.3,
        alpha_end=0.3,
    ),
)

work_dir = str(run_root / "actionformer")

del Path, os, repo_root, run_root, initial_annotation

