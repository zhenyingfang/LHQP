import os
from pathlib import Path

_base_ = ["./anet_i3d.py"]

repo_root = Path(
    os.environ.get("PTAL_REPO_ROOT", "/root/autodl-tmp/code/ptal_v2")
).resolve()
run_root = repo_root / "runs/dynamic_self_train/activitynet/beta1_gamma2_v1"
initial_annotation = (
    repo_root
    / "runs/q_base/activitynet/actionformer_annotations/activity_net.v1-3.min_q_base.json"
)

dataset = dict(
    train=dict(ann_file=str(initial_annotation)),
    val=dict(ann_file=str(initial_annotation)),
    test=dict(ann_file=str(initial_annotation)),
)
evaluation = dict(ground_truth_filename=str(initial_annotation))

# One uninterrupted model/optimizer/scheduler/EMA lifecycle for 20 epochs.
scheduler = dict(
    type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=10
)
workflow = dict(
    end_epoch=10,
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    # Evaluate the validation/test subset after every epoch from epoch 7.
    val_eval_interval=1,
    val_start_epoch=3,
)

dynamic_pseudo = dict(
    enabled=True,
    # Epochs 0-6 use q_base. Update before 7,9,11,13,15.
    start_epoch=4,
    stop_epoch=8,
    interval=1,
    pipeline_config=str(repo_root / "configs/acl_tal_anet.yaml"),
    initial_pseudo=str(
        repo_root / "runs/q_base/activitynet/q_base_pseudo_labels.json"
    ),
    output_dir=str(run_root / "dynamic_updates"),
    updater_module="dynamic_anet_pseudo",
    # Training-video inference cannot use the validation-only CUHK classifier.
    disable_external_cls=True,
    top_k=10,
    beta=1.0,
    gamma=2.0,
    min_agreement=0.30,
    min_agreement_end=1.00,
    update_fraction=0.60,
    update_momentum=0.70,
    updater_options=dict(
        boundary_min_tiou=0.5,
        alpha_start=0.6,
        alpha_end=0.6,
        min_segment_scale=0.01,
    ),
)

# This classifier covers the 4427 validation/test videos only. Dynamic
# training-set inference uses a copied post-processing config and explicitly
# disables external_cls via dynamic_pseudo.disable_external_cls above.
post_processing = dict(
    external_cls=dict(
        type="CUHKANETClassifier",
        path=str(repo_root / "inputs/activitynet/classifier.json"),
        topk=2,
    )
)

work_dir = str(run_root / "actionformer")

del Path, os, repo_root, run_root, initial_annotation
