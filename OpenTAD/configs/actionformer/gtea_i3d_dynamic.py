import os
from pathlib import Path

repo_root = Path(os.environ.get("PTAL_REPO_ROOT", "/root/autodl-tmp/code/point_paper")).resolve()
run_root = repo_root / "runs/dynamic_self_train/gtea/beta1_gamma2_v1"
inputs = repo_root / "inputs/gtea"
initial_annotation = run_root / "initial_labels/gtea_q_base.json"
class_map = run_root / "initial_labels/category_idx.txt"
feature_root = inputs / "features"

# HR-Pro I3D features have one snippet per 16 frames (8-frame center offset).
dataset = dict(
    train=dict(
        type="ThumosPaddingDataset",
        ann_file=str(initial_annotation),
        subset_name="training",
        block_list=None,
        class_map=str(class_map),
        data_path=str(feature_root / "train"),
        filter_gt=False,
        feature_stride=16,
        sample_stride=1,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=256, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="ThumosPaddingDataset",
        ann_file=str(initial_annotation),
        subset_name="validation",
        block_list=None,
        class_map=str(class_map),
        data_path=str(feature_root / "test"),
        filter_gt=False,
        feature_stride=16,
        sample_stride=1,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="ThumosPaddingDataset",
        ann_file=str(initial_annotation),
        subset_name="validation",
        block_list=None,
        class_map=str(class_map),
        data_path=str(feature_root / "test"),
        filter_gt=False,
        test_mode=True,
        feature_stride=16,
        sample_stride=1,
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

model = dict(
    type="ActionFormer",
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=2048,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=9),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=256,
        input_pdrop=0.2,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=7,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(
    type="LinearWarmupCosineAnnealingLR", warmup_epoch=20, max_epoch=60
)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
workflow = dict(
    end_epoch=60,
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=50,
)
dynamic_pseudo = dict(
    enabled=True,
    start_epoch=12,
    stop_epoch=48,
    interval=2,
    pipeline_config=str(repo_root / "configs/acl_tal_gtea.yaml"),
    initial_pseudo=str(run_root / "initial_labels/q_base_pseudo_labels.json"),
    output_dir=str(run_root / "dynamic_updates"),
    top_k=10,
    beta=1.0,
    gamma=2.0,
    min_agreement=1.00,
    min_agreement_end=0.00,
    update_fraction=1.00,
    update_momentum=0.10,
    updater_options=dict(
        boundary_min_tiou=0.5,
        alpha_start=0.5,
        alpha_end=0.5,
    ),
)

post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        iou_threshold=0.1,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=False,
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=str(initial_annotation),
)

work_dir = str(run_root / "actionformer")

del Path, os, repo_root, run_root, inputs, initial_annotation, class_map, feature_root
