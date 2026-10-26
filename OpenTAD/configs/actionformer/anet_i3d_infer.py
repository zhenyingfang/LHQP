import os
from pathlib import Path

_base_ = ["./anet_i3d.py"]
repo_root = Path(os.environ["PTAL_REPO_ROOT"])
inputs = repo_root / "inputs/activitynet"
checkpoint = str(repo_root / "ckpt/activitynet/best.pth")
dataset = dict(test=dict(
    ann_file=str(inputs / "ground_truth.json"),
    class_map=str(repo_root / "ckpt/activitynet/category_idx.txt"),
    data_path=str(repo_root / "OpenTAD/data/activitynet-1.3/features/hrpro_i3d_features"),
    block_list=str(repo_root / "OpenTAD/data/activitynet-1.3/features/hrpro_i3d_features/missing_files.txt"),
))
evaluation = dict(
    ground_truth_filename=str(inputs / "ground_truth.json"),
    blocked_videos=str(repo_root / "OpenTAD/data/activitynet-1.3/annotations/blocked_i3d_hrpro.json"),
)
post_processing = dict(
    save_dict=True,
    external_cls=dict(type="CUHKANETClassifier", path=str(inputs / "classifier.json"), topk=2),
)
work_dir = str(repo_root / "outputs/inference/activitynet")
del Path, os, repo_root, inputs
