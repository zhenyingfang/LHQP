import os
from pathlib import Path

_base_ = ["./thumos_i3d.py"]
repo_root = Path(os.environ["PTAL_REPO_ROOT"])
inputs = repo_root / "inputs/thumos"
checkpoint = str(repo_root / "ckpt/thumos/best.pth")
dataset = dict(test=dict(
    ann_file=str(inputs / "ground_truth.json"),
    class_map=str(repo_root / "ckpt/thumos/category_idx.txt"),
    data_path=str(repo_root / "OpenTAD/data/thumos-14/features/i3d_actionformer_stride4_thumos"),
    block_list=str(repo_root / "OpenTAD/data/thumos-14/features/i3d_actionformer_stride4_thumos/missing_files.txt"),
))
evaluation = dict(
    ground_truth_filename=str(inputs / "ground_truth.json"),
)
post_processing = dict(
    save_dict=True,
)
work_dir = str(repo_root / "outputs/inference/thumos")
del Path, os, repo_root, inputs
