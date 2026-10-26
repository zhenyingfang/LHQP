import os
from pathlib import Path

_base_ = ["./gtea_i3d_dynamic.py"]
repo_root = Path(os.environ["PTAL_REPO_ROOT"])
inputs = repo_root / "inputs/gtea"
checkpoint = str(repo_root / "ckpt/gtea/best.pth")
dataset = dict(test=dict(
    ann_file=str(inputs / "ground_truth.json"),
    class_map=str(repo_root / "ckpt/gtea/category_idx.txt"),
    data_path=str(inputs / "features/test"),
))
evaluation = dict(
    ground_truth_filename=str(inputs / "ground_truth.json"),
)
post_processing = dict(
    save_dict=True,
)
work_dir = str(repo_root / "outputs/inference/gtea")
dynamic_pseudo = dict(enabled=False)
del Path, os, repo_root, inputs
