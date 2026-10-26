"""Copy each dynamic training run's best weights and class map into ckpt/."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DATASETS = ("thumos", "activitynet", "beoid", "gtea")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs/dynamic_self_train")
    parser.add_argument("--run-name", default="beta1_gamma2_v1")
    parser.add_argument("--experiment", default="gpu1_id0")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "ckpt")
    parser.add_argument("--overwrite", action="store_true", help="replace existing weights if they differ")
    args = parser.parse_args()
    copies = []
    # Validate every selected source and destination before copying anything.
    for dataset in args.datasets:
        run = args.run_root / dataset / args.run_name
        source = run / "actionformer" / args.experiment / "checkpoint/best.pth"
        if dataset in ("thumos", "activitynet"):
            data_name = "thumos-14" if dataset == "thumos" else "activitynet-1.3"
            class_map = REPO_ROOT / "OpenTAD/data" / data_name / "annotations/category_idx.txt"
        else:
            class_map = run / "initial_labels/category_idx.txt"
        target = args.ckpt_root / dataset
        for file in (source, class_map):
            if not file.is_file():
                raise FileNotFoundError(file)
        digest = sha256(source)
        for original, dest in ((source, target / "best.pth"), (class_map, target / "category_idx.txt")):
            if dest.exists() and sha256(original) != sha256(dest) and not args.overwrite:
                raise FileExistsError(f"{dest} differs from {original}; use --overwrite to replace it")
        copies.append((dataset, source, class_map, target, digest))
    for dataset, source, class_map, target, digest in copies:
        target.mkdir(parents=True, exist_ok=True)
        for original, dest in ((source, target / "best.pth"), (class_map, target / "category_idx.txt")):
            if not dest.exists() or sha256(original) != sha256(dest):
                shutil.copy2(original, dest)
        manifest = dict(dataset=dataset, source=str(source.resolve()), checkpoint="best.pth",
                        sha256=digest, class_map="category_idx.txt")
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{dataset}: {source} -> {target / 'best.pth'} (SHA256 {digest})")


if __name__ == "__main__":
    main()
