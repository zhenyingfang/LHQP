from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from actionformer_convert import convert_pseudo_to_actionformer_json
from adapters.hrpro_io_adapter import (
    iter_training_points,
    load_points_csv,
    load_stage1_predictions,
    load_thumos_annotations,
)
from config_utils import load_config, repo_path, setup_pipeline_environment
from io_utils import save_json
from q_base import select_r_pseudo_label, select_source_score_pseudo_label


def _selection_score(item: dict) -> float:
    if item.get("selector") == "source_score":
        return float(item["source_score"])
    return float(item["reliability"])


def drop_lowest_reliability(
    pseudo_labels: list[dict], ratio: float
) -> tuple[list[dict], list[dict]]:
    """Remove the global bottom ``ratio`` labels after sorting r descending."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"drop_low_r_ratio must be in [0, 1), got {ratio}")
    if ratio == 0.0 or not pseudo_labels:
        return pseudo_labels, []

    # A tail fraction cannot contain a partial point, so round its size up.
    # Stable point fields make the boundary deterministic when r values tie.
    ranked = sorted(
        pseudo_labels,
        key=lambda item: (
            -_selection_score(item),
            str(item["video_id"]),
            int(item["point_index"]),
        ),
    )
    num_dropped = min(len(ranked), math.ceil(len(ranked) * ratio - 1e-12))
    split = len(ranked) - num_dropped
    kept_keys = {
        (str(item["video_id"]), int(item["point_index"])) for item in ranked[:split]
    }
    kept = [
        item
        for item in pseudo_labels
        if (str(item["video_id"]), int(item["point_index"])) in kept_keys
    ]
    dropped = [
        {
            "video_id": str(item["video_id"]),
            "point_index": int(item["point_index"]),
            "point_time": float(item["point_time"]),
            "label": str(item["label"]),
            "reliability": _selection_score(item),
            "reason": "global_bottom_r",
        }
        for item in ranked[split:]
    ]
    return kept, dropped


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.train_pred:
        cfg.paths.train_pred = args.train_pred
    setup_pipeline_environment(cfg)
    start_time = time.time()
    annotations = load_thumos_annotations(cfg)
    predictions = load_stage1_predictions(cfg)
    points = iter_training_points(annotations, load_points_csv(cfg))
    pseudo = []
    missing = []
    selector = (
        select_r_pseudo_label
        if args.selection_mode == "r"
        else select_source_score_pseudo_label
    )
    selected_points = points if args.max_points is None else points[: args.max_points]
    for point in selected_points:
        label = selector(predictions.get(point["video_id"], []), point, cfg)
        if label is None:
            missing.append(
                {
                    "video_id": point["video_id"],
                    "point_index": int(point["point_index"]),
                    "point_time": float(point["time"]),
                    "label": point["label"],
                }
            )
        else:
            # other point annotations inside the chosen segment (excluding the
            # point itself), split by same/different label.  has_same_point=True
            # pseudo labels get their classification/regression loss skipped
            # (see ThumosPaddingDataset.get_gt); only flagged when loss
            # masking is requested -- by default no pseudo label carries it.
            others = [
                (other_time, other_label)
                for other_index, other_time, other_label in point["point_times"]
                if other_index != point["point_index"] and label["start"] <= other_time <= label["end"]
            ]
            label["other_points_count"] = len(others)
            label["has_diff_point"] = any(other_label != point["label"] for _, other_label in others)
            label["same_points_count"] = sum(1 for _, other_label in others if other_label == point["label"])
            label["has_same_point"] = args.loss_mask and label["same_points_count"] >= 1
            pseudo.append(label)
    pseudo, dropped_points = drop_lowest_reliability(pseudo, args.drop_low_r_ratio)
    pseudo_path = repo_path(cfg, args.output or cfg.paths.pseudo_json)
    save_json(
        pseudo_path,
        {
            "version": f"{args.selection_mode} pseudo labels",
            "selection_mode": args.selection_mode,
            "pseudo_labels": pseudo,
            "num_pseudo_labels": len(pseudo),
            "drop_low_r_ratio": float(args.drop_low_r_ratio),
            "dropped_points": dropped_points,
            "num_dropped_points": len(dropped_points),
            "missing_points": missing,
            "num_missing_points": len(missing),
        },
    )
    actionformer_json = convert_pseudo_to_actionformer_json(pseudo, annotations)
    actionformer_path = repo_path(cfg, args.actionformer_output or cfg.paths.actionformer_annotation_json)
    save_json(actionformer_path, actionformer_json)
    summary = {
        "selection_mode": args.selection_mode,
        "pseudo_labels": len(pseudo),
        "dropped_low_r_points": len(dropped_points),
        "missing_points": len(missing),
        "actionformer_annotation": actionformer_path,
        "pseudo_json": pseudo_path,
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    summary_path = (
        repo_path(cfg, args.summary_output)
        if args.summary_output
        else str(Path(repo_path(cfg, cfg.paths.output_dir)) / "pseudo_generation_summary.json")
    )
    save_json(summary_path, summary)
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select pseudo labels from HR-Pro stage-1 predictions")
    parser.add_argument("--config", default="configs/acl_tal.yaml")
    parser.add_argument(
        "--train-pred",
        default="",
        help="optional stage-1 prediction JSON path overriding paths.train_pred in the config",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--actionformer-output", default="")
    parser.add_argument(
        "--summary-output",
        default="",
        help="optional summary JSON path; defaults to the config output directory",
    )
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=("r", "source_score"),
        default="r",
        help=(
            "r keeps the existing production-reliability selector; "
            "source_score selects the maximum raw HR-Pro prediction score"
        ),
    )
    parser.add_argument(
        "--drop-low-r-ratio",
        type=float,
        default=0.0,
        help="globally sort selected labels by r descending and clear the tail ratio",
    )
    parser.add_argument("--loss-mask", action="store_true", default=False,
                        help="flag has_same_point=True on pseudo labels whose segment contains other same-label point annotations (their loss is then skipped by the dataset); default keeps ALL pseudo labels supervised")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
