"""GT-free dynamic THUMOS/GTEA/BEOID pseudo-label updates for ActionFormer training.

The current EMA ActionFormer supplies a per-point teacher segment.  Fixed
HR-Pro top-k proposals then refine its boundaries when teacher/consensus tIoU
passes the boundary threshold, using the calibrated beta/gamma weights.
Ground truth is used only for
post-hoc diagnostics; no per-point update decision reads its GT segment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from actionformer_convert import convert_pseudo_to_actionformer_json
from adapters.hrpro_io_adapter import (
    iter_training_points,
    load_points_csv,
    load_stage1_predictions,
    load_thumos_annotations,
    point_frame_to_time,
)
from config_utils import load_config
from q_base import collect_covering_predictions


def _read(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _tiou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    intersection = np.maximum(
        0.0,
        np.minimum(first[..., 1], second[..., 1])
        - np.maximum(first[..., 0], second[..., 0]),
    )
    union = (
        first[..., 1] - first[..., 0]
        + second[..., 1] - second[..., 0]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def _metrics(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean_tiou": float(values.mean()) if values.size else 0.0,
        "ge_0_5": float(np.mean(values >= 0.5)) if values.size else 0.0,
        "ge_0_7": float(np.mean(values >= 0.7)) if values.size else 0.0,
    }


def _lookup_results(payload: dict) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = {}
    for video_id, proposals in payload.get("results", {}).items():
        raw = str(video_id).strip()
        lookup[raw] = proposals
        lookup[raw[2:] if raw.startswith("v_") else f"v_{raw}"] = proposals
    return lookup


def update_dynamic_labels(
    *,
    pipeline_config: str,
    teacher_prediction: str,
    current_pseudo: str,
    output_pseudo: str,
    output_annotation: str,
    audit_path: str,
    epoch: int,
    beta: float = 1.0,
    gamma: float = 2.0,
    top_k: int = 10,
    min_agreement: float = 0.30,
    update_fraction: float = 0.60,
    update_momentum: float = 0.70,
    boundary_min_tiou: float = 0.5,
    alpha_start: float = 0.5,
    alpha_end: float = 0.1,
) -> dict:
    """Create one versioned pseudo-label update and return its audit payload."""
    cfg = load_config(pipeline_config)
    # The OpenTAD launcher runs from OpenTAD/, while this pipeline config uses
    # repo_root: ".". Resolve it from the absolute config location instead.
    cfg.paths.repo_root = str(Path(pipeline_config).resolve().parent.parent)
    annotations = load_thumos_annotations(cfg)
    points = iter_training_points(annotations, load_points_csv(cfg))
    hrpro = load_stage1_predictions(cfg)
    teacher = _lookup_results(_read(teacher_prediction))
    current_payload = _read(current_pseudo)
    current_labels = current_payload.get("pseudo_labels", [])
    current_lookup = {
        (str(item["video_id"]), int(item["point_index"])): item
        for item in current_labels
    }

    proposals: dict[tuple[str, int], dict] = {}
    diagnostics: list[dict] = []
    for point in points:
        key = (str(point["video_id"]), int(point["point_index"]))
        old = current_lookup.get(key)
        if old is None:
            continue  # preserve points intentionally empty in round one
        af_candidates = collect_covering_predictions(
            teacher.get(point["video_id"], []),
            point,
            conflict_mode=getattr(cfg.q_base, "conflict_mode", "distance"),
            conflict_power=float(getattr(cfg.q_base, "conflict_power", 2.0)),
            softmax_source=bool(getattr(cfg.q_base, "softmax_source", True)),
        )
        if not af_candidates:
            continue
        af = max(af_candidates, key=lambda item: float(item["reliability"]))
        second = np.asarray([float(af["start"]), float(af["end"])], dtype=np.float64)
        previous = np.asarray([float(old["start"]), float(old["end"])], dtype=np.float64)

        candidates = collect_covering_predictions(
            hrpro.get(point["video_id"], []),
            point,
            conflict_mode=getattr(cfg.q_base, "conflict_mode", "distance"),
            conflict_power=float(getattr(cfg.q_base, "conflict_power", 2.0)),
            softmax_source=bool(getattr(cfg.q_base, "softmax_source", True)),
            match_label=True,
        )
        candidates.sort(key=lambda item: float(item["reliability"]), reverse=True)
        candidates = candidates[:top_k]
        target = second.copy()
        boundary_gate = False
        if candidates:
            segments = np.asarray(
                [[float(item["start"]), float(item["end"])] for item in candidates],
                dtype=np.float64,
            )
            reliability = np.asarray(
                [float(item["reliability"]) for item in candidates], dtype=np.float64
            )
            normalized_r = reliability / max(float(reliability.max()), 1e-12)
            similarity = _tiou(segments, np.broadcast_to(second, segments.shape))
            weights = np.exp(beta * similarity) * np.maximum(normalized_r, 1e-12) ** gamma
            consensus = np.sum(segments * (weights / weights.sum())[:, None], axis=0)
            consensus_agreement = float(
                _tiou(second[None, :], consensus[None, :])[0]
            )
            boundary_gate = bool(consensus_agreement >= boundary_min_tiou)
            if boundary_gate:
                target[0] = (1.0 - alpha_start) * second[0] + alpha_start * consensus[0]
                target[1] = (1.0 - alpha_end) * second[1] + alpha_end * consensus[1]

        point_time = float(point["time"])
        duration = float(point["duration"])
        target[0] = min(max(float(target[0]), 0.0), point_time)
        target[1] = max(min(float(target[1]), duration), point_time)
        agreement = float(_tiou(previous[None, :], second[None, :])[0])
        source_score = float(af.get("source_score", af.get("score", 0.0)))
        # Ranking is fully observable. Agreement prevents unstable early EMA
        # predictions from replacing a sound previous label.
        confidence = agreement * max(source_score, 0.0)
        diagnostics.append(
            {
                "key": key,
                "previous": previous,
                "teacher": second,
                "target": target,
                "agreement": agreement,
                "source_score": source_score,
                "confidence": confidence,
                "boundary_gate": boundary_gate,
                "num_hrpro_candidates": len(candidates),
                "af_reliability": float(af["reliability"]),
            }
        )

    eligible = [item for item in diagnostics if item["agreement"] >= min_agreement]
    eligible.sort(key=lambda item: item["confidence"], reverse=True)
    selected_count = min(len(eligible), int(round(update_fraction * len(current_labels))))
    selected = {item["key"]: item for item in eligible[:selected_count]}

    new_labels = []
    shifts = []
    for old in current_labels:
        updated = dict(old)
        key = (str(old["video_id"]), int(old["point_index"]))
        item = selected.get(key)
        if item is not None:
            previous = item["previous"]
            target = item["target"]
            segment = (1.0 - update_momentum) * previous + update_momentum * target
            point_time = float(old.get("point_time", 0.5 * (segment[0] + segment[1])))
            duration = float(annotations["database"][key[0]]["duration"])
            segment[0] = min(max(float(segment[0]), 0.0), point_time)
            segment[1] = max(min(float(segment[1]), duration), point_time)
            shifts.append(float(np.mean(np.abs(segment - previous))))
            updated.update(
                start=float(segment[0]),
                end=float(segment[1]),
                selector="dynamic_af_hrpro_beta1_gamma2",
                dynamic_epoch=int(epoch),
                dynamic_agreement=float(item["agreement"]),
                dynamic_teacher_score=float(item["source_score"]),
            )
        new_labels.append(updated)

    output_payload = dict(current_payload)
    output_payload["version"] = f"dynamic THUMOS pseudo labels after epoch {epoch}"
    output_payload["pseudo_labels"] = new_labels
    output_payload["num_pseudo_labels"] = len(new_labels)

    # Metrics are post-hoc diagnostics only and never influence selection.
    ground_truth = {}
    for point in points:
        entry = annotations["database"][point["video_id"]]
        ground_truth[(str(point["video_id"]), int(point["point_index"]))] = np.asarray(
            [
                point_frame_to_time(float(point["start_frame"]), entry),
                point_frame_to_time(float(point["stop_frame"]), entry),
            ],
            dtype=np.float64,
        )
    common = [
        (item, ground_truth[(str(item["video_id"]), int(item["point_index"]))])
        for item in new_labels
        if (str(item["video_id"]), int(item["point_index"])) in ground_truth
    ]
    old_lookup = current_lookup
    old_segments = np.asarray(
        [[old_lookup[(str(item["video_id"]), int(item["point_index"]))]["start"],
          old_lookup[(str(item["video_id"]), int(item["point_index"]))]["end"]]
         for item, _ in common], dtype=np.float64
    )
    new_segments = np.asarray(
        [[item["start"], item["end"]] for item, _ in common], dtype=np.float64
    )
    gt_segments = np.asarray([gt for _, gt in common], dtype=np.float64)
    before_metrics = _metrics(_tiou(old_segments, gt_segments))
    after_metrics = _metrics(_tiou(new_segments, gt_segments))
    audit = {
        "epoch": int(epoch),
        "policy": {
            "beta": beta,
            "gamma": gamma,
            "top_k": top_k,
            "min_agreement": min_agreement,
            "update_fraction": update_fraction,
            "update_momentum": update_momentum,
            "boundary_min_tiou": boundary_min_tiou,
            "alpha_start": alpha_start,
            "alpha_end": alpha_end,
            "per_point_gt_used": False,
        },
        "counts": {
            "current_labels": len(current_labels),
            "teacher_candidates": len(diagnostics),
            "eligible": len(eligible),
            "updated": len(selected),
            "kept": len(current_labels) - len(selected),
        },
        "mean_absolute_boundary_shift_seconds": float(np.mean(shifts)) if shifts else 0.0,
        "post_hoc_training_gt_metrics": {
            "before": before_metrics,
            "after": after_metrics,
            "delta_mean_tiou": float(after_metrics["mean_tiou"] - before_metrics["mean_tiou"]),
        },
        "inputs": {
            "teacher_prediction": str(Path(teacher_prediction).resolve()),
            "current_pseudo": str(Path(current_pseudo).resolve()),
        },
        "outputs": {
            "pseudo": str(Path(output_pseudo).resolve()),
            "annotation": str(Path(output_annotation).resolve()),
        },
    }
    _atomic_write(output_pseudo, output_payload)
    _atomic_write(
        output_annotation,
        convert_pseudo_to_actionformer_json(new_labels, annotations),
    )
    _atomic_write(audit_path, audit)
    return audit

