from __future__ import annotations

from typing import Any

import numpy as np

from adapters.hrpro_io_adapter import normalize_label


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    scaled = values / max(float(temperature), 1e-8)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return (weights / weights.sum()).astype(np.float32)


def _temporal_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    return intersection / union if union > 0.0 else 0.0


def _conflicts(
    start: float,
    end: float,
    point_index: int,
    point_times: list[tuple[int, float, str]],
) -> list[tuple[int, float, str]]:
    return [
        (other_index, other_time, other_label)
        for other_index, other_time, other_label in point_times
        if other_index != point_index and start <= other_time <= end
    ]


def _distance_conflict(
    start: float,
    end: float,
    point_time: float,
    point_index: int,
    point_times: list[tuple[int, float, str]],
    power: float = 2.0,
) -> float:
    """Exclusivity of the prediction w.r.t. the point, in (0, 1].

    Collect every other point whose time falls inside [start, end], sum
    their absolute time distance to ``point_time`` -> ratio (no length
    normalisation); conflict = tanh(1 / ratio) ** power. A prediction
    covering no other point gets conflict = 1 (no penalty).
    """
    covered = [other_time for _, other_time, _ in _conflicts(start, end, point_index, point_times)]
    if not covered:
        return 1.0
    ratio = sum(abs(other_time - point_time) for other_time in covered)
    return float(np.tanh(1.0 / max(ratio, 1e-8)) ** power)


def _reliability(source_score: float, centrality: float, conflict: float, conflict_mode: str) -> float:
    """Reliability r from its components, per ``conflict_mode``."""
    if conflict_mode == "count":
        return source_score + centrality - conflict
    return source_score * centrality * conflict  # distance (default)


def collect_covering_predictions(
    predictions: list[dict[str, Any]],
    point: dict[str, Any],
    conflict_mode: str = "count",
    conflict_power: float = 2.0,
    softmax_source: bool = False,
    match_label: bool = True,
) -> list[dict[str, Any]]:
    """Return every valid, same-class HR-Pro prediction that covers ``point``.

    ``conflict_mode`` selects the conflict and reliability formulas:
      - "count" (default, original ACL-TAL): conflict = min(1, weighted count
        of covered other points, 1.0/0.5 for different/same class);
        reliability = source_score + centrality - conflict;
      - "distance" (production r): conflict = tanh(1/ratio) ** conflict_power
        over the summed time distance to covered other points (1.0 when
        nothing covered); reliability = source_score * centrality * conflict.

    With ``softmax_source``, the source scores of the candidates of this
    point are softmax-normalised (temperature 1) *before* computing the
    reliability, so the new s is a distribution over the candidates.

    With ``match_label=False``, the label check is skipped: every prediction
    that covers the point (whatever its label) is a candidate.  Used for
    class-agnostic proposal pools (e.g. ANet ActionFormer inference without
    the external classifier).
    """
    candidates = []
    point_time = float(point["time"])
    point_label = normalize_label(point["label"])
    point_index = int(point["point_index"])
    point_times = list(point.get("point_times", []))

    for prediction_index, prediction in enumerate(predictions or []):
        segment = prediction.get("segment", [prediction.get("start"), prediction.get("end")])
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            continue
        try:
            start, end = float(segment[0]), float(segment[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite([start, end]).all() or end <= start:
            continue
        if match_label and normalize_label(prediction.get("label", "")) != point_label:
            continue
        if not start <= point_time <= end:
            continue

        length = end - start
        centrality = max(0.0, min(1.0, 1.0 - abs(point_time - (start + end) / 2.0) / (length / 2.0)))
        source_score = float(prediction.get("score", 0.0))
        if conflict_mode == "count":
            conflicts = _conflicts(start, end, point_index, point_times)
            conflict = min(
                1.0,
                sum(1.0 if normalize_label(label) != point_label else 0.5 for _, _, label in conflicts),
            )
        else:  # distance
            conflict = _distance_conflict(
                start, end, point_time, point_index, point_times, power=conflict_power
            )
        candidates.append(
            {
                "start": start,
                "end": end,
                "label": point_label,
                "source_score": source_score,
                "source_instance_index": prediction_index,
                "centrality": centrality,
                "conflict": conflict,
                "reliability": _reliability(source_score, centrality, conflict, conflict_mode),
            }
        )

    if softmax_source and candidates:
        softmaxed = _softmax(
            np.asarray([candidate["source_score"] for candidate in candidates], dtype=np.float64), 1.0
        )
        for candidate, new_source in zip(candidates, softmaxed):
            candidate["source_score"] = float(new_source)
            candidate["reliability"] = _reliability(
                float(new_source), candidate["centrality"], candidate["conflict"], conflict_mode
            )
    return candidates


def compute_q_base(
    candidates: list[dict[str, Any]],
    cfg: Any,
    center_reference: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute the original ACL-TAL q_base distribution without its learned branches.

    When *center_reference* is provided (``{"start": c_s, "end": c_e}``), the
    centrality of every candidate is recomputed relative to the consensus-centre
    midpoint ``(c_s + c_e) / 2`` instead of the original per-candidate midpoint.
    """
    if not candidates:
        return np.zeros((0,), dtype=np.float32)

    if center_reference is not None:
        ref_center = (float(center_reference["start"]) + float(center_reference["end"])) / 2.0
        for candidate in candidates:
            length = float(candidate["end"]) - float(candidate["start"])
            half = max(length / 2.0, 1e-8)
            centrality = max(0.0, min(1.0, 1.0 - abs(ref_center - (candidate["start"] + candidate["end"]) / 2.0) / half))
            candidate["centrality"] = float(centrality)
            candidate["reliability"] = float(candidate["source_score"]) + centrality - float(candidate["conflict"])

    reliability = np.asarray([candidate["reliability"] for candidate in candidates], dtype=np.float32)
    teacher = _softmax(reliability, float(cfg.q_base.posterior_temperature))

    segments = [(candidate["start"], candidate["end"]) for candidate in candidates]
    consensus_utility = np.asarray(
        [
            sum(float(teacher[j]) * _temporal_iou(segment, other) for j, other in enumerate(segments))
            for segment in segments
        ],
        dtype=np.float32,
    )
    consensus = _softmax(consensus_utility, float(cfg.q_base.consensus_temperature))
    consensus_weight = float(cfg.q_base.consensus_weight)
    q_base = consensus_weight * consensus + (1.0 - consensus_weight) * teacher
    q_base /= max(float(q_base.sum()), 1e-8)

    for candidate, teacher_score, consensus_score, q_base_score in zip(candidates, teacher, consensus, q_base):
        candidate["teacher_posterior"] = float(teacher_score)
        candidate["consensus_posterior"] = float(consensus_score)
        candidate["q_base"] = float(q_base_score)
    return q_base.astype(np.float32)


def select_q_base_pseudo_label(
    predictions: list[dict[str, Any]],
    point: dict[str, Any],
    cfg: Any,
) -> dict[str, Any] | None:
    candidates = collect_covering_predictions(predictions, point)
    q_base = compute_q_base(candidates, cfg)
    if q_base.size == 0:
        return None

    candidate_index = int(np.argmax(q_base))
    candidate = candidates[candidate_index]
    return {
        "video_id": str(point["video_id"]),
        "label": str(point["label"]),
        "start": float(candidate["start"]),
        "end": float(candidate["end"]),
        "score": float(candidate["q_base"]),
        "q_base": float(candidate["q_base"]),
        "point_index": int(point["point_index"]),
        "point_time": float(point["time"]),
        "candidate_index": candidate_index,
        "source_instance_index": int(candidate["source_instance_index"]),
        "num_covering_predictions": len(candidates),
        "source_score": float(candidate["source_score"]),
    }


def select_r_pseudo_label(
    predictions: list[dict[str, Any]],
    point: dict[str, Any],
    cfg: Any,
) -> dict[str, Any] | None:
    """Select the pseudo label by the production reliability r; argmax of r.

    Settings come from ``cfg.q_base``: conflict_mode (default "distance"),
    conflict_power (default 2.0), softmax_source (default true). With the
    production settings r = softmax(s) * centrality * tanh(1/ratio) ** power.
    """
    candidates = collect_covering_predictions(
        predictions,
        point,
        conflict_mode=getattr(cfg.q_base, "conflict_mode", "distance"),
        conflict_power=float(getattr(cfg.q_base, "conflict_power", 2.0)),
        softmax_source=bool(getattr(cfg.q_base, "softmax_source", True)),
    )
    if not candidates:
        return None

    candidate_index = int(max(range(len(candidates)), key=lambda i: candidates[i]["reliability"]))
    candidate = candidates[candidate_index]
    return {
        "video_id": str(point["video_id"]),
        "label": str(point["label"]),
        "start": float(candidate["start"]),
        "end": float(candidate["end"]),
        "score": float(candidate["reliability"]),
        "reliability": float(candidate["reliability"]),
        "selector": "r",
        "point_index": int(point["point_index"]),
        "point_time": float(point["time"]),
        "candidate_index": candidate_index,
        "source_instance_index": int(candidate["source_instance_index"]),
        "num_covering_predictions": len(candidates),
        "source_score": float(candidate["source_score"]),
    }


def select_source_score_pseudo_label(
    predictions: list[dict[str, Any]],
    point: dict[str, Any],
    cfg: Any,
) -> dict[str, Any] | None:
    """Select the covering same-class HR-Pro proposal with maximum raw score.

    Unlike :func:`select_r_pseudo_label`, this selector does not softmax the
    source scores and does not use centrality or point-conflict penalties for
    ranking. Ties retain the original stage-1 prediction order.
    """
    candidates = collect_covering_predictions(
        predictions,
        point,
        conflict_mode=getattr(cfg.q_base, "conflict_mode", "distance"),
        conflict_power=float(getattr(cfg.q_base, "conflict_power", 2.0)),
        softmax_source=False,
    )
    if not candidates:
        return None

    candidate_index = int(
        max(range(len(candidates)), key=lambda i: candidates[i]["source_score"])
    )
    candidate = candidates[candidate_index]
    source_score = float(candidate["source_score"])
    return {
        "video_id": str(point["video_id"]),
        "label": str(point["label"]),
        "start": float(candidate["start"]),
        "end": float(candidate["end"]),
        "score": source_score,
        "selector": "source_score",
        "point_index": int(point["point_index"]),
        "point_time": float(point["time"]),
        "candidate_index": candidate_index,
        "source_instance_index": int(candidate["source_instance_index"]),
        "num_covering_predictions": len(candidates),
        "source_score": source_score,
    }

