from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from config_utils import ConfigNode, repo_path
from io_utils import load_json


def normalize_label(label: Any) -> str:
    return str(label).strip()


def video_id_variants(video_id: Any) -> list[str]:
    raw = str(video_id).strip()
    if not raw:
        return []
    variants = [raw]
    if raw.startswith("v_"):
        variants.append(raw[2:])
    else:
        variants.append(f"v_{raw}")
    return list(dict.fromkeys(variants))


def load_stage1_predictions(cfg: ConfigNode) -> dict[str, list[dict[str, Any]]]:
    data = load_json(repo_path(cfg, cfg.paths.train_pred))
    results = data.get("results", {})
    lookup: dict[str, list[dict[str, Any]]] = {}
    for video_id, proposals in results.items():
        for key in video_id_variants(video_id):
            lookup[key] = proposals
    return lookup


def load_thumos_annotations(cfg: ConfigNode) -> dict[str, Any]:
    annotations = load_json(repo_path(cfg, cfg.paths.gt_anno))
    fps_json = cfg.paths.get("fps_json", "")
    if not fps_json:
        return annotations

    metadata = load_json(repo_path(cfg, fps_json)).get("database", {})
    for video_id, video_entry in annotations.get("database", {}).items():
        meta = None
        for key in video_id_variants(video_id):
            if key in metadata:
                meta = metadata[key]
                break
        if meta is None:
            continue
        if "frame" not in video_entry and "num_frames" in meta:
            video_entry["frame"] = int(float(meta["num_frames"]))
        if "fps" not in video_entry and "fps" in meta:
            video_entry["fps"] = float(meta["fps"])
    return annotations


def load_points_csv(cfg: ConfigNode) -> dict[str, list[dict[str, Any]]]:
    points_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = Path(repo_path(cfg, cfg.paths.points_csv))
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "point", "class"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            video_id = (row.get("video_id") or "").strip()
            label = (row.get("class") or "").strip()
            point = (row.get("point") or "").strip()
            if not video_id or not label or not point:
                continue
            try:
                point_frame = int(float(point))
            except ValueError:
                continue
            point_item = {
                "point_index": idx,
                "id": idx,
                "video_id": video_id,
                "label": label,
                "point_frame": point_frame,
                "start_frame": row.get("start_frame"),
                "stop_frame": row.get("stop_frame"),
                "class_index": row.get("class_index"),
            }
            for key in video_id_variants(video_id):
                points_by_video[key].append(point_item)
    return points_by_video


def frame_to_time(point_frame: float, duration: float, total_frames: int) -> float | None:
    if total_frames <= 0 or duration <= 0:
        return None
    point_frame = max(0.0, min(float(point_frame), float(total_frames)))
    return point_frame / float(total_frames) * float(duration)


def point_frame_to_time(point_frame: float, video_entry: dict[str, Any]) -> float | None:
    duration = float(video_entry.get("duration", 0.0))
    if point_frame is None or duration <= 0:
        return None
    try:
        fps = float(video_entry.get("fps", 0.0) or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    if fps > 0:
        return max(0.0, min(float(point_frame) / fps, duration))
    total_frames = int(float(video_entry.get("frame", video_entry.get("num_frames", 0)) or 0))
    return frame_to_time(point_frame, duration, total_frames)


def collect_point_times(video_points: list[dict[str, Any]], video_entry: dict[str, Any]) -> list[tuple[int, float, str]]:
    point_times: list[tuple[int, float, str]] = []
    for point in video_points:
        point_time = point_frame_to_time(point["point_frame"], video_entry)
        if point_time is not None:
            point_times.append((int(point["point_index"]), float(point_time), normalize_label(point["label"])))
    return point_times


def iter_training_points(
    annotations: dict[str, Any],
    points_by_video: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for video_id, video_entry in annotations.get("database", {}).items():
        if str(video_entry.get("subset", "")).lower() != "training":
            continue
        duration = float(video_entry.get("duration", 0.0))
        total_frames = int(float(video_entry.get("frame", video_entry.get("num_frames", 0)) or 0))
        point_times = collect_point_times(points_by_video.get(video_id, []), video_entry)
        for point in points_by_video.get(video_id, []):
            point_time = point_frame_to_time(point["point_frame"], video_entry)
            if point_time is None:
                continue
            enriched = dict(point)
            enriched.update(
                {
                    "video_id": video_id,
                    "time": float(point_time),
                    "duration": duration,
                    "frame": total_frames,
                    "point_times": point_times,
                }
            )
            items.append(enriched)
    return items
