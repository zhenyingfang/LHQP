from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any


def convert_pseudo_to_actionformer_json(
    pseudo_labels: list[dict[str, Any]],
    base_annotation: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(base_annotation)
    database = out.get("database", {})
    records_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pseudo_labels:
        records_by_video[item["video_id"]].append(item)

    for video_id, video_entry in database.items():
        if str(video_entry.get("subset", "")).lower() != "training":
            continue
        annotations = []
        for record in records_by_video.get(video_id, []):
            item = {
                "segment": [float(record["start"]), float(record["end"])],
                "label": str(record["label"]),
                "score": float(record.get("score", 0.0)),
                "point_index": int(record.get("point_index", -1)),
                "candidate_index": int(record.get("candidate_index", -1)),
            }
            # ambiguous pseudo label: the segment contains >=1 other point
            # annotation; its classification/regression loss is skipped in
            # training (see ThumosPaddingDataset.get_gt)
            if "ambiguous" in record:
                item["ambiguous"] = bool(record["ambiguous"])
            if "other_points_count" in record:
                item["other_points_count"] = int(record["other_points_count"])
            if "has_diff_point" in record:
                item["has_diff_point"] = bool(record["has_diff_point"])
            if "same_points_count" in record:
                item["same_points_count"] = int(record["same_points_count"])
            if "has_same_point" in record:
                item["has_same_point"] = bool(record["has_same_point"])
            annotations.append(item)
        annotations.sort(key=lambda item: (item["segment"][0], item["segment"][1], item["label"], item["point_index"]))
        video_entry["annotations"] = annotations
    out["version"] = "q_base pseudo labels"
    return out
