#!/usr/bin/env bash
# Dynamic GTEA ActionFormer run with a private initial pseudo-label cohort.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENTAD_ENV="${OPENTAD_ENV:-/root/autodl-tmp/conda/envs/opentad}"
PYTHON="${PYTHON:-${OPENTAD_ENV}/bin/python}"
TORCHRUN="${TORCHRUN:-${OPENTAD_ENV}/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEED="${SEED:-42}"
RESUME="${RESUME:-}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/dynamic_self_train/gtea/beta1_gamma2_v1}"
SHORT_TMP="${SHORT_TMP:-/tmp/ptal_dyn_gtea}"
FEATURE_ROOT="${FEATURE_ROOT:-${REPO_ROOT}/inputs/gtea/features}"
PIPELINE_CONFIG="${REPO_ROOT}/configs/acl_tal_gtea.yaml"
MODEL_CONFIG="${REPO_ROOT}/OpenTAD/configs/actionformer/gtea_i3d_dynamic.py"
STAGE1_PREDICTIONS="${STAGE1_PREDICTIONS:-${REPO_ROOT}/inputs/gtea/stage1_predictions.json}"
LOW_R_DROP_RATIO="${LOW_R_DROP_RATIO:-0.0}"
REGENERATE_INITIAL="${REGENERATE_INITIAL:-1}"
INITIAL_DIR="${INITIAL_DIR:-${RUN_ROOT}/initial_labels}"
INITIAL_PSEUDO="${INITIAL_PSEUDO:-${INITIAL_DIR}/q_base_pseudo_labels.json}"
INITIAL_ANNOTATION="${INITIAL_ANNOTATION:-${INITIAL_DIR}/gtea_q_base.json}"
INITIAL_SUMMARY="${INITIAL_SUMMARY:-${INITIAL_DIR}/pseudo_generation_summary.json}"
CLASS_MAP="${INITIAL_DIR}/category_idx.txt"

for executable in "${PYTHON}" "${TORCHRUN}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "ERROR: executable not found: ${executable}" >&2
    exit 1
  fi
done
for file in "${PIPELINE_CONFIG}" "${MODEL_CONFIG}" "${STAGE1_PREDICTIONS}" \
            "${REPO_ROOT}/inputs/gtea/ground_truth.json" \
            "${REPO_ROOT}/inputs/gtea/point_labels.csv"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: input not found: ${file}" >&2
    exit 1
  fi
done
mkdir -p "${RUN_ROOT}"/{tmp,dynamic_updates,logs} "${INITIAL_DIR}" "${SHORT_TMP}"
cd "${REPO_ROOT}"

# OpenTAD's class indices must include labels from both train and validation.
"${PYTHON}" - "${REPO_ROOT}/inputs/gtea/ground_truth.json" \
  "${CLASS_MAP}" "${FEATURE_ROOT}" 7 <<'PYMAP'
import json
import sys
from pathlib import Path

gt_path, class_map_path, feature_root, count_text = sys.argv[1:]
database = json.loads(Path(gt_path).read_text(encoding="utf-8"))["database"]
classes = sorted({anno["label"] for video in database.values() for anno in video["annotations"]})
assert len(classes) == int(count_text), f"Expected {count_text} classes, found {len(classes)}"
for video_id, video in database.items():
    split = "train" if video["subset"] == "training" else "test"
    feature = Path(feature_root) / split / f"{video_id}.npy"
    if not feature.is_file():
        raise FileNotFoundError(feature)
Path(class_map_path).write_text("\n".join(classes) + "\n", encoding="utf-8")
print(f"Validated {len(database)} GTEA videos and {len(classes)} classes")
PYMAP

if [[ "${REGENERATE_INITIAL}" == "1" || ! -f "${INITIAL_PSEUDO}" || ! -f "${INITIAL_ANNOTATION}" ]]; then
  echo "Generating GTEA initial pseudo labels (drop ratio ${LOW_R_DROP_RATIO})..."
  "${PYTHON}" "${REPO_ROOT}/scripts/generate_pseudo_labels.py" \
    --config "${PIPELINE_CONFIG}" \
    --train-pred "${STAGE1_PREDICTIONS}" \
    --output "${INITIAL_PSEUDO}" \
    --actionformer-output "${INITIAL_ANNOTATION}" \
    --summary-output "${INITIAL_SUMMARY}" \
    --drop-low-r-ratio "${LOW_R_DROP_RATIO}"
fi

"${PYTHON}" - "${INITIAL_PSEUDO}" "${INITIAL_ANNOTATION}" "${LOW_R_DROP_RATIO}" <<'PYVALIDATE'
import json
import math
import sys
from pathlib import Path

pseudo_path, annotation_path, ratio_text = sys.argv[1:]
ratio = float(ratio_text)
payload = json.loads(Path(pseudo_path).read_text(encoding="utf-8"))
annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
labels = payload.get("pseudo_labels", [])
dropped = payload.get("dropped_points", [])
expected_dropped = math.ceil((len(labels) + len(dropped)) * ratio - 1e-12)
training_annotations = sum(
    len(video.get("annotations", []))
    for video in annotation.get("database", {}).values()
    if video.get("subset") == "training"
)
assert abs(float(payload.get("drop_low_r_ratio", -1.0)) - ratio) < 1e-12
assert len(dropped) == expected_dropped
assert int(payload.get("num_pseudo_labels", -1)) == len(labels)
assert int(payload.get("num_dropped_points", -1)) == len(dropped)
assert training_annotations == len(labels)
assert labels, "No initial pseudo labels were generated"
print(f"Validated initial cohort: kept={len(labels)}, dropped={len(dropped)}")
PYVALIDATE

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PTAL_REPO_ROOT="${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/tmp/pycache"
export TMPDIR="${SHORT_TMP}"
export TMP="${SHORT_TMP}"
export TEMP="${SHORT_TMP}"
export XDG_CACHE_HOME="${RUN_ROOT}/tmp/xdg_cache"
export MPLCONFIGDIR="${RUN_ROOT}/tmp/matplotlib"

RESUME_ARGS=()
TEE_ARGS=()
if [[ -n "${RESUME}" ]]; then
  if [[ "${RESUME}" != /* ]]; then
    RESUME="${REPO_ROOT}/${RESUME#./}"
  fi
  if [[ ! -f "${RESUME}" ]]; then
    echo "ERROR: resume checkpoint not found: ${RESUME}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${RESUME}")
  TEE_ARGS=(-a)
fi

echo "Starting GTEA dynamic ActionFormer: ${RUN_ROOT}"
cd "${REPO_ROOT}/OpenTAD"
"${TORCHRUN}" \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  tools/train.py \
  "configs/actionformer/gtea_i3d_dynamic.py" \
  --seed "${SEED}" \
  "${RESUME_ARGS[@]}" \
  --cfg-options \
  "dataset.train.ann_file=${INITIAL_ANNOTATION}" \
  "dataset.val.ann_file=${INITIAL_ANNOTATION}" \
  "dataset.test.ann_file=${INITIAL_ANNOTATION}" \
  "dataset.train.class_map=${CLASS_MAP}" \
  "dataset.val.class_map=${CLASS_MAP}" \
  "dataset.test.class_map=${CLASS_MAP}" \
  "dataset.train.data_path=${FEATURE_ROOT}/train" \
  "dataset.val.data_path=${FEATURE_ROOT}/test" \
  "dataset.test.data_path=${FEATURE_ROOT}/test" \
  "evaluation.ground_truth_filename=${INITIAL_ANNOTATION}" \
  "work_dir=${RUN_ROOT}/actionformer" \
  "dynamic_pseudo.initial_pseudo=${INITIAL_PSEUDO}" \
  "dynamic_pseudo.output_dir=${RUN_ROOT}/dynamic_updates" \
  2>&1 | tee "${TEE_ARGS[@]}" "${RUN_ROOT}/logs/train.log"
