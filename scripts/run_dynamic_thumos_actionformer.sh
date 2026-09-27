#!/usr/bin/env bash
# One config-driven THUMOS ActionFormer run with in-process dynamic labels.
# The schedule and fusion settings come from thumos_i3d_dynamic.py. This shell
# generates a private initial cohort inside RUN_ROOT and overrides only the
# annotation/output paths needed to keep the run self-contained.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENTAD_ENV="${OPENTAD_ENV:-/root/autodl-tmp/conda/envs/opentad}"
PYTHON="${PYTHON:-${OPENTAD_ENV}/bin/python}"
TORCHRUN="${TORCHRUN:-${OPENTAD_ENV}/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEED="${SEED:-42}"
RESUME="${RESUME:-}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/dynamic_self_train/thumos/beta1_gamma2_v1}"
SHORT_TMP="${SHORT_TMP:-/tmp/ptal_dyn_thumos}"
PIPELINE_CONFIG="${REPO_ROOT}/configs/acl_tal.yaml"
LOW_R_DROP_RATIO="${LOW_R_DROP_RATIO:-0.0}"
REGENERATE_INITIAL="${REGENERATE_INITIAL:-1}"
INITIAL_DIR="${INITIAL_DIR:-${RUN_ROOT}/initial_labels}"
INITIAL_PSEUDO="${INITIAL_PSEUDO:-${INITIAL_DIR}/q_base_pseudo_labels.json}"
INITIAL_ANNOTATION="${INITIAL_ANNOTATION:-${INITIAL_DIR}/thumos_14_q_base.json}"
INITIAL_SUMMARY="${INITIAL_SUMMARY:-${INITIAL_DIR}/pseudo_generation_summary.json}"

mkdir -p "${RUN_ROOT}"/{tmp,dynamic_updates,logs} "${INITIAL_DIR}" "${SHORT_TMP}"

if [[ "${REGENERATE_INITIAL}" == "1" || ! -f "${INITIAL_PSEUDO}" || ! -f "${INITIAL_ANNOTATION}" ]]; then
  echo "Generating this run's private THUMOS q_base labels (drop ratio ${LOW_R_DROP_RATIO})..."
  "${PYTHON}" "${REPO_ROOT}/scripts/generate_pseudo_labels.py" \
    --config "${PIPELINE_CONFIG}" \
    --output "${INITIAL_PSEUDO}" \
    --actionformer-output "${INITIAL_ANNOTATION}" \
    --summary-output "${INITIAL_SUMMARY}" \
    --drop-low-r-ratio "${LOW_R_DROP_RATIO}"
fi

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
if [[ -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}")
fi

cd "${REPO_ROOT}/OpenTAD"
"${TORCHRUN}" \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  tools/train.py \
  configs/actionformer/thumos_i3d_dynamic.py \
  --seed "${SEED}" \
  "${RESUME_ARGS[@]}" \
  --cfg-options \
  "dataset.train.ann_file=${INITIAL_ANNOTATION}" \
  "dataset.val.ann_file=${INITIAL_ANNOTATION}" \
  "dataset.test.ann_file=${INITIAL_ANNOTATION}" \
  "evaluation.ground_truth_filename=${INITIAL_ANNOTATION}" \
  "work_dir=${RUN_ROOT}/actionformer" \
  "dynamic_pseudo.initial_pseudo=${INITIAL_PSEUDO}" \
  "dynamic_pseudo.output_dir=${RUN_ROOT}/dynamic_updates" \
  2>&1 | tee "${RUN_ROOT}/logs/train.log"

