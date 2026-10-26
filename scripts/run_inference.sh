#!/usr/bin/env bash
# Infer one dataset or all four, always using weights from ckpt by default.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENTAD_ENV="${OPENTAD_ENV:-/root/autodl-tmp/conda/envs/opentad}"
PYTHON="${PYTHON:-${OPENTAD_ENV}/bin/python}"
DATASET="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "${DATASET}" in
  all) DATASETS=(thumos activitynet beoid gtea) ;;
  thumos|activitynet|beoid|gtea) DATASETS=("${DATASET}") ;;
  *) echo "Usage: $0 [all|thumos|activitynet|beoid|gtea] [test.py options]" >&2; exit 2 ;;
esac
export PTAL_REPO_ROOT="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"
for dataset in "${DATASETS[@]}"; do
  config_dataset="${dataset}"
  if [[ "${dataset}" == "activitynet" ]]; then config_dataset=anet; fi
  "${PYTHON}" "${REPO_ROOT}/OpenTAD/tools/test.py" \
    "${REPO_ROOT}/OpenTAD/configs/actionformer/${config_dataset}_i3d_infer.py" \
    --checkpoint "${CKPT_ROOT:-${REPO_ROOT}/ckpt}/${dataset}/best.pth" \
    --work-dir "${OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference}/${dataset}" "$@"
done
