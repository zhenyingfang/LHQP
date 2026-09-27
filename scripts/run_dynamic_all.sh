#!/usr/bin/env bash
# Run THUMOS, ActivityNet, BEOID, and GTEA dynamic ActionFormer training in order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

THUMOS_RUN_ROOT="${THUMOS_RUN_ROOT:-${REPO_ROOT}/runs/dynamic_self_train/thumos/beta1_gamma2_v1}"
ANET_RUN_ROOT="${ANET_RUN_ROOT:-${REPO_ROOT}/runs/dynamic_self_train/activitynet/beta1_gamma2_v1}"
BEOID_RUN_ROOT="${BEOID_RUN_ROOT:-${REPO_ROOT}/runs/dynamic_self_train/beoid/beta1_gamma2_v1}"
GTEA_RUN_ROOT="${GTEA_RUN_ROOT:-${REPO_ROOT}/runs/dynamic_self_train/gtea/beta1_gamma2_v1}"
THUMOS_RESUME="${THUMOS_RESUME:-}"
ANET_RESUME="${ANET_RESUME:-}"
BEOID_RESUME="${BEOID_RESUME:-}"
GTEA_RESUME="${GTEA_RESUME:-}"
THUMOS_REGENERATE_INITIAL="${THUMOS_REGENERATE_INITIAL:-1}"
ANET_REGENERATE_INITIAL="${ANET_REGENERATE_INITIAL:-1}"
BEOID_REGENERATE_INITIAL="${BEOID_REGENERATE_INITIAL:-1}"
GTEA_REGENERATE_INITIAL="${GTEA_REGENERATE_INITIAL:-1}"

echo "========== [1/4] Starting THUMOS dynamic training =========="
REPO_ROOT="${REPO_ROOT}" \
RUN_ROOT="${THUMOS_RUN_ROOT}" \
RESUME="${THUMOS_RESUME}" \
REGENERATE_INITIAL="${THUMOS_REGENERATE_INITIAL}" \
LOW_R_DROP_RATIO="${THUMOS_LOW_R_DROP_RATIO:-0.0}" \
bash "${SCRIPT_DIR}/run_dynamic_thumos_actionformer.sh"

echo "========== [1/4] THUMOS complete =========="
echo "========== [2/4] Starting ActivityNet dynamic training =========="
REPO_ROOT="${REPO_ROOT}" \
RUN_ROOT="${ANET_RUN_ROOT}" \
RESUME="${ANET_RESUME}" \
REGENERATE_INITIAL="${ANET_REGENERATE_INITIAL}" \
LOW_R_DROP_RATIO="${ANET_LOW_R_DROP_RATIO:-0.10}" \
bash "${SCRIPT_DIR}/run_dynamic_anet_actionformer.sh"

echo "========== [2/4] ActivityNet complete =========="
echo "========== [3/4] Starting BEOID dynamic training =========="
REPO_ROOT="${REPO_ROOT}" \
RUN_ROOT="${BEOID_RUN_ROOT}" \
RESUME="${BEOID_RESUME}" \
REGENERATE_INITIAL="${BEOID_REGENERATE_INITIAL}" \
LOW_R_DROP_RATIO="${BEOID_LOW_R_DROP_RATIO:-0.0}" \
bash "${SCRIPT_DIR}/run_dynamic_beoid_actf.sh"

echo "========== [3/4] BEOID complete =========="
echo "========== [4/4] Starting GTEA dynamic training =========="
REPO_ROOT="${REPO_ROOT}" \
RUN_ROOT="${GTEA_RUN_ROOT}" \
RESUME="${GTEA_RESUME}" \
REGENERATE_INITIAL="${GTEA_REGENERATE_INITIAL}" \
LOW_R_DROP_RATIO="${GTEA_LOW_R_DROP_RATIO:-0.0}" \
bash "${SCRIPT_DIR}/run_dynamic_gtea_actf.sh"

echo "========== [4/4] GTEA complete =========="
echo "========== All dynamic training complete =========="
