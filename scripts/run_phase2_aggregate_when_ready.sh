#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/workspace/SA-mem/halumem-b0-b3-runs"
REPO="/workspace/SA-mem/halumem-b0-b3-work"
LME_ROOT_BASE="/workspace/SA-mem/longmemeval-baselines/runs/samem_2p"
COMBINED="$WORK_ROOT/metadata/halumem_medium_combined.json"

while [[ ! -f "$WORK_ROOT/status/phase2_complete" ]]; do
  sleep 30
done

mkdir -p "$WORK_ROOT/reports"
conda run -n samem-lme python "$REPO/scripts/aggregate_phase2_results.py" \
  --work-root "$WORK_ROOT" \
  --lme-root-base "$LME_ROOT_BASE" \
  --combined "$COMBINED" \
  --output "$WORK_ROOT/reports"
printf '%s\n' "aggregate_complete" > "$WORK_ROOT/status/aggregate_complete"
