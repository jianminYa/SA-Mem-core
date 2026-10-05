#!/usr/bin/env bash
set -euo pipefail

# Persistent Phase-2 orchestrator. It waits for the existing LME B2 run,
# launches LME B0/B1/B2 QA repeats, then runs HaluMem B0/B1/B2/B3 in isolated
# roots and evaluates QA repeats. No existing baseline directory is modified.

REPO="/workspace/SA-mem/halumem-b0-b3-work"
ENV_FILE="/workspace/SA-mem/4omini.txt"
WORK_ROOT="/workspace/SA-mem/halumem-b0-b3-runs"
H_DATA="/workspace/SA-mem/SA-Mem/data/data"
H_PROCESSED="$H_DATA/processed_halumem"
H_PROCESSED_ISO="$WORK_ROOT/metadata/processed_halumem_iso"
H_QA="$H_DATA"
COMBINED="$WORK_ROOT/metadata/halumem_medium_combined.json"
STATS="$WORK_ROOT/metadata/halumem_medium_stats.json"
ORCH_LOG="$WORK_ROOT/orchestrator.log"

mkdir -p "$WORK_ROOT/metadata" "$WORK_ROOT/status"
exec > >(tee -a "$ORCH_LOG") 2>&1

set -a
source "$ENV_FILE"
set +a
export OPENAI_API_KEY
export OPENAI_BASE_URL

echo "phase2 orchestrator started at $(date -u +%FT%TZ)"
echo "API base configured: $([[ -n "${OPENAI_BASE_URL:-}" ]] && echo true || echo false)"
echo "API key configured: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo true || echo false)"

python "$REPO/scripts/prepare_halumem_combined.py" \
  --processed-dir "$H_PROCESSED" \
  --qa-dir "$H_QA" \
  --output "$COMBINED" \
  --stats-output "$STATS"

python "$REPO/scripts/normalize_halumem_processed.py" \
  --input-dir "$H_PROCESSED" \
  --output-dir "$H_PROCESSED_ISO"

echo "waiting for LME B2 completion"
while true; do
  completed=$(python - <<'PY'
import json, glob, os
root='/workspace/SA-mem/longmemeval-baselines/runs/samem_2p/temporal_gate_b2_50q_retrieval/questions'
n=0
for d in glob.glob(root+'/*'):
    if not os.path.isdir(d):
        continue
    p=os.path.join(d,'question_status.json')
    if not os.path.exists(p):
        continue
    try:
        x=json.load(open(p,encoding='utf-8'))
    except Exception:
        continue
    if x.get('status') == 'complete' and os.path.exists(os.path.join(d,'retrieval.jsonl')):
        n += 1
print(n)
PY
)
  echo "LME B2 completed retrieval questions: $completed/50"
  if [[ "$completed" -ge 50 ]]; then
    break
  fi
  sleep 30
done

printf '%s\n' "b2_complete" > "$WORK_ROOT/status/lme_b2_complete"

echo "starting LME B0/B1/B2 QA repeats with 5 workers"
(
  cd "$REPO"
  python scripts/run_lme_qa_repeats.py \
    --output-root "$WORK_ROOT/lme_qa_repeats" \
    --repeats 3 \
    --workers 5 \
    --topk 10 \
    --model gpt-4o-mini \
    --judge-model gpt-4o-2024-08-06 \
    > "$WORK_ROOT/lme_qa_repeats.stdout.log" 2>&1
  printf '%s\n' "complete" > "$WORK_ROOT/status/lme_qa_repeats_complete"
) &
lme_pid=$!

echo "starting HaluMem B0/B1/B2/B3 construction and retrieval with 5 workers"
export HALUMEM_REPO_ROOT="$REPO"
export HALUMEM_PROCESSED_DIR="$H_PROCESSED_ISO"
export HALUMEM_COMBINED_FILE="$COMBINED"
export HALUMEM_OUTPUT_BASE="$WORK_ROOT/variants"
export HALUMEM_BUILD_WORKERS=5
export LLM_MODEL_OVERRIDE=gpt-4o-mini
export EMBEDDING_MODEL_OVERRIDE=text-embedding-3-small

for variant in b0 b1 b2 b3; do
  echo "starting HaluMem $variant"
  bash "$REPO/scripts/run_halumem_variant.sh" "$variant"
  printf '%s\n' "complete" > "$WORK_ROOT/status/halumem_${variant}_construction_retrieval_complete"
done

wait "$lme_pid"

echo "starting HaluMem B0/B1/B2/B3 QA repeats with 10 workers"
for variant in b0 b1 b2 b3; do
  bash "$REPO/scripts/run_halumem_qa_repeats.sh" \
    --variant "$variant" \
    --run-dir "$WORK_ROOT/variants/halumem_${variant}" \
    --combined-file "$COMBINED" \
    --output-root "$WORK_ROOT/qa/${variant}" \
    --repeats 3 \
    --workers 10 \
    --topk 20 \
    --model gpt-4o-mini \
    --judge-model gpt-4o-mini \
    > "$WORK_ROOT/qa_${variant}.stdout.log" 2>&1
  printf '%s\n' "complete" > "$WORK_ROOT/status/halumem_${variant}_qa_complete"
done

printf '%s\n' "complete" > "$WORK_ROOT/status/phase2_complete"
echo "phase2 orchestrator complete at $(date -u +%FT%TZ)"
