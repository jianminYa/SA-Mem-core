#!/usr/bin/env bash
set -euo pipefail

# Run one isolated HaluMem construction/retrieval/generation variant.
# The caller must source the API environment before invoking this script.

variant="${1:?variant must be b0|b1|b2|b3}"
repo_root="${HALUMEM_REPO_ROOT:?HALUMEM_REPO_ROOT is required}"
processed_dir="${HALUMEM_PROCESSED_DIR:?HALUMEM_PROCESSED_DIR is required}"
combined_file="${HALUMEM_COMBINED_FILE:?HALUMEM_COMBINED_FILE is required}"
output_base="${HALUMEM_OUTPUT_BASE:?HALUMEM_OUTPUT_BASE is required}"
llm_model="${LLM_MODEL_OVERRIDE:-gpt-4o-mini}"
embedding_model="${EMBEDDING_MODEL_OVERRIDE:-text-embedding-3-small}"
workers="${HALUMEM_BUILD_WORKERS:-10}"

case "$variant" in
  b0)
    run_id="halumem_b0"
    local_temporal=0
    temporal_gate=0
    merged=0
    ;;
  b1)
    run_id="halumem_b1"
    local_temporal=1
    temporal_gate=0
    merged=0
    ;;
  b2)
    run_id="halumem_b2"
    local_temporal=0
    temporal_gate=1
    merged=0
    ;;
  b3)
    run_id="halumem_b3"
    local_temporal=0
    temporal_gate=0
    merged=1
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

run_dir="$output_base/$run_id"
mkdir -p "$run_dir"
stage_dir="$run_dir/stages"
mkdir -p "$stage_dir"

export SA_MEM_OUTPUT_BASE_DIR="$output_base"
export MEMBLOCK_LOCAL_TEMPORAL_RESOLUTION="$local_temporal"
export MEMBLOCK_TEMPORAL_GATE_B2="$temporal_gate"
export MEMBLOCK_MERGED_EXTRACTION="$merged"
export CONSTRUCTION_CALLS_FILE="$run_dir/construction_calls.jsonl"
export TEMPORAL_GATE_LOG_FILE="$run_dir/temporal_gate.jsonl"

cd "$repo_root"

if [[ ! -f "$stage_dir/build.done" ]]; then
  python memblock_cli.py \
    --stage build \
    --raw-data-file "$combined_file" \
    --raw-data-dir "$processed_dir" \
    --raw-data-glob '*.json' \
    --run-id "$run_id" \
    --workers "$workers" \
    --resume \
    --limit-conversations -1 \
    --limit-sessions -1 \
    --answer-topn 20 \
    --text-modes content \
    --llm-model "$llm_model" \
    --embedding-model "$embedding_model" \
    > "$run_dir/build.stdout.log" 2>&1
  touch "$stage_dir/build.done"
fi

if [[ ! -f "$stage_dir/retrieve.done" ]]; then
  python memblock_cli.py \
    --stage retrieve \
    --raw-data-file "$combined_file" \
    --run-id "$run_id" \
    --limit-conversations -1 \
    --limit-sessions -1 \
    --answer-topn 20 \
    --text-modes content \
    --llm-model "$llm_model" \
    --embedding-model "$embedding_model" \
    > "$run_dir/retrieve.stdout.log" 2>&1
  touch "$stage_dir/retrieve.done"
fi

printf '%s\n' "complete" > "$stage_dir/status"
