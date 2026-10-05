#!/usr/bin/env bash
set -u

ROOT="${PHASE2_WORK_ROOT:-/workspace/SA-mem/halumem-b0-b3-runs-v2}"
LOG="$ROOT/monitor_10m.log"

mkdir -p "$ROOT"

while :; do
  {
    printf 'time=%s\n' "$(date -u +%FT%TZ)"
    for variant in b0 b1 b2 b3; do
      calls_file="$ROOT/variants/halumem_${variant}/construction_calls.jsonl"
      if [[ -f "$calls_file" ]]; then
        calls=$(wc -l < "$calls_file")
        provider_fail=$(jq -s 'map(select(.provider_usage_available==true and .success != true))|length' "$calls_file" 2>/dev/null || echo NA)
        printf '%s calls=%s provider_fail=%s\n' "$variant" "$calls" "$provider_fail"
      else
        printf '%s state=not_started\n' "$variant"
      fi
    done
    printf 'markers='
    find "$ROOT/status" -maxdepth 1 -type f -printf '%f ' 2>/dev/null | sort
    printf '\n'
    build_processes=$(ps -eo cmd | rg 'python memblock_cli.py.*halumem_b[0-3]' | rg -v 'rg ' | wc -l)
    printf 'build_processes=%s\n' "$build_processes"
  } >> "$LOG" 2>&1
  sleep 600
done
