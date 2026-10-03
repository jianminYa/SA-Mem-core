#!/usr/bin/env python3
"""Compare the original SA-Mem construction path with local temporal resolution."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGES = (
    "split_check",
    "pass1_extract",
    "temporal_local_resolve",
    "pass1_tool_followup",
    "pass1_tool_followup_fallback",
    "pass2_classify",
)
RELATIVE_RE = re.compile(
    r"\b(?:yesterday|today|last\s+(?:week|month|year|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)|this\s+(?:morning|afternoon|evening|tonight)|"
    r"two\s+months?\s+ago|a\s+month\s+ago|last\s+night|last\s+weekend)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def token_sum(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows if row.get("success") and row.get(field) is not None)


def stage_stats(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    breakdown: dict[str, dict[str, int]] = {}
    for stage in STAGES:
        subset = [row for row in rows if row.get("stage") == stage]
        breakdown[stage] = {
            "calls": sum(row.get("success") is True for row in subset),
            "input_tokens": token_sum(subset, "prompt_tokens"),
            "output_tokens": token_sum(subset, "completion_tokens"),
            "total_tokens": token_sum(subset, "total_tokens"),
        }
    llm_rows = [row for row in rows if row.get("stage") != "temporal_local_resolve"]
    stats = {
        "rows": len(rows),
        "llm_calls": sum(row.get("success") is True for row in llm_rows),
        "input_tokens": token_sum(llm_rows, "prompt_tokens"),
        "output_tokens": token_sum(llm_rows, "completion_tokens"),
        "total_tokens": token_sum(llm_rows, "total_tokens"),
        "wall_time_sec": sum(float(row.get("latency_sec") or 0.0) for row in rows),
        "retried_calls": sum(int(row.get("retry_index") or 0) for row in llm_rows),
        "failed_llm_calls": sum(row.get("success") is not True for row in llm_rows),
        "local_resolve_successes": sum(
            row.get("stage") == "temporal_local_resolve" and row.get("success") is True
            for row in rows
        ),
        "fallback_event_rows": sum(
            row.get("stage") == "temporal_local_resolve" and row.get("fallback_used") is True
            for row in rows
        ),
        "fallback_followup_calls": sum(
            row.get("stage") == "pass1_tool_followup_fallback" and row.get("success") is True
            for row in rows
        ),
        "breakdown": breakdown,
    }
    fallback_keys = {
        (
            str(row.get("session_id") or ""),
            str(row.get("block_id") or ""),
        )
        for row in rows
        if row.get("stage") == "temporal_local_resolve" and row.get("fallback_used") is True
    }
    stats["fallback_blocks"] = len(fallback_keys)
    return stats


def coverage_key(box: dict[str, Any]) -> tuple[str, int, int]:
    coverage = box.get("coverage") or {}
    return (
        str(coverage.get("session_id") or ""),
        int(coverage.get("start_idx") or 0),
        int(coverage.get("end_idx") or 0),
    )


def load_boxes(root: Path) -> list[dict[str, Any]]:
    return read_jsonl(root / "final_boxes_content.jsonl")


def cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else None


def retrieval_metrics(root: Path, item: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    retrieval_rows = read_jsonl(root / "retrieval.jsonl")
    row = retrieval_rows[0] if retrieval_rows else {}
    ranking = [int(x) for x in ((row.get("rankings") or {}).get("content_event_topic_kw") or [])]
    boxes = {int(box.get("block_id")): box for box in load_boxes(root)}
    gold_sessions = {str(x) for x in item.get("answer_session_ids") or []}

    def hit(k: int) -> bool:
        return any(
            str((boxes.get(block_id) or {}).get("coverage", {}).get("session_id")) in gold_sessions
            for block_id in ranking[:k]
        )

    gold_positions = [
        (rank, block_id)
        for rank, block_id in enumerate(ranking, start=1)
        if str((boxes.get(block_id) or {}).get("coverage", {}).get("session_id")) in gold_sessions
    ]
    gold_rank = gold_positions[0][0] if gold_positions else None
    gold_block = gold_positions[0][1] if gold_positions else None
    gold_score = None

    manifest = read_json(root / "run_manifest.json", {}) or {}
    aliases = manifest.get("retriever_user_aliases") or {}
    alias = str(aliases.get(str(item["question_id"]), item["question_id"]))
    stores = list((root / "vector_store").glob("user_*.json"))
    if gold_block is not None and stores:
        store = read_json(stores[0], {}) or {}
        query = (store.get(f"qa_{alias}_{alias}") or {}).get("question")
        vector = (store.get(f"{alias}_{gold_block}") or {}).get("content_event_topic_kw")
        if query and vector:
            gold_score = cosine(query, vector)

    return {
        "hit_at_1": hit(1),
        "hit_at_5": hit(5),
        "hit_at_10": hit(10),
        "hit_at_20": hit(20),
        "gold_rank": gold_rank,
        "gold_score": gold_score,
        "gold_block_id": gold_block,
        "ranking_size": len(ranking),
        "gold_preserved": bool(gold_positions),
        "gold_in_top20": bool(gold_rank and gold_rank <= 20),
        "gold_in_top10": bool(gold_rank and gold_rank <= 10),
    }


def event_time_signature(event: Any, box: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {"raw": str(event)}
    keys = ("start_time", "end_time", "event_time", "time", "observedTime")
    result = {key: event.get(key) for key in keys if event.get(key) is not None}
    for container_key in ("time_metadata", "temporal_metadata"):
        container = event.get(container_key)
        if isinstance(container, dict):
            result[container_key] = {
                key: container.get(key)
                for key in sorted(container)
                if key in {"start_time", "end_time", "startTime", "endTime", "observedTime"}
            }
    if not result:
        temporal = box.get("temporal_index") or {}
        result = {
            "block_event_start_time": temporal.get("block_event_start_time"),
            "block_event_end_time": temporal.get("block_event_end_time"),
        }
    return result


def event_description(event: Any) -> str:
    return str(event.get("description") or "") if isinstance(event, dict) else str(event or "")


def event_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(overlap, SequenceMatcher(None, left.lower(), right.lower()).ratio())


def temporal_comparison(b0_boxes: list[dict[str, Any]], b1_boxes: list[dict[str, Any]]) -> dict[str, Any]:
    b0 = {coverage_key(box): box for box in b0_boxes}
    b1 = {coverage_key(box): box for box in b1_boxes}
    block_rows = []
    for key in sorted(set(b0) & set(b1)):
        left, right = b0[key], b1[key]
        text = str((left.get("features") or {}).get("content_text") or "")
        descriptions = " ".join(str(event.get("description") or "") for event in left.get("events") or [])
        expressions = sorted(set(RELATIVE_RE.findall(text + " " + descriptions)))
        if not expressions:
            continue
        left_events = left.get("events") or []
        right_events = right.get("events") or []
        left_temporal = [
            event for event in left_events if RELATIVE_RE.search(event_description(event))
        ]
        right_temporal = [
            event for event in right_events if RELATIVE_RE.search(event_description(event))
        ]
        unmatched_right = set(range(len(right_temporal)))
        pairs = []
        for left_event in left_temporal:
            left_expr = {x.lower() for x in RELATIVE_RE.findall(event_description(left_event))}
            match_candidates = [
                (event_similarity(event_description(left_event), event_description(right_temporal[i])), i)
                for i in unmatched_right
                if left_expr & {x.lower() for x in RELATIVE_RE.findall(event_description(right_temporal[i]))}
            ]
            if not match_candidates:
                continue
            score, index = max(match_candidates)
            if score < 0.20:
                continue
            unmatched_right.remove(index)
            pairs.append((left_event, right_temporal[index]))
        pair_count = len(pairs)
        same = sum(
            event_time_signature(left_event, left) == event_time_signature(right_event, right)
            for left_event, right_event in pairs
        )
        block_rows.append({
            "coverage": {"session_id": key[0], "start_idx": key[1], "end_idx": key[2]},
            "expressions": expressions,
            "baseline_event_count": len(left_temporal),
            "optimized_event_count": len(right_temporal),
            "paired_events": pair_count,
            "same_time_metadata": same,
            "different_time_metadata": pair_count - same,
            "unmatched_events": len(left_temporal) + len(right_temporal) - 2 * pair_count,
            "baseline_times": [event_time_signature(left_event, left) for left_event, _ in pairs],
            "optimized_times": [event_time_signature(right_event, right) for _, right_event in pairs],
            "paired_descriptions": [
                {"baseline": event_description(left_event), "optimized": event_description(right_event)}
                for left_event, right_event in pairs
            ],
        })
    pair_count = sum(row["paired_events"] for row in block_rows)
    same = sum(row["same_time_metadata"] for row in block_rows)
    return {
        "candidate_blocks": len(block_rows),
        "paired_events": pair_count,
        "same_time_metadata": same,
        "different_time_metadata": pair_count - same,
        "missing_event_pairs": sum(row["unmatched_events"] for row in block_rows),
        "samples": block_rows[:12],
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pilot-ids", type=Path, required=True)
    parser.add_argument("--b0-root", type=Path, required=True)
    parser.add_argument("--b1-root", type=Path, required=True)
    parser.add_argument("--qa-results", type=Path, required=True)
    parser.add_argument("--b0-question-results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    args = parser.parse_args()

    data = {str(row["question_id"]): row for row in json.loads(args.data.read_text(encoding="utf-8"))}
    ids = [line.strip() for line in args.pilot_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    b1_qa_rows = read_jsonl(args.qa_results)
    b1_correct = {
        str(row["question_id"]): bool((row.get("autoeval_label") or {}).get("label"))
        for row in b1_qa_rows
    }
    b0_correct: dict[str, bool] = {}
    with args.b0_question_results.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["question_id"] in ids:
                b0_correct[row["question_id"]] = row.get("samem_correct", "").lower() == "true"

    rows = []
    temporal_by_question = {}
    for qid in ids:
        item = data[qid]
        b0_dir = args.b0_root / "questions" / qid
        b1_dir = args.b1_root / "questions" / qid
        b0_summary = (read_jsonl(b0_dir / "question_summary.jsonl") or [{}])[0]
        b1_summary = (read_jsonl(b1_dir / "question_summary.jsonl") or [{}])[0]
        b0_stats = stage_stats(b0_dir / "construction_calls.jsonl")
        b1_stats = stage_stats(b1_dir / "construction_calls.jsonl")
        b0_retrieval = retrieval_metrics(b0_dir, item, b0_summary)
        b1_retrieval = retrieval_metrics(b1_dir, item, b1_summary)
        b0_boxes = load_boxes(b0_dir)
        b1_boxes = load_boxes(b1_dir)
        temporal_by_question[qid] = temporal_comparison(b0_boxes, b1_boxes)
        rows.append({
            "question_id": qid,
            "question_type": item["question_type"],
            "b0": {
                "history": b0_summary.get("history", {}),
                "memory_units": len(b0_boxes),
                "construction": b0_stats,
                "retrieval": b0_retrieval,
                "qa_correct": b0_correct.get(qid),
            },
            "b1": {
                "history": b1_summary.get("history", {}),
                "memory_units": len(b1_boxes),
                "construction": b1_stats,
                "retrieval": b1_retrieval,
                "qa_correct": b1_correct.get(qid),
            },
        })

    def aggregate(system: str) -> dict[str, Any]:
        side = [row[system] for row in rows]
        totals = {
            "questions": len(side),
            "history_tokens": sum(int(x["history"].get("estimated_tokens") or 0) for x in side),
            "history_messages": sum(int(x["history"].get("messages") or 0) for x in side),
            "memory_units": sum(x["memory_units"] for x in side),
            "construction_input_tokens": sum(x["construction"]["input_tokens"] for x in side),
            "construction_output_tokens": sum(x["construction"]["output_tokens"] for x in side),
            "construction_total_tokens": sum(x["construction"]["total_tokens"] for x in side),
            "construction_llm_calls": sum(x["construction"]["llm_calls"] for x in side),
            "construction_wall_time_sec": sum(x["construction"]["wall_time_sec"] for x in side),
            "qa_correct": sum(x["qa_correct"] is True for x in side),
            "qa_accuracy": sum(x["qa_correct"] is True for x in side) / len(side) if side else None,
            "hit_at_1": sum(x["retrieval"]["hit_at_1"] for x in side) / len(side) if side else None,
            "hit_at_5": sum(x["retrieval"]["hit_at_5"] for x in side) / len(side) if side else None,
            "hit_at_10": sum(x["retrieval"]["hit_at_10"] for x in side) / len(side) if side else None,
            "hit_at_20": sum(x["retrieval"]["hit_at_20"] for x in side) / len(side) if side else None,
            "gold_rank_mean": sum(x["retrieval"]["gold_rank"] or 0 for x in side) / len(side) if side else None,
            "gold_rank_missing": sum(x["retrieval"]["gold_rank"] is None for x in side),
            "fallback_blocks": sum(x["construction"]["fallback_blocks"] for x in side),
            "fallback_event_rows": sum(x["construction"]["fallback_event_rows"] for x in side),
            "fallback_followup_calls": sum(x["construction"]["fallback_followup_calls"] for x in side),
            "local_resolve_successes": sum(x["construction"]["local_resolve_successes"] for x in side),
            "retried_calls": sum(x["construction"]["retried_calls"] for x in side),
            "failed_llm_calls": sum(x["construction"]["failed_llm_calls"] for x in side),
        }
        stage_totals = {}
        for stage in STAGES:
            stage_totals[stage] = {
                key: sum(x["construction"]["breakdown"][stage][key] for x in side)
                for key in ("calls", "input_tokens", "output_tokens", "total_tokens")
            }
        totals["stage_breakdown"] = stage_totals
        return totals

    aggregate_data = {"b0_original": aggregate("b0"), "b1_optimized": aggregate("b1")}
    aggregate_data["token_saving"] = {
        "absolute": aggregate_data["b0_original"]["construction_total_tokens"] - aggregate_data["b1_optimized"]["construction_total_tokens"],
        "fraction": 1 - aggregate_data["b1_optimized"]["construction_total_tokens"] / aggregate_data["b0_original"]["construction_total_tokens"],
        "ratio_b1_over_b0": aggregate_data["b1_optimized"]["construction_total_tokens"] / aggregate_data["b0_original"]["construction_total_tokens"],
    }
    aggregate_data["temporal"] = {
        "questions_with_candidates": sum(x["candidate_blocks"] > 0 for x in temporal_by_question.values()),
        "candidate_blocks": sum(x["candidate_blocks"] for x in temporal_by_question.values()),
        "paired_events": sum(x["paired_events"] for x in temporal_by_question.values()),
        "same_time_metadata": sum(x["same_time_metadata"] for x in temporal_by_question.values()),
        "different_time_metadata": sum(x["different_time_metadata"] for x in temporal_by_question.values()),
        "missing_event_pairs": sum(x["missing_event_pairs"] for x in temporal_by_question.values()),
        "by_question": temporal_by_question,
    }
    result = {
        "pilot_ids": ids,
        "questions": rows,
        "aggregate": aggregate_data,
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    b0, b1 = aggregate_data["b0_original"], aggregate_data["b1_optimized"]
    lines = [
        "# Temporal Follow-up Optimization Pilot",
        "",
        "This pilot compares the frozen 8-question SA-Mem B0 original two-pass path with B1 local temporal resolution. The baseline behavior, retrieval, QA prompts, MemBlock schema, model configuration, and embedding model were otherwise unchanged.",
        "",
        "## Configuration",
        "",
        "- Dataset: `longmemeval_s_cleaned.json`; IDs are a deterministic subset of the frozen seed-42 50Q set.",
        "- LLM: `gpt-4o-mini`; embedding: `text-embedding-3-small`; temperature: `0.0`.",
        "- SA-Mem: two-pass enabled, merged extraction off, graph off; retrieval candidate Top-20 and QA generation Top-10.",
        "- B0: original Pass1 function-calling and follow-up.",
        "- B1: Pass1 emits `temporal_expressions`; existing `resolve_temporal_expression()` resolves them locally; invalid/unreliable entries use the original tool path with stage `pass1_tool_followup_fallback`.",
        "",
        "## Pilot questions",
        "",
        "| Question ID | Type | Relative-time coverage |",
        "|---|---|---|",
    ]
    for qid in ids:
        item = data[qid]
        coverage = ", ".join(sorted(set(RELATIVE_RE.findall(json.dumps(item, ensure_ascii=False))))) or "none detected in raw sample"
        lines.append(f"| `{qid}` | `{item['question_type']}` | {coverage} |")
    lines += [
        "",
        "## Aggregate comparison",
        "",
        "| Metric | B0 original | B1 optimized |",
        "|---|---:|---:|",
        f"| Construction input tokens | {fmt(b0['construction_input_tokens'])} | {fmt(b1['construction_input_tokens'])} |",
        f"| Construction output tokens | {fmt(b0['construction_output_tokens'])} | {fmt(b1['construction_output_tokens'])} |",
        f"| Construction total tokens | {fmt(b0['construction_total_tokens'])} | {fmt(b1['construction_total_tokens'])} |",
        f"| Construction LLM calls | {fmt(b0['construction_llm_calls'])} | {fmt(b1['construction_llm_calls'])} |",
        f"| Construction wall time (sum) | {fmt(round(b0['construction_wall_time_sec'], 2))} sec | {fmt(round(b1['construction_wall_time_sec'], 2))} sec |",
        f"| Memory units | {fmt(b0['memory_units'])} | {fmt(b1['memory_units'])} |",
        f"| Gold-session Hit@10 | {pct(b0['hit_at_10'])} | {pct(b1['hit_at_10'])} |",
        f"| Mean gold rank | {fmt(b0['gold_rank_mean'])} | {fmt(b1['gold_rank_mean'])} |",
        f"| QA correct | {b0['qa_correct']}/{len(ids)} ({pct(b0['qa_accuracy'])}) | {b1['qa_correct']}/{len(ids)} ({pct(b1['qa_accuracy'])}) |",
        "",
        f"B1 construction-token saving: **{fmt(aggregate_data['token_saving']['absolute'])} tokens ({pct(aggregate_data['token_saving']['fraction'])})**; B1/B0 ratio: **{aggregate_data['token_saving']['ratio_b1_over_b0']:.4f}**.",
        "",
        "## Stage breakdown",
        "",
        "| Stage | B0 calls | B0 total | B1 calls | B1 total |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        lines.append(
            f"| `{stage}` | {fmt(b0['stage_breakdown'][stage]['calls'])} | {fmt(b0['stage_breakdown'][stage]['total_tokens'])} | {fmt(b1['stage_breakdown'][stage]['calls'])} | {fmt(b1['stage_breakdown'][stage]['total_tokens'])} |"
        )
    lines += [
        "",
        f"B1 local resolver successes: **{b1['local_resolve_successes']}**; fallback blocks: **{b1['fallback_blocks']}** (instrumentation rows: {b1['fallback_event_rows']}); fallback tool follow-up calls: **{b1['fallback_followup_calls']}**.",
        "",
        "## Per-question results",
        "",
        "| ID | Type | B0 total | B1 total | B0 units | B1 units | B0 rank | B1 rank | B0 Hit@10 | B1 Hit@10 | B0 QA | B1 QA |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        left, right = row["b0"], row["b1"]
        lines.append(
            f"| `{row['question_id']}` | `{row['question_type']}` | {fmt(left['construction']['total_tokens'])} | {fmt(right['construction']['total_tokens'])} | {left['memory_units']} | {right['memory_units']} | {fmt(left['retrieval']['gold_rank'])} | {fmt(right['retrieval']['gold_rank'])} | {'yes' if left['retrieval']['hit_at_10'] else 'no'} | {'yes' if right['retrieval']['hit_at_10'] else 'no'} | {'yes' if left['qa_correct'] else 'no'} | {'yes' if right['qa_correct'] else 'no'} |"
        )
    temporal = aggregate_data["temporal"]
    lines += [
        "",
        "## Temporal metadata audit",
        "",
        "The runtime schema stores event temporal information in `time_metadata`/`temporal_index` for these artifacts rather than always exposing `events.start_time/end_time`. The audit compares the actual available fields for blocks containing explicit relative-time expressions.",
        "",
        f"- Candidate blocks: **{temporal['candidate_blocks']}** across **{temporal['questions_with_candidates']}** questions.",
        f"- Paired relative-time events: **{temporal['paired_events']}**; identical time metadata: **{temporal['same_time_metadata']}**; different: **{temporal['different_time_metadata']}**; unmatched temporal event records: **{temporal['missing_event_pairs']}**.",
        "",
        "Representative audited blocks:",
        "",
    ]
    for qid in ids:
        samples = temporal_by_question[qid].get("samples") or []
        if not samples:
            continue
        sample = samples[0]
        lines.append(f"- `{qid}` `{sample['coverage']['session_id']}` expressions `{', '.join(sample['expressions'])}`: paired {sample['paired_events']}, same {sample['same_time_metadata']}, different {sample['different_time_metadata']}.")
    lines += [
        "",
        "## QA and failure-stage interpretation",
        "",
        "QA correctness here is a smoke diagnostic only. For each question, extraction is considered preserved when at least one memory covers an answer session; retrieval is considered the limiting stage when the best gold rank is beyond Top-10/Top-20; otherwise a wrong answer is attributed to QA generation. Full per-question details are in `TEMPORAL_FOLLOWUP_PILOT_RESULTS.json` and the local run artifacts.",
        "",
        "## Artifacts",
        "",
        "- B0: `runs/samem_2p/full_50_seed42/questions/<question_id>/`.",
        "- B1: `runs/samem_2p/temporal_followup_b1_8q_parallel/questions/<question_id>/`.",
        "- B1 calls include `temporal_local_resolve` with zero provider tokens and `pass1_tool_followup_fallback` for the original fallback path.",
        "- Exact aggregate JSON: `TEMPORAL_FOLLOWUP_PILOT_RESULTS.json`.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(ids), "b0_tokens": b0["construction_total_tokens"], "b1_tokens": b1["construction_total_tokens"], "saving_fraction": aggregate_data["token_saving"]["fraction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
