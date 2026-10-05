#!/usr/bin/env python3
"""Aggregate the automated LME/HaluMem phase-2 artifacts.

This post-processor is deliberately read-only with respect to construction and
retrieval runs.  It produces a compact machine-readable JSON and a Chinese
Markdown report under a separate report directory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGES = {
    "lme": {
        "b0": ["split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"],
        "b1": ["split_check", "pass1_extract", "pass1_tool_followup", "pass1_tool_followup_fallback", "temporal_local_resolve", "pass2_classify"],
        "b2": ["split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"],
    },
    "halumem": {
        # HaluMem phase 2 uses the same SA-Mem construction runner as LME,
        # with B3 switching to merged extraction.  Keep these names aligned
        # with the actual JSONL instrumentation rather than the earlier
        # Topic-Loom prototype labels.
        "b0": ["split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"],
        "b1": ["split_check", "pass1_extract", "temporal_local_resolve", "pass1_tool_followup_fallback", "pass2_classify"],
        "b2": ["split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"],
        "b3": ["split_check", "merged_extract", "merged_extract_tool_followup"],
    },
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "sum": 0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def usage_sum(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, int]:
    result = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        if prefix and str(row.get("stage", "")) != prefix:
            continue
        if row.get("success") is False:
            continue
        result["calls"] += 1
        result["input_tokens"] += int(row.get("prompt_tokens") or row.get("input_tokens") or 0)
        result["output_tokens"] += int(row.get("completion_tokens") or row.get("output_tokens") or 0)
        result["total_tokens"] += int(row.get("total_tokens") or 0)
    return result


def construction_summary(root: Path, variant: str, family: str) -> dict[str, Any]:
    question_dirs = sorted(p for p in (root / "questions").glob("*") if p.is_dir())
    if not question_dirs:
        # HaluMem variants use one run directory rather than question dirs.
        question_dirs = [root]
    all_rows: list[dict[str, Any]] = []
    unit_counts: list[float] = []
    complete = 0
    for qdir in question_dirs:
        status = read_json(qdir / "question_status.json", {}) or {}
        if status.get("status") == "complete":
            complete += 1
        rows = read_jsonl(qdir / "construction_calls.jsonl")
        if not rows and qdir == root:
            rows = read_jsonl(root / "construction_calls.jsonl")
        all_rows.extend(rows)
        boxes = qdir / "final_boxes_content.jsonl"
        if boxes.exists():
            with boxes.open(encoding="utf-8") as handle:
                unit_counts.append(float(sum(1 for x in handle if x.strip())))
    stage_names = STAGES[family][variant]
    stage_breakdown = {stage: usage_sum(all_rows, stage) for stage in stage_names}
    total = usage_sum(all_rows)
    provider_failures = sum(
        1 for row in all_rows
        if row.get("provider_usage_available") is True and row.get("success") is False
    )
    local_fallback_events = sum(
        1 for row in all_rows
        if row.get("usage_source") == "local_no_llm" and row.get("fallback_used") is True
    )
    return {
        "variant": variant,
        "completed_questions": complete if (root / "questions").exists() else None,
        "memory_units": stats(unit_counts),
        "construction": total,
        "stage_breakdown": stage_breakdown,
        "raw_call_rows": len(all_rows),
        "provider_failures": provider_failures,
        "local_fallback_events": local_fallback_events,
    }


def lme_qa_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path / "qa_results.jsonl")
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[str(row.get("system"))].append(row)
    out: dict[str, Any] = {}
    for system, values in sorted(by_system.items()):
        labels = [int(row.get("correct", 0)) for row in values]
        repeats = defaultdict(list)
        types: dict[str, list[int]] = defaultdict(list)
        for row in values:
            repeats[int(row.get("repeat", 0))].append(int(row.get("correct", 0)))
            types[str(row.get("question_type", "unknown"))].append(int(row.get("correct", 0)))
        out[system] = {
            "rows": len(values),
            "questions": len({row.get("question_id") for row in values}),
            "accuracy_over_all_repeated_rows": sum(labels) / len(labels) if labels else 0.0,
            "repeat_accuracy": {str(k): sum(v) / len(v) for k, v in sorted(repeats.items())},
            "mean_repeat_accuracy": statistics.mean([sum(v) / len(v) for v in repeats.values()]) if repeats else 0.0,
            "question_type_accuracy": {k: sum(v) / len(v) for k, v in sorted(types.items())},
            "qa_retries": sum(int(row.get("qa_retries", 0) or 0) for row in values),
            "judge_retries": sum(int(row.get("judge_retries", 0) or 0) for row in values),
        }
    return out


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def box_search_text(box: dict[str, Any]) -> str:
    return normalize_text(json.dumps(box, ensure_ascii=False))


def halumem_retrieval_summary(variant_root: Path, combined: Path) -> dict[str, Any]:
    data = read_json(combined, []) or []
    qa_map: dict[tuple[str, int], dict[str, Any]] = {}
    for user in data:
        uid = str(user.get("user_id", ""))
        for idx, qa in enumerate(user.get("qa", []) or []):
            qa_map[(uid, idx)] = qa
    boxes = {str(row.get("user_id")): {} for row in read_jsonl(variant_root / "final_boxes_content.jsonl")}
    for row in read_jsonl(variant_root / "final_boxes_content.jsonl"):
        boxes.setdefault(str(row.get("user_id")), {})[int(row.get("block_id", 0))] = row
    rows = read_jsonl(variant_root / "simple_retrieval.jsonl")
    ranks: list[int] = []
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    eligible = 0
    no_evidence = 0
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        uid = str(row.get("user_id", ""))
        idx = int(row.get("qa_idx", 0))
        qa = qa_map.get((uid, idx), {})
        evidence = qa.get("evidence") or []
        if not evidence:
            no_evidence += 1
            continue
        eligible += 1
        gold_texts = [normalize_text(item.get("memory_content")) for item in evidence if item.get("memory_content")]
        ranking = ((row.get("rankings") or {}).get("content_event_topic_kw") or [])
        gold_rank = None
        for rank, bid in enumerate(ranking, start=1):
            box = boxes.get(uid, {}).get(int(bid))
            haystack = box_search_text(box or {})
            if any(text and text in haystack for text in gold_texts):
                gold_rank = rank
                break
        if gold_rank is not None:
            ranks.append(gold_rank)
            for k in hits:
                hits[k] += int(gold_rank <= k)
        by_difficulty[str(qa.get("difficulty", "unknown"))].append(gold_rank is not None and gold_rank <= 20)
    return {
        "queries": len(rows),
        "queries_with_evidence": eligible,
        "queries_without_evidence": no_evidence,
        "queries_with_retrievable_gold_text": len(ranks),
        "hit_rate": {f"hit@{k}": hits[k] / eligible if eligible else 0.0 for k in hits},
        "mrr": sum(1.0 / x for x in ranks) / eligible if eligible else 0.0,
        "mean_gold_rank": statistics.mean(ranks) if ranks else None,
        "rank_count": len(ranks),
        "difficulty_hit_at_20": {k: sum(v) / len(v) if v else 0.0 for k, v in sorted(by_difficulty.items())},
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Phase 2 自动实验汇总报告", "", "本报告由独立 post-processor 从已落盘 artifacts 生成；不修改 construction、retrieval 或 QA 原始结果。", ""]
    lines += ["## LME B0/B1/B2 QA", "", "| Variant | Questions | Mean repeated accuracy | QA retries | Judge retries |", "|---|---:|---:|---:|---:|"]
    for v, row in report.get("lme", {}).get("qa", {}).items():
        lines.append(f"| {v} | {row['questions']} | {row['mean_repeat_accuracy']:.4f} | {row['qa_retries']} | {row['judge_retries']} |")
    lines += ["", "## LME construction", "", "| Variant | Input | Output | Total | Calls |", "|---|---:|---:|---:|---:|"]
    for v, row in report.get("lme", {}).get("construction", {}).items():
        c = row["construction"]
        lines.append(f"| {v} | {c['input_tokens']:,} | {c['output_tokens']:,} | {c['total_tokens']:,} | {c['calls']:,} |")
    lines += ["", "## HaluMem B0/B1/B2/B3", "", "| Variant | Input | Output | Total | Calls | Hit@20 (有 evidence) | Mean rank | QA mean accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v, row in report.get("halumem", {}).get("variants", {}).items():
        c = row["construction"]; r = row["retrieval"]; q = row["qa"]
        lines.append(f"| {v} | {c['input_tokens']:,} | {c['output_tokens']:,} | {c['total_tokens']:,} | {c['calls']:,} | {r.get('hit_rate', {}).get('hit@20', 0.0):.4f} | {r.get('mean_gold_rank') if r.get('mean_gold_rank') is not None else 'n/a'} | {q.get('mean_accuracy', 0.0):.4f} |")
    lines += ["", "## 口径说明", "", "- LME QA 使用 Top-10 context；HaluMem QA 使用 Topic Loom 当前 retrieval 的 Top-20。", "- QA 重复只重复 answer/judge，不重复 construction。", "- HaluMem retrieval 指标只对有 evidence 的问题计算；无 evidence 数量单独保留。", "- provider usage 缺失时不会伪造 token。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--lme-root-base", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lme_roots = {
        "b0": args.lme_root_base / "full_50_seed42",
        "b1": args.lme_root_base / "temporal_followup_b1_50q_retrieval_v2",
        "b2": args.lme_root_base / "temporal_gate_b2_50q_retrieval",
    }
    report: dict[str, Any] = {
        "config": {"llm": "gpt-4o-mini", "embedding": "text-embedding-3-small", "lme_qa_topk": 10, "halumem_qa_topk": 20},
        "lme": {"construction": {}, "qa": lme_qa_summary(args.work_root / "lme_qa_repeats")},
        "halumem": {"variants": {}},
    }
    for variant, root in lme_roots.items():
        report["lme"]["construction"][variant] = construction_summary(root, variant, "lme")
    for variant in ("b0", "b1", "b2", "b3"):
        root = args.work_root / "variants" / f"halumem_{variant}"
        qa_summary = read_json(args.work_root / "qa" / variant / "qa_summary.json", {}) or {}
        qacc = float(qa_summary.get("mean_accuracy", 0.0) or 0.0)
        report["halumem"]["variants"][variant] = {
            "construction": construction_summary(root, variant, "halumem"),
            "retrieval": halumem_retrieval_summary(root, args.combined),
            "qa": {"mean_accuracy": qacc, **qa_summary},
        }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "PHASE2_RESULTS.json", report)
    (args.output / "PHASE2_REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "lme_variants": list(report["lme"]["construction"]), "halumem_variants": list(report["halumem"]["variants"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
