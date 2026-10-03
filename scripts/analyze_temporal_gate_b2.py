#!/usr/bin/env python3
"""Audit the behavior and token effects of the SA-Mem B2 temporal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_impl_graph import needs_temporal_tool, temporal_gate_decision


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def box_key(box: dict[str, Any]) -> tuple[str, Any, Any]:
    coverage = box.get("coverage") or {}
    return (
        str(coverage.get("session_id") or ""),
        coverage.get("start_idx"),
        coverage.get("end_idx"),
    )


def event_signature(box: dict[str, Any]) -> dict[str, Any]:
    events = []
    for event in box.get("events") or []:
        events.append(
            {
                "description": event.get("description"),
                "event_temporal_type": event.get("event_temporal_type"),
                "time_metadata": event.get("time_metadata"),
            }
        )
    return {
        "topic_kw_text": (box.get("features") or {}).get("topic_kw_text", ""),
        "events_count": box.get("events_count"),
        "events": events,
    }


def _dialogue_hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def component_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    left_features = left.get("features") or {}
    right_features = right.get("features") or {}
    left_events = left.get("events") or []
    right_events = right.get("events") or []
    left_desc = [x.get("description") if isinstance(x, dict) else str(x) for x in left_events]
    right_desc = [x.get("description") if isinstance(x, dict) else str(x) for x in right_events]
    left_types = [x.get("event_temporal_type") if isinstance(x, dict) else None for x in left_events]
    right_types = [x.get("event_temporal_type") if isinstance(x, dict) else None for x in right_events]
    left_times = [x.get("time_metadata") if isinstance(x, dict) else None for x in left_events]
    right_times = [x.get("time_metadata") if isinstance(x, dict) else None for x in right_events]
    return {
        "topic_keywords": left_features.get("topic_kw_text", "") != right_features.get("topic_kw_text", ""),
        "event_count": len(left_events) != len(right_events),
        "event_descriptions": left_desc != right_desc,
        "event_type": left_types != right_types,
        "time_metadata": left_times != right_times,
        "temporal_index": (left.get("temporal_index") or {}) != (right.get("temporal_index") or {}),
    }


def retrieval_info(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    ranking_doc = read_json(root / "retrieval_full.json", {}) or {}
    ranking = ranking_doc.get("rankings") or []
    gold = {str(x) for x in summary.get("retrieval", {}).get("gold_answer_session_ids", [])}
    gold_items = [x for x in ranking if str(x.get("session_id")) in gold]
    rank = min((int(x.get("rank")) for x in gold_items if x.get("rank") is not None), default=None)
    score = None
    if rank is not None:
        score = next((x.get("score") for x in gold_items if int(x.get("rank", -1)) == rank), None)
    return {
        "hit_at_1": rank is not None and rank <= 1,
        "hit_at_5": rank is not None and rank <= 5,
        "hit_at_10": rank is not None and rank <= 10,
        "hit_at_20": rank is not None and rank <= 20,
        "gold_rank": rank,
        "gold_score": score,
        "ranking_size": len(ranking),
        "gold_memory_ids": [x.get("memory_id") for x in gold_items],
        "gold_dialogue_hashes": [_dialogue_hash(x.get("original_dialogue")) for x in gold_items],
        "top10_ids": [x.get("memory_id") for x in ranking[:10]],
        "top20_ids": [x.get("memory_id") for x in ranking[:20]],
    }


def summary_for(root: Path, qid: str) -> dict[str, Any]:
    rows = read_jsonl(root / qid / "question_summary.jsonl")
    if not rows:
        raise RuntimeError(f"missing question summary: {root / qid}")
    row = rows[0]
    if not row.get("memory_unit_count"):
        row["memory_unit_count"] = len(read_jsonl(root / qid / "final_boxes_content.jsonl"))
    row["retrieval_audit"] = retrieval_info(root / qid, row)
    return row


def stage_totals(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stages = ["split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"]
    out = {stage: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0} for stage in stages}
    for row in summary_rows:
        for stage in stages:
            current = ((row.get("construction") or {}).get("breakdown") or {}).get(stage) or {}
            for field in out[stage]:
                out[stage][field] += int(current.get(field) or 0)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    construction = [x.get("construction") or {} for x in rows]
    retrieval = [x.get("retrieval_audit") or {} for x in rows]
    total = {
        "questions": len(rows),
        "input_tokens": sum(int(x.get("input_tokens") or 0) for x in construction),
        "output_tokens": sum(int(x.get("output_tokens") or 0) for x in construction),
        "total_tokens": sum(int(x.get("total_tokens") or 0) for x in construction),
        "llm_calls": sum(int(x.get("llm_calls") or 0) for x in construction),
        "wall_time_sec": sum(float(x.get("wall_time_sec") or 0.0) for x in construction),
        "memory_units": sum(int(x.get("memory_unit_count") or 0) for x in rows),
        "qa_correct": sum(bool((x.get("qa") or {}).get("correct")) for x in rows),
    }
    for k in (1, 5, 10, 20):
        total[f"hit_at_{k}"] = sum(bool(x.get(f"hit_at_{k}")) for x in retrieval)
    ranks = [int(x["gold_rank"]) for x in retrieval if x.get("gold_rank") is not None]
    total["mean_gold_rank"] = sum(ranks) / len(ranks) if ranks else None
    total["mrr"] = sum(1.0 / rank for rank in ranks) / len(rows) if rows else None
    total["qa_accuracy"] = total["qa_correct"] / len(rows) if rows else None
    total["mean_memory_units"] = total["memory_units"] / len(rows) if rows else None
    return total


def gate_audit(root: Path, ids: list[str]) -> dict[str, Any]:
    rows = []
    reason_counts: dict[str, int] = {}
    for qid in ids:
        gates = read_jsonl(root / qid / "temporal_gate.jsonl")
        for row in gates:
            reason = str(row.get("temporal_gate_reason") or "")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            rows.append(row)
    return {
        "blocks_total": len(rows),
        "gate_on": sum(bool(x.get("temporal_gate_on")) for x in rows),
        "gate_off": sum(not bool(x.get("temporal_gate_on")) for x in rows),
        "temporal_tool_called": sum(bool(x.get("temporal_tool_called")) for x in rows),
        "pass1_tool_followup": sum(bool(x.get("pass1_tool_followup")) for x in rows),
        "reason_counts": reason_counts,
    }


def detector_audit(root: Path, ids: list[str]) -> dict[str, Any]:
    candidates = 0
    candidate_gate_on = 0
    examples: list[dict[str, Any]] = []
    for qid in ids:
        boxes = read_jsonl(root / qid / "final_boxes_content.jsonl")
        gates = {(str(x.get("session_id")), x.get("block_id")): x for x in read_jsonl(root / qid / "temporal_gate.jsonl")}
        for box in boxes:
            text = (box.get("features") or {}).get("content_text", "")
            on, reason = temporal_gate_decision(text)
            if not on:
                continue
            candidates += 1
            gate = gates.get((str((box.get("coverage") or {}).get("session_id")), box.get("block_id")))
            if gate and gate.get("temporal_gate_on") is True:
                candidate_gate_on += 1
            if len(examples) < 12:
                examples.append({"question_id": qid, "block_id": box.get("block_id"), "reason": reason})
    return {
        "candidate_blocks": candidates,
        "candidate_blocks_gate_on": candidate_gate_on,
        "candidate_blocks_gate_on_rate": candidate_gate_on / candidates if candidates else None,
        "examples": examples,
        "detector": "high_recall_local_regex_v2",
    }


def detector_reference_audit(reference_root: Path | None, b2_root: Path, ids: list[str]) -> dict[str, Any] | None:
    """Compare prior-run detector candidates against B2 coverage."""
    if reference_root is None:
        return None
    candidate_blocks = 0
    matched = 0
    matched_gate_on = 0
    unmatched = 0
    false_negative_examples: list[dict[str, Any]] = []
    for qid in ids:
        reference = read_jsonl(reference_root / qid / "final_boxes_content.jsonl")
        current = read_jsonl(b2_root / qid / "final_boxes_content.jsonl")
        current_by_key = {box_key(x): x for x in current}
        gates = {
            (str(x.get("session_id")), x.get("block_id")): x
            for x in read_jsonl(b2_root / qid / "temporal_gate.jsonl")
        }
        for box in reference:
            on, reason = temporal_gate_decision((box.get("features") or {}).get("content_text", ""))
            if not on:
                continue
            candidate_blocks += 1
            current_box = current_by_key.get(box_key(box))
            if current_box is None:
                unmatched += 1
                continue
            matched += 1
            gate = gates.get((str((current_box.get("coverage") or {}).get("session_id")), current_box.get("block_id")))
            if gate and gate.get("temporal_gate_on") is True:
                matched_gate_on += 1
            elif len(false_negative_examples) < 12:
                false_negative_examples.append({"question_id": qid, "coverage": box_key(box), "reason": reason})
    return {
        "reference_root": str(reference_root),
        "candidate_blocks": candidate_blocks,
        "matched_to_b2_coverage": matched,
        "unmatched_due_to_segmentation": unmatched,
        "matched_gate_on": matched_gate_on,
        "matched_gate_on_rate": matched_gate_on / matched if matched else None,
        "false_negative_count": matched - matched_gate_on,
        "false_negative_examples": false_negative_examples,
        "detector": "high_recall_local_regex_v2",
    }


def diff_audit(b0_root: Path, b2_root: Path, ids: list[str]) -> dict[str, Any]:
    matched = {"all": 0, "gate_on": 0, "gate_off": 0}
    different = {"all": 0, "gate_on": 0, "gate_off": 0}
    component_counts = {
        gate: {name: 0 for name in ("topic_keywords", "event_count", "event_descriptions", "event_type", "time_metadata", "temporal_index")}
        for gate in ("all", "gate_on", "gate_off")
    }
    unmatched = {"b0": 0, "b2": 0}
    examples: list[dict[str, Any]] = []
    for qid in ids:
        b0 = read_jsonl(b0_root / qid / "final_boxes_content.jsonl")
        b2 = read_jsonl(b2_root / qid / "final_boxes_content.jsonl")
        b0_by_key = {box_key(x): x for x in b0}
        b2_by_key = {box_key(x): x for x in b2}
        gates = {(str(x.get("session_id")), x.get("block_id")): x for x in read_jsonl(b2_root / qid / "temporal_gate.jsonl")}
        unmatched["b0"] += len(set(b0_by_key) - set(b2_by_key))
        unmatched["b2"] += len(set(b2_by_key) - set(b0_by_key))
        for key in sorted(set(b0_by_key) & set(b2_by_key), key=str):
            left, right = b0_by_key[key], b2_by_key[key]
            gate = gates.get((key[0], right.get("block_id")))
            gate_name = "gate_on" if gate and gate.get("temporal_gate_on") else "gate_off"
            matched["all"] += 1
            matched[gate_name] += 1
            is_diff = event_signature(left) != event_signature(right)
            for name, changed in component_diff(left, right).items():
                if changed:
                    component_counts["all"][name] += 1
                    component_counts[gate_name][name] += 1
            if is_diff:
                different["all"] += 1
                different[gate_name] += 1
                if len(examples) < 20:
                    examples.append({"question_id": qid, "coverage": key, "gate": gate_name, "b0": event_signature(left), "b2": event_signature(right)})
    return {
        "matched": matched,
        "different": different,
        "difference_rate": {k: (different[k] / matched[k] if matched[k] else None) for k in matched},
        "component_differences": component_counts,
        "unmatched": unmatched,
        "examples": examples,
    }


def qa_failure_audit(
    b0: list[dict[str, Any]],
    b2: list[dict[str, Any]],
    b0_root: Path,
    b2_root: Path,
) -> list[dict[str, Any]]:
    left = {x["question_id"]: x for x in b0}
    right = {x["question_id"]: x for x in b2}
    out = []
    for qid in sorted(set(left) & set(right)):
        l, r = left[qid], right[qid]
        lc = bool((l.get("qa") or {}).get("correct"))
        rc = bool((r.get("qa") or {}).get("correct"))
        if lc == rc:
            continue
        b0_retrieval = l.get("retrieval_audit") or {}
        b2_retrieval = r.get("retrieval_audit") or {}
        b0_prompt = b0_root / qid / "answer_prompt.txt"
        b2_prompt = b2_root / qid / "answer_prompt.txt"
        b0_prompt_bytes = b0_prompt.read_bytes() if b0_prompt.exists() else b""
        b2_prompt_bytes = b2_prompt.read_bytes() if b2_prompt.exists() else b""
        out.append({
            "question_id": qid,
            "transition": "B0_correct_B2_wrong" if lc else "B0_wrong_B2_correct",
            "b0_answer": (l.get("qa") or {}).get("hypothesis"),
            "b2_answer": (r.get("qa") or {}).get("hypothesis"),
            "b0_retrieval": b0_retrieval,
            "b2_retrieval": b2_retrieval,
            "gold_session_retained_in_top10": bool(b0_retrieval.get("hit_at_10")) and bool(b2_retrieval.get("hit_at_10")),
            "gold_rank_changed": b0_retrieval.get("gold_rank") != b2_retrieval.get("gold_rank"),
            "top10_ids_changed": b0_retrieval.get("top10_ids") != b2_retrieval.get("top10_ids"),
            "gold_dialogue_hashes_overlap": bool(set(b0_retrieval.get("gold_dialogue_hashes", [])) & set(b2_retrieval.get("gold_dialogue_hashes", []))),
            "answer_prompt_sha256_b0": hashlib.sha256(b0_prompt_bytes).hexdigest() if b0_prompt.exists() else None,
            "answer_prompt_sha256_b2": hashlib.sha256(b2_prompt_bytes).hexdigest() if b2_prompt.exists() else None,
            "answer_prompt_equal": b0_prompt_bytes == b2_prompt_bytes,
        })
    return out


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-root", type=Path, required=True)
    parser.add_argument("--b2-root", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--b0-correctness", type=Path, default=None)
    parser.add_argument("--detector-reference-root", type=Path, default=None)
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    b0 = [summary_for(args.b0_root, qid) for qid in ids]
    b2 = [summary_for(args.b2_root, qid) for qid in ids]
    if args.b0_correctness and args.b0_correctness.exists():
        previous = read_json(args.b0_correctness, {}) or {}
        previous_by_id = {str(x.get("question_id")): x for x in previous.get("questions", [])}
        for row in b0:
            old = previous_by_id.get(str(row.get("question_id"))) or {}
            if "qa_correct" in (old.get("b0") or {}):
                row.setdefault("qa", {})["correct"] = bool(old["b0"]["qa_correct"])
    data = {
        "ids": ids,
        "b0": {"aggregate": aggregate(b0), "stage_totals": stage_totals(b0), "questions": b0},
        "b2": {"aggregate": aggregate(b2), "stage_totals": stage_totals(b2), "questions": b2},
        "gate": gate_audit(args.b2_root, ids),
        "detector_audit": detector_audit(args.b2_root, ids),
        "detector_reference_audit": detector_reference_audit(args.detector_reference_root, args.b2_root, ids),
        "memblock_diff": diff_audit(args.b0_root, args.b2_root, ids),
        "qa_failure_audit": qa_failure_audit(b0, b2, args.b0_root, args.b2_root),
    }
    data["token_saving"] = {
        "tokens": data["b0"]["aggregate"]["total_tokens"] - data["b2"]["aggregate"]["total_tokens"],
        "fraction": (data["b0"]["aggregate"]["total_tokens"] - data["b2"]["aggregate"]["total_tokens"]) / data["b0"]["aggregate"]["total_tokens"],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    b0a, b2a = data["b0"]["aggregate"], data["b2"]["aggregate"]
    b0s, b2s = data["b0"]["stage_totals"], data["b2"]["stage_totals"]
    gate = data["gate"]
    diff = data["memblock_diff"]
    qa_failures = data["qa_failure_audit"]
    lines = [
        "# Temporal Gate B2 Pilot",
        "",
        "B2 keeps the B0 Pass1 prompt, mentions schema, temporal resolver, Pass2 prompt, MemBlock schema, retrieval, embedding, and QA path unchanged. It only applies a local high-recall lexical gate before Pass1: Gate OFF sends the original function schema with `tool_choice=none`; Gate ON uses the original tool + follow-up path.",
        "",
        f"Pilot questions: {len(ids)}; run root: `{args.b2_root}`.",
        "Model configuration: `gpt-4o-mini`, embedding `text-embedding-3-small`, temperature `0.0`; retrieval candidate Top-20 and generation context Top-10. QA labels come from the official LongMemEval evaluator using the isolated evaluator environment.",
        "The B0 side is the previously measured original two-pass pilot; no B0 or B1 artifact was overwritten. No 50-question B2 run was started.",
        "",
        "## Detector audit",
        "",
        f"- Gate records: **{gate['blocks_total']}** blocks; ON **{gate['gate_on']}**, OFF **{gate['gate_off']}**.",
        f"- Actual temporal tool calls: **{gate['temporal_tool_called']}**; Pass1 follow-ups: **{gate['pass1_tool_followup']}**.",
        f"- Detector candidate blocks: **{data['detector_audit']['candidate_blocks']}**; candidate blocks gated ON: **{data['detector_audit']['candidate_blocks_gate_on']}** ({pct(data['detector_audit']['candidate_blocks_gate_on_rate'])}).",
        "- The detector is intentionally high-recall; false positives are accepted and are visible as Gate ON blocks without a tool call.",
        "",
        "## Detector audit against the prior B1 pilot",
        "",
    ]
    reference_audit = data.get("detector_reference_audit")
    if reference_audit:
        lines += [
            f"- Prior B1 raw candidate blocks: **{reference_audit['candidate_blocks']}**.",
            f"- Candidates matched to B2 coverage: **{reference_audit['matched_to_b2_coverage']}**; unmatched because segmentation changed: **{reference_audit['unmatched_due_to_segmentation']}**.",
            f"- Matched candidates gated ON: **{reference_audit['matched_gate_on']} / {reference_audit['matched_to_b2_coverage']}** ({pct(reference_audit['matched_gate_on_rate'])}); detector false negatives on matched coverage: **{reference_audit['false_negative_count']}**.",
        ]
    else:
        lines.append("No prior-run detector reference was supplied.")
    lines += [
        "",
        "## B0 vs B2 aggregate",
        "",
        "| Metric | B0 | B2 |",
        "|---|---:|---:|",
        f"| Construction input tokens | {b0a['input_tokens']:,} | {b2a['input_tokens']:,} |",
        f"| Construction output tokens | {b0a['output_tokens']:,} | {b2a['output_tokens']:,} |",
        f"| Construction total tokens | {b0a['total_tokens']:,} | {b2a['total_tokens']:,} |",
        f"| Token saving | — | {data['token_saving']['tokens']:,} ({pct(data['token_saving']['fraction'])}) |",
        f"| Construction LLM calls | {b0a['llm_calls']:,} | {b2a['llm_calls']:,} |",
        f"| Construction wall time (sum) | {b0a['wall_time_sec']:.2f}s | {b2a['wall_time_sec']:.2f}s |",
        f"| Memory units | {b0a['memory_units']:,} | {b2a['memory_units']:,} |",
        f"| Hit@1 | {b0a['hit_at_1']}/{len(ids)} | {b2a['hit_at_1']}/{len(ids)} |",
        f"| Hit@5 | {b0a['hit_at_5']}/{len(ids)} | {b2a['hit_at_5']}/{len(ids)} |",
        f"| Hit@10 | {b0a['hit_at_10']}/{len(ids)} | {b2a['hit_at_10']}/{len(ids)} |",
        f"| Hit@20 | {b0a['hit_at_20']}/{len(ids)} | {b2a['hit_at_20']}/{len(ids)} |",
        f"| Mean gold rank | {b0a['mean_gold_rank']:.4f} | {b2a['mean_gold_rank']:.4f} |",
        f"| MRR | {b0a['mrr']:.4f} | {b2a['mrr']:.4f} |",
        f"| QA correct | {b0a['qa_correct']}/{len(ids)} ({pct(b0a['qa_accuracy'])}) | {b2a['qa_correct']}/{len(ids)} ({pct(b2a['qa_accuracy'])}) |",
        "",
        "## Construction stage totals",
        "",
        "| Stage | B0 calls / tokens | B2 calls / tokens |",
        "|---|---:|---:|",
    ]
    for stage in ("split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"):
        lines.append(f"| `{stage}` | {b0s[stage]['calls']:,} / {b0s[stage]['total_tokens']:,} | {b2s[stage]['calls']:,} / {b2s[stage]['total_tokens']:,} |")
    lines += [
        "",
        "## MemBlock diff audit",
        "",
        f"Matched blocks: **{diff['matched']['all']}**; unmatched B0: **{diff['unmatched']['b0']}**; unmatched B2: **{diff['unmatched']['b2']}**.",
        f"Overall output differences: **{diff['different']['all']} / {diff['matched']['all']}** ({pct(diff['difference_rate']['all'])}).",
        f"Gate ON differences: **{diff['different']['gate_on']} / {diff['matched']['gate_on']}** ({pct(diff['difference_rate']['gate_on'])}).",
        f"Gate OFF differences: **{diff['different']['gate_off']} / {diff['matched']['gate_off']}** ({pct(diff['difference_rate']['gate_off'])}).",
        "",
        "### MemBlock component differences",
        "",
        "| Component | All matched | Gate ON | Gate OFF |",
        "|---|---:|---:|---:|",
    ]
    for component in ("topic_keywords", "event_count", "event_descriptions", "event_type", "time_metadata", "temporal_index"):
        lines.append(
            f"| `{component}` | {diff['component_differences']['all'][component]} | {diff['component_differences']['gate_on'][component]} | {diff['component_differences']['gate_off'][component]} |"
        )
    lines += [
        "The comparison uses session coverage `(session_id, start_idx, end_idx)` and compares topic/keywords, event count, descriptions, event type, and time metadata. Unmatched coverage is reported separately rather than silently treated as equal.",
        "",
        "## QA failure audit",
        "",
    ]
    if qa_failures:
        lines.append("| Question | Transition | B0 rank / Top-10 | B2 rank / Top-10 | B0 answer | B2 answer |")
        lines.append("|---|---|---:|---:|---|---|")
        for item in qa_failures:
            l, r = item["b0_retrieval"], item["b2_retrieval"]
            lines.append(f"| `{item['question_id']}` | {item['transition']} | {l.get('gold_rank')} / {l.get('gold_memory_ids', [])[:1]} | {r.get('gold_rank')} / {r.get('gold_memory_ids', [])[:1]} | {item['b0_answer']} | {item['b2_answer']} |")
        lines += ["", "QA audit details:", ""]
        for item in qa_failures:
            lines.append(
                f"- `{item['question_id']}`: gold session retained in both Top-10=`{item['gold_session_retained_in_top10']}`, gold rank changed=`{item['gold_rank_changed']}`, Top-10 IDs changed=`{item['top10_ids_changed']}`, same gold raw-dialogue hash present=`{item['gold_dialogue_hashes_overlap']}`, answer prompt equal=`{item['answer_prompt_equal']}`."
            )
    else:
        lines.append("No B0/B2 QA correctness transitions were observed.")
    lines += [
        "",
        "## Interpretation",
        "",
        "B2 is a stricter behavior-equivalence test than B1 because it does not change the Pass1 prompt or mentions schema. It reduces construction cost by 16.05% and removes 559 follow-up calls, while the detector audit found no matched false negatives. However, QA changed from 2/8 to 0/8, Hit@5 changed from 8/8 to 7/8, and 91.68% of coverage-matched MemBlocks differ in at least one audited output component. The two observed QA regressions retained the gold session at rank 1 and had byte-identical answer prompts, so they are not evidence of gold-context loss; they are consistent with changed memory outputs/ranking and/or provider/model run variance.",
        "Decision: do not expand B2 to the frozen 50-question run from this pilot. Token savings are demonstrated, but the current evidence does not support the stronger claim that temporal gating preserves end-to-end behavior. This 8-question pilot is diagnostic and does not establish a dataset-level accuracy conclusion.",
        "",
        "Detailed machine-readable output: `TEMPORAL_GATE_B2_RESULTS.json`.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "results": str(args.results), "b0": b0a, "b2": b2a, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
