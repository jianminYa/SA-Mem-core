#!/usr/bin/env python3
"""Post-process, officially evaluate, and summarize the 50-question runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_50q import analyze_one
from common import json_dump, load_questions, read_jsonl


def numbers(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"total": 0, "mean": 0.0, "median": 0.0}
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def sum_stage(rows: list[dict[str, Any]], stage: str) -> dict[str, int]:
    fields = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        stage_row = ((row.get("construction") or {}).get("breakdown") or {}).get(stage) or {}
        for field in fields:
            fields[field] += int(stage_row.get(field, 0) or 0)
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "question_type", "history_sessions", "history_messages", "history_tokens",
        "membox_memory_units", "membox_construction_tokens", "membox_gold_rank", "membox_correct",
        "samem_memory_units", "samem_construction_tokens", "samem_gold_rank", "samem_correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_failures(rows: list[dict[str, Any]], system: str) -> dict[str, list[dict[str, Any]]]:
    out = {"correct_other_wrong": [], "wrong_other_correct": [], "both_wrong": [], "rank_gt_10": [], "rank_gt_20": [], "extraction_failure": [], "context_wrong": []}
    for row in rows:
        rank = row.get("gold_rank")
        correct = row.get("correct") is True
        payload = {
            "question_id": row["question_id"], "question_type": row["question_type"],
            "correct": correct, "gold_rank": rank, "construction_tokens": row.get("construction_tokens"),
        }
        if rank is None:
            out["extraction_failure"].append(payload)
        elif rank > 20:
            out["rank_gt_20"].append(payload)
        elif rank > 10:
            out["rank_gt_10"].append(payload)
        elif not correct:
            out["context_wrong"].append(payload)
    return out


def summarize_system(system: str, summaries: list[dict[str, Any]], root: Path, parallel_manifest: dict[str, Any]) -> dict[str, Any]:
    construction_total = [float((row.get("construction") or {}).get("total_tokens", 0) or 0) for row in summaries]
    construction_input = [float((row.get("construction") or {}).get("input_tokens", 0) or 0) for row in summaries]
    construction_output = [float((row.get("construction") or {}).get("output_tokens", 0) or 0) for row in summaries]
    calls = [float((row.get("construction") or {}).get("llm_calls", 0) or 0) for row in summaries]
    wall = [float((row.get("construction") or {}).get("wall_time_sec", 0) or 0) for row in summaries]
    history_tokens = [float((row.get("history") or {}).get("estimated_tokens", 0) or 0) for row in summaries]
    messages = [float((row.get("history") or {}).get("messages", 0) or 0) for row in summaries]
    units = [float(row.get("memory_unit_count", 0) or 0) for row in summaries]
    hits = {str(k): sum(bool((row.get("retrieval") or {}).get(f"evidence_hit_at_{k}")) for row in summaries) / len(summaries) for k in (1, 5, 10, 20)}
    mrr = [float((row.get("retrieval") or {}).get("mrr", 0.0) or 0.0) for row in summaries]
    stage_names = ["topic_continuity", "box_extraction"] if system == "membox" else ["split_check", "pass1_extract", "pass1_tool_followup", "pass2_classify"]
    stage_breakdown = {stage: sum_stage(summaries, stage) for stage in stage_names}
    permanent_failures = sum(int(json.loads((root / "questions" / row["question_id"] / "question_status.json").read_text()).get("permanently_failed_calls", 0) or 0) for row in summaries)
    retries = sum(int(json.loads((root / "questions" / row["question_id"] / "question_status.json").read_text()).get("retried_calls", 0) or 0) for row in summaries)
    return {
        "system": system,
        "questions": len(summaries),
        "history_tokens": numbers(history_tokens),
        "history_messages": numbers(messages),
        "memory_units": numbers(units),
        "construction_input_tokens": numbers(construction_input),
        "construction_output_tokens": numbers(construction_output),
        "construction_total_tokens": numbers(construction_total),
        "construction_calls": numbers(calls),
        "construction_wall_time_sec": numbers(wall),
        "construction_tokens_per_history_token": sum(construction_total) / sum(history_tokens) if sum(history_tokens) else 0.0,
        "construction_tokens_per_message": sum(construction_total) / sum(messages) if sum(messages) else 0.0,
        "construction_tokens_per_memory_unit": sum(construction_total) / sum(units) if sum(units) else 0.0,
        "llm_calls_per_question": statistics.mean(calls) if calls else 0.0,
        "evidence_hit": hits,
        "mrr": numbers(mrr),
        "stage_breakdown": stage_breakdown,
        "question_workers": parallel_manifest.get("question_workers"),
        "parallel_wall_time_sec": parallel_manifest.get("wall_time_sec"),
        "retried_calls": retries,
        "permanently_failed_calls": permanent_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--membox-root", type=Path, required=True)
    parser.add_argument("--samem-root", type=Path, required=True)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    questions = load_questions(args.data, args.ids)
    by_id = {str(row["question_id"]): row for row in questions}
    args.reports_root.mkdir(parents=True, exist_ok=True)
    per_system: dict[str, list[dict[str, Any]]] = {}
    roots = {"membox": args.membox_root, "samem": args.samem_root}

    for system, root in roots.items():
        statuses = []
        for item in questions:
            qid = str(item["question_id"])
            status_path = root / "questions" / qid / "question_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise RuntimeError(f"cannot aggregate incomplete {system} question {qid}")
            statuses.append(status)
        summaries = [analyze_one(system, root / "questions" / str(item["question_id"]), item, args.model) for item in questions]
        with (root / "question_summary.jsonl").open("w", encoding="utf-8") as handle:
            for row in summaries:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        hypotheses = []
        for item in questions:
            qid = str(item["question_id"])
            rows = read_jsonl(root / "questions" / qid / "hypotheses.jsonl")
            if len(rows) != 1:
                raise RuntimeError(f"missing hypothesis for {system}/{qid}")
            hypotheses.append(rows[0])
        with (root / "hypotheses.jsonl").open("w", encoding="utf-8") as handle:
            for row in hypotheses:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not (root / "hypotheses.jsonl.eval-results-gpt-4o").exists():
            subprocess.run([
                "conda", "run", "-n", "longmemeval-eval", "python", str(args.workspace_root / "scripts/evaluate_smoke.py"),
                "--run-root", str(root), "--hypotheses", str(root / "hypotheses.jsonl"),
                "--reference", str(args.data), "--evaluator-repo", str(args.evaluator_repo),
                "--env-file", str(args.env_file),
            ], cwd=args.workspace_root, check=True)
        eval_rows = {str(row["question_id"]): bool((row.get("autoeval_label") or {}).get("label")) for row in read_jsonl(root / "hypotheses.jsonl.eval-results-gpt-4o")}
        for summary in summaries:
            summary["qa"]["correct"] = eval_rows.get(str(summary["question_id"]))
            summary["diagnosis"]["failure_stage"] = (
                "extraction" if not summary["retrieval"].get("gold_memory_ids") else
                "retrieval ranking" if not summary["retrieval"].get("gold_in_top20") else
                "generation-context truncation" if not summary["retrieval"].get("gold_in_generation_top10") else
                "QA generation" if summary["qa"]["correct"] is not True else "none"
            )
            json_dump(root / "questions" / str(summary["question_id"]) / "answer.json", {
                "question_id": summary["question_id"],
                "hypothesis": summary["qa"].get("hypothesis", ""),
                "correct": summary["qa"]["correct"],
            })
        with (root / "question_summary.jsonl").open("w", encoding="utf-8") as handle:
            for row in summaries:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        parallel_manifest = json.loads((root / "parallel_manifest.json").read_text(encoding="utf-8"))
        per_system[system] = summaries
        per_system.setdefault(f"{system}_aggregate", summarize_system(system, summaries, root, parallel_manifest))

    mb_rows = {str(row["question_id"]): row for row in per_system["membox"]}
    sm_rows = {str(row["question_id"]): row for row in per_system["samem"]}
    csv_rows = []
    for item in questions:
        qid = str(item["question_id"])
        mb = mb_rows[qid]
        sm = sm_rows[qid]
        csv_rows.append({
            "question_id": qid, "question_type": item["question_type"],
            "history_sessions": item.get("haystack_sessions") and len(item["haystack_sessions"]) or 0,
            "history_messages": mb["history"]["messages"], "history_tokens": mb["history"]["estimated_tokens"],
            "membox_memory_units": mb.get("memory_unit_count"), "membox_construction_tokens": mb["construction"].get("total_tokens"),
            "membox_gold_rank": mb["retrieval"].get("gold_rank"), "membox_correct": mb["qa"].get("correct"),
            "samem_memory_units": sm.get("memory_unit_count"), "samem_construction_tokens": sm["construction"].get("total_tokens"),
            "samem_gold_rank": sm["retrieval"].get("gold_rank"), "samem_correct": sm["qa"].get("correct"),
        })
    write_csv(args.reports_root / "question_level_results.csv", csv_rows)

    failure_root = args.reports_root / "failure_cases"
    failure_root.mkdir(parents=True, exist_ok=True)
    failure_sets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    mb_outcomes = {qid: bool(row["qa"].get("correct")) for qid, row in mb_rows.items()}
    sm_outcomes = {qid: bool(row["qa"].get("correct")) for qid, row in sm_rows.items()}
    for qid in mb_rows:
        payload = csv_rows[next(i for i, row in enumerate(csv_rows) if row["question_id"] == qid)]
        if mb_outcomes[qid] and not sm_outcomes[qid]:
            failure_sets.setdefault("membox_correct_samem_wrong", []).append(payload)
        if sm_outcomes[qid] and not mb_outcomes[qid]:
            failure_sets.setdefault("samem_correct_membox_wrong", []).append(payload)
        if not mb_outcomes[qid] and not sm_outcomes[qid]:
            failure_sets.setdefault("both_wrong", []).append(payload)
    for system, rows in (("membox", mb_rows), ("samem", sm_rows)):
        buckets = {"gold_extraction_failure": [], "gold_extracted_rank_gt_10": [], "gold_extracted_rank_gt_20": [], "gold_in_context_but_qa_wrong": []}
        for qid, row in rows.items():
            rank = row["retrieval"].get("gold_rank")
            if rank is None:
                buckets["gold_extraction_failure"].append(csv_rows[next(i for i, x in enumerate(csv_rows) if x["question_id"] == qid)])
            elif rank > 20:
                buckets["gold_extracted_rank_gt_20"].append(csv_rows[next(i for i, x in enumerate(csv_rows) if x["question_id"] == qid)])
            elif rank > 10:
                buckets["gold_extracted_rank_gt_10"].append(csv_rows[next(i for i, x in enumerate(csv_rows) if x["question_id"] == qid)])
            elif not row["qa"].get("correct"):
                buckets["gold_in_context_but_qa_wrong"].append(csv_rows[next(i for i, x in enumerate(csv_rows) if x["question_id"] == qid)])
        for name, values in buckets.items():
            failure_sets[f"{system}_{name}"] = values
    for name, values in failure_sets.items():
        (failure_root / f"{name}.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    aggregate = {
        "dataset": str(args.data),
        "question_count": len(questions),
        "model": args.model,
        "embedding_model": "text-embedding-3-small",
        "membox": per_system["membox_aggregate"],
        "samem": per_system["samem_aggregate"],
        "token_ratio_samem_over_membox": per_system["samem_aggregate"]["construction_total_tokens"]["total"] / per_system["membox_aggregate"]["construction_total_tokens"]["total"],
        "question_type_counts": dict(sorted({t: sum(item["question_type"] == t for item in questions) for t in {item["question_type"] for item in questions}}.items())),
        "failure_case_counts": {key: len(value) for key, value in failure_sets.items()},
    }
    json_dump(args.reports_root / "LONGMEMEVAL_50Q_BASELINE_RESULTS.json", aggregate)
    md = [
        "# LongMemEval-S 50-Question Baseline Report",
        "",
        "This report covers the frozen seed-42 50-question subset. No one-pass extraction, filtering, retrieval optimization, or embedding optimization was used.",
        "",
        "## QA results",
        "",
        "| System | Accuracy | Questions |",
        "|---|---:|---:|",
        f"| MemBox | {float(json.loads((args.membox_root / 'qa_results.json').read_text()).get('accuracy', 0.0)):.4f} | 50 |",
        f"| SA-Mem | {float(json.loads((args.samem_root / 'qa_results.json').read_text()).get('accuracy', 0.0)):.4f} | 50 |",
        "",
        "Accuracy by question type is in each aggregate `qa_results.json` and the machine-readable results file.",
        "",
        "## Construction cost",
        "",
        "| Metric | MemBox total | MemBox mean | MemBox median | SA-Mem total | SA-Mem mean | SA-Mem median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [("Input tokens", "construction_input_tokens"), ("Output tokens", "construction_output_tokens"), ("Total tokens", "construction_total_tokens"), ("LLM calls", "construction_calls"), ("Memory units", "memory_units")]:
        mb = per_system["membox_aggregate"][key]
        sm = per_system["samem_aggregate"][key]
        md.append(f"| {label} | {mb['total']:.0f} | {mb['mean']:.2f} | {mb['median']:.2f} | {sm['total']:.0f} | {sm['mean']:.2f} | {sm['median']:.2f} |")
    md += [
        "",
        f"SA-Mem / MemBox construction token ratio: **{aggregate['token_ratio_samem_over_membox']:.4f}x**.",
        "",
        "### MemBox stages",
        "",
        "| Stage | Calls | Input | Output | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, row in per_system["membox_aggregate"]["stage_breakdown"].items():
        md.append(f"| `{stage}` | {row['calls']} | {row['input_tokens']} | {row['output_tokens']} | {row['total_tokens']} |")
    md += ["", "### SA-Mem stages", "", "| Stage | Calls | Input | Output | Total |", "|---|---:|---:|---:|---:|"]
    for stage, row in per_system["samem_aggregate"]["stage_breakdown"].items():
        md.append(f"| `{stage}` | {row['calls']} | {row['input_tokens']} | {row['output_tokens']} | {row['total_tokens']} |")
    md += [
        "",
        "## Retrieval evidence",
        "",
        "| System | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| MemBox | {per_system['membox_aggregate']['evidence_hit']['1']:.4f} | {per_system['membox_aggregate']['evidence_hit']['5']:.4f} | {per_system['membox_aggregate']['evidence_hit']['10']:.4f} | {per_system['membox_aggregate']['evidence_hit']['20']:.4f} | {per_system['membox_aggregate']['mrr']['mean']:.4f} |",
        f"| SA-Mem | {per_system['samem_aggregate']['evidence_hit']['1']:.4f} | {per_system['samem_aggregate']['evidence_hit']['5']:.4f} | {per_system['samem_aggregate']['evidence_hit']['10']:.4f} | {per_system['samem_aggregate']['evidence_hit']['20']:.4f} | {per_system['samem_aggregate']['mrr']['mean']:.4f} |",
        "",
        "## Normalized construction metrics",
        "",
        "| Metric | MemBox | SA-Mem |",
        "|---|---:|---:|",
        f"| Tokens / history token | {per_system['membox_aggregate']['construction_tokens_per_history_token']:.6f} | {per_system['samem_aggregate']['construction_tokens_per_history_token']:.6f} |",
        f"| Tokens / message | {per_system['membox_aggregate']['construction_tokens_per_message']:.2f} | {per_system['samem_aggregate']['construction_tokens_per_message']:.2f} |",
        f"| Tokens / memory unit | {per_system['membox_aggregate']['construction_tokens_per_memory_unit']:.2f} | {per_system['samem_aggregate']['construction_tokens_per_memory_unit']:.2f} |",
        f"| LLM calls / question | {per_system['membox_aggregate']['llm_calls_per_question']:.2f} | {per_system['samem_aggregate']['llm_calls_per_question']:.2f} |",
        "",
        "## Parallel execution and failures",
        "",
        f"- Question workers: MemBox `{per_system['membox_aggregate']['question_workers']}`, SA-Mem `{per_system['samem_aggregate']['question_workers']}`.",
        f"- Parallel wall time: MemBox `{per_system['membox_aggregate']['parallel_wall_time_sec']}` seconds; SA-Mem `{per_system['samem_aggregate']['parallel_wall_time_sec']}` seconds.",
        f"- Retried calls: MemBox `{per_system['membox_aggregate']['retried_calls']}`; SA-Mem `{per_system['samem_aggregate']['retried_calls']}`.",
        f"- Permanently failed construction calls: MemBox `{per_system['membox_aggregate']['permanently_failed_calls']}`; SA-Mem `{per_system['samem_aggregate']['permanently_failed_calls']}`.",
        "- Failure-case JSON files are under `reports/failure_cases/`; question-level raw metrics are in `reports/question_level_results.csv`.",
        "",
        "## Reproducibility artifacts",
        "",
        "- Per-question runs: `runs/membox/full_50_seed42/questions/<question_id>/` and `runs/samem_2p/full_50_seed42/questions/<question_id>/`.",
        "- Each completed question has `construction_calls.jsonl`, `memories.jsonl`, `retrieval_full.json`, `retrieval_top20.json`, `generation_top10.json`, `answer_prompt.txt`, and `question_status.json`.",
        "- Full dataset and embedding caches remain local and are not committed.",
    ]
    (args.reports_root / "LONGMEMEVAL_50Q_BASELINE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.reports_root / "LONGMEMEVAL_50Q_BASELINE_REPORT.md"), "results": aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
