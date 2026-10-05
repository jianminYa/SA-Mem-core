#!/usr/bin/env python3
"""Run repeatable LongMemEval QA/evaluator calls over frozen retrieval artifacts.

Construction and retrieval artifacts are read-only.  Each system/repeat/question
gets its own directory so the runner can resume without touching B0/B1/B2 data.
At most ``--workers`` question tasks are active; each task performs one QA call
and one judge call sequentially.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from generate_lme import build_memories_string, get_anscheck_prompt, load_boxes, true_or_false


SYSTEM_ROOTS = {
    "b0": "/workspace/SA-mem/longmemeval-baselines/runs/samem_2p/full_50_seed42",
    "b1": "/workspace/SA-mem/longmemeval-baselines/runs/samem_2p/temporal_followup_b1_50q_retrieval_v2",
    "b2": "/workspace/SA-mem/longmemeval-baselines/runs/samem_2p/temporal_gate_b2_50q_retrieval",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        tmp = Path(handle.name)
    tmp.replace(path)


def call_with_retry(client: OpenAI, model: str, messages: list[dict[str, str]], retries: int) -> tuple[str, dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=0.8,
                max_tokens=2000,
            )
            usage = getattr(response, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            return (response.choices[0].message.content or "", usage_dict, attempt)
        except Exception as exc:  # provider-specific exception classes vary
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(30.0, 0.75 * (2**attempt) + random.random() * 0.5))
    raise RuntimeError(f"API call failed after retries: {type(last_error).__name__}") from last_error


def load_retrieval(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"empty retrieval file: {path}")


def run_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    system = task["system"]
    repeat = task["repeat"]
    qid = task["question_id"]
    out_dir = Path(args.output_root) / system / f"repeat_{repeat:02d}" / "questions" / qid
    answer_path = out_dir / "answer.json"
    if answer_path.exists() and not args.overwrite:
        return json.loads(answer_path.read_text(encoding="utf-8"))

    question_dir = Path(args.system_roots[system]) / "questions" / qid
    retrieval_path = question_dir / "retrieval.jsonl"
    boxes_path = question_dir / "final_boxes_content.jsonl"
    retrieval = load_retrieval(retrieval_path)
    boxes = load_boxes(str(boxes_path))
    user_id = str(retrieval.get("user_id", ""))
    ranking = (retrieval.get("rankings") or {}).get("content_event_topic_kw") or []
    memories = build_memories_string(boxes.get(user_id, {}), ranking, args.topk)
    if not memories:
        raise RuntimeError(f"empty QA context for {system}/{qid}")

    question = str(retrieval.get("question", ""))
    question_date = str(retrieval.get("question_date", ""))
    question_type = str(retrieval.get("question_type", ""))
    gold = str(retrieval.get("answer", ""))
    abstention = qid.endswith("_abs")
    prompt = f"Question time:{question_date} and question:{question}\nPlease answer the question based on the following memories: {memories}"
    qa_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), base_url=os.environ.get("OPENAI_BASE_URL"))
    generated, qa_usage, qa_retries = call_with_retry(client, args.model, qa_messages, args.max_retries)

    judge_prompt = get_anscheck_prompt(question_type, question, gold, generated, abstention=abstention)
    judged, judge_usage, judge_retries = call_with_retry(
        client,
        args.judge_model,
        [{"role": "user", "content": judge_prompt}],
        args.max_retries,
    )
    result = {
        "system": system,
        "repeat": repeat,
        "question_id": qid,
        "question_type": question_type,
        "question": question,
        "gold": gold,
        "hypothesis": generated,
        "judge_response": judged,
        "correct": int(true_or_false(judged)),
        "qa_usage": qa_usage,
        "judge_usage": judge_usage,
        "qa_retries": qa_retries,
        "judge_retries": judge_retries,
        "topk": args.topk,
        "retrieval_source": str(retrieval_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer_prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "judge_prompt.txt").write_text(judge_prompt, encoding="utf-8")
    atomic_json(answer_path, result)
    return result


def aggregate(output_root: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    results = sorted(results, key=lambda x: (x["system"], int(x["repeat"]), x["question_id"]))
    jsonl = output_root / "qa_results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {"systems": {}, "config": {"model": args.model, "judge_model": args.judge_model, "topk": args.topk, "repeats": args.repeats, "workers": args.workers}}
    for system in sorted({x["system"] for x in results}):
        by_repeat: dict[int, list[dict[str, Any]]] = {}
        for item in results:
            if item["system"] == system:
                by_repeat.setdefault(int(item["repeat"]), []).append(item)
        repeat_rows = []
        per_question: dict[str, list[int]] = {}
        for repeat, rows in sorted(by_repeat.items()):
            vals = [int(row["correct"]) for row in rows]
            for row in rows:
                per_question.setdefault(row["question_id"], []).append(int(row["correct"]))
            repeat_rows.append({"repeat": repeat, "questions": len(vals), "correct": sum(vals), "accuracy": sum(vals) / len(vals) if vals else 0.0})
        accuracies = [row["accuracy"] for row in repeat_rows]
        consistency = sum(1 for vals in per_question.values() if len(set(vals)) == 1) / len(per_question) if per_question else 0.0
        summary["systems"][system] = {
            "repeat_results": repeat_rows,
            "mean_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
            "std_accuracy": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
            "question_answer_consistency": consistency,
            "questions": len(per_question),
        }
    atomic_json(output_root / "qa_summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--system-roots", nargs="+", default=[f"{k}={v}" for k, v in SYSTEM_ROOTS.items()])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-2024-08-06")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.system_roots = dict(item.split("=", 1) for item in args.system_roots)
    missing = [name for name in ("b0", "b1", "b2") if name not in args.system_roots]
    if missing:
        raise SystemExit(f"missing systems: {missing}")

    tasks: list[dict[str, Any]] = []
    for system, root in args.system_roots.items():
        if system not in ("b0", "b1", "b2"):
            continue
        qroot = Path(root) / "questions"
        qids = sorted(p.name for p in qroot.iterdir() if p.is_dir() and (p / "retrieval.jsonl").exists())
        if len(qids) < 50:
            raise SystemExit(f"{system}: expected 50 completed retrieval questions, found {len(qids)}")
        for repeat in range(1, args.repeats + 1):
            tasks.extend({"system": system, "repeat": repeat, "question_id": qid} for qid in qids)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one, task, args): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"complete {task['system']} r{task['repeat']} {task['question_id']}", flush=True)
            except Exception as exc:
                failures.append({**task, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {task['system']} r{task['repeat']} {task['question_id']} {type(exc).__name__}", flush=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "failures.json", failures)
    aggregate(output_root, results, args)
    if failures:
        raise SystemExit(f"{len(failures)} QA tasks failed; rerun to resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
