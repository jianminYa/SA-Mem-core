#!/usr/bin/env python3
"""Parallel HaluMem QA + official-style judge evaluation over one variant.

The memory construction and retrieval artifacts are read-only.  QA repeats are
written under a separate repeat/question tree and can be resumed independently.
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

# The runner is invoked as ``python scripts/run_halumem_qa_repeats.py``.
# Make repository-root modules importable without relying on the caller's cwd.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_prompts import PROMPT_QA_ANSWER
from llm_judge_eval import EVALUATION_PROMPT_FOR_QUESTION


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
            return (
                response.choices[0].message.content or "",
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                attempt,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(30.0, 0.75 * (2**attempt) + random.random() * 0.5))
    raise RuntimeError(f"API call failed after retries: {type(last_error).__name__}") from last_error


def parse_judge(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(raw[start : end + 1])
            label = str(obj.get("evaluation_result", "Error"))
            reasoning = str(obj.get("reasoning", ""))
            if label in {"Correct", "Hallucination", "Omission"}:
                return label, reasoning
    except Exception:
        pass
    return "Error", raw[:1000]


def load_combined_qa(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for user in data:
        uid = str(user.get("user_id", ""))
        for index, question in enumerate(user.get("qa", []) or []):
            result[(uid, index)] = question
    return result


def load_boxes(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            box = json.loads(line)
            uid = str(box.get("user_id", ""))
            bid = int(box.get("block_id"))
            result.setdefault(uid, {})[bid] = box
    return result


def box_text(box: dict[str, Any]) -> str:
    features = box.get("features", {}) or {}
    if features.get("content_text"):
        return str(features["content_text"])
    pieces = []
    topic = features.get("topic_kw_text", "")
    if topic:
        pieces.append(f"Topics: {topic}")
    events = box.get("events", []) or []
    descriptions = [str(event.get("description", "")) for event in events if event.get("description")]
    if descriptions:
        pieces.append("Events:\n" + "\n".join(f"- {item}" for item in descriptions))
    return "\n\n".join(pieces)


def load_retrieval(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_one(task: dict[str, Any], args: argparse.Namespace, qa_map: dict[tuple[str, int], dict[str, Any]], boxes: dict[str, dict[int, dict[str, Any]]], retrieval: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    repeat = task["repeat"]
    uid = task["user_id"]
    qa_idx = task["qa_idx"]
    qid = f"{uid}__qa_{qa_idx}"
    out_dir = Path(args.output_root) / f"repeat_{repeat:02d}" / "questions" / qid
    answer_path = out_dir / "answer.json"
    if answer_path.exists() and not args.overwrite:
        return json.loads(answer_path.read_text(encoding="utf-8"))

    qa = qa_map[(uid, qa_idx)]
    ret = retrieval[(uid, qa_idx)]
    ranking = ((ret.get("rankings") or {}).get("content_event_topic_kw") or [])[: args.topk]
    context = "\n\n".join(box_text(boxes[uid][int(bid)]) for bid in ranking if int(bid) in boxes.get(uid, {}))
    if not context:
        raise RuntimeError(f"empty context for {uid}/{qa_idx}")
    question = str(qa.get("question", ""))
    gold = qa.get("answer", "")
    prompt = PROMPT_QA_ANSWER.format(memories=context, question=question)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), base_url=os.environ.get("OPENAI_BASE_URL"))
    generated, qa_usage, qa_retries = call_with_retry(
        client,
        args.model,
        [{"role": "user", "content": prompt}],
        args.max_retries,
    )
    evidence = qa.get("evidence") or []
    key_points = "\n".join(str(item.get("memory_content", "")) for item in evidence)
    judge_prompt = EVALUATION_PROMPT_FOR_QUESTION.format(
        question=question,
        reference_answer=gold,
        key_memory_points=key_points or "No specific memory points provided.",
        response=generated,
    )
    judged, judge_usage, judge_retries = call_with_retry(
        client,
        args.judge_model,
        [{"role": "user", "content": judge_prompt}],
        args.max_retries,
    )
    label, reasoning = parse_judge(judged)
    result = {
        "variant": args.variant,
        "repeat": repeat,
        "user_id": uid,
        "qa_idx": qa_idx,
        "question_id": qid,
        "difficulty": qa.get("difficulty"),
        "evidence_count": len(evidence),
        "evidence_types": [item.get("memory_type") for item in evidence],
        "question": question,
        "gold": gold,
        "hypothesis": generated,
        "judge_label": label,
        "judge_reasoning": reasoning,
        "qa_usage": qa_usage,
        "judge_usage": judge_usage,
        "qa_retries": qa_retries,
        "judge_retries": judge_retries,
        "topk": args.topk,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer_prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "judge_prompt.txt").write_text(judge_prompt, encoding="utf-8")
    atomic_json(answer_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--combined-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    qa_map = load_combined_qa(Path(args.combined_file))
    boxes = load_boxes(run_dir / "final_boxes_content.jsonl")
    retrieval_rows = load_retrieval(run_dir / "simple_retrieval.jsonl")
    retrieval = {(str(row["user_id"]), int(row["qa_idx"])): row for row in retrieval_rows}
    keys = sorted(set(qa_map) & set(retrieval))
    if not keys:
        raise SystemExit("no matching HaluMem QA/retrieval rows")

    tasks = [{"repeat": repeat, "user_id": uid, "qa_idx": qidx} for repeat in range(1, args.repeats + 1) for uid, qidx in keys]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one, task, args, qa_map, boxes, retrieval): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
                print(f"complete {args.variant} r{task['repeat']} {task['user_id']} qa{task['qa_idx']}", flush=True)
            except Exception as exc:
                failures.append({**task, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {args.variant} r{task['repeat']} {task['user_id']} qa{task['qa_idx']} {type(exc).__name__}", flush=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda row: (int(row["repeat"]), row["user_id"], int(row["qa_idx"])))
    with (output_root / "qa_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(output_root / "failures.json", failures)

    summary: dict[str, Any] = {"variant": args.variant, "repeats": args.repeats, "questions": len(keys), "repeat_results": []}
    for repeat in range(1, args.repeats + 1):
        rows = [row for row in results if int(row["repeat"]) == repeat]
        labels = [row["judge_label"] for row in rows]
        correct = sum(label == "Correct" for label in labels)
        summary["repeat_results"].append({
            "repeat": repeat,
            "questions": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "hallucination": labels.count("Hallucination"),
            "omission": labels.count("Omission"),
            "error": labels.count("Error"),
        })
    accuracies = [row["accuracy"] for row in summary["repeat_results"]]
    summary["mean_accuracy"] = statistics.mean(accuracies) if accuracies else 0.0
    summary["std_accuracy"] = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
    atomic_json(output_root / "qa_summary.json", summary)
    if failures:
        raise SystemExit(f"{len(failures)} HaluMem QA tasks failed; rerun to resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
