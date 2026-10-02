#!/usr/bin/env python3
"""Run the official LongMemEval QA evaluator against one smoke run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from common import json_dump, load_env_file, normalize_base_url, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()

    env_values = load_env_file(args.env_file)
    child_env = os.environ.copy()
    child_env.update(env_values)
    child_env["OPENAI_BASE_URL"] = normalize_base_url(env_values.get("OPENAI_BASE_URL", ""))
    judge_calls = args.run_root / "judge_calls.jsonl"
    child_env["JUDGE_CALLS_FILE"] = str(judge_calls)
    result_file = Path(str(args.hypotheses) + ".eval-results-gpt-4o")
    if result_file.exists():
        raise RuntimeError(f"refusing to overwrite existing evaluator result: {result_file}")

    evaluator = args.evaluator_repo / "src" / "evaluation" / "evaluate_qa.py"
    log_path = args.run_root / "evaluator_stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(evaluator), "gpt-4o", str(args.hypotheses), str(args.reference)],
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0 or not result_file.exists():
        raise RuntimeError(f"official evaluator failed; inspect {log_path}")

    reference = {x["question_id"]: x for x in json.loads(args.reference.read_text(encoding="utf-8"))}
    results = read_jsonl(result_file)
    by_type: dict[str, list[int]] = defaultdict(list)
    abstention: list[int] = []
    for row in results:
        label = bool((row.get("autoeval_label") or {}).get("label"))
        qid = row["question_id"]
        qtype = reference[qid]["question_type"]
        by_type[qtype].append(int(label))
        if str(qid).endswith("_abs"):
            abstention.append(int(label))
    all_labels = [value for values in by_type.values() for value in values]
    judge_rows = read_jsonl(judge_calls)
    qa_results = {
        "evaluator": "LongMemEval/src/evaluation/evaluate_qa.py",
        "judge_model": "gpt-4o-2024-08-06",
        "question_count": len(results),
        "accuracy": (sum(all_labels) / len(all_labels)) if all_labels else None,
        "by_question_type": {key: (sum(values) / len(values) if values else None) for key, values in sorted(by_type.items())},
        "abstention_accuracy": (sum(abstention) / len(abstention)) if abstention else None,
        "judge_calls": len(judge_rows),
        "judge_input_tokens": sum(x.get("prompt_tokens", 0) or 0 for x in judge_rows),
        "judge_output_tokens": sum(x.get("completion_tokens", 0) or 0 for x in judge_rows),
        "judge_total_tokens": sum(x.get("total_tokens", 0) or 0 for x in judge_rows),
        "judge_usage_available": all(x.get("provider_usage_available") for x in judge_rows) if judge_rows else False,
        "result_file": str(result_file),
    }
    json_dump(args.run_root / "qa_results.json", qa_results)

    # Fill the nullable smoke summary correctness field after judging.
    labels = {row["question_id"]: bool((row.get("autoeval_label") or {}).get("label")) for row in results}
    summary_path = args.run_root / "question_summary.jsonl"
    updated = []
    for row in read_jsonl(summary_path):
        row.setdefault("qa", {})["correct"] = labels.get(row.get("question_id"))
        updated.append(row)
    with summary_path.open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"run": str(args.run_root), "question_count": len(results), "accuracy": qa_results["accuracy"], "judge_calls": len(judge_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
