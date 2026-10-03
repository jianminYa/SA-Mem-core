#!/usr/bin/env python3
"""Run one LongMemEval question per isolated subprocess with resume support."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import load_questions, read_jsonl, sha256


SYSTEMS = {
    "membox": {
        "env": "membox-lme",
        "script": "scripts/run_membox_smoke.py",
        "repo_name": "Membox",
    },
    "samem": {
        "env": "samem-lme",
        "script": "scripts/run_samem_smoke.py",
        "repo_name": "SA-Mem-core",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def vector_cache_valid(run_root: Path) -> tuple[bool, str]:
    stores = list((run_root / "vector_store").glob("*.json"))
    if not stores:
        return False, "missing vector store"
    for path in stores:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"invalid vector store: {type(exc).__name__}"
        if not data:
            return False, "empty vector store"
        for key, fields in data.items():
            for field, vector in fields.items():
                if field not in {"question", "content_event_topic_kw"}:
                    continue
                if not isinstance(vector, list) or not vector or all(float(x) == 0.0 for x in vector):
                    return False, f"zero or missing embedding at {key}:{field}"
    return True, "ok"


def validate_question(run_root: Path, system: str) -> dict[str, Any]:
    required = [
        "run_manifest.json",
        "final_boxes_content.jsonl",
        "construction_calls.jsonl",
        "retrieval.jsonl",
        "question_summary.jsonl",
        "hypotheses.jsonl",
        "qa_calls.jsonl",
    ]
    missing = [name for name in required if not (run_root / name).exists()]
    if missing:
        return {"complete": False, "reason": f"missing files: {missing}", "failed_calls": 0, "retried_calls": 0}
    calls = read_jsonl(run_root / "construction_calls.jsonl")
    llm_calls = [row for row in calls if row.get("stage") != "temporal_local_resolve"]
    failed_calls = sum(row.get("success") is not True for row in llm_calls)
    usage_gaps = sum(
        row.get("success") is True and row.get("provider_usage_available") is not True
        for row in llm_calls
    )
    retried_calls = sum(max(0, int(row.get("retry_index", 0) or 0)) for row in llm_calls)
    boxes = read_jsonl(run_root / "final_boxes_content.jsonl")
    retrieval = read_jsonl(run_root / "retrieval.jsonl")
    summaries = read_jsonl(run_root / "question_summary.jsonl")
    qa_calls = read_jsonl(run_root / "qa_calls.jsonl")
    hypotheses = read_jsonl(run_root / "hypotheses.jsonl")
    bad_content = sum(
        not str((row.get("features") or {}).get("content_text", "") or "").strip()
        for row in boxes
    )
    qa_failures = sum(row.get("success") is not True for row in qa_calls)
    empty_hypotheses = sum(not str(row.get("hypothesis", "") or "").strip() for row in hypotheses)
    vector_ok, vector_reason = vector_cache_valid(run_root)
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    mode_ok = (
        manifest.get("merged_extraction_enabled") is False
        and manifest.get("graph_enabled") is False
        and (system != "membox" or manifest.get("trace_construction_enabled") is False)
    )
    complete = bool(
        calls and not failed_calls and not usage_gaps and boxes and not bad_content
        and len(retrieval) == 1 and len(summaries) == 1 and len(hypotheses) == 1
        and not qa_failures and not empty_hypotheses and vector_ok and mode_ok
    )
    reason = "ok" if complete else "; ".join(
        part for part in [
            f"failed_calls={failed_calls}" if failed_calls else "",
            f"usage_gaps={usage_gaps}" if usage_gaps else "",
            f"bad_content={bad_content}" if bad_content else "",
            f"retrieval_rows={len(retrieval)}" if len(retrieval) != 1 else "",
            f"summary_rows={len(summaries)}" if len(summaries) != 1 else "",
            f"hypothesis_rows={len(hypotheses)}" if len(hypotheses) != 1 else "",
            f"qa_failures={qa_failures}" if qa_failures else "",
            f"empty_hypotheses={empty_hypotheses}" if empty_hypotheses else "",
            vector_reason if not vector_ok else "",
            "baseline mode mismatch" if not mode_ok else "",
        ] if part
    ) or "incomplete"
    return {
        "complete": complete,
        "reason": reason,
        "failed_calls": failed_calls,
        "usage_gaps": usage_gaps,
        "retried_calls": retried_calls,
        "construction_calls": len(calls),
        "memory_units": len(boxes),
        "qa_failures": qa_failures,
        "permanently_failed_calls": failed_calls,
    }


def quarantine(run_root: Path, failed_root: Path, label: str) -> None:
    if not run_root.exists() or not any(run_root.iterdir()):
        return
    target = failed_root / f"{run_root.name}_{label}_{int(time.time())}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_root), str(target))


def run_one(
    *,
    qid: str,
    args: argparse.Namespace,
    qid_file: Path,
    question_root: Path,
    system_cfg: dict[str, str],
    print_lock: threading.Lock,
) -> dict[str, Any]:
    status_path = question_root / "question_status.json"
    if status_path.exists():
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
            if old.get("status") == "complete" and validate_question(question_root, args.system).get("complete"):
                return {"question_id": qid, "status": "skipped", "attempts": old.get("attempts", 0)}
        except Exception:
            pass
    if question_root.exists() and any(question_root.iterdir()):
        # A child can finish all artifacts but the parent can race with an
        # invalid intermediate JSONL read before the next reconciliation.
        # Preserve a fully valid existing run instead of paying for another
        # full question replay.
        try:
            existing_validation = validate_question(question_root, args.system)
        except Exception:
            existing_validation = {"complete": False}
        if existing_validation.get("complete"):
            status = {
                "question_id": qid,
                "status": "complete",
                "attempts": 0,
                "started_at": utc_now(),
                "finished_at": utc_now(),
                "failed_calls": 0,
                "retried_calls": int(existing_validation.get("retried_calls", 0) or 0),
                "permanently_failed_calls": 0,
                "reconciled_existing_artifacts": True,
                "validation": existing_validation,
            }
            json_write(status_path, status)
            with print_lock:
                print(json.dumps({"system": args.system, "question_id": qid, "status": "complete", "attempts": 0, "reconciled": True}, ensure_ascii=False), flush=True)
            return status
        quarantine(question_root, args.run_root / "_failed_attempts" / qid, "incomplete")

    attempts = 0
    total_retry_calls = 0
    last_validation: dict[str, Any] = {}
    for attempt in range(1, args.max_question_retries + 2):
        attempts = attempt
        question_root.mkdir(parents=True, exist_ok=True)
        log_path = args.run_root / "_launcher_logs" / f"{qid}.attempt{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_script = Path(args.repo) / system_cfg["script"]
        cmd = [
            "conda", "run", "-n", system_cfg["env"], "python", str(child_script),
            "--data", str(args.data), "--smoke-ids", str(qid_file),
            "--env-file", str(args.env_file), "--repo", str(args.repo),
            "--run-root", str(question_root), "--workspace-root", str(args.workspace_root),
            "--model", args.model, "--embedding-model", args.embedding_model,
        ]
        if args.system == "samem" and args.temporal_followup_opt:
            cmd.append("--temporal-followup-opt")
        started = utc_now()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    cmd, cwd=args.workspace_root, stdout=log, stderr=subprocess.STDOUT,
                    text=True, timeout=args.question_timeout_sec,
                )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            log_path.open("a", encoding="utf-8").write("\nquestion timeout\n")
        except OSError as exc:
            returncode = 125
            log_path.open("a", encoding="utf-8").write(f"\nlauncher error: {type(exc).__name__}\n")
        if returncode == 0:
            last_validation = validate_question(question_root, args.system)
        else:
            last_validation = {"complete": False, "reason": f"runner returncode={returncode}", "failed_calls": 0, "retried_calls": 0}
        total_retry_calls += int(last_validation.get("retried_calls", 0) or 0)
        if last_validation.get("complete"):
            shutil.copyfile(log_path, question_root / "stdout.log")
            status = {
                "question_id": qid,
                "status": "complete",
                "attempts": attempts,
                "started_at": started,
                "finished_at": utc_now(),
                "failed_calls": 0,
                "retried_calls": total_retry_calls,
                "permanently_failed_calls": 0,
                "validation": last_validation,
            }
            json_write(status_path, status)
            with print_lock:
                print(json.dumps({"system": args.system, "question_id": qid, "status": "complete", "attempts": attempts}, ensure_ascii=False), flush=True)
            return status
        if attempt <= args.max_question_retries:
            quarantine(question_root, args.run_root / "_failed_attempts" / qid, f"attempt{attempt}")
            time.sleep(min(30, 2 ** min(attempt, 4)))

    status = {
        "question_id": qid,
        "status": "failed",
        "attempts": attempts,
        "finished_at": utc_now(),
        "failed_calls": int(last_validation.get("failed_calls", 0) or 0),
        "retried_calls": total_retry_calls,
        "permanently_failed_calls": int(last_validation.get("failed_calls", 0) or 0),
        "validation": last_validation,
    }
    json_write(status_path, status)
    with print_lock:
        print(json.dumps({"system": args.system, "question_id": qid, "status": "failed", "reason": last_validation.get("reason")}, ensure_ascii=False), flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-question-retries", type=int, default=5)
    parser.add_argument("--question-timeout-sec", type=int, default=7200)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--temporal-followup-opt",
        action="store_true",
        help="Pass the local temporal follow-up optimization switch to SA-Mem workers.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise ValueError("workers must be between 1 and 6")
    system_cfg = SYSTEMS[args.system]
    process_started = time.perf_counter()
    questions = load_questions(args.data, args.ids)
    qids = [str(item["question_id"]) for item in questions]
    args.run_root.mkdir(parents=True, exist_ok=True)
    question_root = args.run_root / "questions"
    failed_root = args.run_root / "_failed_attempts"
    qid_dir = args.run_root / "_question_ids"
    question_root.mkdir(parents=True, exist_ok=True)
    qid_dir.mkdir(parents=True, exist_ok=True)
    for qid in qids:
        (qid_dir / f"{qid}.txt").write_text(qid + "\n", encoding="utf-8")
    json_write(args.run_root / "parallel_manifest.json", {
        "timestamp": utc_now(),
        "system": args.system,
        "dataset": str(args.data),
        "dataset_sha256": sha256(args.data),
        "question_ids_file": str(args.ids),
        "question_ids": qids,
        "question_count": len(qids),
        "question_workers": args.workers,
        "max_question_retries": args.max_question_retries,
        "question_timeout_sec": args.question_timeout_sec,
        "llm_model": args.model,
        "embedding_model": args.embedding_model,
        "temporal_followup_optimization_enabled": bool(args.temporal_followup_opt),
        "resume_enabled": True,
        "session_parallelism": False,
        "started_at": utc_now(),
    })
    print(json.dumps({"system": args.system, "questions": len(qids), "workers": args.workers, "resume": True}, ensure_ascii=False), flush=True)
    print_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                qid=qid,
                args=args,
                qid_file=qid_dir / f"{qid}.txt",
                question_root=question_root / qid,
                system_cfg=system_cfg,
                print_lock=print_lock,
            ): qid
            for qid in qids
        }
        for future in as_completed(futures):
            results.append(future.result())
    complete = sum(row.get("status") in {"complete", "skipped"} for row in results)
    failed = len(results) - complete
    manifest_path = args.run_root / "parallel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "finished_at": utc_now(),
        "complete_questions": complete,
        "failed_questions": failed,
        "wall_time_sec": round(time.perf_counter() - process_started, 6),
    })
    json_write(manifest_path, manifest)
    print(json.dumps({"system": args.system, "complete_questions": complete, "failed_questions": failed}, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
