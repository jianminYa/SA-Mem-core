#!/usr/bin/env python3
"""Run MemBox Topic-Loom-only on the frozen four-question smoke set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import (
    QAClient,
    append_jsonl,
    evidence_hit,
    history_stats,
    json_dump,
    load_env_file,
    load_questions,
    memories_from_boxes,
    normalize_base_url,
    read_jsonl,
    sha256,
    encoder_for_model,
)


def token_sum(rows: list[dict], field: str):
    values = [row.get(field) for row in rows if row.get("success") and row.get(field) is not None]
    return sum(values) if values else None


def atomic_write_boxes(path: Path, boxes: list[dict]) -> None:
    """Persist the official builder output without exposing a partial JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".atomic.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for box in boxes:
            handle.write(json.dumps(box, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--smoke-ids", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()

    env = load_env_file(args.env_file)
    api_key = env.get("OPENAI_API_KEY", "")
    base_url = env.get("OPENAI_BASE_URL", "")
    model = args.model
    embedding_model = args.embedding_model
    if not api_key or not base_url:
        raise RuntimeError("the model configuration file must define OPENAI_API_KEY and OPENAI_BASE_URL")

    questions = load_questions(args.data, args.smoke_ids)
    model_encoder = encoder_for_model(model)
    run_root = args.run_root
    if (run_root / "final_boxes_content.jsonl").exists():
        raise RuntimeError(f"refusing to overwrite existing run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    calls_path = run_root / "construction_calls.jsonl"

    # Import the official MemBox implementation from the isolated clone.
    sys.path.insert(0, str(args.repo))
    import membox  # type: ignore

    Config = membox.Config
    Config.API_KEY = api_key
    Config.BASE_URL = normalize_base_url(base_url)
    Config.LLM_MODEL = model
    Config.EMBEDDING_MODEL = embedding_model
    Config.OUTPUT_BASE_DIR = str(run_root.parent)
    Config.RAW_DATA_FILE = str(run_root / "membox_lme_smoke_input.json")
    Config.LIMIT_CONVERSATIONS = len(questions)
    Config.LIMIT_SESSIONS = None
    Config.TOP_K_RETRIEVE = 10
    Config.ANSWER_TOP_N = 10
    Config.API_MAX_RETRIES = 5
    Config.CHECKPOINT_EVERY_SAMPLE = False
    Config.GEN_TEXT_MODES = ["content"]
    Config.CONSTRUCTION_CALLS_FILE = str(calls_path)
    Config.apply_run_id(run_root.name)
    membox.TokenAnalyzer.stage_stats.clear()

    transformed = []
    session_maps: dict[int, dict[str, str]] = {}
    for sample_idx, item in enumerate(questions):
        conversation: dict[str, object] = {
            "speaker_a": "user",
            "speaker_b": "assistant",
        }
        mapping: dict[str, str] = {}
        for idx, (sid, date, session) in enumerate(
            zip(item.get("haystack_session_ids", []), item.get("haystack_dates", []), item.get("haystack_sessions", [])),
            start=1,
        ):
            local_sid = f"session_{idx}"
            mapping[local_sid] = str(sid)
            conversation[local_sid] = [
                {"speaker": str(message.get("role", "")), "text": str(message.get("content", "") or "")}
                for message in session
                if isinstance(message, dict)
            ]
            conversation[f"{local_sid}_date_time"] = date
        session_maps[sample_idx] = mapping
        transformed.append(
            {
                "sample_id": item["question_id"],
                "conversation": conversation,
                "qa": [{"id": 0, "question": item["question"], "answer": item["answer"], "category": 0, "evidence": []}],
            }
        )
    (run_root / "membox_lme_smoke_input.json").write_text(json.dumps(transformed, ensure_ascii=False), encoding="utf-8")

    worker = membox.LLMWorker()
    builder = membox.MemoryBuilder(worker)
    start_build = time.perf_counter()
    boxes = builder.build_all()
    builder.save(boxes)
    # Preserve the official box contents while making the artifact itself
    # atomic for the question-level runner and later audit scripts.
    atomic_write_boxes(run_root / "final_boxes_content.jsonl", boxes)
    builder.summarize_and_log()
    construction_wall = time.perf_counter() - start_build

    # No trace stage is called. The explicit content mode is retained in the
    # manifest, and absence of trace output is checked below.
    retriever = membox.SimpleRetriever(worker, top_k=10)
    retriever.run(str(run_root / "retrieval.jsonl"), str(run_root / "retrieval.csv"))

    boxes_by_sample: dict[int, dict[int, dict]] = {}
    # Use the just-built in-memory objects for the QA-side lookup.  The
    # official builder persists the same objects to JSONL, but some runs have
    # exposed a short-lived read-after-write race in that artifact.  Retrieval
    # still consumes the persisted file above; this avoids treating a transient
    # persistence read as a failed construction result.
    for box in boxes:
        boxes_by_sample.setdefault(int(box["sample_id"]), {})[int(box["box_id"])] = box
    retrieval_rows = read_jsonl(run_root / "retrieval.jsonl")
    qa_client = QAClient(model=model, base_url=base_url, api_key=api_key, log_path=run_root / "qa_calls.jsonl")
    hypotheses_path = run_root / "hypotheses.jsonl"
    summaries = []

    for sample_idx, item in enumerate(questions):
        row = next((x for x in retrieval_rows if int(x.get("sample_id", -1)) == sample_idx), None)
        ranking = [int(x) for x in ((row or {}).get("rankings", {}) or {}).get("content_event_topic_kw", [])]
        selected = ranking[:10]
        boxes = boxes_by_sample.get(sample_idx, {})
        retrieved_sessions = []
        for bid in selected:
            local_sid = str((boxes.get(bid) or {}).get("coverage", {}).get("session_id", ""))
            if local_sid in session_maps[sample_idx]:
                retrieved_sessions.append(session_maps[sample_idx][local_sid])
        memories = memories_from_boxes(boxes, ranking, 10)
        qa_item = dict(item)
        qa_item["_system"] = "membox_compact"
        hypothesis, qa_row = qa_client.call(qa_item, memories)
        append_jsonl(hypotheses_path, {"question_id": item["question_id"], "hypothesis": hypothesis})

        calls = [x for x in read_jsonl(calls_path) if int(x.get("user_id", -1)) == sample_idx]
        exact = all(x.get("provider_usage_available") for x in calls if x.get("success")) and bool(calls)
        summary = {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "system": "membox_compact",
            "history": history_stats(item, model_encoder),
            "construction": {
                "input_tokens": token_sum(calls, "prompt_tokens"),
                "output_tokens": token_sum(calls, "completion_tokens"),
                "total_tokens": token_sum(calls, "total_tokens"),
                "llm_calls": len(calls),
                "wall_time_sec": round(construction_wall / len(questions), 6),
                "provider_usage_available": exact,
                "breakdown": {
                    stage: {
                        "calls": sum(x.get("success") is True for x in calls if x.get("stage") == stage),
                        "input_tokens": token_sum([x for x in calls if x.get("stage") == stage], "prompt_tokens"),
                        "output_tokens": token_sum([x for x in calls if x.get("stage") == stage], "completion_tokens"),
                        "total_tokens": token_sum([x for x in calls if x.get("stage") == stage], "total_tokens"),
                    }
                    for stage in ("topic_continuity", "box_extraction", "other_topic_loom_calls")
                },
                "trace_calls": sum(x.get("stage", "").startswith("trace") for x in calls),
                "trace_tokens": token_sum([x for x in calls if x.get("stage", "").startswith("trace")], "total_tokens") or 0,
            },
            "retrieval": {
                "top_k": 10,
                "retrieved_session_ids": retrieved_sessions,
                "gold_answer_session_ids": item.get("answer_session_ids", []),
                "evidence_hit": evidence_hit(retrieved_sessions, item.get("answer_session_ids", [])),
            },
            "qa": {"hypothesis": hypothesis, "correct": None, "usage": qa_row},
        }
        summaries.append(summary)
    with (run_root / "question_summary.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    trace_files = [run_root / "time_traces.jsonl", run_root / "trace_prompts.jsonl", run_root / "trace_stats.jsonl"]
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "membox_compact",
        "dataset": "longmemeval_s_cleaned.json",
        "dataset_sha256": sha256(args.data),
        "subset_file": str(args.smoke_ids),
        "subset_file_sha256": sha256(args.smoke_ids),
        "question_ids": [x["question_id"] for x in questions],
        "question_ids_sha256": hashlib.sha256("\n".join(x["question_id"] for x in questions).encode()).hexdigest(),
        "source_git_commit": os.popen(f"git -C {args.repo} rev-parse HEAD").read().strip(),
        "llm_model": model,
        "embedding_model": embedding_model,
        "api_base_configured": bool(base_url),
        "temperature": 0.0,
        "top_k": 10,
        "trace_enabled": False,
        "trace_construction_enabled": False,
        "graph_enabled": False,
        "merged_extraction_enabled": False,
        "text_modes": ["content"],
        "environment_name": "membox-lme",
        "python_version": sys.version,
        "provider_usage_available": all(x.get("provider_usage_available") for x in read_jsonl(calls_path) if x.get("success")),
        "trace_artifacts_present": [str(p.name) for p in trace_files if p.exists()],
        "construction_wall_time_sec": round(construction_wall, 6),
    }
    json_dump(run_root / "run_manifest.json", manifest)
    json_dump(run_root / "qa_results.json", {"system": "membox_compact", "question_ids": [x["question_id"] for x in questions], "hypotheses_file": str(hypotheses_path), "evaluator": "pending"})
    print(json.dumps({"system": "membox_compact", "questions": len(questions), "construction_calls": len(read_jsonl(calls_path)), "construction_wall_time_sec": round(construction_wall, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
