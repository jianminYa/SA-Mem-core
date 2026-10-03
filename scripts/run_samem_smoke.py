#!/usr/bin/env python3
"""Run SA-Mem's native LongMemEval adapter in the default two-pass mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import (
    QAClient,
    append_jsonl,
    encoder_for_model,
    evidence_hit,
    history_stats,
    json_dump,
    load_env_file,
    load_questions,
    memories_from_boxes,
    normalize_base_url,
    read_jsonl,
    sha256,
)


def token_sum(rows: list[dict], field: str):
    values = [row.get(field) for row in rows if row.get("success") and row.get(field) is not None]
    return sum(values) if values else None


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".atomic.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    parser.add_argument(
        "--temporal-followup-opt",
        action="store_true",
        help="Use local temporal resolution in Pass 1 with original tool-calling fallback.",
    )
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

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "retrieval"))
    import memblock_extractor as mx  # type: ignore
    from build_impl_graph import MemoryBuilder  # type: ignore
    from retrieval.retrieval_lme import LMERetriever  # type: ignore

    Config = mx.Config
    Config.LLM_PROVIDER = "openai"
    Config.API_KEY = api_key
    Config.BASE_URL = normalize_base_url(base_url)
    Config.LLM_MODEL = model
    Config.EMBEDDING_MODEL = embedding_model
    Config.OUTPUT_BASE_DIR = str(run_root.parent)
    Config.RAW_DATA_FILE = str(args.data)
    Config.LIMIT_CONVERSATIONS = len(questions)
    Config.LIMIT_SESSIONS = None
    Config.TOP_K_RETRIEVE = 20
    Config.ANSWER_TOP_N = 10
    Config.API_MAX_RETRIES = 5
    Config.CHECKPOINT_EVERY_SAMPLE = False
    Config.ENABLE_MERGED_EXTRACTION = False
    Config.ENABLE_LOCAL_TEMPORAL_RESOLUTION = bool(args.temporal_followup_opt)
    Config.ENABLE_EVENT_CLASSIFICATION = True
    Config.GEN_TEXT_MODES = ["content"]
    Config.CONSTRUCTION_CALLS_FILE = str(calls_path)
    Config.apply_run_id(run_root.name)
    mx.TokenAnalyzer.stage_stats.clear()
    mx.setup_logging(str(run_root / "builder.log"))

    # The native LME retriever intentionally accepts only 8-character
    # lowercase hexadecimal user IDs.  LongMemEval cleaned data also contains
    # synthetic IDs such as gpt4_59c863d7, so use a deterministic valid alias
    # internally and restore the original question IDs in saved diagnostics.
    aliases = {
        str(item["question_id"]): hashlib.sha256(str(item["question_id"]).encode()).hexdigest()[:8]
        for item in questions
    }
    if len(set(aliases.values())) != len(aliases):
        raise RuntimeError("question ID alias collision")
    alias_to_question = {alias: qid for qid, alias in aliases.items()}

    worker = mx.LLMWorker()
    builder = MemoryBuilder(worker)
    start_build = time.perf_counter()
    boxes = builder.build_all_longmemeval(raw_list_override=questions, user_id_start=0, write_incremental=False)
    build_wall = time.perf_counter() - start_build

    # The native LME builder uses numeric user IDs. Re-key only the persisted
    # artifacts so the existing LME retriever can address each question by its
    # stable question_id; this does not affect construction prompts or output.
    for box in boxes:
        numeric_id = int(box.get("user_id", -1))
        question_id = str(questions[numeric_id]["question_id"])
        box["question_id"] = question_id
        box["user_id"] = aliases[question_id]
    mx._write_boxes_jsonl(Config.FINAL_CONTENT_FILE, boxes)

    lme_dir = run_root / "lme_questions"
    lme_dir.mkdir(parents=True, exist_ok=True)
    for item in questions:
        retrieval_item = dict(item)
        retrieval_item["question_id"] = aliases[str(item["question_id"])]
        (lme_dir / f"{item['question_id']}.json").write_text(
            json.dumps([retrieval_item], ensure_ascii=False), encoding="utf-8"
        )

    Config.TOP_K_RETRIEVE = 20
    retriever = LMERetriever(
        worker=worker,
        boxes_dirs=[(run_root.name, str(run_root))],
        lme_data_dir=str(lme_dir),
        top_k=20,
        graph_expand=False,
        graph_include_relations=False,
        use_anchor=False,
        axis_mode="auto",
    )
    retriever.run(
        str(run_root / "retrieval.jsonl"),
        str(run_root / "retrieval.csv"),
        use_enhanced=True,
        limit=len(questions),
    )

    # Restore stable LongMemEval IDs in the externally visible retrieval
    # artifact after the native retriever has used its internal aliases.
    retrieval_rows = read_jsonl(run_root / "retrieval.jsonl")
    for row in retrieval_rows:
        alias = str(row.get("question_id") or row.get("user_id") or "")
        original = alias_to_question.get(alias, alias)
        row["question_id"] = original
        row["user_id"] = original
    atomic_write_jsonl(run_root / "retrieval.jsonl", retrieval_rows)

    boxes_by_question: dict[str, dict[int, dict]] = {}
    for box in read_jsonl(run_root / "final_boxes_content.jsonl"):
        boxes_by_question.setdefault(str(box["user_id"]), {})[int(box["block_id"])] = box
    rows_by_question = {str(x.get("question_id")): x for x in retrieval_rows}
    qa_client = QAClient(model=model, base_url=base_url, api_key=api_key, log_path=run_root / "qa_calls.jsonl")
    hypotheses_path = run_root / "hypotheses.jsonl"
    summaries = []
    all_calls = read_jsonl(calls_path)

    for item in questions:
        qid = str(item["question_id"])
        retrieval_uid = aliases[qid]
        row = rows_by_question.get(qid, {})
        ranking = [int(x) for x in ((row.get("rankings", {}) or {}).get("content_event_topic_kw", []) or [])]
        boxes = boxes_by_question.get(retrieval_uid, {})
        selected = ranking[:10]
        retrieved_sessions = [str((boxes.get(bid) or {}).get("coverage", {}).get("session_id")) for bid in selected if boxes.get(bid)]
        memories = memories_from_boxes(boxes, ranking, 10)
        qa_item = dict(item)
        qa_item["_system"] = "samem_2p"
        hypothesis, qa_row = qa_client.call(qa_item, memories)
        append_jsonl(hypotheses_path, {"question_id": qid, "hypothesis": hypothesis})

        calls = [x for x in all_calls if x.get("question_id") == qid]
        llm_calls = [
            x for x in calls
            if x.get("stage") != "temporal_local_resolve"
        ]
        exact = all(
            x.get("provider_usage_available")
            for x in llm_calls
            if x.get("success")
        ) and bool(llm_calls)
        stage_names = (
            "split_check",
            "pass1_extract",
            "temporal_local_resolve",
            "pass1_tool_followup",
            "pass1_tool_followup_fallback",
            "pass2_classify",
        )
        breakdown = {}
        for stage in stage_names:
            stage_rows = [x for x in calls if x.get("stage") == stage]
            breakdown[stage] = {
                "calls": sum(x.get("success") is True for x in stage_rows),
                "input_tokens": token_sum(stage_rows, "prompt_tokens"),
                "output_tokens": token_sum(stage_rows, "completion_tokens"),
                "total_tokens": token_sum(stage_rows, "total_tokens"),
            }
        summary = {
            "question_id": qid,
            "question_type": item["question_type"],
            "system": "samem_2p",
            "history": history_stats(item, model_encoder),
            "construction": {
                "input_tokens": token_sum(calls, "prompt_tokens"),
                "output_tokens": token_sum(calls, "completion_tokens"),
                "total_tokens": token_sum(calls, "total_tokens"),
                "llm_calls": sum(x.get("success") is True for x in llm_calls),
                "wall_time_sec": round(sum(float(x.get("latency_sec", 0.0) or 0.0) for x in calls), 6),
                "provider_usage_available": exact,
                "breakdown": breakdown,
                "qa_tokens_in_construction": False,
                "judge_tokens_in_construction": False,
                "retrieval_parser_tokens_in_construction": False,
                "embedding_tokens_in_construction": False,
            },
            "retrieval": {
                "candidate_top_k": 20,
                "generation_top_k": 10,
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

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "samem_2p",
        "dataset": "longmemeval_s_cleaned.json",
        "dataset_sha256": sha256(args.data),
        "subset_file": str(args.smoke_ids),
        "subset_file_sha256": sha256(args.smoke_ids),
        "question_ids": [x["question_id"] for x in questions],
        "question_ids_sha256": hashlib.sha256("\n".join(x["question_id"] for x in questions).encode()).hexdigest(),
        "source_git_commit": git_commit(args.repo),
        "llm_model": model,
        "embedding_model": embedding_model,
        "api_base_configured": bool(base_url),
        "temperature": 0.0,
        "top_k": 20,
        "generation_top_k": 10,
        "trace_enabled": False,
        "trace_construction_enabled": False,
        "graph_enabled": False,
        "merged_extraction_enabled": False,
        "temporal_followup_optimization_enabled": bool(args.temporal_followup_opt),
        "temporal_local_resolve_enabled": bool(args.temporal_followup_opt),
        "temporal_fallback_stage": "pass1_tool_followup_fallback",
        "event_classification_enabled": True,
        "retrieval_mode": "native_lme_enhanced_no_graph",
        "retriever_user_aliases": aliases,
        "retrieval_query_parser_tokens_counted_as_construction": False,
        "environment_name": "samem-lme",
        "python_version": sys.version,
        "provider_usage_available": all(
            x.get("provider_usage_available")
            for x in all_calls
            if x.get("success") and x.get("stage") != "temporal_local_resolve"
        ),
        "construction_wall_time_sec": round(build_wall, 6),
    }
    json_dump(run_root / "run_manifest.json", manifest)
    json_dump(run_root / "qa_results.json", {"system": "samem_2p", "question_ids": [x["question_id"] for x in questions], "hypotheses_file": str(hypotheses_path), "evaluator": "pending"})
    print(json.dumps({"system": "samem_2p", "questions": len(questions), "construction_calls": len(all_calls), "construction_wall_time_sec": round(build_wall, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
