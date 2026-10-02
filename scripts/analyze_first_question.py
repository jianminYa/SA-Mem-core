#!/usr/bin/env python3
"""Post-process the one-question LongMemEval baseline runs.

This script only reads the completed runs and cached embedding vectors.  It
does not call an LLM or embedding endpoint, and it does not alter either
baseline's construction or retrieval behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from common import build_qa_prompt, encoder_for_model, history_stats, json_dump, read_jsonl


ANSWER = "Business Administration"


def load_one(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def contains(text: Any, needle: str = ANSWER) -> bool:
    return needle.casefold() in str(text or "").casefold()


def mem_id(box: dict[str, Any], system: str) -> int:
    return int(box["box_id"] if system == "membox" else box["block_id"])


def session_id(box: dict[str, Any]) -> str:
    return str((box.get("coverage") or {}).get("session_id", ""))


def topic_keywords(box: dict[str, Any]) -> str:
    return str((box.get("features") or {}).get("topic_kw_text", "") or "")


def event_text(box: dict[str, Any], system: str) -> str:
    features = box.get("features") or {}
    if system == "membox":
        return str(features.get("events_text", "") or "")
    return " | ".join(
        str(event.get("description", "") or "")
        for event in (box.get("events") or [])
        if isinstance(event, dict) and event.get("description")
    )


def content_text(box: dict[str, Any]) -> str:
    return str((box.get("features") or {}).get("content_text", "") or "")


def embedding_text(box: dict[str, Any], system: str) -> str:
    content = content_text(box)
    topic = topic_keywords(box)
    if system == "membox":
        event = str((box.get("features") or {}).get("events_text", "") or "")
    else:
        event = event_text(box, system)
    return f"{content} {event} {topic}".strip()


def cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return None
    return dot / (na * nb)


def normalize_memory(box: dict[str, Any], system: str) -> dict[str, Any]:
    record = dict(box)
    record["memory_id"] = mem_id(box, system)
    record["session_id"] = session_id(box)
    record["topic"] = topic_keywords(box)
    record["keywords"] = topic_keywords(box)
    record["events_text"] = event_text(box, system)
    record["original_dialogue"] = content_text(box)
    record["embedding_representation_text"] = embedding_text(box, system)
    return record


def vector_map(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_rows(
    *,
    system: str,
    boxes: dict[int, dict[str, Any]],
    ranking: list[int],
    vectors: dict[str, dict[str, Any]],
    query_key: str,
    query_text: str,
    question_id: str,
    gold_sessions: set[str],
) -> list[dict[str, Any]]:
    query_vec = (vectors.get(query_key) or {}).get("question")
    rows: list[dict[str, Any]] = []
    for rank, bid in enumerate(ranking, start=1):
        box = boxes.get(int(bid))
        if box is None:
            continue
        key_prefix = "0" if system == "membox" else question_id
        memory_key = f"{key_prefix}_{int(bid)}"
        memory_vec = (vectors.get(memory_key) or {}).get("content_event_topic_kw")
        score = cosine(query_vec, memory_vec)
        record = normalize_memory(box, system)
        row = {
            "rank": rank,
            "memory_id": int(bid),
            "score": score,
            "session_id": session_id(box),
            "topic": record["topic"],
            "keywords": record["keywords"],
            "events": record["events_text"],
            "original_dialogue": record["original_dialogue"],
            "embedding_representation_text": record["embedding_representation_text"],
            "gold_session": session_id(box) in gold_sessions,
            "answer_in_original_dialogue": contains(record["original_dialogue"]),
            "answer_in_extracted_fields": contains(
                f"{record['topic']} {record['events_text']}"
            ),
            "embedding_cache_key": memory_key,
            "embedding_provider_usage": "cached vector from completed retrieval",
        }
        rows.append(row)
    return rows


def write_embedding_inputs(run_root: Path, rows: list[dict[str, Any]], query_text: str, gold_ids: list[int]) -> None:
    out = run_root / "embedding_inputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "query.txt").write_text(query_text, encoding="utf-8")
    by_id = {int(row["memory_id"]): row for row in rows}
    gold_row = next((by_id[x] for x in gold_ids if x in by_id), None)
    (out / "gold_memory.txt").write_text(
        str((gold_row or {}).get("embedding_representation_text", "")), encoding="utf-8"
    )
    for index in range(1, 4):
        row = rows[index - 1] if len(rows) >= index else {}
        (out / f"top{index}.txt").write_text(
            str(row.get("embedding_representation_text", "")), encoding="utf-8"
        )


def build_context(boxes: dict[int, dict[str, Any]], ranking: list[int], top_k: int) -> str:
    rows = []
    for bid in ranking[:top_k]:
        box = boxes.get(int(bid))
        if box is None:
            continue
        text = content_text(box).strip()
        if not text:
            continue
        temporal = box.get("start_time") or (box.get("temporal_index") or {}).get("start", "")
        rows.append((str(temporal), int(bid), text))
    rows.sort(key=lambda x: (x[0], x[1]))
    return "\n\n".join(row[2] for row in rows)


def construction_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    success = [row for row in rows if row.get("success")]
    def total(field: str) -> int | None:
        values = [row.get(field) for row in success if row.get(field) is not None]
        return sum(values) if values else None
    stages: dict[str, dict[str, Any]] = {}
    for stage in sorted({str(row.get("stage")) for row in rows}):
        stage_rows = [row for row in success if row.get("stage") == stage]
        stages[stage] = {
            "calls": len(stage_rows),
            "input_tokens": sum(row.get("prompt_tokens", 0) or 0 for row in stage_rows),
            "output_tokens": sum(row.get("completion_tokens", 0) or 0 for row in stage_rows),
            "total_tokens": sum(row.get("total_tokens", 0) or 0 for row in stage_rows),
            "provider_usage_available": all(
                row.get("provider_usage_available") is True for row in stage_rows
            ),
        }
    return {
        "calls": len(rows),
        "successful_calls": len(success),
        "failed_calls": len(rows) - len(success),
        "input_tokens": total("prompt_tokens"),
        "output_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "provider_usage_available": bool(success) and all(
            row.get("provider_usage_available") is True for row in success
        ),
        "stages": stages,
    }


def analyze_system(
    *,
    system: str,
    run_root: Path,
    item: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    raw_boxes = read_jsonl(run_root / "final_boxes_content.jsonl")
    boxes = {mem_id(box, system): box for box in raw_boxes}
    session_map = {
        f"session_{index}": str(session_id)
        for index, session_id in enumerate(item.get("haystack_session_ids", []), start=1)
    }
    native_gold_sessions = set(str(x) for x in item.get("answer_session_ids", []))
    if system == "membox":
        native_gold_sessions = {
            local for local, original in session_map.items()
            if original in set(str(x) for x in item.get("answer_session_ids", []))
        }
    for box in raw_boxes:
        # Keep the original native record and add the requested normalized
        # fields; no source fields are discarded.
        pass
    with (run_root / "memories.jsonl").open("w", encoding="utf-8") as handle:
        for box in raw_boxes:
            record = normalize_memory(box, system)
            if system == "membox":
                record["native_session_id"] = record["session_id"]
                record["session_id"] = session_map.get(record["session_id"], record["session_id"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    retrieval = load_one(run_root / "retrieval.jsonl")
    ranking = [int(x) for x in ((retrieval.get("rankings") or {}).get("content_event_topic_kw") or [])]
    vector_file = run_root / "vector_store" / (
        "sample_0.json" if system == "membox" else f"user_{item['question_id']}.json"
    )
    vectors = vector_map(vector_file)
    query_text = str(item["question"])
    # MemBox uses the raw question. SA-Mem's parser reported time=NONE for
    # this question, so its rewrite is the unchanged question; this is also
    # the text represented by the completed query cache key.
    rows = score_rows(
        system=system,
        boxes=boxes,
        ranking=ranking,
        vectors=vectors,
        query_key=("qa_0_0" if system == "membox" else f"qa_{item['question_id']}_{item['question_id']}"),
        query_text=query_text,
        question_id=str(item["question_id"]),
        gold_sessions=native_gold_sessions,
    )
    if system == "membox":
        for row in rows:
            row["native_session_id"] = row["session_id"]
            row["session_id"] = session_map.get(row["session_id"], row["session_id"])
    json_dump(run_root / "retrieval_full.json", {
        "system": system,
        "question_id": item["question_id"],
        "query_embedding_input": query_text,
        "embedding_model": model,
        "candidate_top_k": 20 if system == "samem" else None,
        "ranking_count": len(rows),
        "rankings": rows,
    })
    json_dump(run_root / "retrieval_top20.json", {
        "system": system,
        "question_id": item["question_id"],
        "top_k": 20,
        "rankings": rows[:20],
    })
    context_rows = rows[:10]
    json_dump(run_root / "generation_top10.json", {
        "system": system,
        "question_id": item["question_id"],
        "top_k": 10,
        "rankings": context_rows,
        "gold_in_final_context": any(row["gold_session"] for row in context_rows),
    })
    gold_ids = [row["memory_id"] for row in rows if row["gold_session"]]
    write_embedding_inputs(run_root, rows, query_text, gold_ids)

    context = build_context(boxes, ranking, 10)
    (run_root / "retrieved_context.txt").write_text(context, encoding="utf-8")
    prompt = build_qa_prompt(item, context)
    (run_root / "answer_prompt.txt").write_text(prompt, encoding="utf-8")
    hypothesis_rows = read_jsonl(run_root / "hypotheses.jsonl")
    hypothesis = str((hypothesis_rows[0] if hypothesis_rows else {}).get("hypothesis", ""))
    eval_rows = read_jsonl(run_root / "hypotheses.jsonl.eval-results-gpt-4o")
    autoeval = (eval_rows[0] if eval_rows else {}).get("autoeval_label", {})
    answer = {
        "question_id": item["question_id"],
        "question": item["question"],
        "answer": ANSWER,
        "hypothesis": hypothesis,
        "autoeval": autoeval,
        "retrieved_context_file": "retrieved_context.txt",
        "prompt_file": "answer_prompt.txt",
    }
    json_dump(run_root / "answer.json", answer)

    gold_rows = [row for row in rows if row["gold_session"]]
    gold_rank = min((row["rank"] for row in gold_rows), default=None)
    gold_row = next((row for row in gold_rows if row["rank"] == gold_rank), None)
    original_answer_rows = [row for row in gold_rows if row["answer_in_original_dialogue"]]
    extracted_answer_rows = [row for row in gold_rows if row["answer_in_extracted_fields"]]
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    construction = construction_summary(run_root / "construction_calls.jsonl")
    construction["wall_time_sec"] = manifest.get("construction_wall_time_sec")
    result = {
        "system": system,
        "run_root": str(run_root),
        "memory_unit_count": len(raw_boxes),
        "construction": construction,
        "gold_memory_ids": gold_ids,
        "gold_memory_preserved": bool(original_answer_rows),
        "gold_extracted_answer_preserved": bool(extracted_answer_rows),
        "gold_memory_count": len(gold_rows),
        "gold_rank": gold_rank,
        "gold_score": gold_row.get("score") if gold_row else None,
        "gold_in_top20": gold_rank is not None and gold_rank <= 20,
        "gold_in_final_context": gold_rank is not None and gold_rank <= 10,
        "gold_rows": gold_rows,
        "original_answer_memory_ids": [row["memory_id"] for row in original_answer_rows],
        "extracted_answer_memory_ids": [row["memory_id"] for row in extracted_answer_rows],
        "hypothesis": hypothesis,
        "autoeval": autoeval,
        "failure_stage": (
            "extraction" if not original_answer_rows else
            "retrieval ranking" if gold_rank is None or gold_rank > 20 else
            "generation-context truncation" if gold_rank > 10 else
            "QA generation" if not autoeval.get("label", False) else
            "none"
        ),
        "query_embedding_input": query_text,
        "embedding_model": model,
    }
    existing_summaries = read_jsonl(run_root / "question_summary.jsonl")
    summary = existing_summaries[0] if existing_summaries else {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "system": "membox_compact" if system == "membox" else "samem_2p",
    }
    summary["memory_unit_count"] = len(raw_boxes)
    summary["construction"] = construction
    summary["retrieval"] = {
        "candidate_top_k": 20,
        "generation_top_k": 10,
        "retrieved_session_ids": [row["session_id"] for row in rows[:10]],
        "gold_answer_session_ids": item.get("answer_session_ids", []),
        "evidence_hit_at_10": any(row["gold_session"] for row in rows[:10]),
        "gold_memory_ids": gold_ids,
        "gold_rank": gold_rank,
        "gold_score": gold_row.get("score") if gold_row else None,
        "gold_in_top20": gold_rank is not None and gold_rank <= 20,
        "gold_in_generation_top10": gold_rank is not None and gold_rank <= 10,
    }
    summary["qa"] = {
        "hypothesis": hypothesis,
        "correct": bool(autoeval.get("label")) if "label" in autoeval else None,
        "evaluator": autoeval,
    }
    summary["diagnosis"] = {
        "gold_memory_preserved": bool(original_answer_rows),
        "gold_extracted_answer_preserved": bool(extracted_answer_rows),
        "failure_stage": result["failure_stage"],
    }
    with (run_root / "question_summary.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--membox-run", type=Path, required=True)
    parser.add_argument("--samem-run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    item = next(x for x in data if x["question_id"] == args.question_id)
    encoding = encoder_for_model(args.model)
    history = history_stats(item, encoding)
    results = {
        "question_id": args.question_id,
        "question_type": item["question_type"],
        "question": item["question"],
        "answer": ANSWER,
        "history": history,
        "gold_answer_session_ids": item.get("answer_session_ids", []),
        "membox": analyze_system(system="membox", run_root=args.membox_run, item=item, model=args.model),
        "samem": analyze_system(system="samem", run_root=args.samem_run, item=item, model=args.model),
    }
    json_dump(args.report.with_suffix(".json"), results)
    print(json.dumps({
        "question_id": args.question_id,
        "history": history,
        "membox": {
            "memory_units": results["membox"]["memory_unit_count"],
            "construction": results["membox"]["construction"],
            "gold_rank": results["membox"]["gold_rank"],
            "gold_score": results["membox"]["gold_score"],
        },
        "samem": {
            "memory_units": results["samem"]["memory_unit_count"],
            "construction": results["samem"]["construction"],
            "gold_rank": results["samem"]["gold_rank"],
            "gold_score": results["samem"]["gold_score"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
