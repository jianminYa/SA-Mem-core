#!/usr/bin/env python3
"""Create per-question retrieval/QA diagnostics from completed 50Q runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_first_question import (
    build_context,
    content_text,
    embedding_text,
    mem_id,
    normalize_memory,
    score_rows,
    session_id,
    vector_map,
    write_embedding_inputs,
)
from common import build_qa_prompt, encoder_for_model, history_stats, json_dump, load_questions, read_jsonl


def analyze_one(system: str, run_root: Path, item: dict[str, Any], model: str) -> dict[str, Any]:
    raw_boxes = read_jsonl(run_root / "final_boxes_content.jsonl")
    boxes = {mem_id(row, system): row for row in raw_boxes}
    session_map = {
        f"session_{index}": str(sid)
        for index, sid in enumerate(item.get("haystack_session_ids", []), start=1)
    }
    native_gold = set(str(x) for x in item.get("answer_session_ids", []))
    if system == "membox":
        native_gold = {local for local, original in session_map.items() if original in native_gold}
    retrieval = read_jsonl(run_root / "retrieval.jsonl")
    if len(retrieval) != 1:
        raise RuntimeError(f"expected one retrieval row for {item['question_id']}")
    retrieval_row = retrieval[0]
    ranking = [int(x) for x in ((retrieval_row.get("rankings") or {}).get("content_event_topic_kw") or [])]
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    aliases = manifest.get("retriever_user_aliases") or {}
    retrieval_user_id = str(aliases.get(str(item["question_id"]), item["question_id"])) if system == "samem" else str(item["question_id"])
    vector_file = run_root / "vector_store" / (
        "sample_0.json" if system == "membox" else f"user_{item['question_id']}.json"
    )
    if system == "samem":
        vector_file = run_root / "vector_store" / f"user_{retrieval_user_id}.json"
    vectors = vector_map(vector_file)
    query_text = str(retrieval_row.get("query_embedding_input") or item["question"])
    rows = score_rows(
        system=system,
        boxes=boxes,
        ranking=ranking,
        vectors=vectors,
        query_key=("qa_0_0" if system == "membox" else f"qa_{retrieval_user_id}_{retrieval_user_id}"),
        query_text=query_text,
        question_id=(str(item["question_id"]) if system == "membox" else retrieval_user_id),
        gold_sessions=native_gold,
    )
    if system == "membox":
        for row in rows:
            row["native_session_id"] = row["session_id"]
            row["session_id"] = session_map.get(row["session_id"], row["session_id"])
    with (run_root / "memories.jsonl").open("w", encoding="utf-8") as handle:
        for row in raw_boxes:
            handle.write(json.dumps(normalize_memory(row, system), ensure_ascii=False) + "\n")
    json_dump(run_root / "retrieval_full.json", {
        "system": system,
        "question_id": item["question_id"],
        "query_embedding_input": query_text,
        "embedding_model": model,
        "ranking_count": len(rows),
        "rankings": rows,
    })
    json_dump(run_root / "retrieval_top20.json", {
        "system": system, "question_id": item["question_id"], "top_k": 20, "rankings": rows[:20],
    })
    context_rows = rows[:10]
    json_dump(run_root / "generation_top10.json", {
        "system": system, "question_id": item["question_id"], "top_k": 10,
        "rankings": context_rows,
        "gold_in_final_context": any(row["gold_session"] for row in context_rows),
    })
    gold_ids = [row["memory_id"] for row in rows if row["gold_session"]]
    write_embedding_inputs(run_root, rows, query_text, gold_ids)
    context = build_context(boxes, ranking, 10)
    (run_root / "retrieved_context.txt").write_text(context, encoding="utf-8")
    (run_root / "answer_prompt.txt").write_text(build_qa_prompt(item, context), encoding="utf-8")
    hypothesis_rows = read_jsonl(run_root / "hypotheses.jsonl")
    hypothesis = str((hypothesis_rows[0] if hypothesis_rows else {}).get("hypothesis", ""))
    gold_rows = [row for row in rows if row["gold_session"]]
    gold_rank = min((row["rank"] for row in gold_rows), default=None)
    gold_row = next((row for row in gold_rows if row["rank"] == gold_rank), None)
    existing = read_jsonl(run_root / "question_summary.jsonl")
    summary = existing[0] if existing else {"question_id": item["question_id"], "system": system}
    summary["memory_unit_count"] = len(raw_boxes)
    summary["history"] = history_stats(item, encoder_for_model(model))
    summary["retrieval"] = {
        "candidate_top_k": 20,
        "generation_top_k": 10,
        "retrieved_session_ids": [row["session_id"] for row in rows[:10]],
        "gold_answer_session_ids": item.get("answer_session_ids", []),
        "evidence_hit_at_1": any(row["gold_session"] for row in rows[:1]),
        "evidence_hit_at_5": any(row["gold_session"] for row in rows[:5]),
        "evidence_hit_at_10": any(row["gold_session"] for row in rows[:10]),
        "evidence_hit_at_20": any(row["gold_session"] for row in rows[:20]),
        "gold_memory_ids": gold_ids,
        "gold_rank": gold_rank,
        "gold_score": gold_row.get("score") if gold_row else None,
        "gold_in_top20": gold_rank is not None and gold_rank <= 20,
        "gold_in_generation_top10": gold_rank is not None and gold_rank <= 10,
        "mrr": (1.0 / gold_rank) if gold_rank else 0.0,
    }
    summary["qa"] = {"hypothesis": hypothesis, "correct": None}
    summary["diagnosis"] = {
        "gold_memory_preserved": bool(gold_rows),
        "gold_extracted_but_rank_gt_10": bool(gold_rank and gold_rank > 10),
        "gold_extracted_but_rank_gt_20": bool(gold_rank and gold_rank > 20),
        "gold_extraction_failure": not bool(gold_rows),
    }
    (run_root / "question_summary.jsonl").write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    json_dump(run_root / "answer.json", {
        "question_id": item["question_id"], "question": item["question"],
        "hypothesis": hypothesis, "correct": None,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--system", choices=("membox", "samem"), required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    questions = load_questions(args.data, args.ids)
    summaries = []
    for item in questions:
        qroot = args.run_root / "questions" / str(item["question_id"])
        summaries.append(analyze_one(args.system, qroot, item, args.model))
    with (args.run_root / "question_summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"system": args.system, "questions": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
