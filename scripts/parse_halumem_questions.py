#!/usr/bin/env python3
"""Parse HaluMem questions with QueryParser and export classifications."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from query_pasing_byllm import FastParser, QueryParser


_TIME_FORMATS = ["%b %d, %Y, %H:%M:%S", "%B %d, %Y, %H:%M:%S"]

FIELDNAMES = [
    "source_file",
    "user_id",
    "session_index",
    "question_index",
    "session_start_time",
    "session_end_time",
    "base_time",
    "question",
    "answer",
    "difficulty",
    "question_type",
    "evidence_count",
    "intent",
    "time_type",
    "time_text",
    "time_axis",
    "time_start",
    "time_end",
    "anchor_event",
    "anchor_relation",
    "parse_source",
    "rewritten_query",
]


def _parse_session_time(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _iter_users(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for key in ("users", "data", "items"):
            val = data.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        yield item
                return
    return


def _get_user_id(user: Dict[str, Any], fallback: str) -> str:
    for key in ("user_id", "uuid", "id"):
        val = user.get(key)
        if isinstance(val, str) and val:
            return val
    return fallback


def _get_sessions(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("conversation", "sessions", "dialogs"):
        val = user.get(key)
        if isinstance(val, list):
            return val
    return []


def _parse_question(
    parser_obj: QueryParser,
    question: str,
    base_time: Optional[datetime],
    fast_only: bool,
):
    if not question:
        return parser_obj._fallback_directive(question or "")
    if fast_only:
        fast_result = FastParser.try_parse(question, base_time=base_time)
        if fast_result:
            return parser_obj._construct_directive(
                question, fast_result, source="FAST", base_time=base_time
            )
        return parser_obj._fallback_directive(question)
    return parser_obj.parse(question, base_time=base_time)


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: str(kv[0])))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse HaluMem questions with QueryParser and export CSV/JSONL."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/data/wjl/SA-Mem/data/data/qa_1b846c59.json",
            "/data/wjl/SA-Mem/data/data/qa_2e36c193.json",
            "/data/wjl/SA-Mem/data/data/qa_348492a7.json",
            "/data/wjl/SA-Mem/data/data/qa_2f1f897e.json",
        ],
        help="Input HaluMem QA JSON files",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/wjl/SA-Mem/data/data/halumem_query_parse",
        help="Output directory for CSV/JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of questions per input file",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Disable LLM fallback and use FAST parsing only",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON summary output path",
    )
    args = parser.parse_args()

    missing = [path for path in args.inputs if not os.path.exists(path)]
    if missing:
        print("Missing input files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    summary_out = args.summary_out or os.path.join(
        args.out_dir, "halumem_query_parse_summary.json"
    )

    parser_obj = QueryParser()

    total_questions = 0
    global_intent = Counter()
    global_time_type = Counter()
    global_time_axis = Counter()
    global_question_type = Counter()
    global_difficulty = Counter()
    global_source = Counter()

    per_file_summary: Dict[str, Dict[str, Any]] = {}

    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        rows = []
        intent_counts = Counter()
        time_type_counts = Counter()
        time_axis_counts = Counter()
        question_type_counts = Counter()
        difficulty_counts = Counter()
        source_counts = Counter()

        users = list(_iter_users(data))
        if not users:
            print(f"No user records in {input_path}", file=sys.stderr)
            continue

        for user_idx, user in enumerate(users):
            user_id = _get_user_id(user, f"user_{user_idx}")
            sessions = _get_sessions(user)

            if sessions:
                for session_idx, session in enumerate(sessions):
                    questions = session.get("questions", [])
                    if not isinstance(questions, list) or not questions:
                        continue

                    base_time = _parse_session_time(session.get("end_time"))
                    if base_time is None:
                        base_time = _parse_session_time(session.get("start_time"))
                    for q_idx, q in enumerate(questions):
                        if args.limit is not None and q_idx >= args.limit:
                            break
                        question = q.get("question")
                        if not question:
                            continue

                        directive = _parse_question(
                            parser_obj, question, base_time, args.fast_only
                        )
                        tc = directive.time_constraint

                        row = {
                            "source_file": input_path,
                            "user_id": user_id,
                            "session_index": session_idx,
                            "question_index": q_idx,
                            "session_start_time": session.get("start_time"),
                            "session_end_time": session.get("end_time"),
                            "base_time": base_time.isoformat() if base_time else None,
                            "question": question,
                            "answer": q.get("answer"),
                            "difficulty": q.get("difficulty"),
                            "question_type": q.get("question_type"),
                            "evidence_count": len(q.get("evidence", []) or []),
                            "intent": directive.intent,
                            "time_type": tc.type,
                            "time_text": tc.raw_text,
                            "time_axis": directive.time_axis,
                            "time_start": tc.start,
                            "time_end": tc.end,
                            "anchor_event": tc.anchor_event,
                            "anchor_relation": tc.anchor_relation,
                            "parse_source": directive.parse_source,
                            "rewritten_query": directive.rewritten_query,
                        }
                        rows.append(row)

                        intent_counts[directive.intent] += 1
                        time_type_counts[tc.type] += 1
                        time_axis_counts[directive.time_axis] += 1
                        question_type_counts[q.get("question_type")] += 1
                        difficulty_counts[q.get("difficulty")] += 1
                        source_counts[directive.parse_source] += 1

                        total_questions += 1
                        global_intent[directive.intent] += 1
                        global_time_type[tc.type] += 1
                        global_time_axis[directive.time_axis] += 1
                        global_question_type[q.get("question_type")] += 1
                        global_difficulty[q.get("difficulty")] += 1
                        global_source[directive.parse_source] += 1
            else:
                qa_list = user.get("qa", [])
                if isinstance(qa_list, list):
                    for q_idx, q in enumerate(qa_list):
                        if args.limit is not None and q_idx >= args.limit:
                            break
                        question = q.get("question")
                        if not question:
                            continue
                        directive = _parse_question(
                            parser_obj, question, None, args.fast_only
                        )
                        tc = directive.time_constraint

                        row = {
                            "source_file": input_path,
                            "user_id": user_id,
                            "session_index": None,
                            "question_index": q_idx,
                            "session_start_time": None,
                            "session_end_time": None,
                            "base_time": None,
                            "question": question,
                            "answer": q.get("answer"),
                            "difficulty": q.get("difficulty"),
                            "question_type": q.get("question_type")
                            or q.get("category"),
                            "evidence_count": len(q.get("evidence", []) or []),
                            "intent": directive.intent,
                            "time_type": tc.type,
                            "time_text": tc.raw_text,
                            "time_axis": directive.time_axis,
                            "time_start": tc.start,
                            "time_end": tc.end,
                            "anchor_event": tc.anchor_event,
                            "anchor_relation": tc.anchor_relation,
                            "parse_source": directive.parse_source,
                            "rewritten_query": directive.rewritten_query,
                        }
                        rows.append(row)

                        intent_counts[directive.intent] += 1
                        time_type_counts[tc.type] += 1
                        time_axis_counts[directive.time_axis] += 1
                        question_type_counts[row["question_type"]] += 1
                        difficulty_counts[q.get("difficulty")] += 1
                        source_counts[directive.parse_source] += 1

                        total_questions += 1
                        global_intent[directive.intent] += 1
                        global_time_type[tc.type] += 1
                        global_time_axis[directive.time_axis] += 1
                        global_question_type[row["question_type"]] += 1
                        global_difficulty[q.get("difficulty")] += 1
                        global_source[directive.parse_source] += 1

        if not rows:
            print(f"No questions found in {input_path}", file=sys.stderr)
            continue

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        csv_out = os.path.join(args.out_dir, f"{base_name}_query_parse.csv")
        jsonl_out = os.path.join(args.out_dir, f"{base_name}_query_parse.jsonl")

        with open(csv_out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        with open(jsonl_out, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        per_file_summary[base_name] = {
            "input": input_path,
            "total_questions": len(rows),
            "intent_counts": _counter_dict(intent_counts),
            "time_type_counts": _counter_dict(time_type_counts),
            "time_axis_counts": _counter_dict(time_axis_counts),
            "question_type_counts": _counter_dict(question_type_counts),
            "difficulty_counts": _counter_dict(difficulty_counts),
            "parse_source_counts": _counter_dict(source_counts),
            "csv_out": csv_out,
            "jsonl_out": jsonl_out,
        }

        print(f"Processed {len(rows)} questions from {input_path}")
        print("  Intent counts:", _counter_dict(intent_counts))
        print("  Time type counts:", _counter_dict(time_type_counts))
        print("  Parse source counts:", _counter_dict(source_counts))

    summary = {
        "total_questions": total_questions,
        "intent_counts": _counter_dict(global_intent),
        "time_type_counts": _counter_dict(global_time_type),
        "time_axis_counts": _counter_dict(global_time_axis),
        "question_type_counts": _counter_dict(global_question_type),
        "difficulty_counts": _counter_dict(global_difficulty),
        "parse_source_counts": _counter_dict(global_source),
        "files": per_file_summary,
    }

    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Total questions processed: {total_questions}")
    print(f"Summary written to: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
