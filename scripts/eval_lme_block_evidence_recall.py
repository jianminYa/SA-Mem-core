#!/usr/bin/env python3
"""
Evaluate LongMemEval block-level evidence-gold recall for SA-Mem outputs.

This script treats each question's answer_session_ids as gold evidence sessions
and checks whether final_boxes_content.jsonl contains at least one extracted
block whose user_id == question_id and coverage.session_id == answer_session_id.

Usage:
    python scripts/eval_lme_block_evidence_recall.py \
        --boxes-jsonl /data/lyc/SA-Mem/out/longmemeval_s_merged/final_boxes_content.jsonl \
        --lme-data-dir /data/wjl/SA-Mem/data/lme_preprocessed

Focus only on two known failure cases:
    python scripts/eval_lme_block_evidence_recall.py --only-focus
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_BOXES_JSONL = "/data/lyc/SA-Mem/out/longmemeval_s_merged/final_boxes_content.jsonl"
DEFAULT_LME_DATA_DIR = "/data/wjl/SA-Mem/data/lme_preprocessed"
DEFAULT_FOCUS_QIDS = ["0bc8ad93", "0db4c65d"]
HEX_QID_RE = re.compile(r"^[0-9a-f]{8}$")


BlockSummary = Dict[str, Any]
QuestionEntry = Dict[str, Any]
QuestionResult = Dict[str, Any]
BoxesIndex = Dict[str, Dict[str, List[BlockSummary]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LME answer_session_ids coverage in extracted SA-Mem blocks."
    )
    parser.add_argument(
        "--boxes-jsonl",
        default=DEFAULT_BOXES_JSONL,
        help=f"Path to final_boxes_content.jsonl (default: {DEFAULT_BOXES_JSONL})",
    )
    parser.add_argument(
        "--lme-data-dir",
        default=DEFAULT_LME_DATA_DIR,
        help=f"Directory containing lme_preprocessed/*.json (default: {DEFAULT_LME_DATA_DIR})",
    )
    parser.add_argument(
        "--focus-qids",
        nargs="*",
        default=DEFAULT_FOCUS_QIDS,
        help="Question IDs to print detailed diagnostics for. Default: 0bc8ad93 0db4c65d",
    )
    parser.add_argument(
        "--only-focus",
        action="store_true",
        help="Evaluate only --focus-qids instead of all eligible questions.",
    )
    parser.add_argument(
        "--include-non-hex",
        action="store_true",
        help="Include non-8-hex question files such as *_abs.json. Default matches retrieval_lme.py and skips them.",
    )
    parser.add_argument(
        "--filter-qids-from",
        default=None,
        help="Optional JSONL file whose question_id/user_id values define the question subset, e.g. retrieval_enhanced.jsonl.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write detailed JSON results.",
    )
    parser.add_argument(
        "--max-blocks-per-question",
        type=int,
        default=50,
        help="Maximum matched block summaries to print per focus question. Use 0 for no limit.",
    )
    return parser.parse_args()


def _unique_strings(values: Iterable[Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_filter_qids(path: str) -> Set[str]:
    qids: Set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = item.get("question_id") or item.get("user_id")
            if qid:
                qids.add(str(qid))
    return qids


def load_questions(
    lme_data_dir: str,
    focus_qids: Optional[Set[str]] = None,
    only_focus: bool = False,
    include_non_hex: bool = False,
) -> Tuple[List[QuestionEntry], Dict[str, int]]:
    data_dir = Path(lme_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"LME data dir not found: {data_dir}")

    questions: List[QuestionEntry] = []
    stats = {
        "files_seen": 0,
        "files_loaded": 0,
        "files_skipped_non_hex": 0,
        "files_skipped_focus": 0,
        "entries_loaded": 0,
    }

    for path in sorted(data_dir.glob("*.json")):
        stats["files_seen"] += 1
        stem = path.stem

        if only_focus and focus_qids is not None and stem not in focus_qids:
            stats["files_skipped_focus"] += 1
            continue
        if not include_non_hex and not HEX_QID_RE.fullmatch(stem):
            stats["files_skipped_non_hex"] += 1
            continue

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else [data]
        stats["files_loaded"] += 1
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qid = str(entry.get("question_id") or stem)
            if only_focus and focus_qids is not None and qid not in focus_qids:
                continue
            if not include_non_hex and not HEX_QID_RE.fullmatch(qid):
                continue

            questions.append(
                {
                    "question_id": qid,
                    "question_type": entry.get("question_type", "unknown"),
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                    "question_date": entry.get("question_date", ""),
                    "answer_session_ids": _unique_strings(entry.get("answer_session_ids", [])),
                    "source_file": str(path),
                }
            )
            stats["entries_loaded"] += 1

    return questions, stats


def load_boxes_index(
    boxes_jsonl: str,
    allowed_qids: Optional[Set[str]] = None,
    include_non_hex: bool = False,
) -> Tuple[BoxesIndex, Dict[str, int]]:
    path = Path(boxes_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Boxes JSONL not found: {path}")

    index: BoxesIndex = defaultdict(lambda: defaultdict(list))
    stats = {
        "lines_seen": 0,
        "blocks_indexed": 0,
        "blocks_skipped_user": 0,
        "blocks_skipped_coverage": 0,
        "blocks_skipped_non_hex": 0,
    }

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["lines_seen"] += 1
            try:
                block = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            user_id = str(block.get("user_id") or "")
            if not user_id:
                stats["blocks_skipped_user"] += 1
                continue
            if allowed_qids is not None and user_id not in allowed_qids:
                stats["blocks_skipped_user"] += 1
                continue
            if not include_non_hex and not HEX_QID_RE.fullmatch(user_id):
                stats["blocks_skipped_non_hex"] += 1
                continue

            coverage = block.get("coverage") or {}
            session_id = coverage.get("session_id")
            if not session_id:
                stats["blocks_skipped_coverage"] += 1
                continue

            summary: BlockSummary = {
                "block_id": block.get("block_id"),
                "session_id": str(session_id),
                "start_idx": coverage.get("start_idx"),
                "end_idx": coverage.get("end_idx"),
                "events_n": len(block.get("events") or []),
            }
            temporal_index = block.get("temporal_index") or {}
            if temporal_index.get("start") or temporal_index.get("end"):
                summary["temporal_start"] = temporal_index.get("start")
                summary["temporal_end"] = temporal_index.get("end")

            index[user_id][str(session_id)].append(summary)
            stats["blocks_indexed"] += 1

    for session_map in index.values():
        for blocks in session_map.values():
            blocks.sort(key=lambda b: (b.get("start_idx") is None, b.get("start_idx") or -1, b.get("end_idx") or -1))

    return {uid: dict(session_map) for uid, session_map in index.items()}, stats


def evaluate_questions(questions: List[QuestionEntry], boxes_index: BoxesIndex) -> Tuple[List[QuestionResult], Dict[str, Any]]:
    results: List[QuestionResult] = []
    summary = {
        "questions_total": len(questions),
        "questions_evaluated": 0,
        "questions_skipped_no_gold": 0,
        "questions_full_covered": 0,
        "questions_partial_covered": 0,
        "questions_zero_covered": 0,
        "gold_sessions_total": 0,
        "gold_sessions_covered": 0,
        "macro_block_recall_sum": 0.0,
    }

    for question in questions:
        qid = question["question_id"]
        gold_sessions = question.get("answer_session_ids", [])
        session_map = boxes_index.get(qid, {})

        if not gold_sessions:
            result = _base_result(question, [], [], [], [], "skipped_no_gold_sessions")
            result["block_level_recall"] = None
            result["gold_session_count"] = 0
            result["covered_session_count"] = 0
            results.append(result)
            summary["questions_skipped_no_gold"] += 1
            continue

        covered_sessions = [sid for sid in gold_sessions if sid in session_map and session_map[sid]]
        missing_sessions = [sid for sid in gold_sessions if sid not in set(covered_sessions)]
        matched_blocks: List[BlockSummary] = []
        for session_id in covered_sessions:
            matched_blocks.extend(session_map[session_id])

        recall = len(covered_sessions) / len(gold_sessions)
        if recall >= 1.0:
            diagnosis = "block_gold_fully_covered"
            summary["questions_full_covered"] += 1
        elif covered_sessions:
            diagnosis = "block_gold_partially_missing"
            summary["questions_partial_covered"] += 1
        else:
            diagnosis = "block_gold_missing_all_sessions"
            if not session_map:
                diagnosis = "no_blocks_for_question_user_id"
            summary["questions_zero_covered"] += 1

        result = _base_result(
            question,
            gold_sessions,
            covered_sessions,
            missing_sessions,
            matched_blocks,
            diagnosis,
        )
        result["block_level_recall"] = recall
        result["gold_session_count"] = len(gold_sessions)
        result["covered_session_count"] = len(covered_sessions)
        result["matched_block_count"] = len(matched_blocks)
        result["available_session_count_for_question"] = len(session_map)
        results.append(result)

        summary["questions_evaluated"] += 1
        summary["gold_sessions_total"] += len(gold_sessions)
        summary["gold_sessions_covered"] += len(covered_sessions)
        summary["macro_block_recall_sum"] += recall

    evaluated = summary["questions_evaluated"]
    summary["macro_block_recall"] = summary["macro_block_recall_sum"] / evaluated if evaluated else 0.0
    summary["micro_block_recall"] = (
        summary["gold_sessions_covered"] / summary["gold_sessions_total"]
        if summary["gold_sessions_total"]
        else 0.0
    )
    summary.pop("macro_block_recall_sum", None)
    return results, summary


def _base_result(
    question: QuestionEntry,
    gold_sessions: List[str],
    covered_sessions: List[str],
    missing_sessions: List[str],
    matched_blocks: List[BlockSummary],
    diagnosis: str,
) -> QuestionResult:
    return {
        "question_id": question["question_id"],
        "question_type": question.get("question_type", "unknown"),
        "question_date": question.get("question_date", ""),
        "question": question.get("question", ""),
        "answer": question.get("answer", ""),
        "answer_session_ids": list(gold_sessions),
        "covered_answer_session_ids": list(covered_sessions),
        "missing_answer_session_ids": list(missing_sessions),
        "matched_blocks": matched_blocks,
        "diagnosis": diagnosis,
    }


def aggregate_by_type(results: List[QuestionResult]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "questions_evaluated": 0,
            "questions_full_covered": 0,
            "questions_partial_covered": 0,
            "questions_zero_covered": 0,
            "gold_sessions_total": 0,
            "gold_sessions_covered": 0,
            "macro_block_recall_sum": 0.0,
        }
    )

    for result in results:
        recall = result.get("block_level_recall")
        if recall is None:
            continue
        row = grouped[str(result.get("question_type") or "unknown")]
        row["questions_evaluated"] += 1
        row["gold_sessions_total"] += result.get("gold_session_count", 0)
        row["gold_sessions_covered"] += result.get("covered_session_count", 0)
        row["macro_block_recall_sum"] += float(recall)
        if recall >= 1.0:
            row["questions_full_covered"] += 1
        elif recall > 0:
            row["questions_partial_covered"] += 1
        else:
            row["questions_zero_covered"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for question_type, row in sorted(grouped.items()):
        n = row["questions_evaluated"]
        gold_total = row["gold_sessions_total"]
        out[question_type] = {
            "questions_evaluated": n,
            "questions_full_covered": row["questions_full_covered"],
            "questions_partial_covered": row["questions_partial_covered"],
            "questions_zero_covered": row["questions_zero_covered"],
            "macro_block_recall": row["macro_block_recall_sum"] / n if n else 0.0,
            "micro_block_recall": row["gold_sessions_covered"] / gold_total if gold_total else 0.0,
            "gold_sessions_total": gold_total,
            "gold_sessions_covered": row["gold_sessions_covered"],
        }
    return out


def print_report(
    results: List[QuestionResult],
    summary: Dict[str, Any],
    by_type: Dict[str, Dict[str, Any]],
    question_stats: Dict[str, int],
    boxes_stats: Dict[str, int],
    focus_qids: List[str],
    max_blocks_per_question: int,
) -> None:
    print("=" * 80)
    print("LME Block-level Evidence-Gold Recall")
    print("=" * 80)
    filter_text = ""
    if "entries_filtered_by_jsonl" in question_stats:
        filter_text = (
            f" filtered_by_jsonl={question_stats['entries_filtered_by_jsonl']}"
            f" filter_qids={question_stats.get('filter_qids_loaded', 0)}"
        )
    print(
        f"Questions: loaded={question_stats['entries_loaded']} files_loaded={question_stats['files_loaded']} "
        f"skipped_non_hex={question_stats['files_skipped_non_hex']} skipped_focus={question_stats['files_skipped_focus']}"
        f"{filter_text}"
    )
    print(
        f"Boxes: lines_seen={boxes_stats['lines_seen']} indexed={boxes_stats['blocks_indexed']} "
        f"skipped_user/filter={boxes_stats['blocks_skipped_user']} skipped_coverage={boxes_stats['blocks_skipped_coverage']}"
    )
    print()
    print("Overall:")
    print(f"  evaluated_questions: {summary['questions_evaluated']} / {summary['questions_total']}")
    print(f"  skipped_no_gold:     {summary['questions_skipped_no_gold']}")
    print(f"  full_covered:        {summary['questions_full_covered']}")
    print(f"  partial_covered:     {summary['questions_partial_covered']}")
    print(f"  zero_covered:        {summary['questions_zero_covered']}")
    print(f"  macro_block_recall:  {summary['macro_block_recall']:.4f}")
    print(f"  micro_block_recall:  {summary['micro_block_recall']:.4f}")
    print(f"  gold_sessions:       {summary['gold_sessions_covered']} / {summary['gold_sessions_total']}")

    if by_type:
        print("\nBy question_type:")
        header = f"{'type':<28} {'n':>5} {'full':>6} {'partial':>8} {'zero':>6} {'macroR':>8} {'microR':>8} {'sessions':>13}"
        print(header)
        print("-" * len(header))
        for question_type, row in by_type.items():
            sessions = f"{row['gold_sessions_covered']}/{row['gold_sessions_total']}"
            print(
                f"{question_type:<28} {row['questions_evaluated']:>5} "
                f"{row['questions_full_covered']:>6} {row['questions_partial_covered']:>8} "
                f"{row['questions_zero_covered']:>6} {row['macro_block_recall']:>8.4f} "
                f"{row['micro_block_recall']:>8.4f} {sessions:>13}"
            )

    result_by_qid = {result["question_id"]: result for result in results}
    if focus_qids:
        print("\nFocus question diagnostics:")
        for qid in focus_qids:
            result = result_by_qid.get(qid)
            if not result:
                print(f"\n[{qid}] NOT LOADED")
                continue
            print_focus_result(result, max_blocks_per_question)


def print_focus_result(result: QuestionResult, max_blocks_per_question: int) -> None:
    recall = result.get("block_level_recall")
    recall_text = "N/A" if recall is None else f"{recall:.4f}"
    print("\n" + "-" * 80)
    print(f"question_id: {result['question_id']}")
    print(f"question_type: {result.get('question_type')}")
    print(f"question: {result.get('question')}")
    print(f"answer: {result.get('answer')}")
    print(f"diagnosis: {result.get('diagnosis')}")
    print(
        f"block_level_recall: {recall_text} "
        f"({result.get('covered_session_count', 0)}/{result.get('gold_session_count', 0)} gold sessions)"
    )
    print(f"answer_session_ids: {result.get('answer_session_ids', [])}")
    print(f"covered_answer_session_ids: {result.get('covered_answer_session_ids', [])}")
    print(f"missing_answer_session_ids: {result.get('missing_answer_session_ids', [])}")

    blocks = result.get("matched_blocks", [])
    print(f"matched_block_count: {len(blocks)}")
    if not blocks:
        return
    limit = len(blocks) if max_blocks_per_question <= 0 else min(len(blocks), max_blocks_per_question)
    print("matched_blocks:")
    for block in blocks[:limit]:
        print(
            f"  - block_id={block.get('block_id')} session_id={block.get('session_id')} "
            f"range={block.get('start_idx')}-{block.get('end_idx')} events_n={block.get('events_n')}"
        )
    if limit < len(blocks):
        print(f"  ... {len(blocks) - limit} more block(s) omitted")


def write_json_output(
    output_json: str,
    summary: Dict[str, Any],
    by_type: Dict[str, Dict[str, Any]],
    question_stats: Dict[str, int],
    boxes_stats: Dict[str, int],
    results: List[QuestionResult],
) -> None:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "by_question_type": by_type,
        "question_loader_stats": question_stats,
        "boxes_loader_stats": boxes_stats,
        "results": results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed JSON written to: {path}")


def main() -> None:
    args = parse_args()
    focus_qids = _unique_strings(args.focus_qids)
    focus_set = set(focus_qids) if focus_qids else None

    questions, question_stats = load_questions(
        args.lme_data_dir,
        focus_qids=focus_set,
        only_focus=args.only_focus,
        include_non_hex=args.include_non_hex,
    )
    if args.filter_qids_from:
        filter_qids = load_filter_qids(args.filter_qids_from)
        before = len(questions)
        questions = [q for q in questions if q["question_id"] in filter_qids]
        question_stats["entries_filtered_by_jsonl"] = before - len(questions)
        question_stats["filter_qids_loaded"] = len(filter_qids)

    allowed_qids = {q["question_id"] for q in questions}
    boxes_index, boxes_stats = load_boxes_index(
        args.boxes_jsonl,
        allowed_qids=allowed_qids,
        include_non_hex=args.include_non_hex,
    )
    results, summary = evaluate_questions(questions, boxes_index)
    by_type = aggregate_by_type(results)

    print_report(
        results,
        summary,
        by_type,
        question_stats,
        boxes_stats,
        focus_qids,
        args.max_blocks_per_question,
    )

    if args.output_json:
        write_json_output(args.output_json, summary, by_type, question_stats, boxes_stats, results)


if __name__ == "__main__":
    main()
