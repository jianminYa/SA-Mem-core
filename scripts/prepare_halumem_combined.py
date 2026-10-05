#!/usr/bin/env python3
"""Build a local HaluMem-Medium QA/conversation manifest without changing source data.

The processed conversation files and QA files are kept separate in the existing
workspace.  The native generator expects one JSON list containing both the
conversation and the QA list, so this script joins them by the UUID prefix and
writes a derived, local-only manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_single(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"expected one-record JSON list: {path}")
    return value[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--qa-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    qa_dir = Path(args.qa_dir)
    processed_files = sorted(processed_dir.glob("*.json"))
    if not processed_files:
        raise SystemExit(f"no processed HaluMem files found: {processed_dir}")

    merged: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "processed_files": len(processed_files),
        "users": 0,
        "sessions": 0,
        "messages": 0,
        "questions": 0,
        "evidence_entries": 0,
        "questions_without_evidence": 0,
        "difficulty_counts": Counter(),
        "evidence_type_counts": Counter(),
        "source_files": [],
    }

    for processed_path in processed_files:
        stem = processed_path.stem
        short_id = stem.split("-")[0]
        qa_path = qa_dir / f"qa_{short_id}.json"
        if not qa_path.exists():
            raise SystemExit(f"missing matching QA file for {processed_path.name}: {qa_path}")

        conversation_record = load_single(processed_path)
        qa_record = load_single(qa_path)
        user_id = str(qa_record.get("user_id") or short_id)
        conversation_record["user_id"] = user_id
        conversation_record["qa"] = qa_record.get("qa", []) or []
        conversation_record["persona_info"] = qa_record.get("persona_info")
        merged.append(conversation_record)

        sessions = conversation_record.get("conversation", [])
        # Processed HaluMem uses a dict with session_1, session_2, ... fields.
        if isinstance(sessions, dict):
            session_keys = [k for k in sessions if k.startswith("session_") and k.endswith("_date_time")]
            session_count = len(session_keys)
            message_count = sum(
                len(sessions.get(k.replace("_date_time", ""), []) or [])
                for k in session_keys
            )
        else:
            session_count = len(sessions) if isinstance(sessions, list) else 0
            message_count = 0
        questions = conversation_record["qa"]
        stats["users"] += 1
        stats["sessions"] += session_count
        stats["messages"] += message_count
        stats["questions"] += len(questions)
        stats["source_files"].append({"processed": str(processed_path), "qa": str(qa_path)})
        for question in questions:
            evidence = question.get("evidence") or []
            stats["evidence_entries"] += len(evidence)
            if not evidence:
                stats["questions_without_evidence"] += 1
            stats["difficulty_counts"][str(question.get("difficulty"))] += 1
            for item in evidence:
                stats["evidence_type_counts"][str(item.get("memory_type"))] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False)

    stats["difficulty_counts"] = dict(sorted(stats["difficulty_counts"].items()))
    stats["evidence_type_counts"] = dict(sorted(stats["evidence_type_counts"].items()))
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in stats.items() if k != "source_files"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
