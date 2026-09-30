#!/usr/bin/env python3
"""Parse LongMemEval questions with QueryParser and export classifications."""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from query_pasing_byllm import QueryParser


_DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2}).*?(\d{2}:\d{2})")


def _parse_question_date(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M")
    except ValueError:
        return None


def _iter_items(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for key in ("data", "samples", "items", "dialogs", "conversations"):
            val = data.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        yield item
                return
    return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse LongMemEval queries with QueryParser and export CSV/JSONL."
    )
    parser.add_argument(
        "--input",
        default="/data/LightMem/longmemeval/longmemeval_s.json",
        help="Path to longmemeval_s.json",
    )
    parser.add_argument(
        "--csv-out",
        default="/data/LightMem/longmemeval/longmemeval_query_parse.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--jsonl-out",
        default="/data/LightMem/longmemeval/longmemeval_query_parse.jsonl",
        help="JSONL output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of items to process",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = list(_iter_items(data))
    if not items:
        print("No items found in input JSON.", file=sys.stderr)
        return 1

    parser_obj = QueryParser()

    rows = []
    intent_counts = Counter()
    time_type_counts = Counter()
    source_counts = Counter()

    total = 0
    for item in items:
        if args.limit is not None and total >= args.limit:
            break
        question = item.get("question")
        if not question:
            print("Skipping item with missing question.", file=sys.stderr)
            continue

        base_time = _parse_question_date(item.get("question_date"))
        if base_time is None:
            base_time = datetime.now()
            print(
                f"Warning: failed to parse question_date for {item.get('question_id')}, "
                "using current time.",
                file=sys.stderr,
            )

        directive = parser_obj.parse(question, base_time=base_time)
        tc = directive.time_constraint

        row = {
            "question_id": item.get("question_id"),
            "question": question,
            "question_type": item.get("question_type"),
            "intent": directive.intent,
            "time_type": tc.type,
            "time_text": tc.raw_text,
            "time_axis": directive.time_axis,
            "parse_source": directive.parse_source,
            "anchor_event": tc.anchor_event,
            "anchor_relation": tc.anchor_relation,
        }
        rows.append(row)

        intent_counts[directive.intent] += 1
        time_type_counts[tc.type] += 1
        source_counts[directive.parse_source] += 1
        total += 1

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(args.jsonl_out), exist_ok=True)
    with open(args.jsonl_out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Processed {total} questions")
    print("Intent counts:", dict(intent_counts))
    print("Time type counts:", dict(time_type_counts))
    print("Parse source counts:", dict(source_counts))
    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote JSONL: {args.jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
