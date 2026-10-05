#!/usr/bin/env python3
"""Normalize only HaluMem session metadata timestamps for the experiment adapter."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path


FORMATS = (
    "%b %d, %Y, %H:%M:%S",
    "%B %d, %Y, %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def normalize(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    for fmt in FORMATS:
        try:
            return datetime.strptime(text, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for source in sorted(input_dir.glob("*.json")):
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        copied = copy.deepcopy(data)
        for record in copied:
            conversation = record.get("conversation", {}) if isinstance(record, dict) else {}
            if not isinstance(conversation, dict):
                continue
            for key, value in list(conversation.items()):
                if key.endswith("_start_time") or key.endswith("_end_time") or key.endswith("_date_time"):
                    conversation[key] = normalize(value)
        target = output_dir / source.name
        with target.open("w", encoding="utf-8") as handle:
            json.dump(copied, handle, ensure_ascii=False)
        count += 1
    print(f"normalized_files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
