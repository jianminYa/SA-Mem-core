#!/usr/bin/env python3
"""
build_lme_reference.py  —  从 lme_preprocessed 构建 LongMemEval 评测 reference 文件

LongMemEval 的 evaluate_qa.py 和 print_qa_metrics.py 需要一个 JSON 列表文件
作为 reference，每个条目包含 question_id, question, answer, question_type 等字段。

此脚本从 lme_preprocessed 目录中的 per-question JSON 文件合并为一个 reference 文件。

用法:
    python scripts/build_lme_reference.py \
        --lme-data-dir /data/wjl/SA-Mem/data/lme_preprocessed \
        --output-json  out/longmemeval_s_merged/lme_reference.json

    # 仅包含你有 retrieval 结果的 question_id:
    python scripts/build_lme_reference.py \
        --lme-data-dir /data/wjl/SA-Mem/data/lme_preprocessed \
        --output-json  out/longmemeval_s_merged/lme_reference.json \
        --filter-qids-from out/longmemeval_s_merged/retrieval_enhanced.jsonl
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Set


def parse_args():
    p = argparse.ArgumentParser(description="Build LongMemEval reference JSON from lme_preprocessed")
    p.add_argument("--lme-data-dir", required=True, help="lme_preprocessed directory")
    p.add_argument("--output-json", required=True, help="Output reference JSON file")
    p.add_argument(
        "--filter-qids-from",
        type=str,
        default=None,
        help="Only include question_ids found in this JSONL (e.g., retrieval_enhanced.jsonl)",
    )
    return p.parse_args()


def load_filter_qids(path: str) -> Set[str]:
    qids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    e = json.loads(line)
                    qids.add(e["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return qids


def main():
    args = parse_args()

    # Load filter qids if specified
    filter_qids: Optional[Set[str]] = None
    if args.filter_qids_from:
        filter_qids = load_filter_qids(args.filter_qids_from)
        print(f"Filtering to {len(filter_qids)} question_ids from {args.filter_qids_from}")

    # Scan lme_preprocessed
    data_dir = args.lme_data_dir
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])
    print(f"Found {len(all_files)} JSON files in {data_dir}")

    references: List[dict] = []
    skipped = 0
    for fname in all_files:
        qid = fname.replace(".json", "")
        if filter_qids is not None and qid not in filter_qids:
            skipped += 1
            continue

        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # lme_preprocessed 格式: 每个文件是一个 list，通常只有 1 个条目
        if isinstance(data, list):
            for entry in data:
                ref = {
                    "question_id": entry.get("question_id", qid),
                    "question_type": entry.get("question_type", ""),
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                    "question_date": entry.get("question_date", ""),
                    "answer_session_ids": entry.get("answer_session_ids", []),
                }
                references.append(ref)
        elif isinstance(data, dict):
            ref = {
                "question_id": data.get("question_id", qid),
                "question_type": data.get("question_type", ""),
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "question_date": data.get("question_date", ""),
                "answer_session_ids": data.get("answer_session_ids", []),
            }
            references.append(ref)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt reference file: {args.output_json}")
    print(f"  Total questions: {len(references)}")
    print(f"  Skipped (not in filter): {skipped}")

    # Show question type distribution
    from collections import Counter
    type_counts = Counter(r["question_type"] for r in references)
    print("\nQuestion type distribution:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
