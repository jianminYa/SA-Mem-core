#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

def analyze_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    row_count = 0
    total_context_tokens = 0.0

    # key = (user_id, qa_idx)
    context_by_user_qa = defaultdict(list)
    context_by_user = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_no}: invalid json, skipped")
                continue

            ct = obj.get("context_tokens")
            user_id = obj.get("user_id")
            qa_idx = obj.get("qa_idx")

            if not isinstance(ct, (int, float)):
                continue
            if user_id is None or qa_idx is None:
                # 没有 user_id/qa_idx 的记录不计入“每个QA”统计
                continue

            ct = float(ct)
            row_count += 1
            total_context_tokens += ct

            context_by_user_qa[(str(user_id), int(qa_idx))].append(ct)
            context_by_user[str(user_id)].append(ct)

    if row_count == 0:
        return {
            "rows": 0,
            "unique_user_qa": 0,
            "avg_per_row": None,
            "avg_per_user_qa": None,
            "sum_context_tokens": 0.0,
            "per_user_avg": {},
        }

    # 1) 按行平均
    avg_per_row = total_context_tokens / row_count

    # 2) 按 (user_id, qa_idx) 先聚合后平均
    # 同一个 (user_id, qa_idx) 如果有多条记录，先取其均值
    per_user_qa_means = []
    for _, vals in context_by_user_qa.items():
        per_user_qa_means.append(sum(vals) / len(vals))
    avg_per_user_qa = sum(per_user_qa_means) / len(per_user_qa_means)

    # 3) 每个 user 的按行平均（可选辅助）
    per_user_avg = {}
    for uid, vals in context_by_user.items():
        per_user_avg[uid] = sum(vals) / len(vals)

    return {
        "rows": row_count,
        "unique_user_qa": len(context_by_user_qa),
        "avg_per_row": avg_per_row,
        "avg_per_user_qa": avg_per_user_qa,
        "sum_context_tokens": total_context_tokens,
        "per_user_avg": per_user_avg,
    }

def fmt(x):
    return "N/A" if x is None else f"{x:.4f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="/data/HaluMem/yjm/SAM_halumem/SA-Mem/out/test_all/generation_baseline.jsonl",
        help="path to generation jsonl",
    )
    args = parser.parse_args()

    path = Path(args.file)
    res = analyze_file(path)

    print("=" * 80)
    print(f"File: {path}")
    print("=" * 80)
    print(f"rows (valid):            {res['rows']}")
    print(f"unique (user_id,qa_idx): {res['unique_user_qa']}")
    print(f"sum context_tokens:      {res['sum_context_tokens']:.0f}")
    print(f"avg context_tokens/row:  {fmt(res['avg_per_row'])}")
    print(f"avg context_tokens/QA:   {fmt(res['avg_per_user_qa'])}")
    print("-" * 80)
    print("Per-user avg context_tokens/row:")
    for uid, v in sorted(res["per_user_avg"].items(), key=lambda x: x[0]):
        print(f"  {uid}: {v:.4f}")

if __name__ == "__main__":
    main()