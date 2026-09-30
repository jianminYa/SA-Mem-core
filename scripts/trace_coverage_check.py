#!/usr/bin/env python3
"""Quick coverage check: how many QAs have at least one of their top-5 retrieval
results covered by some trace's box_ids? This predicts whether trace injection
will actually take effect for `content_trace_event` mode.

Usage:
    python scripts/trace_coverage_check.py \
        --run-id locomo-all-user \
        --topn 5

Outputs:
    out/<run-id>/trace_coverage_check.json
"""
import argparse
import json
import os
from collections import defaultdict


def load_traces(path: str):
    """Build map: user_id -> metric -> list of set(box_ids)."""
    traces = defaultdict(lambda: defaultdict(list))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            uid = str(t.get("user_id"))
            metric = t.get("metric") or "content_event_topic_kw"
            box_ids = set(int(b) for b in t.get("box_ids", []))
            traces[uid][metric].append(box_ids)
    return traces


def coverage_for_retrieval(retrieval_path: str, traces, topn: int, metric: str):
    total = 0
    covered = 0
    per_cat = defaultdict(lambda: {"total": 0, "covered": 0})
    hit_counts = []  # how many of top-k blocks land in any trace
    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row.get("user_id"))
            cat = str(row.get("category", "?"))
            rankings = row.get("rankings") or {}
            # rankings can be a dict (json) or a stringified dict (loaded already to dict by json)
            top_ids = rankings.get(metric, [])[:topn]
            top_ids = [int(b) for b in top_ids]

            user_traces = traces.get(uid, {}).get(metric, [])
            # Union of all trace box_ids for this user
            union_box_ids = set()
            for s in user_traces:
                union_box_ids.update(s)

            hits = sum(1 for b in top_ids if b in union_box_ids)
            total += 1
            per_cat[cat]["total"] += 1
            if hits > 0:
                covered += 1
                per_cat[cat]["covered"] += 1
            hit_counts.append(hits)
    return {
        "total_qa": total,
        "covered_qa": covered,
        "coverage_ratio": (covered / total) if total else 0.0,
        "avg_hit_in_topn": (sum(hit_counts) / len(hit_counts)) if hit_counts else 0.0,
        "max_hit_in_topn": max(hit_counts) if hit_counts else 0,
        "by_category": {
            c: {
                "total": v["total"],
                "covered": v["covered"],
                "coverage_ratio": (v["covered"] / v["total"]) if v["total"] else 0.0,
            }
            for c, v in sorted(per_cat.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="locomo-all-user")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--metric", default="content_event_topic_kw")
    parser.add_argument(
        "--out-base",
        default="/data/lyc/SA-Mem/out",
        help="Base out dir; final dir is <out-base>/<run-id>/",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.out_base, args.run_id)
    trace_path = os.path.join(out_dir, "time_traces.jsonl")
    baseline_path = os.path.join(out_dir, "retrieval_baseline.jsonl")
    enhanced_path = os.path.join(out_dir, "retrieval_enhanced.jsonl")

    print(f"Loading traces from {trace_path} ...")
    traces = load_traces(trace_path)
    n_users = len(traces)
    n_traces = sum(len(lst) for u in traces.values() for lst in u.values())
    print(f"  users with traces: {n_users}, total trace entries: {n_traces}")

    result = {
        "run_id": args.run_id,
        "topn": args.topn,
        "metric": args.metric,
        "n_users_with_traces": n_users,
        "n_trace_entries": n_traces,
        "baseline": coverage_for_retrieval(baseline_path, traces, args.topn, args.metric),
        "enhanced": coverage_for_retrieval(enhanced_path, traces, args.topn, args.metric),
    }

    out_path = os.path.join(out_dir, "trace_coverage_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}\n")

    for tag in ("baseline", "enhanced"):
        r = result[tag]
        print(f"[{tag}] total={r['total_qa']} covered={r['covered_qa']} "
              f"ratio={r['coverage_ratio']:.2%} avg_hits={r['avg_hit_in_topn']:.2f}/{args.topn}")
        for c, v in r["by_category"].items():
            print(f"    cat={c}: covered={v['covered']}/{v['total']} ({v['coverage_ratio']:.2%})")


if __name__ == "__main__":
    main()
