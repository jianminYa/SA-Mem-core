"""
Bi-temporal axis ablation analysis — LongMemEval (51 query subset).

LME-specific notes:
- target_boxes is a list of multiple block ids (median ~6, range 1-13)
- We compute Recall@K = (#hit targets) / (#targets) averaged across queries (set-recall)
- Also report at least-1 hit ratio for completeness
"""
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
from typing import Any, Dict, List

OUT_DIR = "/data/lyc/SA-Mem/out/longmemeval_s_merged"
SETTINGS = ["none", "auto", "session", "event"]
TOPK_LIST = [5, 10, 20]


def load(fp: str):
    with open(fp) as f:
        return [json.loads(l) for l in f if l.strip()]


def recall_at_k(targets, ranked, k):
    """Average set-recall: |targets ∩ topK| / |targets|."""
    if not targets:
        return None
    topk = set(ranked[:k])
    hits = len(set(targets) & topk)
    return hits / len(targets)


def hit_at_k(targets, ranked, k):
    """At-least-1 hit."""
    if not targets:
        return None
    return float(bool(set(ranked[:k]) & set(targets)))


def analyze(label, fp):
    if not os.path.exists(fp):
        return {"label": label, "missing": True}
    recs = load(fp)
    n = len(recs)
    overall = {f"R@{k}": 0.0 for k in TOPK_LIST}
    overall.update({f"H@{k}": 0.0 for k in TOPK_LIST})
    by_qtype = defaultdict(lambda: {"n": 0, **{f"R@{k}": 0.0 for k in TOPK_LIST}})
    by_tc = defaultdict(lambda: {"n": 0, **{f"R@{k}": 0.0 for k in TOPK_LIST}})

    for rec in recs:
        targets = rec.get("target_boxes") or []
        if not targets:
            continue
        ranked = rec.get("rankings", {}).get("content_event_topic_kw", [])
        qt = str(rec.get("question_type") or "?")
        tc = str(rec.get("time_constraint_type") or "NONE")
        for k in TOPK_LIST:
            r = recall_at_k(targets, ranked, k)
            h = hit_at_k(targets, ranked, k)
            overall[f"R@{k}"] += r
            overall[f"H@{k}"] += h
            by_qtype[qt][f"R@{k}"] += r
            by_tc[tc][f"R@{k}"] += r
        by_qtype[qt]["n"] += 1
        by_tc[tc]["n"] += 1

    valid = sum(1 for r in recs if r.get("target_boxes"))
    for k in TOPK_LIST:
        overall[f"R@{k}"] /= valid if valid else 1
        overall[f"H@{k}"] /= valid if valid else 1
    for d in (by_qtype, by_tc):
        for key in d:
            nb = d[key]["n"]
            for k in TOPK_LIST:
                d[key][f"R@{k}"] /= nb if nb else 1

    return {"label": label, "n": valid, "overall": overall,
            "by_qtype": dict(by_qtype), "by_tc": dict(by_tc)}


def fmt(x):
    return f"{x*100:.2f}" if x is not None else "-"


def main():
    print("# Bi-Temporal Axis Ablation — LongMemEval (n=51)\n")
    print("Set-recall metric: |targets ∩ top-K| / |targets|, averaged over queries.")
    print("(LME's target_boxes is a list of multiple blocks per query, median ~6)\n")

    results = {}
    for s in SETTINGS:
        fp = os.path.join(OUT_DIR, f"retrieval_enhanced_axis_{s}.jsonl")
        results[s] = analyze(s, fp)

    # Overall recall
    print("## Overall Recall@K (set-recall)\n")
    print("| Setting | n | R@5 | R@10 | R@20 | Hit@5 | Hit@10 |")
    print("|---|---|---|---|---|---|---|")
    label_map = {"none": "No-filter", "auto": "Both (auto)",
                 "session": "Session-only", "event": "Event-only"}
    for s in ["none", "auto", "session", "event"]:
        r = results[s]
        if r.get("missing"):
            print(f"| {label_map[s]} | _missing_ |")
            continue
        ov = r["overall"]
        print(f"| {label_map[s]} | {r['n']} | {fmt(ov['R@5'])} | {fmt(ov['R@10'])} | "
              f"{fmt(ov['R@20'])} | {fmt(ov['H@5'])} | {fmt(ov['H@10'])} |")

    # By question_type
    print("\n## R@5 by question_type\n")
    all_qt = sorted({qt for r in results.values() if not r.get("missing")
                     for qt in r["by_qtype"]})
    print("| Question type | n | " + " | ".join(label_map[s] for s in SETTINGS) + " |")
    print("|---|---|" + "|".join(["---"]*len(SETTINGS)) + "|")
    ref = results["auto"]["by_qtype"] if not results["auto"].get("missing") else {}
    for qt in all_qt:
        n_qt = ref.get(qt, {}).get("n", "?")
        cells = [str(n_qt)]
        for s in SETTINGS:
            r = results[s]
            if r.get("missing"):
                cells.append("-")
                continue
            v = r["by_qtype"].get(qt, {}).get("R@5")
            cells.append(fmt(v))
        print(f"| {qt} | {' | '.join(cells)} |")

    # By time_constraint_type (constrained subset)
    print("\n## R@5 by time_constraint_type\n")
    all_tc = sorted({tc for r in results.values() if not r.get("missing")
                     for tc in r["by_tc"]}, key=lambda x: (x != "NONE", x))
    print("| tc_type | n | " + " | ".join(label_map[s] for s in SETTINGS) + " |")
    print("|---|---|" + "|".join(["---"]*len(SETTINGS)) + "|")
    ref_tc = results["auto"]["by_tc"] if not results["auto"].get("missing") else {}
    for tc in all_tc:
        n_tc = ref_tc.get(tc, {}).get("n", "?")
        cells = [str(n_tc)]
        for s in SETTINGS:
            r = results[s]
            if r.get("missing"):
                cells.append("-")
                continue
            v = r["by_tc"].get(tc, {}).get("R@5")
            cells.append(fmt(v))
        print(f"| {tc} | {' | '.join(cells)} |")

    # JSON summary
    out_json = os.path.join(OUT_DIR, "axis_ablation_lme_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n_JSON saved to {out_json}_")


if __name__ == "__main__":
    main()
