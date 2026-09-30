# merged_eval.py

from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memblock_extractor as mx


# =========================
# M/R_max score
# =========================
def compute_score(gold_block_ids, ranking, top_k):
    M = len(gold_block_ids)
    if M == 0:
        return 0.0

    ranking = ranking[:top_k]
    gold_set = set(gold_block_ids)

    positions = []
    for i, b in enumerate(ranking, start=1):
        if b in gold_set:
            positions.append(i)

    if len(positions) < M:
        return 0.0

    return M / max(positions)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-file", required=True)
    parser.add_argument("--retrieval-jsonl", required=True)
    parser.add_argument("--final-content-file", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temporal-json", type=str, default=None)
    args = parser.parse_args()

    # load
    raw = json.load(open(args.raw_data_file))
    retrieval = [json.loads(l) for l in open(args.retrieval_jsonl)]
    boxes = {}

    for l in open(args.final_content_file):
        b = json.loads(l)
        uid = str(b["user_id"])
        boxes.setdefault(uid, []).append(b)

    raw_map = {str(i): r for i, r in enumerate(raw)}

    # stats
    hit_sum = recall_sum = precision_sum = 0.0
    scores_all = []
    scores_by_cat = defaultdict(list)
    scores_by_temp = defaultdict(list)
    scores_by_temp_merge = defaultdict(list)

    # ===== temporal =====
    def load_temporal(path):
        data = json.load(open(path))
        m = {}
        for uid, conv in enumerate(data):
            for i, qa in enumerate(conv["qa"]):
                if qa.get("category") == 5:
                    continue
                m[(str(uid), i)] = qa.get("temporal_category", "none")
        return m

    temporal_map = load_temporal(args.temporal_json) if args.temporal_json else {}

    def merge_temp(x):
        if x in ["event_timing","duration","existence_check","aggregation"]:
            return "explicit_temporal"
        if x in ["event_attributes","order","other"]:
            return "implicit_temporal"
        return "non_temporal"

    # =========================
    # LOOP
    # =========================
    for ent in retrieval:
        uid = str(ent["user_id"])
        qa_idx = int(ent["qa_idx"])

        qa = raw_map[uid]["qa"][qa_idx]
        category = qa.get("category")

        ranked = ent["rankings"]["content_event_topic_kw"][:args.k]

        gold = mx.evidence_to_targets(qa["evidence"], boxes[uid])
        gold_set = set(gold)

        overlap = set(ranked) & gold_set

        hit = 1 if overlap else 0
        recall = len(overlap) / len(gold) if gold else 0
        precision = len(overlap) / args.k

        hit_sum += hit
        recall_sum += recall
        precision_sum += precision

        # ===== score =====
        score = compute_score(gold, ent["rankings"]["content_event_topic_kw"], args.k)
        scores_all.append(score)

        if category is not None:
            scores_by_cat[category].append(score)

        if temporal_map:
            tcat = temporal_map.get((uid, qa_idx), "none")
            scores_by_temp[tcat].append(score)
            scores_by_temp_merge[merge_temp(tcat)].append(score)

    # =========================
    # PRINT
    # =========================
    n = len(scores_all)

    print("=== Retrieval Quality (Overall) ===")
    print(f"k={args.k}")
    print(f"evaluated_queries={n}")
    print(f"hit@k={hit_sum/n:.6f}")
    print(f"recall@k={recall_sum/n:.6f}")
    print(f"precision@k={precision_sum/n:.6f}")

    print(f"\n=== M/R_max Score Summary (top_k={args.k}) ===")
    avg = sum(scores_all)/n
    perfect = sum(1 for s in scores_all if s>=1.0)
    zero = sum(1 for s in scores_all if s==0)

    print(f"Total queries : {n}")
    print(f"Average Score : {avg:.4f}")
    print(f"Score = 1.0   : {perfect}  ({100*perfect/n:.1f}%)")
    print(f"Score = 0.0   : {zero}  ({100*zero/n:.1f}%)")

    print("\nPer-category average:")
    for c in sorted(scores_by_cat):
        s = scores_by_cat[c]
        print(f"  Category {c}: {sum(s)/len(s):.4f}  (n={len(s)})")

    if temporal_map:
        print("\nPer-temporal-category average:")
        for t in sorted(scores_by_temp):
            s = scores_by_temp[t]
            print(f"  Temporal {t}: {sum(s)/len(s):.4f}  (n={len(s)})")

        print("\nPer-temporal-merged-category average:")
        for t in sorted(scores_by_temp_merge):
            s = scores_by_temp_merge[t]
            print(f"  TemporalMerged {t}: {sum(s)/len(s):.4f}  (n={len(s)})")


if __name__ == "__main__":
    main()