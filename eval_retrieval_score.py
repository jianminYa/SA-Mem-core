"""
Evaluate retrieval with Score = M / R_max metric (optionally capped at top-k).

For each query:
  M = number of gold blocks
  Only the top-k ranked blocks are considered (--top-k, default: all)
  r1 < r2 < ... < rM = positions (1-indexed) of gold blocks within top-k
  R_max = rM (position of the last hit gold block)
  Score = M / R_max  if all M gold blocks appear within top-k
        = 0          otherwise
"""

import json
import argparse
import re
from collections import defaultdict


RETRIEVAL_FILE = "out/locomo-all-user/retrieval_enhanced_lolol.jsonl"
METRICS_FILE   = "out/locomo-all-user/metrics/retrieval_quality_per_query_retrieval_enhanced_lolol_top5.jsonl"
OUTPUT_FILE    = "out/locomo-all-user/metrics/score_mover_rmax.jsonl"
RANKING_MODE   = "content_event_topic_kw"
DEFAULT_TEMPORAL_JSON = "/data/wjp/REMem/reproduce/dataset/locomo/locomo_temporal.json"

TEMPORAL_MERGED_MAP = {
    # Explicit temporal queries
    "event_timing": "explicit_temporal",
    "duration": "explicit_temporal",
    "existence_check": "explicit_temporal",
    "aggregation": "explicit_temporal",
    # Implicit temporal queries
    "event_attributes": "implicit_temporal",
    "order": "implicit_temporal",
    "other": "implicit_temporal",
    # Non-temporal queries
    "none": "non_temporal",
    
}


def _merge_temporal_category(temporal_category):
    return TEMPORAL_MERGED_MAP.get(str(temporal_category or "").strip().lower(), "non_temporal")


def _normalize_text(text):
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _build_temporal_indexes(temporal_json_path):
    """
    Build temporal indexes from LoCoMo temporal file.

    Important: category=5 queries are excluded by design to match current eval set.
    """
    with open(temporal_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("temporal_json must be a list of conversations")

    idx_by_uid_qa = {}
    idx_by_uid_question = {}
    idx_by_question_global_unique = {}
    global_question_all = {}
    global_question_ambiguous = set()

    label_index_stats = {
        "conversations": 0,
        "qa_total": 0,
        "qa_dropped_category5": 0,
        "qa_dropped_missing_temporal": 0,
        "qa_dropped_missing_question": 0,
        "qa_kept": 0,
        "conflict_uid_qa": 0,
        "conflict_uid_question": 0,
        "conflict_global_question": 0,
    }

    for conv_idx, conv in enumerate(data):
        if not isinstance(conv, dict):
            continue
        label_index_stats["conversations"] += 1
        qa_list = conv.get("qa", [])
        if not isinstance(qa_list, list):
            continue

        uid = str(conv_idx)
        for qa_idx, qa in enumerate(qa_list):
            label_index_stats["qa_total"] += 1
            if not isinstance(qa, dict):
                continue

            if qa.get("category") == 5:
                label_index_stats["qa_dropped_category5"] += 1
                continue

            temporal_category = _normalize_text(qa.get("temporal_category"))
            if not temporal_category:
                label_index_stats["qa_dropped_missing_temporal"] += 1
                continue

            qnorm = _normalize_text(qa.get("question"))
            if not qnorm:
                label_index_stats["qa_dropped_missing_question"] += 1
                continue

            label_index_stats["qa_kept"] += 1

            k_uid_qa = (uid, qa_idx)
            old_uid_qa = idx_by_uid_qa.get(k_uid_qa)
            if old_uid_qa is None:
                idx_by_uid_qa[k_uid_qa] = temporal_category
            elif old_uid_qa != temporal_category:
                label_index_stats["conflict_uid_qa"] += 1

            k_uid_q = (uid, qnorm)
            old_uid_q = idx_by_uid_question.get(k_uid_q)
            if old_uid_q is None:
                idx_by_uid_question[k_uid_q] = temporal_category
            elif old_uid_q != temporal_category:
                label_index_stats["conflict_uid_question"] += 1

            old_global = global_question_all.get(qnorm)
            if old_global is None:
                global_question_all[qnorm] = temporal_category
            elif old_global != temporal_category:
                global_question_ambiguous.add(qnorm)
                label_index_stats["conflict_global_question"] += 1

    for qnorm, cat in global_question_all.items():
        if qnorm not in global_question_ambiguous:
            idx_by_question_global_unique[qnorm] = cat

    label_index_stats["global_unique_questions"] = len(idx_by_question_global_unique)
    label_index_stats["global_ambiguous_questions"] = len(global_question_ambiguous)

    return idx_by_uid_qa, idx_by_uid_question, idx_by_question_global_unique, label_index_stats


def _map_temporal_category(record, idx_by_uid_qa, idx_by_uid_question, idx_by_question_global):
    """Map query to temporal category with fallback order.

    Fallback order:
    1) (user_id, qa_idx)
    2) (user_id, normalized question)
    3) global normalized question (unique only)
    """
    uid = str(record.get("user_id", "")).strip()
    qa_idx = record.get("qa_idx")
    question_norm = _normalize_text(record.get("question", ""))

    if isinstance(qa_idx, int) and (uid, qa_idx) in idx_by_uid_qa:
        return idx_by_uid_qa[(uid, qa_idx)], "uid_qa_idx"

    if question_norm and (uid, question_norm) in idx_by_uid_question:
        return idx_by_uid_question[(uid, question_norm)], "uid_question"

    if question_norm and question_norm in idx_by_question_global:
        return idx_by_question_global[question_norm], "global_question"

    return "none", "none"


def compute_score(gold_block_ids: list, full_ranking: list, top_k: int | None = None) -> tuple[float, list[int]]:
    """Return (score, hit_positions).

    top_k: if set, only the first top_k items of full_ranking are considered.
    """
    M = len(gold_block_ids)
    if M == 0:
        return 0.0, []

    ranking = full_ranking[:top_k] if top_k is not None else full_ranking

    gold_set = set(gold_block_ids)
    positions = []
    for rank, block_id in enumerate(ranking, start=1):
        if block_id in gold_set:
            positions.append(rank)

    if len(positions) < M:
        return 0.0, positions   # not all gold found within top-k → score 0

    R_max = max(positions)
    return M / R_max, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-file", default=RETRIEVAL_FILE)
    parser.add_argument("--metrics-file",   default=METRICS_FILE)
    parser.add_argument("--output-file",    default=OUTPUT_FILE)
    parser.add_argument("--ranking-mode",   default=RANKING_MODE)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Only consider top-k ranked blocks (default: all)")
    parser.add_argument(
        "--temporal-json",
        default=None,
        help=(
            "Optional temporal annotation file (LoCoMo format). "
            "When provided, outputs temporal-category breakdown and maps each query to one of 8 classes."
        ),
    )
    parser.add_argument(
        "--temporal-summary-file",
        default=None,
        help="Optional output path for temporal summary JSON.",
    )
    args = parser.parse_args()

    use_temporal = bool(args.temporal_json)
    idx_by_uid_qa = {}
    idx_by_uid_question = {}
    idx_by_question_global = {}
    label_index_stats = None
    mapping_stats = {
        "total_queries": 0,
        "mapped_uid_qa_idx": 0,
        "mapped_uid_question": 0,
        "mapped_global_question": 0,
        "mapped_none": 0,
    }

    if use_temporal:
        temporal_json_path = args.temporal_json or DEFAULT_TEMPORAL_JSON
        (
            idx_by_uid_qa,
            idx_by_uid_question,
            idx_by_question_global,
            label_index_stats,
        ) = _build_temporal_indexes(temporal_json_path)

    # Load full rankings indexed by (user_id, qa_idx)
    rankings_map: dict[tuple, list] = {}
    with open(args.retrieval_file) as f:
        for line in f:
            item = json.loads(line)
            key = (str(item["user_id"]), int(item["qa_idx"]))
            mode_rankings = item.get("rankings", {})
            # Fall back to first available mode if requested mode missing
            if args.ranking_mode in mode_rankings:
                rankings_map[key] = mode_rankings[args.ranking_mode]
            elif mode_rankings:
                rankings_map[key] = next(iter(mode_rankings.values()))

    # Compute score per query
    results = []
    scores_all = []
    scores_by_cat: dict[int, list] = defaultdict(list)
    scores_by_temporal: dict[str, list] = defaultdict(list)
    scores_by_temporal_merged: dict[str, list] = defaultdict(list)

    with open(args.metrics_file) as f:
        for line in f:
            item = json.loads(line)
            key = (str(item["user_id"]), int(item["qa_idx"]))
            gold_block_ids = item["gold_block_ids"]
            category = item.get("category")

            if key not in rankings_map:
                print(f"[WARN] no ranking for user={item['user_id']} qa={item['qa_idx']}")
                continue

            score, positions = compute_score(gold_block_ids, rankings_map[key], top_k=args.top_k)

            record = {
                "user_id":        item["user_id"],
                "qa_idx":         item["qa_idx"],
                "question":       item.get("question", ""),
                "category":       category,
                "gold_block_ids": gold_block_ids,
                "hit_positions":  positions,
                "R_max":          max(positions) if positions else None,
                "M":              len(gold_block_ids),
                "score":          round(score, 6),
            }

            if use_temporal:
                mapping_stats["total_queries"] += 1
                temporal_category, match_source = _map_temporal_category(
                    record,
                    idx_by_uid_qa,
                    idx_by_uid_question,
                    idx_by_question_global,
                )
                record["temporal_category"] = temporal_category
                record["temporal_match_source"] = match_source
                merged_temporal_category = _merge_temporal_category(temporal_category)
                record["temporal_category_merged"] = merged_temporal_category

                if match_source == "uid_qa_idx":
                    mapping_stats["mapped_uid_qa_idx"] += 1
                elif match_source == "uid_question":
                    mapping_stats["mapped_uid_question"] += 1
                elif match_source == "global_question":
                    mapping_stats["mapped_global_question"] += 1
                else:
                    mapping_stats["mapped_none"] += 1

                scores_by_temporal[temporal_category].append(score)
                scores_by_temporal_merged[merged_temporal_category].append(score)

            results.append(record)
            scores_all.append(score)
            if category is not None:
                scores_by_cat[category].append(score)

    # Write per-query results
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    n = len(scores_all)
    avg = sum(scores_all) / n if n else 0.0
    perfect = sum(1 for s in scores_all if s >= 1.0 - 1e-9)
    zero    = sum(1 for s in scores_all if s == 0.0)

    print(f"\n=== M/R_max Score Summary (top_k={args.top_k if args.top_k else 'all'}) ===")
    print(f"Total queries : {n}")
    print(f"Average Score : {avg:.4f}")
    print(f"Score = 1.0   : {perfect}  ({100*perfect/n:.1f}%)")
    print(f"Score = 0.0   : {zero}  ({100*zero/n:.1f}%)")

    if scores_by_cat:
        print(f"\nPer-category average:")
        for cat in sorted(scores_by_cat):
            cat_scores = scores_by_cat[cat]
            print(f"  Category {cat}: {sum(cat_scores)/len(cat_scores):.4f}  (n={len(cat_scores)})")

    if use_temporal and scores_by_temporal:
        print("\nPer-temporal-category average:")
        for tcat in sorted(scores_by_temporal):
            tc_scores = scores_by_temporal[tcat]
            print(f"  Temporal {tcat}: {sum(tc_scores)/len(tc_scores):.4f}  (n={len(tc_scores)})")

    if use_temporal and scores_by_temporal_merged:
        print("\nPer-temporal-merged-category average:")
        for tcatm in sorted(scores_by_temporal_merged):
            tcm_scores = scores_by_temporal_merged[tcatm]
            print(f"  TemporalMerged {tcatm}: {sum(tcm_scores)/len(tcm_scores):.4f}  (n={len(tcm_scores)})")

        mt = float(mapping_stats["total_queries"]) if mapping_stats["total_queries"] else 0.0
        print("\nTemporal mapping stats:")
        if mt > 0:
            print(
                "  "
                f"uid_qa_idx={mapping_stats['mapped_uid_qa_idx']} ({mapping_stats['mapped_uid_qa_idx']/mt:.6f}) | "
                f"uid_question={mapping_stats['mapped_uid_question']} ({mapping_stats['mapped_uid_question']/mt:.6f}) | "
                f"global_question={mapping_stats['mapped_global_question']} ({mapping_stats['mapped_global_question']/mt:.6f}) | "
                f"none={mapping_stats['mapped_none']} ({mapping_stats['mapped_none']/mt:.6f})"
            )
        else:
            print("  total_queries=0")

        # Save temporal summary JSON for downstream analysis.
        temporal_summary_file = args.temporal_summary_file
        if not temporal_summary_file:
            if args.output_file.endswith(".jsonl"):
                temporal_summary_file = args.output_file[:-6] + "_temporal_summary.json"
            else:
                temporal_summary_file = args.output_file + "_temporal_summary.json"

        temporal_summary = {
            "k": args.top_k,
            "score_metric": "M/R_max",
            "retrieval_file": args.retrieval_file,
            "metrics_file": args.metrics_file,
            "output_file": args.output_file,
            "temporal_json": args.temporal_json,
            "label_index_stats": label_index_stats,
            "mapping_stats": {
                **mapping_stats,
                "mapped_ratio_uid_qa_idx": round(mapping_stats["mapped_uid_qa_idx"] / mt, 6) if mt > 0 else 0.0,
                "mapped_ratio_uid_question": round(mapping_stats["mapped_uid_question"] / mt, 6) if mt > 0 else 0.0,
                "mapped_ratio_global_question": round(mapping_stats["mapped_global_question"] / mt, 6) if mt > 0 else 0.0,
                "mapped_ratio_none": round(mapping_stats["mapped_none"] / mt, 6) if mt > 0 else 0.0,
            },
            "overall": {
                "query_count": n,
                "average_score": round(avg, 6),
                "score_eq_1_count": perfect,
                "score_eq_0_count": zero,
            },
            "by_temporal_category": {
                tcat: {
                    "query_count": len(tc_scores),
                    "query_ratio": round(len(tc_scores) / mt, 6) if mt > 0 else 0.0,
                    "average_score": round(sum(tc_scores) / len(tc_scores), 6) if tc_scores else 0.0,
                    "score_eq_1_rate": round(sum(1 for x in tc_scores if x >= 1.0 - 1e-9) / len(tc_scores), 6)
                    if tc_scores
                    else 0.0,
                    "score_eq_0_rate": round(sum(1 for x in tc_scores if x == 0.0) / len(tc_scores), 6)
                    if tc_scores
                    else 0.0,
                }
                for tcat, tc_scores in sorted(scores_by_temporal.items(), key=lambda kv: kv[0])
            },
            "temporal_merged_definition": {
                "explicit_temporal": ["event_timing", "duration", "existence_check"],
                "implicit_temporal": ["event_attributes", "order", "aggregation"],
                "non_temporal": ["none", "other"],
            },
            "by_temporal_category_merged": {
                tcatm: {
                    "query_count": len(tcm_scores),
                    "query_ratio": round(len(tcm_scores) / mt, 6) if mt > 0 else 0.0,
                    "average_score": round(sum(tcm_scores) / len(tcm_scores), 6) if tcm_scores else 0.0,
                    "score_eq_1_rate": round(sum(1 for x in tcm_scores if x >= 1.0 - 1e-9) / len(tcm_scores), 6)
                    if tcm_scores
                    else 0.0,
                    "score_eq_0_rate": round(sum(1 for x in tcm_scores if x == 0.0) / len(tcm_scores), 6)
                    if tcm_scores
                    else 0.0,
                }
                for tcatm, tcm_scores in sorted(scores_by_temporal_merged.items(), key=lambda kv: kv[0])
            },
        }

        with open(temporal_summary_file, "w", encoding="utf-8") as tf:
            json.dump(temporal_summary, tf, ensure_ascii=False, indent=2)

        print(f"Temporal summary saved to: {temporal_summary_file}")

    print(f"\nPer-query results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
