#!/usr/bin/env python3
"""
Memory Extraction Evaluation for LoCoMo dataset.

Metrics
-------
Recall    = N_correct_gold_blocks / N_total_gold_blocks
            "Of all gold blocks referenced by QA queries, how many did the
             extraction system actually produce?"
            gold_block_ids in the retrieval-quality file are already in the
            coordinate space of the extracted blocks, so recall = 1.0 when
            running on the SAME extraction output that produced those IDs.
            Use this metric to compare a NEW extraction system against the
            saved gold block IDs.

Precision = N_gold_unique_blocks / N_total_extracted_blocks  (per-user avg)
            "Of all extracted blocks, what fraction is referenced as gold by
             at least one query?"
            Measures over-extraction / noise.  A system that dumps the whole
            dialogue as one giant block scores Recall=1 but Precision=low.

F1        = harmonic mean of Recall and Precision.

Granularity
-----------
Gold blocks are analyzed for their dialogue-turn span.
Fine-grained extraction (span ≈ 1) is better for retrieval precision.
Coarse extraction (span >> 1) bundles many turns together, which makes
retrieval harder because the block may also contain irrelevant turns.

Evidence-text coverage  (requires --locomo-file)
------------------------------------------------
For each QA query we know which dialogue turn(s) (e.g. "D1:3") are the
evidence.  We check whether the gold block that *covers* that turn also
contains the verbatim dialogue text.  This verifies that extraction
faithfully preserved the raw text, not just the turn position.

Usage
-----
# Minimal (no LoCoMo file needed for Recall/Precision/F1):
python eval_memory_extraction.py \\
    --blocks-file  out/locomo-all-user/final_boxes_content.jsonl \\
    --queries-file out/locomo-all-user/metrics/retrieval_quality_per_query_retrieval_enhanced_lolol_top5.jsonl

# Full (with evidence-text coverage check):
python eval_memory_extraction.py \\
    --blocks-file  out/locomo-all-user/final_boxes_content.jsonl \\
    --queries-file out/locomo-all-user/metrics/retrieval_quality_per_query_retrieval_enhanced_lolol_top5.jsonl \\
    --locomo-file  /data/locomo/data/locomo10.json \\
    --output-dir   out/locomo-all-user/metrics
"""

import json
import os
import re
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_blocks(path):
    blocks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                blocks.append(json.loads(line))
    return blocks


def load_queries(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_locomo(path):
    """Return dict: user_idx (str "0".."9") -> dia_id -> turn text."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for idx, user in enumerate(data):
        conv = user["conversation"]
        dia_map = {}
        for key, turns in conv.items():
            if not isinstance(turns, list):
                continue
            for turn in turns:
                dia_id = turn.get("dia_id")
                if dia_id:
                    dia_map[dia_id] = turn.get("text", "")
        result[str(idx)] = dia_map

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_evidences(ev_list):
    """Parse 'D1:3' or 'D8:6; D9:17' -> list of dia_id strings."""
    result = []
    for ev in ev_list:
        for part in re.split(r"[;,]", ev):
            part = part.strip()
            if re.match(r"D\d+:\d+", part):
                result.append(part)
    return result


def block_span(block):
    return block["coverage"]["end_idx"] - block["coverage"]["start_idx"] + 1


def dia_id_to_session_turn(dia_id):
    """'D3:7' -> ('session_3', 7)"""
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if m:
        return f"session_{m.group(1)}", int(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_recall_precision(blocks, rows):
    """Return per-user and aggregate recall / precision dicts."""
    block_map = {(b["user_id"], b["block_id"]): b for b in blocks}

    # Unique gold blocks per user (de-duplicate across queries)
    gold_per_user = defaultdict(set)
    for r in rows:
        for bid in r["gold_block_ids"]:
            gold_per_user[r["user_id"]].add(bid)

    ext_per_user = defaultdict(set)
    for b in blocks:
        ext_per_user[b["user_id"]].add(b["block_id"])

    gold_set = set()
    for uid, bids in gold_per_user.items():
        for bid in bids:
            gold_set.add((uid, bid))

    recall_per_user = {}
    for uid, gold in gold_per_user.items():
        found = gold & ext_per_user[uid]
        recall_per_user[uid] = (len(found), len(gold))

    precision_per_user = {}
    for uid, extracted in ext_per_user.items():
        hit = sum(1 for bid in extracted if (uid, bid) in gold_set)
        precision_per_user[uid] = (hit, len(extracted))

    return recall_per_user, precision_per_user, gold_set, block_map


def compute_granularity(blocks, gold_set):
    gold_spans = [block_span(b) for b in blocks if (b["user_id"], b["block_id"]) in gold_set]
    non_gold = [b for b in blocks if (b["user_id"], b["block_id"]) not in gold_set]
    non_gold_spans = [block_span(b) for b in non_gold]
    avg_gold = sum(gold_spans) / len(gold_spans) if gold_spans else 0.0
    avg_non_gold = sum(non_gold_spans) / len(non_gold_spans) if non_gold_spans else 0.0
    return avg_gold, avg_non_gold


def compute_evidence_turn_coverage(rows, block_map):
    """Check if evidence turns fall inside a gold block's coverage range."""
    single, multi, uncov = 0, 0, 0
    total = 0
    for r in rows:
        uid = r["user_id"]
        for dia_id in parse_evidences(r["evidence"]):
            sess_id, turn_idx = dia_id_to_session_turn(dia_id)
            if sess_id is None:
                continue
            total += 1
            covered = False
            for bid in r["gold_block_ids"]:
                b = block_map.get((uid, bid))
                if (
                    b
                    and b["coverage"]["session_id"] == sess_id
                    and b["coverage"]["start_idx"] <= turn_idx <= b["coverage"]["end_idx"]
                ):
                    if block_span(b) == 1:
                        single += 1
                    else:
                        multi += 1
                    covered = True
                    break
            if not covered:
                uncov += 1
    return single, multi, uncov, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate memory extraction quality on LoCoMo.")
    parser.add_argument(
        "--blocks-file",
        default="out/locomo-all-user/final_boxes_content.jsonl",
        help="Extracted memory blocks (final_boxes_content.jsonl)",
    )
    parser.add_argument(
        "--queries-file",
        default="out/locomo-all-user/metrics/retrieval_quality_per_query_retrieval_enhanced_lolol_top5.jsonl",
        help="Per-query retrieval quality file with gold_block_ids",
    )
    parser.add_argument(
        "--locomo-file",
        default=None,
        help="(Optional) LoCoMo raw data JSON for evidence-text coverage check",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="(Optional) Save results JSON to this directory",
    )
    args = parser.parse_args()

    blocks = load_blocks(args.blocks_file)
    rows = load_queries(args.queries_file)

    recall_per_user, precision_per_user, gold_set, block_map = compute_recall_precision(blocks, rows)

    # Aggregate recall
    total_found = sum(v[0] for v in recall_per_user.values())
    total_gold = sum(v[1] for v in recall_per_user.values())
    micro_recall = total_found / total_gold if total_gold else 0.0
    macro_recall = (
        sum(v[0] / v[1] for v in recall_per_user.values()) / len(recall_per_user)
        if recall_per_user else 0.0
    )

    # Aggregate precision
    total_hit = sum(v[0] for v in precision_per_user.values())
    total_ext = sum(v[1] for v in precision_per_user.values())
    micro_precision = total_hit / total_ext if total_ext else 0.0
    macro_precision = (
        sum(v[0] / v[1] for v in precision_per_user.values()) / len(precision_per_user)
        if precision_per_user else 0.0
    )

    micro_f1 = (
        2 * micro_recall * micro_precision / (micro_recall + micro_precision)
        if (micro_recall + micro_precision) > 0 else 0.0
    )
    macro_f1 = (
        2 * macro_recall * macro_precision / (macro_recall + macro_precision)
        if (macro_recall + macro_precision) > 0 else 0.0
    )

    avg_gold_span, avg_nongold_span = compute_granularity(blocks, gold_set)
    single, multi, uncov, ev_total = compute_evidence_turn_coverage(rows, block_map)

    # ---------------------------------------------------------------------------
    # Print
    # ---------------------------------------------------------------------------
    sep = "=" * 62

    print(sep)
    print("MEMORY EXTRACTION EVALUATION  (LoCoMo)")
    print(sep)
    print(f"  Extracted blocks : {len(blocks)}")
    print(f"  Unique gold blocks: {len(gold_set)}")
    print(f"  QA queries        : {len(rows)}")
    print()

    print("── RECALL  (gold blocks extracted / gold blocks needed) ──")
    print(f"  Micro recall : {total_found}/{total_gold} = {micro_recall:.4f}")
    print(f"  Macro recall : {macro_recall:.4f}  (avg over users)")
    print()
    print("  Per-user:")
    for uid in sorted(recall_per_user):
        found, gold = recall_per_user[uid]
        print(f"    user {uid}: {found}/{gold} = {found/gold:.4f}")
    print()

    print("── PRECISION  (gold blocks / all extracted blocks) ──")
    print(f"  Micro precision : {total_hit}/{total_ext} = {micro_precision:.4f}")
    print(f"  Macro precision : {macro_precision:.4f}  (avg over users)")
    print()
    print("  Per-user:")
    for uid in sorted(precision_per_user):
        hit, ext = precision_per_user[uid]
        print(f"    user {uid}: {hit}/{ext} = {hit/ext:.4f}")
    print()

    print("── F1 ──")
    print(f"  Micro F1 : {micro_f1:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print()

    print("── GRANULARITY ──")
    print(f"  Avg span of gold blocks     : {avg_gold_span:.2f} turns")
    print(f"  Avg span of non-gold blocks : {avg_nongold_span:.2f} turns")
    print()

    print("── EVIDENCE TURN COVERAGE ──")
    if ev_total:
        print(f"  Turns in single-turn gold block (fine-grained) : {single}/{ev_total} = {single/ev_total:.4f}")
        print(f"  Turns in multi-turn  gold block (coarse)       : {multi}/{ev_total} = {multi/ev_total:.4f}")
        print(f"  Turns not covered by any gold block            : {uncov}/{ev_total} = {uncov/ev_total:.4f}")
    print()

    print(sep)
    print("INTERPRETATION")
    print(sep)
    print("""
  Recall = 1.0 is expected when the gold_block_ids in the queries file
  were derived from THIS extraction run's own block IDs.  To get a
  meaningful recall number, run a DIFFERENT extraction system and compare
  against the same gold_block_ids.

  Precision < 1.0 means the system extracted blocks not needed by any
  query.  Lower precision = more noise / over-segmentation.

  Granularity: gold blocks averaging >>1 turn per block means the system
  bundles multiple facts per block (coarse extraction).  Ideal extraction
  produces fine-grained, single-fact blocks (span ≈ 1) to maximise
  retrieval precision.
""")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "recall": {
                "micro": micro_recall,
                "macro": macro_recall,
                "total_found": total_found,
                "total_gold": total_gold,
                "per_user": {uid: {"found": v[0], "gold": v[1], "recall": v[0]/v[1]}
                             for uid, v in recall_per_user.items()},
            },
            "precision": {
                "micro": micro_precision,
                "macro": macro_precision,
                "total_hit": total_hit,
                "total_extracted": total_ext,
                "per_user": {uid: {"hit": v[0], "extracted": v[1], "precision": v[0]/v[1]}
                             for uid, v in precision_per_user.items()},
            },
            "f1": {"micro": micro_f1, "macro": macro_f1},
            "granularity": {
                "avg_gold_block_span_turns": avg_gold_span,
                "avg_nongold_block_span_turns": avg_nongold_span,
            },
            "evidence_turn_coverage": {
                "single_turn_block": single,
                "multi_turn_block": multi,
                "not_covered": uncov,
                "total": ev_total,
            },
        }
        out_path = os.path.join(args.output_dir, "memory_extraction_eval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
