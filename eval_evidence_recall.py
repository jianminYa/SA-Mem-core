"""
Evidence Recall 评估脚本
用法:
  python eval_evidence_recall.py \
    --retrieval-file out/locomo_hitRatio/hitRatio0.8/retrieval_enhanced_hitRatio0.8.jsonl \
    --generation-file out/locomo_hitRatio/generation_results_locomo_enhanced_top5_content_hitRatio0.8.jsonl \
    --ks 5 10 20
"""
import argparse
import json
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_recall_at_k(rankings, target_boxes, k):
    if not target_boxes:
        return None
    top_k = set(rankings[:k])
    hits = sum(1 for t in target_boxes if t in top_k)
    return hits / len(target_boxes)


def main():
    parser = argparse.ArgumentParser(description="Evidence Recall 评估")
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--generation-file", default=None,
                        help="generation results (可选，用于验证 target_boxes)")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--metric", default="content_event_topic_kw",
                        help="ranking metric key")
    parser.add_argument("--output", default=None,
                        help="输出文件路径 (可选，同时输出到终端和文件)")
    args = parser.parse_args()

    out_file = open(args.output, "w", encoding="utf-8") if args.output else None

    def log(msg=""):
        print(msg)
        if out_file:
            out_file.write(msg + "\n")

    retrieval_data = load_jsonl(args.retrieval_file)

    # --- 按 (user_id, qa_idx) 计算 ---
    results_by_category = defaultdict(lambda: defaultdict(list))
    all_results = defaultdict(list)
    hit_counts = defaultdict(int)
    total = 0
    skipped = 0

    for item in retrieval_data:
        uid = item["user_id"]
        qa_idx = item["qa_idx"]
        category = item.get("category", "unknown")
        target_boxes = item.get("target_boxes", [])
        rankings = item.get("rankings", {}).get(args.metric, [])

        if not target_boxes:
            skipped += 1
            continue

        total += 1
        for k in args.ks:
            recall = compute_recall_at_k(rankings, target_boxes, k)
            all_results[k].append(recall)
            results_by_category[category][k].append(recall)
            if recall is not None and recall > 0:
                hit_counts[k] += 1

    # --- 输出总体指标 ---
    log("=" * 60)
    log(f"Evidence Recall 评估结果")
    log(f"总 QA 数: {total + skipped}, 有 target_boxes 的: {total}, 跳过: {skipped}")
    log(f"Ranking metric: {args.metric}")
    log("=" * 60)

    log(f"\n{'K':>5}  {'Recall@K':>10}  {'HitRate@K':>10}")
    log("-" * 35)
    for k in args.ks:
        avg_recall = sum(all_results[k]) / len(all_results[k]) if all_results[k] else 0
        hit_rate = hit_counts[k] / total if total > 0 else 0
        log(f"{k:>5}  {avg_recall:>10.4f}  {hit_rate:>10.4f}")

    # --- 按 category 分组 ---
    log(f"\n按 category 分组:")
    header = f"{'Cat':>5}  {'Count':>6}"
    for k in args.ks:
        header += f"  {'R@'+str(k):>8}  {'Hit@'+str(k):>8}"
    log(header)
    log("-" * (14 + len(args.ks) * 20))

    for cat in sorted(results_by_category.keys()):
        cat_data = results_by_category[cat]
        n = len(cat_data[args.ks[0]])
        row = f"{cat:>5}  {n:>6}"
        for k in args.ks:
            vals = cat_data[k]
            avg_r = sum(vals) / len(vals) if vals else 0
            hits = sum(1 for v in vals if v > 0)
            hr = hits / len(vals) if vals else 0
            row += f"  {avg_r:>8.4f}  {hr:>8.4f}"
        log(row)

    # --- 可选: 对比 generation file 的 topk ---
    if args.generation_file:
        gen_data = load_jsonl(args.generation_file)
        log(f"\n--- Generation file topk 对比 (topn from generation) ---")
        gen_hits = 0
        gen_total = 0
        gen_recall_sum = 0.0
        for item in gen_data:
            target_boxes = item.get("target_boxes", [])
            topk = item.get("topk", [])
            if not target_boxes:
                continue
            gen_total += 1
            top_set = set(topk)
            hits = sum(1 for t in target_boxes if t in top_set)
            r = hits / len(target_boxes)
            gen_recall_sum += r
            if hits > 0:
                gen_hits += 1

        if gen_total > 0:
            topn = len(gen_data[0].get("topk", []))
            log(f"  TopN={topn}, Total={gen_total}")
            log(f"  Recall@{topn}: {gen_recall_sum / gen_total:.4f}")
            log(f"  HitRate@{topn}: {gen_hits / gen_total:.4f}")

    if out_file:
        out_file.close()
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
