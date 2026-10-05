#!/usr/bin/env python3
"""Package completed LongMemEval artifacts without making API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean, median


def first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"empty JSONL: {path}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def load_ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def normalize_b2_question(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    copies = {
        "final_boxes_content.jsonl": "memories.jsonl",
        "construction_calls.jsonl": "construction_calls.jsonl",
        "temporal_gate.jsonl": "temporal_gate.jsonl",
        "question_summary.jsonl": "question_summary.jsonl",
        "retrieval_timings.jsonl": "retrieval_timings.jsonl",
        "question_status.json": "question_status.json",
        "run_manifest.json": "run_manifest.json",
    }
    for source_name, target_name in copies.items():
        copy_file(source / source_name, target / target_name)

    native = first_jsonl(source / "retrieval.jsonl")
    write_json(target / "retrieval_native.json", native)

    memories = {}
    with (source / "final_boxes_content.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                memories[str(row["block_id"])] = row

    method = "content_event_topic_kw"
    ordered_ids = native.get("rankings", {}).get(method, [])
    gold_sessions = set(native.get("answer_session_ids", []))
    ranking = []
    for rank, memory_id in enumerate(ordered_ids, start=1):
        row = memories.get(str(memory_id), {})
        features = row.get("features", {})
        coverage = row.get("coverage", {})
        ranking.append({
            "rank": rank,
            "memory_id": memory_id,
            "score": None,
            "score_available": False,
            "score_source": "native retrieval artifact stores ordered IDs but no similarity score",
            "session_id": coverage.get("session_id"),
            "coverage": coverage,
            "temporal_index": row.get("temporal_index"),
            "topic_kw_text": features.get("topic_kw_text"),
            "content_text": features.get("content_text"),
            "events": row.get("events", []),
            "events_count": row.get("events_count"),
            "gold_session": coverage.get("session_id") in gold_sessions,
        })
    metadata = {
        "question_id": native.get("question_id"),
        "question_type": native.get("question_type"),
        "question": native.get("question"),
        "answer_session_ids": native.get("answer_session_ids", []),
        "question_date": native.get("question_date"),
        "query_embedding_input": native.get("query_embedding_input"),
        "retrieval_method": method,
        "score_note": "Similarity scores are unavailable in the native artifact and were not reconstructed.",
        "ranking": ranking,
    }
    write_json(target / "retrieval_full.json", metadata)
    write_json(target / "retrieval_top20.json", {**metadata, "ranking": ranking[:20]})


STAGES = (
    "split_check",
    "pass1_extract",
    "temporal_local_resolve",
    "pass1_tool_followup",
    "pass1_tool_followup_fallback",
    "pass2_classify",
)


def aggregate_construction(root: Path, ids: list[str]) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0, "wall_time_sec": 0.0, "memory_units": 0}
    stages = {name: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0} for name in STAGES}
    gate = {"blocks_total": 0, "gate_on_count": 0, "gate_off_count": 0, "temporal_tool_called_count": 0, "pass1_tool_followup_count": 0}
    for qid in ids:
        summary = first_jsonl(root / "questions" / qid / "question_summary.jsonl")
        construction = summary.get("construction", {})
        for key in ("input_tokens", "output_tokens", "total_tokens", "llm_calls"):
            destination = "calls" if key == "llm_calls" else key
            total[destination] += construction.get(key, 0) or 0
        total["wall_time_sec"] += construction.get("wall_time_sec", 0) or 0
        total["memory_units"] += summary.get("memory_unit_count", 0) or 0
        for name in STAGES:
            stage = construction.get("breakdown", {}).get(name, {}) or {}
            for key in ("calls", "input_tokens", "output_tokens", "total_tokens"):
                stages[name][key] += stage.get(key, 0) or 0
        for key in gate:
            gate[key] += (summary.get("temporal_gate", {}) or {}).get(key, 0) or 0
    total["stage_breakdown"] = stages
    total["temporal_gate"] = gate
    total["questions"] = len(ids)
    return total


def retrieval_metrics(root: Path, ids: list[str]) -> dict:
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    ranks = []
    for qid in ids:
        source = root / "questions" / qid
        native = first_jsonl(source / "retrieval.jsonl")
        memories = {}
        with (source / "final_boxes_content.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    memories[str(row["block_id"])] = row
        gold_sessions = set(native.get("answer_session_ids", []))
        order = native.get("rankings", {}).get("content_event_topic_kw", [])
        rank = None
        for i, block_id in enumerate(order, start=1):
            session = memories.get(str(block_id), {}).get("coverage", {}).get("session_id")
            if session in gold_sessions:
                rank = i
                break
        if rank is not None:
            ranks.append(rank)
            for k in hits:
                if rank <= k:
                    hits[k] += 1
    n = len(ids)
    return {
        "questions": n,
        "hit_at_1": hits[1] / n,
        "hit_at_5": hits[5] / n,
        "hit_at_10": hits[10] / n,
        "hit_at_20": hits[20] / n,
        "hit_counts": {f"hit_at_{k}": v for k, v in hits.items()},
        "mean_gold_rank": mean(ranks) if ranks else None,
        "median_gold_rank": median(ranks) if ranks else None,
        "mrr": mean(1.0 / x for x in ranks) if ranks else 0.0,
        "missing_gold_rank": n - len(ranks),
        "native_similarity_scores_available": False,
    }


def copy_qa(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("qa_summary.json", "qa_results.jsonl", "failures.json"):
        copy_file(source / name, target / name)
    for system in ("b0", "b1", "b2"):
        for repeat in ("repeat_01", "repeat_02", "repeat_03"):
            source_questions = source / system / repeat / "questions"
            if not source_questions.exists():
                continue
            for qdir in sorted(source_questions.iterdir()):
                if not qdir.is_dir():
                    continue
                for name in ("answer.json", "answer_prompt.txt", "judge_prompt.txt"):
                    copy_file(qdir / name, target / system / repeat / "questions" / qdir.name / name)


def build_report(systems: dict, qa: dict) -> str:
    def fmt(value):
        return f"{value:,}" if isinstance(value, int) else f"{value:.4f}" if isinstance(value, float) else str(value)
    lines = [
        "# LongMemEval-S B0 / B1 / B2：50Q 构建、检索与重复 QA 结果",
        "",
        "本报告补充 frozen seed-42 50-question subset 的最新 B2 temporal-gate 构建/检索结果，以及基于冻结 memory/retrieval artifacts 的 B0、B1、B2 三次重复 QA。B1/B2 没有因本次 QA 补充而重新抽取；QA token 不计入 construction cost。",
        "",
        "## 配置与口径",
        "",
        "- Dataset：官方 `longmemeval_s_cleaned.json` 的固定 50Q subset。",
        "- LLM：`gpt-4o-mini`；embedding：`text-embedding-3-small`；temperature：0。",
        "- Retrieval candidate Top-20；最终 QA context Top-10；QA repeats：3 次。",
        "- B0：原始 SA-Mem two-pass；B1：local temporal-resolution path；B2：B0 行为等价的本地 temporal gate。",
        "- provider token 仅来自 API usage；不包含 API key、配置文件或完整 dataset。",
        "",
        "## 50Q construction",
        "",
        "| 指标 | B0 | B1 | B2 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Memory units", "memory_units"), ("Construction input tokens", "input_tokens"), ("Construction output tokens", "output_tokens"), ("Construction total tokens", "total_tokens"), ("Construction LLM calls", "calls")):
        lines.append(f"| {label} | {fmt(systems['b0']['construction'][key])} | {fmt(systems['b1']['construction'][key])} | {fmt(systems['b2']['construction'][key])} |")
    lines += [
        f"| Construction wall time (sum sec) | {systems['b0']['construction']['wall_time_sec']:.1f} | {systems['b1']['construction']['wall_time_sec']:.1f} | {systems['b2']['construction']['wall_time_sec']:.1f} |",
        "",
        "## Retrieval",
        "",
        "| 指标 | B0 | B1 | B2 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Hit@1", "hit_at_1"), ("Hit@5", "hit_at_5"), ("Hit@10", "hit_at_10"), ("Hit@20", "hit_at_20"), ("Mean gold rank", "mean_gold_rank"), ("MRR", "mrr")):
        lines.append(f"| {label} | {systems['b0']['retrieval'][key]:.4f} | {systems['b1']['retrieval'][key]:.4f} | {systems['b2']['retrieval'][key]:.4f} |")
    saving = (1 - systems["b2"]["construction"]["total_tokens"] / systems["b0"]["construction"]["total_tokens"]) * 100
    lines += [
        "",
        f"B2 相对 B0 的 construction total token 节省为 **{saving:.2f}%**。B2 的 native SA-Mem retrieval artifact 只保存有序 memory IDs，没有 provider similarity score；上传的规范化 `score` 明确为 `null`，没有重建或伪造分数。",
        "",
        "## Construction stage breakdown",
        "",
        "| Stage | B0 total | B1 total | B2 total |",
        "|---|---:|---:|---:|",
    ]
    for stage in STAGES:
        lines.append(f"| `{stage}` | {fmt(systems['b0']['construction']['stage_breakdown'][stage]['total_tokens'])} | {fmt(systems['b1']['construction']['stage_breakdown'][stage]['total_tokens'])} | {fmt(systems['b2']['construction']['stage_breakdown'][stage]['total_tokens'])} |")
    lines += [
        "",
        "B2 gate aggregate：`" + json.dumps(systems["b2"]["construction"]["temporal_gate"], ensure_ascii=False) + "`。gate OFF blocks 不调用 temporal tool；gate ON blocks 保留 B0 原始 tool/follow-up 路径。",
        "",
        "## 三次重复 QA",
        "",
        "| System | Repeat 1 | Repeat 2 | Repeat 3 | Mean accuracy | Std | Question consistency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("b0", "B0"), ("b1", "B1"), ("b2", "B2")):
        result = qa["systems"][name]
        reps = result["repeat_results"]
        lines.append(f"| {label} | {reps[0]['correct']}/50 ({reps[0]['accuracy']:.2f}) | {reps[1]['correct']}/50 ({reps[1]['accuracy']:.2f}) | {reps[2]['correct']}/50 ({reps[2]['accuracy']:.2f}) | {result['mean_accuracy']:.4f} | {result['std_accuracy']:.4f} | {result['question_answer_consistency']:.2f} |")
    lines += [
        "",
        "QA 是独立的 generation/judge 重复实验：B0 mean 0.6200，B1 mean 0.5733，B2 mean 0.6267。它们用于 baseline sanity comparison；B1/B2 与 B0 的 QA 调用存在模型生成随机性，不能把三次 repeats 当作新的 construction 实验。Judge tokens、QA generation tokens、embedding tokens 均未计入 construction totals。",
        "",
        "## 文件导航",
        "",
        "- B2 逐题 memory / construction / gate / retrieval：[`50q_artifacts/samem_2p_b2/`](50q_artifacts/samem_2p_b2/)。每个 `questions/<qid>/` 保存 `memories.jsonl`、`construction_calls.jsonl`、`temporal_gate.jsonl`、`retrieval_native.json`、`retrieval_full.json`、`retrieval_top20.json`、summary 和 manifest。",
        "- 三个系统全部 QA 原始可审计文件：[`qa_repeats/`](qa_repeats/)，包括 3 个 repeat 的每题 `answer.json`、`answer_prompt.txt`、`judge_prompt.txt`，以及 `qa_summary.json` / `qa_results.jsonl`。",
        "- 可复现入口与字段说明：[`README.md`](README.md) 和 [`50q_artifacts/README.md`](50q_artifacts/README.md)。",
        "",
        "## 限制与注意事项",
        "",
        "1. B1 50Q construction/retrieval 使用已完成的 B1 v2 server run；本次没有重新抽取。",
        "2. B2 construction/retrieval 使用已完成的 50Q temporal-gate run；其 source manifest 的 `qa_enabled=false`，因此 B2 QA 来自独立 QA repeat tree。",
        "3. 本次提交没有上传完整 LongMemEval dataset、embedding cache、launcher/builder logs、token streams、trace artifacts 或任何凭证。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--b0-root", type=Path, required=True)
    parser.add_argument("--b1-root", type=Path, required=True)
    parser.add_argument("--b2-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    ids = load_ids(repo / "experiments/longmemeval/longmemeval_s_50_ids.txt")

    artifact_root = repo / "experiments/longmemeval/50q_artifacts/samem_2p_b2"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    for qid in ids:
        normalize_b2_question(args.b2_root / "questions" / qid, artifact_root / "questions" / qid)

    qa_dest = repo / "experiments/longmemeval/qa_repeats"
    if qa_dest.exists():
        shutil.rmtree(qa_dest)
    copy_qa(args.qa_root, qa_dest)

    qa_summary = json.loads((args.qa_root / "qa_summary.json").read_text(encoding="utf-8"))
    systems = {}
    for name, root in (("b0", args.b0_root), ("b1", args.b1_root), ("b2", args.b2_root)):
        systems[name] = {"construction": aggregate_construction(root, ids), "retrieval": retrieval_metrics(root, ids)}
    results = {
        "dataset": "LongMemEval-S cleaned",
        "subset": {"file": "longmemeval_s_50_seed42.json", "seed": 42, "questions": len(ids), "question_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
        "models": {"llm": "gpt-4o-mini", "embedding": "text-embedding-3-small", "temperature": 0.0, "retrieval_candidate_top_k": 20, "generation_top_k": 10},
        "qa": qa_summary,
        "systems": systems,
        "notes": [
            "Packaging copied completed server-side artifacts; it made no API calls.",
            "QA repeats use frozen B0/B1/B2 artifacts and are separate from construction tokens.",
            "Judge and QA generation tokens are not included in construction totals.",
        ],
    }
    write_json(repo / "experiments/longmemeval/LONGMEMEVAL_B0_B1_B2_QA_RESULTS.json", results)
    (repo / "experiments/longmemeval/LONGMEMEVAL_B0_B1_B2_QA_REPORT.md").write_text(build_report(systems, qa_summary), encoding="utf-8")
    write_json(qa_dest / "qa_manifest.json", {
        "source": "server-side independent QA repeat run",
        "systems": ["b0", "b1", "b2"],
        "repeats": 3,
        "questions": len(ids),
        "model": "gpt-4o-mini",
        "judge_model": qa_summary.get("config", {}).get("judge_model"),
        "top_k": qa_summary.get("config", {}).get("topk", 10),
        "workers": qa_summary.get("config", {}).get("workers"),
        "construction_tokens_included": False,
        "api_keys_included": False,
    })
    print(json.dumps({"questions": len(ids), "b2": str(artifact_root), "qa": str(qa_dest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
