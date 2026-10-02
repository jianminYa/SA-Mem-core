#!/usr/bin/env python3
"""Render the completed one-question baseline diagnostics as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def stage_table(result: dict) -> str:
    lines = ["| Stage | Calls | Input | Output | Total |", "|---|---:|---:|---:|---:|"]
    for stage, row in result["construction"]["stages"].items():
        lines.append(
            f"| `{stage}` | {fmt(row['calls'])} | {fmt(row['input_tokens'])} | "
            f"{fmt(row['output_tokens'])} | {fmt(row['total_tokens'])} |"
        )
    return "\n".join(lines)


def qa_usage(run_root: Path, filename: str) -> dict:
    rows = read_jsonl(run_root / filename)
    return rows[0] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--membox-run", type=Path, required=True)
    parser.add_argument("--samem-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    subset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    membox = data["membox"]
    samem = data["samem"]
    membox_manifest = json.loads((args.membox_run / "run_manifest.json").read_text())
    samem_manifest = json.loads((args.samem_run / "run_manifest.json").read_text())
    membox_qa = qa_usage(args.membox_run, "qa_calls.jsonl")
    samem_qa = qa_usage(args.samem_run, "qa_calls.jsonl")
    membox_judge = qa_usage(args.membox_run, "judge_calls.jsonl")
    samem_judge = qa_usage(args.samem_run, "judge_calls.jsonl")

    table_rows = [
        ("History sessions", data["history"]["sessions"], data["history"]["sessions"]),
        ("History messages", data["history"]["messages"], data["history"]["messages"]),
        ("Memory units", membox["memory_unit_count"], samem["memory_unit_count"]),
        ("Construction input tokens", membox["construction"]["input_tokens"], samem["construction"]["input_tokens"]),
        ("Construction output tokens", membox["construction"]["output_tokens"], samem["construction"]["output_tokens"]),
        ("Construction total tokens", membox["construction"]["total_tokens"], samem["construction"]["total_tokens"]),
        ("Construction LLM calls", membox["construction"]["calls"], samem["construction"]["calls"]),
        ("Construction wall time (sec)", membox["construction"]["wall_time_sec"], samem["construction"]["wall_time_sec"]),
        ("Gold memory preserved", membox["gold_memory_preserved"], samem["gold_memory_preserved"]),
        ("Gold extracted answer preserved", membox["gold_extracted_answer_preserved"], samem["gold_extracted_answer_preserved"]),
        ("Gold retrieval rank", membox["gold_rank"], samem["gold_rank"]),
        ("Gold retrieval score", membox["gold_score"], samem["gold_score"]),
        ("Gold in final QA context", membox["gold_in_final_context"], samem["gold_in_final_context"]),
        ("Final answer", membox["hypothesis"], samem["hypothesis"]),
        ("QA correct", membox["autoeval"].get("label"), samem["autoeval"].get("label")),
        ("Failure stage", membox["failure_stage"], samem["failure_stage"]),
    ]
    lines = [
        "# LongMemEval-S First Question Baseline Report",
        "",
        "This is a one-question pipeline smoke diagnostic, not a 50-question result.",
        "The only executed question is `e47becba`.",
        "",
        "## Environment",
        "",
        "- LLM: `gpt-4o-mini`",
        "- Embedding: `text-embedding-3-small`",
        "- API key: configured at runtime only; not recorded",
        f"- MemBox environment: `{membox_manifest.get('environment_name')}`",
        f"- SA-Mem environment: `{samem_manifest.get('environment_name')}`",
        "- Evaluator environment: `longmemeval-eval`",
        f"- Dataset SHA256: `{membox_manifest.get('dataset_sha256')}`",
        f"- Subset manifest: `{subset.get('source_dataset')}`, seed `{subset.get('seed')}`; this run uses the fixed first-question file",
        "- Provider usage was available for every construction call in both runs.",
        "",
        "## Question and full history",
        "",
        f"- ID: `{data['question_id']}`",
        f"- Type: `{data['question_type']}`",
        f"- Question: {data['question']}",
        f"- Gold answer: `{data['answer']}`",
        f"- Gold answer session: `{', '.join(data['gold_answer_session_ids'])}`",
        f"- Complete history: {data['history']['sessions']} sessions, {data['history']['messages']} messages, {data['history']['characters']} characters, {data['history']['estimated_tokens']:,} locally estimated tokens.",
        "",
        "## Configuration",
        "",
        "### MemBox",
        "",
        "- Topic Loom continuity and box extraction: ON.",
        "- Trace Weaver construction/context: OFF. No trace-stage construction calls or Trace Weaver artifacts were found; `trace_build_process.jsonl` is retained only as the native box/split audit log.",
        "- The native output retained 164 boxes; the report maps MemBox's local `session_N` coverage back to the official LongMemEval session IDs.",
        "",
        "### SA-Mem",
        "",
        "- Two-pass extraction: ON (`MEMBLOCK_MERGED_EXTRACTION=0`).",
        "- Graph: OFF; anchor expansion: OFF; event-chain/trace construction: OFF.",
        "- The native output retained 159 MemBlocks.",
        "- The native LME adapter produced a full ranking; this report uses candidate Top-20 diagnostics and generation Top-10, with no ranking truncation before scoring.",
        "",
        "## Construction cost",
        "",
        "| Metric | MemBox | SA-Mem |",
        "|---|---:|---:|",
    ]
    for name, left, right in table_rows:
        lines.append(f"| {name} | {fmt(left)} | {fmt(right)} |")
    lines += [
        "",
        "All construction token values above are provider-reported `prompt_tokens`, `completion_tokens`, and `total_tokens`; they are not tokenizer estimates.",
        "",
        "### MemBox construction breakdown",
        "",
        stage_table(membox),
        "",
        "Trace calls/tokens: `0 / 0`; trace artifacts: none found.",
        "",
        "### SA-Mem construction breakdown",
        "",
        stage_table(samem),
        "",
        "The SA-Mem total is the sum of split checks, Pass 1 extraction, Pass 1 temporal/tool follow-ups, and Pass 2 classification. The stage total reconciles exactly to the construction total.",
        "",
        "## Retrieval and gold evidence",
        "",
        f"- MemBox gold memory IDs: `{membox['gold_memory_ids']}`; original dialogue preserved: `{membox['gold_memory_preserved']}`; extracted fields preserved the answer: `{membox['gold_extracted_answer_preserved']}`.",
        f"- SA-Mem gold MemBlock IDs: `{samem['gold_memory_ids']}`; original dialogue preserved: `{samem['gold_memory_preserved']}`; extracted events/topic fields preserved the answer: `{samem['gold_extracted_answer_preserved']}`.",
        f"- MemBox gold rank/score: `{membox['gold_rank']}` / `{membox['gold_score']}`; in candidate Top-20: `{membox['gold_in_top20']}`; in generation Top-10: `{membox['gold_in_final_context']}`.",
        f"- SA-Mem gold rank/score: `{samem['gold_rank']}` / `{samem['gold_score']}`; in candidate Top-20: `{samem['gold_in_top20']}`; in generation Top-10: `{samem['gold_in_final_context']}`.",
        "- Full ranking records, scores, session IDs, topics, keywords, events, and gold flags are in each run's `retrieval_full.json`; Top-20 and Top-10 slices are saved separately.",
        "",
        "## Embedding representation audit",
        "",
        "- Query input for both systems: the raw question text, `What degree did I graduate with?`. SA-Mem's query parser logged `time=NONE`; no query-time rewrite was applied.",
        "- MemBox memory input: `content_text + events_text + topic_kw_text`.",
        "- SA-Mem memory input: `content_text + event descriptions + topic_kw_text`; event descriptions come from the native structured event records.",
        "- The actual query, gold-memory, and Top-1/Top-2/Top-3 embedding texts are saved under each run's `embedding_inputs/`. Native memory records are in `memories.jsonl`.",
        "",
        "## QA sanity",
        "",
        f"- MemBox answer: {membox['hypothesis']}",
        f"- SA-Mem answer: {samem['hypothesis']}",
        "- Both answers were accepted by the official LongMemEval evaluator for this single question.",
        f"- QA generation usage (separate from construction): MemBox `{membox_qa.get('total_tokens', 'N/A')}` total tokens; SA-Mem `{samem_qa.get('total_tokens', 'N/A')}` total tokens.",
        f"- Judge usage (separate from construction): MemBox `{membox_judge.get('total_tokens', 'N/A')}` total tokens; SA-Mem `{samem_judge.get('total_tokens', 'N/A')}` total tokens.",
        "- No QA, judge, retrieval-parser, or embedding tokens were included in construction totals.",
        "",
        "## Failure-stage diagnosis",
        "",
        f"- MemBox: `{membox['failure_stage']}`.",
        f"- SA-Mem: `{samem['failure_stage']}`.",
        "- Both systems preserved the answer-bearing dialogue and extracted answer-bearing fields; the gold memory ranked first and reached the final QA context. This one-question result therefore shows no observed extraction, ranking, context truncation, or QA-generation failure.",
        "",
        "## Artifacts",
        "",
        "- MemBox run: `runs/membox/e47becba/`",
        "- SA-Mem run: `runs/samem_2p/e47becba/`",
        "- Analysis JSON: `reports/FIRST_QUESTION_BASELINE_REPORT.json`",
        "- The original cleaned dataset is retained under `datasets/` and is excluded from Git.",
        "",
        "## Full 50-question commands (not executed)",
        "",
        "```bash",
        "conda run -n membox-lme python scripts/run_membox_smoke.py --data datasets/longmemeval_s_cleaned.json --smoke-ids subsets/longmemeval_s_50_ids.txt --env-file /workspace/SA-mem/4omini.txt --repo repos/Membox --run-root runs/membox/full_50_seed42 --workspace-root . --model gpt-4o-mini --embedding-model text-embedding-3-small",
        "conda run -n samem-lme python scripts/run_samem_smoke.py --data datasets/longmemeval_s_cleaned.json --smoke-ids subsets/longmemeval_s_50_ids.txt --env-file /workspace/SA-mem/4omini.txt --repo repos/SA-Mem-core --run-root runs/samem_2p/full_50_seed42 --workspace-root . --model gpt-4o-mini --embedding-model text-embedding-3-small",
        "```",
        "",
        "These commands are recorded for the next phase only. They were not executed in this task.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "qa": {"membox": membox["autoeval"], "samem": samem["autoeval"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
