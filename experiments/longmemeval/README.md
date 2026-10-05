# LongMemEval-S B0 / B1 / B2 experiments

This branch is `longmemeval-b1-b2-50q-retrieval`. It contains the Chinese
phase-1 progress note plus the B2 implementation and audit. It is based on
the completed phase-1 baseline branch `longmemeval-50q-baseline` and keeps the
B0 baseline, B1 local-resolution experiment, and the single-question snapshot
`membox/samem-lme-firstquestion` conceptually separate.

中文阶段汇报（包含 B1 call 口径、Pass1 输出格式、50Q construction 和
retrieval 指标）：

- [`B0_B1_PHASE1_PROGRESS_ZH.md`](B0_B1_PHASE1_PROGRESS_ZH.md)

The original 50-question baseline remains the B0 reference. This branch now
also contains the completed 50-question B2 construction/retrieval artifacts
and the independent three-repeat QA results for B0, B1, and B2.

最新统一结果见：

- [`LONGMEMEVAL_B0_B1_B2_QA_REPORT.md`](LONGMEMEVAL_B0_B1_B2_QA_REPORT.md)
- [`LONGMEMEVAL_B0_B1_B2_QA_RESULTS.json`](LONGMEMEVAL_B0_B1_B2_QA_RESULTS.json)

## B2 temporal gate

本 README 主要说明代码和 artifact 位置；B0/B1 当前阶段的详细中文解释请
先阅读上面的 [`B0_B1_PHASE1_PROGRESS_ZH.md`](B0_B1_PHASE1_PROGRESS_ZH.md)。

B2 keeps the B0 Pass1 prompt, function schema, mentions schema, temporal
resolver, Pass2 prompt, MemBlock schema, retrieval, embedding, and QA path.
The only opt-in change is a local high-recall gate before Pass1:

```text
raw block -> local needs_temporal_tool()
  no relative-time expression -> original tool schema + tool_choice=none
  uncertain/relative expression -> original B0 tool + follow-up path
-> original Pass2
```

The gate is enabled only with `--temporal-gate-b2` in the experiment runner.
It is disabled by default. Gate decisions and whether a tool/follow-up was
actually observed are written to `temporal_gate.jsonl`; provider usage for
construction remains in `construction_calls.jsonl`. The detector is
high-recall and intentionally accepts false positives. The measured v2 pilot
used the expanded `high_recall_local_regex_v2` detector.

The committed B2 audit files are:

- [`TEMPORAL_GATE_B2_REPORT.md`](TEMPORAL_GATE_B2_REPORT.md)
- [`TEMPORAL_GATE_B2_RESULTS.json`](TEMPORAL_GATE_B2_RESULTS.json)
- [`scripts/analyze_temporal_gate_b2.py`](../../scripts/analyze_temporal_gate_b2.py)

The complete server-side B2 artifacts are under:

```text
runs/samem_2p/temporal_gate_b2_8q_v2/questions/<question_id>/
├── final_boxes_content.jsonl
├── construction_calls.jsonl
├── temporal_gate.jsonl
├── retrieval_full.json
├── retrieval_top20.json
├── generation_top10.json
├── answer_prompt.txt
├── answer.json
└── run_manifest.json
```

`final_boxes_content.jsonl` is the constructed MemBlock output;
`retrieval_full.json` is the complete ranking with memory IDs, scores,
session IDs, raw dialogue and embedding representation text;
`retrieval_top20.json` and `generation_top10.json` are the candidate and QA
context slices. The B2 run directory is intentionally not committed because
it contains large raw per-question artifacts.

本 branch 现在上传了一个去除日志、token stream、trace 和缓存后的可审计副本：

```text
50q_artifacts/samem_2p_b2/questions/<question_id>/
├── memories.jsonl
├── construction_calls.jsonl
├── temporal_gate.jsonl
├── retrieval_native.json
├── retrieval_full.json
├── retrieval_top20.json
├── question_summary.jsonl
├── question_status.json
├── run_manifest.json
└── retrieval_timings.jsonl
```

其中 `memories.jsonl` 是 B2 实际生成的全部 MemBlock；`retrieval_native.json`
保留原始 SA-Mem retrieval object；`retrieval_full.json` 是按 native ranking
展开的可读版本。当前 native SA-Mem artifact 没有保存 similarity score，
所以规范化结果中的 `score` 为 `null`，不会伪造分数。字段和查看方式见
[`50q_artifacts/README.md`](50q_artifacts/README.md)。

## Experiment definition

```text
Dataset: longmemeval_s_cleaned.json
Subset: seed 42, 50 fixed question IDs
LLM: gpt-4o-mini
Embedding: text-embedding-3-small
Question workers: 4
Parallelism: question-level only
Temperature: 0.0
Retrieval candidate Top-K: 20
Generation context Top-K: 10
QA model: gpt-4o-mini
Evaluator/judge model: gpt-4o-2024-08-06
MemBox environment: membox-lme
SA-Mem environment: samem-lme
```

MemBox uses Topic Loom / box construction without Trace Weaver. SA-Mem B0 uses
the native two-pass MemBlock construction with:

```text
split_check -> pass1_extract -> pass1_tool_followup -> pass2_classify
```

For SA-Mem, merged extraction and graph/trace construction are disabled.
Construction token totals use provider-reported usage. QA generation, judge,
retrieval-parser, and embedding calls are not included in construction cost.

## B1 temporal follow-up switch

The optimization is off by default. Enable it only with:

```bash
--temporal-followup-opt
```

The B1 path is:

```text
Pass1 JSON (mentions + temporal_expressions)
    -> existing local resolve_temporal_expression()
    -> Pass2 classification
```

When a mention contains an unmapped relative expression, an unsupported
expression, invalid mention index, or invalid observation time, B1 falls back
to the original function-calling path. Fallback follow-ups are recorded as
`pass1_tool_followup_fallback`; local resolution is recorded as
`temporal_local_resolve` with zero provider LLM tokens. Raw dialogue and the
MemBlock schema are unchanged.

The 8-question pilot IDs are in
[`temporal_followup_pilot_8_ids.txt`](temporal_followup_pilot_8_ids.txt).
The final measured pilot uses the server-side run root
`runs/samem_2p/temporal_followup_b1_8q_parallel_v2/`; its per-question raw
artifacts are intentionally kept off GitHub because they are large. The
auditable committed outputs are:

- [`TEMPORAL_FOLLOWUP_PILOT.md`](TEMPORAL_FOLLOWUP_PILOT.md)
- [`TEMPORAL_FOLLOWUP_PILOT_RESULTS.json`](TEMPORAL_FOLLOWUP_PILOT_RESULTS.json)
- [`scripts/analyze_temporal_followup_pilot.py`](../../scripts/analyze_temporal_followup_pilot.py)

The report explicitly separates B0 original follow-ups, B1 fallback
follow-ups, local resolver calls, construction tokens, QA, gold-session
retrieval, and paired relative-time metadata.

## Main results

- [`LONGMEMEVAL_50Q_BASELINE_REPORT.md`](LONGMEMEVAL_50Q_BASELINE_REPORT.md)
- [`LONGMEMEVAL_50Q_BASELINE_RESULTS.json`](LONGMEMEVAL_50Q_BASELINE_RESULTS.json)
- [`question_level_results.csv`](question_level_results.csv)
- [`failure_cases/`](failure_cases/)
- [`RETRIEVAL_AUDIT.md`](RETRIEVAL_AUDIT.md)

The report contains QA accuracy, question-type accuracy, construction token
totals and stage breakdowns, evidence Hit@1/5/10/20, MRR, normalized token
metrics, retry status, and representative failure cases.

## B0 / B1 / B2 repeated QA

三套冻结 memory/retrieval 结果各运行 3 次 QA，并保存每个 question 的实际
答案和 prompt：

```text
qa_repeats/
├── b0/repeat_01..03/questions/<question_id>/
├── b1/repeat_01..03/questions/<question_id>/
└── b2/repeat_01..03/questions/<question_id>/
```

每个 question 目录包含 `answer.json`、`answer_prompt.txt` 和
`judge_prompt.txt`。汇总结果在 `qa_summary.json`、`qa_results.jsonl`；
配置说明在 `qa_manifest.json`。本次 QA 补充不重新执行 memory construction，
也不把 QA/judge token 计入 construction cost。

## Reproducibility metadata

- [`longmemeval_s_50_ids.txt`](longmemeval_s_50_ids.txt): frozen question IDs.
- [`longmemeval_s_50_manifest.json`](longmemeval_s_50_manifest.json): subset
  provenance and distribution.
- `scripts/run_question_parallel.py`: resumable question-level runner.
- `scripts/aggregate_50q.py`: aggregate report generator.
- `scripts/analyze_50q.py`: per-question memory/retrieval diagnostics.
- `scripts/common.py`: shared QA, history-size, and artifact helpers.

The actual 50-question run directories remain on the experiment server at:

```text
runs/membox/full_50_seed42/questions/<question_id>/
runs/samem_2p/full_50_seed42/questions/<question_id>/
```

Each local completed question contains its memory units, construction call
log, full retrieval ranking, Top-20, generation Top-10, QA prompt, answer,
and manifest. These raw per-question directories are intentionally not
committed in the original `runs/` layout. A deduplicated, browsable copy of
the audit files is now included in:

[`50q_artifacts/`](50q_artifacts/)

The layout is:

```text
50q_artifacts/
├── membox/questions/<question_id>/
│   ├── memories.jsonl
│   ├── retrieval_full.json
│   ├── retrieval_top20.json
│   ├── generation_top10.json
│   ├── answer_prompt.txt
│   ├── answer.json
│   ├── question_summary.jsonl
│   ├── run_manifest.json
│   └── construction_calls.jsonl
└── samem_2p/questions/<question_id>/
    └── same artifact set
└── samem_2p_b2/questions/<question_id>/
    └── B2 construction/retrieval audit set
```

`memories.jsonl` records every constructed box/MemBlock for that question.
`retrieval_full.json` records the complete ranked memory IDs and similarity
scores for the query; `retrieval_top20.json` and `generation_top10.json` are
the candidate and QA-context slices. The per-question artifact README gives
the exact field conventions.

## Single-question audit reference

The first question is also retained as a fully inspectable example:

- [`first_question/FIRST_QUESTION_DIAGNOSTIC.md`](first_question/FIRST_QUESTION_DIAGNOSTIC.md)
- [`first_question/artifacts/`](first_question/artifacts/)
- [`first_question/README.md`](first_question/README.md)

Those artifacts show the complete chain from source dialogue to boxes/
MemBlocks, full ranking, Top-10 QA context, prompt, and final answer for
`e47becba`.

## Exclusions

The repository does not contain the complete LongMemEval dataset, API
configuration files, API keys, authorization headers, embedding caches, or
the full raw 50-question output directories.
