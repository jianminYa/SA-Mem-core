# LongMemEval-S 50-question baseline

This branch is the completed phase-1 baseline for the frozen 50-question
subset. It is separate from the single-question snapshot
`membox/samem-lme-firstquestion`.

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

MemBox uses Topic Loom / box construction without Trace Weaver. SA-Mem uses
the native two-pass MemBlock construction with:

```text
split_check -> pass1_extract -> pass1_tool_followup -> pass2_classify
```

For SA-Mem, merged extraction and graph/trace construction are disabled.
Construction token totals use provider-reported usage. QA generation, judge,
retrieval-parser, and embedding calls are not included in construction cost.

## Main results

- [`LONGMEMEVAL_50Q_BASELINE_REPORT.md`](LONGMEMEVAL_50Q_BASELINE_REPORT.md)
- [`LONGMEMEVAL_50Q_BASELINE_RESULTS.json`](LONGMEMEVAL_50Q_BASELINE_RESULTS.json)
- [`question_level_results.csv`](question_level_results.csv)
- [`failure_cases/`](failure_cases/)
- [`RETRIEVAL_AUDIT.md`](RETRIEVAL_AUDIT.md)

The report contains QA accuracy, question-type accuracy, construction token
totals and stage breakdowns, evidence Hit@1/5/10/20, MRR, normalized token
metrics, retry status, and representative failure cases.

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
