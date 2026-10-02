# LongMemEval baseline artifacts

This branch is the immutable first-question diagnostic snapshot for
`e47becba`:

```text
Question: What degree did I graduate with?
Gold: Business Administration
LLM: gpt-4o-mini
Embedding: text-embedding-3-small
```

## Where to look

Start with the human-readable report:

- [`FIRST_QUESTION_BASELINE_REPORT.md`](FIRST_QUESTION_BASELINE_REPORT.md)
- [`first_question/FIRST_QUESTION_DIAGNOSTIC.md`](first_question/FIRST_QUESTION_DIAGNOSTIC.md)

The diagnostic chain is deliberately visible:

```text
source dialogue
  -> extracted boxes / MemBlocks
  -> full embedding retrieval ranking
  -> candidate Top-20
  -> generation Top-10
  -> exact QA prompt
  -> answer and evaluator result
```

The complete single-question intermediate artifacts are under
[`first_question/artifacts/`](first_question/artifacts/):

| Artifact | Contents |
|---|---|
| `membox_boxes.jsonl` | All MemBox memory boxes for the question, including coverage, original dialogue, topic/keywords, events, and embedding representation text. |
| `samem_memblocks.jsonl` | All SA-Mem MemBlocks, including coverage, original dialogue, events, event descriptions, and temporal metadata. |
| `membox_retrieval_full.json` | Full MemBox ranking with IDs, scores, sessions, topics, events, and gold flags. |
| `samem_retrieval_full.json` | Full SA-Mem ranking with IDs, scores, sessions, topics, events, and gold flags. |
| `membox_generation_top10.json` / `samem_generation_top10.json` | The exact Top-10 records passed to QA. |
| `membox_answer_prompt.txt` / `samem_answer_prompt.txt` | The final prompts sent to `gpt-4o-mini`. |
| `*_answer.json` / `*_qa_results.json` | Generated answer and official evaluator result. |
| `*_construction_calls.jsonl` | Sidecar token/latency records for every construction call; prompts and credentials are not stored. |
| `*_run_manifest.json` | Non-secret model, dataset, environment, and pipeline configuration. |

The compact, readable Top-20 files are also available directly as
[`membox_retrieval_top20.json`](first_question/membox_retrieval_top20.json)
and [`samem_retrieval_top20.json`](first_question/samem_retrieval_top20.json).

## Pipeline configuration

- MemBox Topic Loom and box extraction: enabled.
- MemBox Trace Weaver construction/context: disabled.
- SA-Mem native two-pass extraction: enabled.
- `MEMBLOCK_MERGED_EXTRACTION=0`.
- Graph, anchor expansion, and event-chain/trace construction: disabled.
- Construction token usage comes from provider-reported usage and is kept
  separate from QA and judge usage.

## Scope and exclusions

This repository contains the one-question audit package and small analysis
artifacts. It does not contain the complete LongMemEval dataset, the 50Q raw
run directories, embedding caches, `.env` files, API keys, or authorization
headers. The 50-question aggregate branch is
[`longmemeval-50q-baseline`](https://github.com/jianminYa/SA-Mem-core/tree/longmemeval-50q-baseline).
