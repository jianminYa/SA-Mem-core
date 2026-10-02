# First-question intermediate artifacts

These files are the auditable intermediate outputs for exactly one question:
`e47becba` (`What degree did I graduate with?`). They are copied from the
completed isolated runs without changing the memory or retrieval output.

## File conventions

- `membox_boxes.jsonl`: one normalized MemBox box per JSONL line. The record
  retains the native `features.content_text`, `features.topic_kw_text`,
  `features.events`, coverage, and the saved embedding representation text.
- `samem_memblocks.jsonl`: one normalized SA-Mem MemBlock per JSONL line. The
  record retains native content, coverage, event descriptions, event types,
  and temporal metadata.
- `*_retrieval_full.json`: the complete ranking, not only the displayed
  Top-20 slice. Scores are the saved cosine-similarity scores.
- `*_generation_top10.json`: the exact ten ranking records selected for QA.
- `*_answer_prompt.txt`: the final QA prompt sent to `gpt-4o-mini`.
- `*_construction_calls.jsonl`: read-only call accounting with stage, model,
  provider token usage, latency, success, and retry index. Request bodies and
  credentials are intentionally absent.
- `*_run_manifest.json`: non-secret reproducibility metadata.

For the readable evidence path, see the parent
[`FIRST_QUESTION_DIAGNOSTIC.md`](../FIRST_QUESTION_DIAGNOSTIC.md), which links
the source dialogue, gold memory, ranking table, and final context.

The full LongMemEval dataset, raw 50-question outputs, vector caches, `.env`
files, API keys, and authorization headers are not included.
