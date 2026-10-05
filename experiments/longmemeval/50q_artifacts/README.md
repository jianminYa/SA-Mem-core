# 50-question per-question artifacts

This directory contains the deduplicated audit files for all 50 questions in
the frozen seed-42 subset. It intentionally excludes native duplicate box
files, vector caches, builder logs, token streams, and trace artifacts.

## Layout

```text
membox/questions/<question_id>/...
samem_2p/questions/<question_id>/...
samem_2p_b2/questions/<question_id>/...
```

Each question directory contains:

- `memories.jsonl`: every constructed memory unit. MemBox records have box
  fields; SA-Mem records have MemBlock fields. The `coverage.session_id`
  identifies the source conversation/session.
- `retrieval_full.json`: the complete query ranking. The `rankings` array
  contains memory ID, rank, similarity score, session ID, topic, keywords,
  events, and gold-evidence flags.
- `retrieval_top20.json`: the candidate Top-20 slice.
- `generation_top10.json`: the exact Top-10 memory records selected for QA.
- `answer_prompt.txt`: the actual QA prompt sent to `gpt-4o-mini`.
- `answer.json`: generated answer and question metadata.
- `question_summary.jsonl`: compact per-question construction, retrieval, and
  QA summary.
- `run_manifest.json`: non-secret configuration and provenance.
- `construction_calls.jsonl`: per-call stage and provider token accounting;
  request bodies, API keys, and authorization headers are not stored.

For the native retrieval artifact, the ordered content IDs are also available
under `retrieval.jsonl` in the server-side run directory. In the uploaded
normalized artifact, use:

```text
retrieval_full.json -> rankings[*].memory_id
```

and use `rank`, `score`, and `session_id` alongside each ID.

The `samem_2p_b2` tree is the completed 50-question temporal-gate run. It uses
the same normalized memory/retrieval convention, and additionally contains
`temporal_gate.jsonl`, `retrieval_native.json`, `question_status.json`, and
`retrieval_timings.jsonl`. The gate log records whether each block stayed on
the original B0 temporal-tool path or used the local zero-token gate.

The B2 native retrieval output stores an ordered list of block IDs but does
not expose provider similarity values. Therefore the uploaded B2
`retrieval_full.json` and `retrieval_top20.json` use `score: null` and state
the score provenance explicitly; no similarity score was estimated or
fabricated. The exact native object is preserved in `retrieval_native.json`.

Repeated QA is separate from construction artifacts and is available at
`../qa_repeats/`. It contains B0/B1/B2 × three repeats × 50 questions, with
the final answer, QA prompt, judge prompt, and aggregate JSONL results.

The original uploaded B0/MemBox bundle is approximately 366.6 MiB. The new
B2 normalized bundle is approximately 98 MiB and the repeated-QA bundle is
approximately 25 MiB. The complete dataset, `.env` files, API keys, embedding
caches, launcher logs, token streams, and trace artifacts are not included.
