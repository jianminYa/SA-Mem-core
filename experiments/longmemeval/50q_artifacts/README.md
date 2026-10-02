# 50-question per-question artifacts

This directory contains the deduplicated audit files for all 50 questions in
the frozen seed-42 subset. It intentionally excludes native duplicate box
files, vector caches, builder logs, token streams, and trace artifacts.

## Layout

```text
membox/questions/<question_id>/...
samem_2p/questions/<question_id>/...
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

The uploaded bundle is approximately 366.6 MiB. The complete dataset, `.env`
files, API keys, embedding caches, and raw logs are not included.
