# LongMemEval-S First Question Baseline Report

This is a one-question pipeline smoke diagnostic, not a 50-question result.
The only executed question is `e47becba`.

## Environment

- LLM: `gpt-4o-mini`
- Embedding: `text-embedding-3-small`
- API key: configured at runtime only; not recorded
- MemBox environment: `membox-lme`
- SA-Mem environment: `samem-lme`
- Evaluator environment: `longmemeval-eval`
- Dataset SHA256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Subset manifest: `longmemeval_s_cleaned.json`, seed `42`; this run uses the fixed first-question file
- Provider usage was available for every construction call in both runs.

## Question and full history

- ID: `e47becba`
- Type: `single-session-user`
- Question: What degree did I graduate with?
- Gold answer: `Business Administration`
- Gold answer session: `answer_280352e9`
- Complete history: 53 sessions, 550 messages, 485494 characters, 102,177 locally estimated tokens.

## Configuration

### MemBox

- Topic Loom continuity and box extraction: ON.
- Trace Weaver construction/context: OFF. No trace-stage construction calls or Trace Weaver artifacts were found; `trace_build_process.jsonl` is retained only as the native box/split audit log.
- The native output retained 164 boxes; the report maps MemBox's local `session_N` coverage back to the official LongMemEval session IDs.

### SA-Mem

- Two-pass extraction: ON (`MEMBLOCK_MERGED_EXTRACTION=0`).
- Graph: OFF; anchor expansion: OFF; event-chain/trace construction: OFF.
- The native output retained 159 MemBlocks.
- The native LME adapter produced a full ranking; this report uses candidate Top-20 diagnostics and generation Top-10, with no ranking truncation before scoring.

## Construction cost

| Metric | MemBox | SA-Mem |
|---|---:|---:|
| History sessions | 53 | 53 |
| History messages | 550 | 550 |
| Memory units | 164 | 159 |
| Construction input tokens | 377,622 | 842,329 |
| Construction output tokens | 26,109 | 98,760 |
| Construction total tokens | 403,731 | 941,089 |
| Construction LLM calls | 501 | 815 |
| Construction wall time (sec) | 892.083769 | 2220.793856 |
| Gold memory preserved | yes | yes |
| Gold extracted answer preserved | yes | yes |
| Gold retrieval rank | 1 | 1 |
| Gold retrieval score | 0.259557 | 0.291137 |
| Gold in final QA context | yes | yes |
| Final answer | You graduated with a degree in Business Administration. | You graduated with a degree in Business Administration. |
| QA correct | yes | yes |
| Failure stage | none | none |

All construction token values above are provider-reported `prompt_tokens`, `completion_tokens`, and `total_tokens`; they are not tokenizer estimates.

### MemBox construction breakdown

| Stage | Calls | Input | Output | Total |
|---|---:|---:|---:|---:|
| `box_extraction` | 164 | 188,693 | 25,445 | 214,138 |
| `topic_continuity` | 337 | 188,929 | 664 | 189,593 |

Trace calls/tokens: `0 / 0`; trace artifacts: none found.

### SA-Mem construction breakdown

| Stage | Calls | Input | Output | Total |
|---|---:|---:|---:|---:|
| `pass1_extract` | 159 | 254,176 | 18,373 | 272,549 |
| `pass1_tool_followup` | 159 | 275,980 | 26,686 | 302,666 |
| `pass2_classify` | 155 | 121,306 | 53,044 | 174,350 |
| `split_check` | 342 | 190,867 | 657 | 191,524 |

The SA-Mem total is the sum of split checks, Pass 1 extraction, Pass 1 temporal/tool follow-ups, and Pass 2 classification. The stage total reconciles exactly to the construction total.

## Retrieval and gold evidence

- MemBox gold memory IDs: `[157, 156, 158, 155, 159]`; original dialogue preserved: `True`; extracted fields preserved the answer: `True`.
- SA-Mem gold MemBlock IDs: `[152, 151, 153, 150, 154]`; original dialogue preserved: `True`; extracted events/topic fields preserved the answer: `True`.
- MemBox gold rank/score: `1` / `0.2595568534544348`; in candidate Top-20: `True`; in generation Top-10: `True`.
- SA-Mem gold rank/score: `1` / `0.2911372022324682`; in candidate Top-20: `True`; in generation Top-10: `True`.
- Full ranking records, scores, session IDs, topics, keywords, events, and gold flags are in each run's `retrieval_full.json`; Top-20 and Top-10 slices are saved separately.

## Embedding representation audit

- Query input for both systems: the raw question text, `What degree did I graduate with?`. SA-Mem's query parser logged `time=NONE`; no query-time rewrite was applied.
- MemBox memory input: `content_text + events_text + topic_kw_text`.
- SA-Mem memory input: `content_text + event descriptions + topic_kw_text`; event descriptions come from the native structured event records.
- The actual query, gold-memory, and Top-1/Top-2/Top-3 embedding texts are saved under each run's `embedding_inputs/`. Native memory records are in `memories.jsonl`.

## QA sanity

- MemBox answer: You graduated with a degree in Business Administration.
- SA-Mem answer: You graduated with a degree in Business Administration.
- Both answers were accepted by the official LongMemEval evaluator for this single question.
- QA generation usage (separate from construction): MemBox `5507` total tokens; SA-Mem `5667` total tokens.
- Judge usage (separate from construction): MemBox `127` total tokens; SA-Mem `127` total tokens.
- No QA, judge, retrieval-parser, or embedding tokens were included in construction totals.

## Failure-stage diagnosis

- MemBox: `none`.
- SA-Mem: `none`.
- Both systems preserved the answer-bearing dialogue and extracted answer-bearing fields; the gold memory ranked first and reached the final QA context. This one-question result therefore shows no observed extraction, ranking, context truncation, or QA-generation failure.

## Artifacts

- MemBox run: `runs/membox/e47becba/`
- SA-Mem run: `runs/samem_2p/e47becba/`
- Analysis JSON: `reports/FIRST_QUESTION_BASELINE_REPORT.json`
- The original cleaned dataset is retained under `datasets/` and is excluded from Git.

## Full 50-question commands (not executed)

```bash
conda run -n membox-lme python scripts/run_membox_smoke.py --data datasets/longmemeval_s_cleaned.json --smoke-ids subsets/longmemeval_s_50_ids.txt --env-file /workspace/SA-mem/4omini.txt --repo repos/Membox --run-root runs/membox/full_50_seed42 --workspace-root . --model gpt-4o-mini --embedding-model text-embedding-3-small
conda run -n samem-lme python scripts/run_samem_smoke.py --data datasets/longmemeval_s_cleaned.json --smoke-ids subsets/longmemeval_s_50_ids.txt --env-file /workspace/SA-mem/4omini.txt --repo repos/SA-Mem-core --run-root runs/samem_2p/full_50_seed42 --workspace-root . --model gpt-4o-mini --embedding-model text-embedding-3-small
```

These commands are recorded for the next phase only. They were not executed in this task.

## Detailed evidence audit

The answer-bearing source session is `answer_280352e9`. A local relevant
excerpt is saved in `first_question/gold_source_dialogue.txt` and reproduced
in `first_question/FIRST_QUESTION_DIAGNOSTIC.md`.

The main MemBox gold memory is box `157` and the main
SA-Mem gold MemBlock is block `152`. Both native records,
including content, topics, events, and temporal metadata, are saved in the
first-question diagnostic directory.

The full embedding ranking is saved in each run's `retrieval_full.json`.
Readable Top-20 tables are in `first_question/FIRST_QUESTION_DIAGNOSTIC.md`.
The gold memory is rank 1 in both systems and is included in both final
generation Top-10 contexts. The exact contexts and prompts remain in the run
directories as `retrieved_context.txt` and `answer_prompt.txt`.

### Construction Cost Breakdown

MemBox construction is 403,731 provider-reported tokens: topic continuity
189,593 and box extraction 214,138. SA-Mem construction is 941,089 tokens:
split check 191,524, Pass 1 272,549, Pass 1 tool follow-up 302,666, and
Pass 2 174,350. The SA-Mem / MemBox ratio is **2.3310x**. The continuity
and split costs are close here; the current difference is mainly the SA-Mem
extraction pipeline. This remains a one-question diagnostic.
