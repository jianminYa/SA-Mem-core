# LongMemEval-S 50-Question Baseline Report

This report covers the frozen seed-42 50-question subset. No one-pass extraction, filtering, retrieval optimization, or embedding optimization was used.

## QA results

| System | Accuracy | Questions |
|---|---:|---:|
| MemBox | 0.3600 | 50 |
| SA-Mem | 0.3200 | 50 |

Accuracy by question type:

| Question type | MemBox | SA-Mem |
|---|---:|---:|
| `knowledge-update` | 0.5000 | 0.6250 |
| `multi-session` | 0.0769 | 0.1538 |
| `single-session-assistant` | 0.6000 | 0.6000 |
| `single-session-preference` | 0.0000 | 0.0000 |
| `single-session-user` | 0.8571 | 0.5714 |
| `temporal-reasoning` | 0.2857 | 0.1429 |

The official evaluator used 50 judge calls per system. Judge tokens are stored in each run's `qa_results.json` and are not included in construction cost.

## Construction cost

| Metric | MemBox total | MemBox mean | MemBox median | SA-Mem total | SA-Mem mean | SA-Mem median |
|---|---:|---:|---:|---:|---:|---:|
| Input tokens | 18213112 | 364262.24 | 366260.50 | 39188476 | 783769.52 | 784431.50 |
| Output tokens | 1260289 | 25205.78 | 25093.50 | 4768153 | 95363.06 | 96045.50 |
| Total tokens | 19473401 | 389468.02 | 390992.00 | 43956629 | 879132.58 | 879279.50 |
| LLM calls | 22088 | 441.76 | 445.00 | 35669 | 713.38 | 714.50 |
| Memory units | 6876 | 137.52 | 138.00 | 6857 | 137.14 | 138.50 |

SA-Mem / MemBox construction token ratio: **2.2573x**.

### MemBox stages

| Stage | Calls | Input | Output | Total |
|---|---:|---:|---:|---:|
| `topic_continuity` | 15212 | 9439930 | 28501 | 9468431 |
| `box_extraction` | 6876 | 8773182 | 1231788 | 10004970 |

### SA-Mem stages

| Stage | Calls | Input | Output | Total |
|---|---:|---:|---:|---:|
| `split_check` | 15229 | 9455886 | 28458 | 9484344 |
| `pass1_extract` | 6857 | 11692190 | 806242 | 12498432 |
| `pass1_tool_followup` | 6850 | 12648636 | 1290647 | 13939283 |
| `pass2_classify` | 6733 | 5391764 | 2642806 | 8034570 |

## Retrieval evidence

| System | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|
| MemBox | 0.8800 | 0.9600 | 1.0000 | 1.0000 | 0.9152 |
| SA-Mem | 0.8200 | 0.8800 | 0.9600 | 0.9800 | 0.8532 |

## Normalized construction metrics

| Metric | MemBox | SA-Mem |
|---|---:|---:|
| Tokens / history token | 3.796733 | 8.570233 |
| Tokens / message | 799.93 | 1805.65 |
| Tokens / memory unit | 2832.08 | 6410.48 |
| LLM calls / question | 441.76 | 713.38 |

## Parallel execution and failures

- Question workers: MemBox `4`, SA-Mem `4`.
- SA-Mem parallel wall time: `16817.631808` seconds (`4` workers, question-level parallelism).
- MemBox's recorded `6.364938` seconds is a resume/reconciliation manifest duration, not the original full compute duration; it must not be compared as end-to-end wall time. The comparable per-question construction wall-time sums are MemBox `42891.860329` seconds and SA-Mem `107577.297619` seconds.
- Retried calls: MemBox `13`; SA-Mem `0`.
- Question-level retries: MemBox `0`; SA-Mem `9` (all resumed questions completed successfully).
- Permanently failed construction calls: MemBox `0`; SA-Mem `0`.
- Failure-case JSON files are under `reports/failure_cases/`; question-level raw metrics are in `reports/question_level_results.csv`.

## Comparison caveats

- Both construction and QA generation used `gpt-4o-mini`; embeddings used `text-embedding-3-small`.
- The official LongMemEval evaluator used its protocol judge model `gpt-4o-2024-08-06`, separately from QA generation and excluded from construction cost.
- MemBox was run in Topic Loom-only mode with Trace Weaver construction disabled. SA-Mem used two-pass extraction with merged extraction and graph disabled.
- The subset contains 50 frozen IDs from the cleaned dataset. These results are a baseline run, not a claim about general performance beyond this subset.

## Reproducibility artifacts

- Per-question runs: `runs/membox/full_50_seed42/questions/<question_id>/` and `runs/samem_2p/full_50_seed42/questions/<question_id>/`.
- Each completed question has `construction_calls.jsonl`, `memories.jsonl`, `retrieval_full.json`, `retrieval_top20.json`, `generation_top10.json`, `answer_prompt.txt`, and `question_status.json`.
- Full dataset and embedding caches remain local and are not committed.
