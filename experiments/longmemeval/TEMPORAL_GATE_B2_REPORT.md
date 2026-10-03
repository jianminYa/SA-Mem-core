# Temporal Gate B2 Pilot

B2 keeps the B0 Pass1 prompt, mentions schema, temporal resolver, Pass2 prompt, MemBlock schema, retrieval, embedding, and QA path unchanged. It only applies a local high-recall lexical gate before Pass1: Gate OFF sends the original function schema with `tool_choice=none`; Gate ON uses the original tool + follow-up path.

Pilot questions: 8; run root: `/workspace/SA-mem/longmemeval-baselines/runs/samem_2p/temporal_gate_b2_8q_v2/questions`.
Model configuration: `gpt-4o-mini`, embedding `text-embedding-3-small`, temperature `0.0`; retrieval candidate Top-20 and generation context Top-10. QA labels come from the official LongMemEval evaluator using the isolated evaluator environment.
The B0 side is the previously measured original two-pass pilot; no B0 or B1 artifact was overwritten. No 50-question B2 run was started.

## Detector audit

- Gate records: **1066** blocks; ON **509**, OFF **557**.
- Actual temporal tool calls: **508**; Pass1 follow-ups: **508**.
- Detector candidate blocks: **509**; candidate blocks gated ON: **509** (100.00%).
- The detector is intentionally high-recall; false positives are accepted and are visible as Gate ON blocks without a tool call.

## Detector audit against the prior B1 pilot

- Prior B1 raw candidate blocks: **505**.
- Candidates matched to B2 coverage: **486**; unmatched because segmentation changed: **19**.
- Matched candidates gated ON: **486 / 486** (100.00%); detector false negatives on matched coverage: **0**.

## B0 vs B2 aggregate

| Metric | B0 | B2 |
|---|---:|---:|
| Construction input tokens | 6,227,229 | 5,243,186 |
| Construction output tokens | 755,431 | 618,558 |
| Construction total tokens | 6,982,660 | 5,861,744 |
| Token saving | — | 1,120,916 (16.05%) |
| Construction LLM calls | 5,618 | 5,044 |
| Construction wall time (sum) | 16564.57s | 12709.40s |
| Memory units | 1,069 | 1,066 |
| Hit@1 | 7/8 | 7/8 |
| Hit@5 | 8/8 | 7/8 |
| Hit@10 | 8/8 | 8/8 |
| Hit@20 | 8/8 | 8/8 |
| Mean gold rank | 1.3750 | 1.7500 |
| MRR | 0.9062 | 0.8929 |
| QA correct | 2/8 (25.00%) | 0/8 (0.00%) |

## Construction stage totals

| Stage | B0 calls / tokens | B2 calls / tokens |
|---|---:|---:|
| `split_check` | 2,431 / 1,549,469 | 2,434 / 1,548,016 |
| `pass1_extract` | 1,069 / 1,971,672 | 1,066 / 1,986,234 |
| `pass1_tool_followup` | 1,067 / 2,196,675 | 508 / 1,139,382 |
| `pass2_classify` | 1,051 / 1,264,844 | 1,036 / 1,188,112 |

## MemBlock diff audit

Matched blocks: **1022**; unmatched B0: **46**; unmatched B2: **43**.
Overall output differences: **937 / 1022** (91.68%).
Gate ON differences: **406 / 485** (83.71%).
Gate OFF differences: **531 / 537** (98.88%).

### MemBlock component differences

| Component | All matched | Gate ON | Gate OFF |
|---|---:|---:|---:|
| `topic_keywords` | 665 | 213 | 452 |
| `event_count` | 559 | 207 | 352 |
| `event_descriptions` | 824 | 330 | 494 |
| `event_type` | 712 | 282 | 430 |
| `time_metadata` | 641 | 258 | 383 |
| `temporal_index` | 60 | 33 | 27 |
The comparison uses session coverage `(session_id, start_idx, end_idx)` and compares topic/keywords, event count, descriptions, event type, and time metadata. Unmatched coverage is reported separately rather than silently treated as equal.

## QA failure audit

| Question | Transition | B0 rank / Top-10 | B2 rank / Top-10 | B0 answer | B2 answer |
|---|---|---:|---:|---|---|
| `d682f1a2` | B0_correct_B2_wrong | 1 / [87] | 1 / [87] | You have used at least two different types of food delivery services recently: Domino's Pizza and Uber Eats. Additionally, you mentioned a new service called Fresh Fusion. | The information is unavailable. |
| `gpt4_f420262d` | B0_correct_B2_wrong | 1 / [65] | 1 / [67] | You flew with American Airlines on Valentine's Day. | The information is unavailable. |

QA audit details:

- `d682f1a2`: gold session retained in both Top-10=`True`, gold rank changed=`False`, Top-10 IDs changed=`True`, same gold raw-dialogue hash present=`True`, answer prompt equal=`True`.
- `gpt4_f420262d`: gold session retained in both Top-10=`True`, gold rank changed=`False`, Top-10 IDs changed=`True`, same gold raw-dialogue hash present=`True`, answer prompt equal=`True`.

## Interpretation

B2 is a stricter behavior-equivalence test than B1 because it does not change the Pass1 prompt or mentions schema. It reduces construction cost by 16.05% and removes 559 follow-up calls, while the detector audit found no matched false negatives. However, QA changed from 2/8 to 0/8, Hit@5 changed from 8/8 to 7/8, and 91.68% of coverage-matched MemBlocks differ in at least one audited output component. The two observed QA regressions retained the gold session at rank 1 and had byte-identical answer prompts, so they are not evidence of gold-context loss; they are consistent with changed memory outputs/ranking and/or provider/model run variance.
Decision: do not expand B2 to the frozen 50-question run from this pilot. Token savings are demonstrated, but the current evidence does not support the stronger claim that temporal gating preserves end-to-end behavior. This 8-question pilot is diagnostic and does not establish a dataset-level accuracy conclusion.

Detailed machine-readable output: `TEMPORAL_GATE_B2_RESULTS.json`.
