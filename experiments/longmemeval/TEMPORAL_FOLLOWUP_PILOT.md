# Temporal Follow-up Optimization Pilot

This pilot compares the frozen 8-question SA-Mem B0 original two-pass path with B1 local temporal resolution. The baseline behavior, retrieval, QA prompts, MemBlock schema, model configuration, and embedding model were otherwise unchanged.

## Configuration

- Dataset: `longmemeval_s_cleaned.json`; IDs are a deterministic subset of the frozen seed-42 50Q set.
- LLM: `gpt-4o-mini`; embedding: `text-embedding-3-small`; temperature: `0.0`.
- Measured B0 source commit: `ca4f58d93321d92675b3190ebaa5e16c9e62c78b`; measured B1 source commit: `07d5a389b811ee7e998e3bfd0a8ca9432fd26143`.
- SA-Mem: two-pass enabled, merged extraction off, graph off; retrieval candidate Top-20 and QA generation Top-10.
- B0: original Pass1 function-calling and follow-up.
- B1: Pass1 emits `temporal_expressions`; existing `resolve_temporal_expression()` resolves them locally; invalid/unreliable entries use the original tool path with stage `pass1_tool_followup_fallback`.

## Pilot questions

| Question ID | Type | Relative-time coverage |
|---|---|---|
| `94f70d80` | `single-session-user` | Last weekend, last Saturday, last night, last week, last weekend, this evening, today |
| `1d4da289` | `single-session-assistant` | Today, Yesterday, a month ago, last Saturday, last Sunday, last month, last night, last week, last weekend, today |
| `d682f1a2` | `multi-session` | Today, last Sunday, last month, last week, last weekend, today |
| `41698283` | `knowledge-update` | Today, last month, last week, last weekend, today, yesterday |
| `gpt4_468eb064` | `temporal-reasoning` | Today, last Tuesday, last month, last week, last weekend, today |
| `gpt4_b5700ca0` | `temporal-reasoning` | Today, last week, today |
| `0a34ad58` | `single-session-preference` | Today, last Saturday, last month, last week, today |
| `gpt4_f420262d` | `temporal-reasoning` | TODAY, Today, last Saturday, last Thursday, last Wednesday, last month, last night, last week, last weekend, today, two months ago |

## Aggregate comparison

| Metric | B0 original | B1 optimized |
|---|---:|---:|
| Construction input tokens | 6,227,229 | 4,646,178 |
| Construction output tokens | 755,431 | 884,887 |
| Construction total tokens | 6,982,660 | 5,531,065 |
| Construction LLM calls | 5,618 | 5,024 |
| Construction wall time (sum) | 16564.5700 sec | 16418.6500 sec |
| Memory units | 1,069 | 1,069 |
| Gold-session Hit@10 | 100.00% | 100.00% |
| Mean gold rank | 1.3750 | 1.7500 |
| QA correct | 2/8 (25.00%) | 1/8 (12.50%) |

B1 construction-token saving: **1,451,595 tokens (20.79%)**; B1/B0 ratio: **0.7921**.

## Stage breakdown

| Stage | B0 calls | B0 total | B1 calls | B1 total |
|---|---:|---:|---:|---:|
| `split_check` | 2,431 | 1,549,469 | 2,430 | 1,549,275 |
| `pass1_extract` | 1,069 | 1,971,672 | 1,298 | 2,019,955 |
| `temporal_local_resolve` | 0 | 0 | 58 | 0 |
| `pass1_tool_followup` | 1,067 | 2,196,675 | 0 | 0 |
| `pass1_tool_followup_fallback` | 0 | 0 | 229 | 519,432 |
| `pass2_classify` | 1,051 | 1,264,844 | 1,067 | 1,442,403 |

B1 local resolver successes: **58**; fallback blocks: **229** (instrumentation rows: 458); fallback tool follow-up calls: **229**.

## Per-question results

| ID | Type | B0 total | B1 total | B0 units | B1 units | B0 rank | B1 rank | B0 Hit@10 | B1 Hit@10 | B0 QA | B1 QA |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| `94f70d80` | `single-session-user` | 866,058 | 694,427 | 129 | 128 | 1 | 1 | yes | yes | no | yes |
| `1d4da289` | `single-session-assistant` | 906,767 | 689,802 | 141 | 140 | 1 | 1 | yes | yes | no | no |
| `d682f1a2` | `multi-session` | 876,032 | 689,364 | 138 | 141 | 1 | 1 | yes | yes | yes | no |
| `41698283` | `knowledge-update` | 877,788 | 690,689 | 135 | 138 | 1 | 1 | yes | yes | no | no |
| `gpt4_468eb064` | `temporal-reasoning` | 848,902 | 670,494 | 134 | 131 | 4 | 7 | yes | yes | no | no |
| `gpt4_b5700ca0` | `temporal-reasoning` | 876,889 | 730,854 | 129 | 127 | 1 | 1 | yes | yes | no | no |
| `0a34ad58` | `single-session-preference` | 866,941 | 695,240 | 129 | 128 | 1 | 1 | yes | yes | no | no |
| `gpt4_f420262d` | `temporal-reasoning` | 863,283 | 670,195 | 134 | 136 | 1 | 1 | yes | yes | yes | no |

## Temporal metadata audit

The runtime schema stores event temporal information in `time_metadata`/`temporal_index` for these artifacts rather than always exposing `events.start_time/end_time`. The audit compares the actual available fields for blocks containing explicit relative-time expressions.

- Candidate blocks: **130** across **8** questions.
- Paired relative-time events: **84**; identical time metadata: **73**; different: **11**; unmatched temporal event records: **24**.

Representative audited blocks:

- `94f70d80` `34deeb0c_2` expressions `Last weekend`: paired 0, same 0, different 0.
- `1d4da289` `1e01dbcc_2` expressions `today`: paired 1, same 1, different 0.
- `d682f1a2` `0aa8f6bf` expressions `last month`: paired 1, same 1, different 0.
- `41698283` `0e8ba03b_1` expressions `last month, yesterday`: paired 2, same 2, different 0.
- `gpt4_468eb064` `21ab8a2c_1` expressions `last month`: paired 1, same 1, different 0.
- `gpt4_b5700ca0` `01dc2b54_2` expressions `today`: paired 1, same 0, different 1.
- `0a34ad58` `02f2ab8e_1` expressions `today`: paired 0, same 0, different 0.
- `gpt4_f420262d` `00c7b769_1` expressions `last weekend`: paired 1, same 1, different 0.

## QA and failure-stage interpretation

QA correctness here is a smoke diagnostic only. For each question, extraction is considered preserved when at least one memory covers an answer session; retrieval is considered the limiting stage when the best gold rank is beyond Top-10/Top-20; otherwise a wrong answer is attributed to QA generation. Full per-question details are in `TEMPORAL_FOLLOWUP_PILOT_RESULTS.json` and the local run artifacts.

## Pilot decision

B1 saves **20.79%** construction tokens and keeps gold-session Hit@10 at **100.00%**, but it does not clear the expansion gate: QA accuracy changes from **25.00%** to **12.50%**, mean gold rank changes from **1.3750** to **1.7500**, and **11** of **84** paired relative-time events have different available temporal metadata. Do not expand B1 to the full 50Q run from this pilot; investigate the Pass1 temporal annotation/context handling and QA differences first.

## Artifacts

- B0: `runs/samem_2p/full_50_seed42/questions/<question_id>/`.
- B1: `runs/samem_2p/temporal_followup_b1_8q_parallel_v2/questions/<question_id>/`.
- B1 calls include `temporal_local_resolve` with zero provider tokens and `pass1_tool_followup_fallback` for the original fallback path.
- Exact aggregate JSON: `TEMPORAL_FOLLOWUP_PILOT_RESULTS.json`.
