# LongMemEval-S B0 / B1 / B2：50Q 构建、检索与重复 QA 结果

本报告补充 frozen seed-42 50-question subset 的最新 B2 temporal-gate 构建/检索结果，以及基于冻结 memory/retrieval artifacts 的 B0、B1、B2 三次重复 QA。B1/B2 没有因本次 QA 补充而重新抽取；QA token 不计入 construction cost。

## 配置与口径

- Dataset：官方 `longmemeval_s_cleaned.json` 的固定 50Q subset。
- LLM：`gpt-4o-mini`；embedding：`text-embedding-3-small`；temperature：0。
- Retrieval candidate Top-20；最终 QA context Top-10；QA repeats：3 次。
- B0：原始 SA-Mem two-pass；B1：local temporal-resolution path；B2：B0 行为等价的本地 temporal gate。
- provider token 仅来自 API usage；不包含 API key、配置文件或完整 dataset。

## 50Q construction

| 指标 | B0 | B1 | B2 |
|---|---:|---:|---:|
| Memory units | 6,857 | 5,784 | 6,848 |
| Construction input tokens | 39,188,476 | 32,031,233 | 32,702,025 |
| Construction output tokens | 4,768,153 | 5,737,839 | 3,869,880 |
| Construction total tokens | 43,956,629 | 37,769,072 | 36,571,905 |
| Construction LLM calls | 35,669 | 33,477 | 31,859 |
| Construction wall time (sum sec) | 107577.3 | 104228.3 | 86146.2 |

## Retrieval

| 指标 | B0 | B1 | B2 |
|---|---:|---:|---:|
| Hit@1 | 0.8200 | 0.8200 | 0.8200 |
| Hit@5 | 0.8800 | 0.8600 | 0.8800 |
| Hit@10 | 0.9600 | 0.9600 | 0.9600 |
| Hit@20 | 0.9800 | 0.9800 | 0.9800 |
| Mean gold rank | 2.3600 | 2.2600 | 2.2000 |
| MRR | 0.8532 | 0.8525 | 0.8560 |

B2 相对 B0 的 construction total token 节省为 **16.80%**。B2 的 native SA-Mem retrieval artifact 只保存有序 memory IDs，没有 provider similarity score；上传的规范化 `score` 明确为 `null`，没有重建或伪造分数。

## Construction stage breakdown

| Stage | B0 total | B1 total | B2 total |
|---|---:|---:|---:|
| `split_check` | 9,484,344 | 9,493,357 | 9,487,622 |
| `pass1_extract` | 12,498,432 | 14,275,259 | 12,598,546 |
| `temporal_local_resolve` | 0 | 0 | 0 |
| `pass1_tool_followup` | 13,939,283 | 0 | 6,972,753 |
| `pass1_tool_followup_fallback` | 0 | 4,964,302 | 0 |
| `pass2_classify` | 8,034,570 | 9,036,154 | 7,512,984 |

B2 gate aggregate：`{"blocks_total": 6848, "gate_on_count": 3168, "gate_off_count": 3680, "temporal_tool_called_count": 3167, "pass1_tool_followup_count": 3167}`。gate OFF blocks 不调用 temporal tool；gate ON blocks 保留 B0 原始 tool/follow-up 路径。

## 三次重复 QA

| System | Repeat 1 | Repeat 2 | Repeat 3 | Mean accuracy | Std | Question consistency |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 32/50 (0.64) | 30/50 (0.60) | 31/50 (0.62) | 0.6200 | 0.0163 | 0.96 |
| B1 | 30/50 (0.60) | 27/50 (0.54) | 29/50 (0.58) | 0.5733 | 0.0249 | 0.92 |
| B2 | 31/50 (0.62) | 31/50 (0.62) | 32/50 (0.64) | 0.6267 | 0.0094 | 0.96 |

QA 是独立的 generation/judge 重复实验：B0 mean 0.6200，B1 mean 0.5733，B2 mean 0.6267。它们用于 baseline sanity comparison；B1/B2 与 B0 的 QA 调用存在模型生成随机性，不能把三次 repeats 当作新的 construction 实验。Judge tokens、QA generation tokens、embedding tokens 均未计入 construction totals。

## 文件导航

- B2 逐题 memory / construction / gate / retrieval：[`50q_artifacts/samem_2p_b2/`](50q_artifacts/samem_2p_b2/)。每个 `questions/<qid>/` 保存 `memories.jsonl`、`construction_calls.jsonl`、`temporal_gate.jsonl`、`retrieval_native.json`、`retrieval_full.json`、`retrieval_top20.json`、summary 和 manifest。
- 三个系统全部 QA 原始可审计文件：[`qa_repeats/`](qa_repeats/)，包括 3 个 repeat 的每题 `answer.json`、`answer_prompt.txt`、`judge_prompt.txt`，以及 `qa_summary.json` / `qa_results.jsonl`。
- 可复现入口与字段说明：[`README.md`](README.md) 和 [`50q_artifacts/README.md`](50q_artifacts/README.md)。

## 限制与注意事项

1. B1 50Q construction/retrieval 使用已完成的 B1 v2 server run；本次没有重新抽取。
2. B2 construction/retrieval 使用已完成的 50Q temporal-gate run；其 source manifest 的 `qa_enabled=false`，因此 B2 QA 来自独立 QA repeat tree。
3. 本次提交没有上传完整 LongMemEval dataset、embedding cache、launcher/builder logs、token streams、trace artifacts 或任何凭证。
