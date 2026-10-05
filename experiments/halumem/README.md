# HaluMem-Medium B0 / B1 / B2 / B3 实验

本目录记录 SA-Mem 在 HaluMem-Medium 上的时序记忆对照实验设计。实验使用独立
branch 和独立服务器 run root，不覆盖 LongMemEval B0/B1/B2 结果。

## 版本定义

| 版本 | construction 行为 |
|---|---|
| B0 | 原始 two-pass：Pass1 → temporal tool/follow-up → Pass2 |
| B1 | Pass1 输出 `temporal_expressions`，本地 resolver 解析，异常时 fallback 到 B0 |
| B2 | 保持 B0 prompt/schema；本地 high-recall gate 在无 relative time 时禁止 temporal tool |
| B3 | merged/one-pass extraction：一次 LLM 同时输出 topic、keywords、events 和 temporal metadata |

四个版本使用相同的：

```text
LLM: gpt-4o-mini
Embedding: text-embedding-3-small
temperature: 0
retrieval: full ranking，QA context Top-20
```

Trace Weaver、graph enhancement 和其它 retrieval 优化均不启用。

## 数据与 adapter

当前服务器数据统计：

```text
users: 20
sessions: 1,387
messages: 60,146
questions: 3,467
evidence entries: 4,651
questions without evidence: 828
```

原始 HaluMem 文件不会被修改。实验脚本会生成：

```text
halumem-b0-b3-runs/metadata/halumem_medium_combined.json
halumem-b0-b3-runs/metadata/processed_halumem_iso/
```

其中 ISO 副本只规范化 session start/end metadata，解决 HaluMem 原始日期格式与
temporal resolver 输入格式不一致的问题；对话文本保持不变。

## 自动运行方式

```bash
bash scripts/run_phase2_orchestrator.sh
```

orchestrator 会：

1. 等待 LongMemEval B2 50Q 完成；
2. 自动运行 LME B0/B1/B2 QA，每个问题重复 3 次；
3. 自动运行 HaluMem B0/B1/B2/B3 construction 和 retrieval；
4. 自动运行 HaluMem 四个版本 QA，每个问题重复 3 次；
5. 保存 checkpoint、失败列表、逐问题结果和 aggregate summary。

实验完成后由独立 watcher 自动执行：

```bash
bash scripts/run_phase2_aggregate_when_ready.sh
```

最终汇总位置：

```text
/workspace/SA-mem/halumem-b0-b3-runs/reports/PHASE2_REPORT.md
/workspace/SA-mem/halumem-b0-b3-runs/reports/PHASE2_RESULTS.json
```

对应生成器为 `scripts/aggregate_phase2_results.py`。它只读取 LME 与
HaluMem 的已完成 artifacts，不覆盖任何 B0/B1/B2 construction、retrieval 或
QA 原始文件。

LME QA 使用 Top-10 context；HaluMem 使用当前 HaluMem baseline 的 Top-20 context。
QA repeat 只重复 generation/judge，不重复 construction。

## 服务器 run 目录

```text
/workspace/SA-mem/halumem-b0-b3-runs/
├── metadata/
├── variants/
│   ├── halumem_b0/
│   ├── halumem_b1/
│   ├── halumem_b2/
│   └── halumem_b3/
├── lme_qa_repeats/
├── qa/
└── status/
```

每个 HaluMem variant 的 construction/retrieval 目录包含：

```text
final_boxes_content.jsonl
construction_calls.jsonl
simple_retrieval.jsonl
simple_retrieval.csv
build_stats.jsonl
token_stream.jsonl
```

QA repeat 目录保存实际 QA prompt、judge prompt、答案、provider usage 和 retry
信息。API key、配置文件、完整原始数据和 embedding cache 不提交 Git。

## 主要指标

- construction input/output/total tokens、LLM calls、wall time；
- B0/B1/B2/B3 各阶段 breakdown；
- gold evidence preservation / extraction recall；
- evidence Hit@1/5/10/20、MRR、gold rank；
- HaluMem `Correct / Hallucination / Omission / Error`；
- easy/medium/hard、evidence/no-evidence、Persona/Event/Relationship 分组；
- QA repeat 的 mean accuracy、standard deviation 和逐问题一致率。

实验运行完成后，会在本目录补充最终中文报告和机器可读 aggregate JSON；当前
README 只描述固定配置和自动化入口。
