# B0 / B1 LongMemEval-S Phase 1 阶段汇报

本文记录当前 SA-Mem LongMemEval-S 实验中 B0 原始 two-pass baseline 与 B1 temporal follow-up 优化的设计、测试方法和 50-question 结果。

## 1. 实验设置

- 数据集：官方 `longmemeval_s_cleaned.json`
- 问题集合：固定 seed-42 的 50 个 question IDs
- LLM：`gpt-4o-mini`
- Embedding：`text-embedding-3-small`
- temperature：`0`
- retrieval candidate：Top-20
- QA generation context：Top-10
- 并行方式：question-level parallelism，4 workers；同一个 question 内的 session 保持原始顺序
- SA-Mem：two-pass，merged extraction 关闭，graph/trace 关闭

本阶段先完成 construction 和 retrieval 对比。B1 的 50Q run 使用 `--skip-qa`，因此这里不把 B1 的 50Q 结果解释为完整端到端 QA 结果。

## 2. B0 与 B1 的流程

### B0：原始 two-pass 路径

```text
raw conversation block
    ↓
Pass 1 LLM：keywords / topic / mentions
    ↓
如果模型发起 temporal function call：
    本地执行 resolve_temporal_expression()
    ↓
第二次 LLM follow-up，读取 tool result 并完成 Pass 1 JSON
    ↓
Pass 2 LLM：event description / type / temporal metadata
    ↓
MemBlock
```

### B1：local temporal resolution 路径

```text
raw conversation block
    ↓
Pass 1 LLM：keywords / topic / mentions / temporal_expressions
    ↓
本地执行现有 resolve_temporal_expression()
    ↓
将绝对日期补回对应 mention
    ↓
Pass 2 LLM
    ↓
MemBlock
```

当 B1 的 JSON 不合法、relative expression 没有映射到 mention、mention index 无效或本地解析不可靠时，B1 会 fallback 到原始 B0 tool-calling 路径。raw dialogue、Pass2 schema、MemBlock schema、retrieval 和 QA prompt 没有因 B1 改写。

## 3. 为什么 follow-up 是 LLM call，但总 call 数仍然接近？

是的，follow-up 是 LLM call，而且在 token instrumentation 中被单独记录。

一次 B0 temporal tool 流程实际包含：

```text
第 1 次 chat completion：Pass1 请求，模型决定调用 tool
本地函数执行：resolve_temporal_expression()，不是 LLM call
第 2 次 chat completion：把 tool result 放回 messages 后的 follow-up
```

因此 B0 的 `pass1_tool_followup` 每一行对应一次真实的第二次 provider chat completion。tool 本身不产生 LLM token，但 follow-up 会产生真实的 prompt/completion/total tokens。

总 LLM call 数接近，主要有四个原因：

1. 总 call 数还包括 `split_check` 和 `pass2_classify`。这两类调用在 B1 中仍然存在，合计约占 B0 全部调用的 61%，没有被 B1 消除。
2. B1 的每个 block 仍然必须先执行一次 Pass1 LLM。local resolver 是零 LLM token 的本地步骤，并不会把 Pass1 本身变成非 LLM 流程。
3. B1 的 fallback block 仍然保留原始路径。它们会经历 B1 的初始 Pass1、fallback 的原始 Pass1，以及原始 follow-up；因此 `pass1_extract` 不只是“成功的一次 Pass1”。
4. B1 中 memory unit 数量和 split/pass2 调用数量也有小幅变化，所以总 calls 的变化不能简单等同于 follow-up calls 的变化。

50Q 的实际 call 统计如下：

| 阶段 | B0 calls | B1 calls | 说明 |
|---|---:|---:|---|
| `split_check` | 15,229 | 15,235 | 两者基本相同 |
| `pass1_extract` | 6,857 | 9,130 | B1 包含 local Pass1 和 fallback 重新执行的原始 Pass1 |
| 原始 `pass1_tool_followup` | 6,850 | 0 | B1 成功 local resolve 的路径不走该阶段 |
| B1 `pass1_tool_followup_fallback` | 0 | 2,277 | B1 fallback 的真实 follow-up |
| `pass2_classify` | 6,733 | 6,835 | 两者都保留 |
| **总 calls** | **35,669** | **33,477** | **减少 2,192，约 6.15%** |

所以，B1 真正减少的是大量 follow-up 的“第二次请求”及其高输入 token 成本，而不是让所有 construction LLM call 消失。token 节省比例高于 call 节省比例，正是因为 follow-up 会重新携带较长的前序上下文和 tool result。

## 4. Phase 1 的 Pass1 输出格式示例

B1 的 Pass1 仍然是一个 JSON-only 输出。与 B0 相比，B1 新增了 `temporal_expressions` 字段，用于把 relative expression 映射回具体 mention。下面是格式示例，字段结构与当前代码一致，内容为说明性示例：

```json
{
  "keywords": [
    "travel",
    "Seattle",
    "conference"
  ],
  "topic": "planned conference travel",
  "mentions": [
    "The user attended a conference in Seattle last Sunday.",
    "The user planned to review the conference notes this week."
  ],
  "temporal_expressions": [
    {
      "mention_index": 0,
      "expression": "last Sunday"
    },
    {
      "mention_index": 1,
      "expression": "this week"
    }
  ]
}
```

其中：

- `keywords` 和 `topic` 直接进入当前 MemBlock 的检索相关字段；
- `mentions` 是 Pass1 抽取的原子事实，必须能够在原始对话中找到依据；
- `mention_index` 是从 0 开始的数组下标；
- `expression` 保留原始 relative wording，不要求 Pass1 自己计算日期；
- B1 本地 resolver 根据该 block 的 `session_end` 计算绝对日期，再将日期补回对应 mention；
- Pass2 继续接收处理后的 mentions，并生成最终 `explicit_mentions`、事件类型和时间元数据。

没有 relative temporal expression 时，B1 仍保持同一 schema，只是返回空列表：

```json
{
  "keywords": ["social media", "analytics", "engagement"],
  "topic": "improving social media analytics",
  "mentions": [
    "The user is thinking of revamping their social media strategy.",
    "Improving social media analytics can help make data-driven decisions."
  ],
  "temporal_expressions": []
}
```

当前运行日志中可以看到同样的实际输出形状，例如 B1 question `94f70d80` 的 `builder.log` 中记录了 `keywords`、`topic`、`mentions` 和 `temporal_expressions` 四个字段。由于原始 server-side builder log 不作为 GitHub artifact 上传，本文只保留 schema 示例，不上传完整 history 或完整日志。

## 5. 50Q construction 结果

| 指标 | B0 原始 | B1 优化 | 变化 |
|---|---:|---:|---:|
| Construction input tokens | 39.19M | 32.03M | -18.27% |
| Construction output tokens | 4.77M | 5.74M | +20.33% |
| Construction total tokens | 43.96M | 37.77M | **-14.08%** |
| Construction LLM calls | 35,669 | 33,477 | -6.15% |
| Memory units | 6,857 | 5,784 | -15.65% |

阶段 token breakdown：

| 阶段 | B0 total tokens | B1 total tokens |
|---|---:|---:|
| `split_check` | 9,484,344 | 9,493,357 |
| `pass1_extract` | 12,498,432 | 14,275,259 |
| `temporal_local_resolve` | 0 | 0 provider tokens |
| 原始 `pass1_tool_followup` | 13,939,283 | 0 |
| B1 `pass1_tool_followup_fallback` | 0 | 4,964,302 |
| `pass2_classify` | 8,034,570 | 9,036,154 |
| **总计** | **43,956,629** | **37,769,072** |

B1 的 fallback follow-up 次数从 B0 的 6,850 降到 2,277，减少约 66.8%。但 B1 为了让 Pass1 同时输出 temporal annotations，Pass1 和 Pass2 的输出 token 增加，所以总节省最终为 14.08%，不是简单按 follow-up token 比例计算。

## 6. Retrieval 结果

50Q 使用相同的 frozen question IDs、embedding model 和 Top-20 candidate retrieval。基于 `answer_session_ids` 的 gold-session 结果如下：

| 指标 | B0 | B1 |
|---|---:|---:|
| Hit@1 | 41/50 | 41/50 |
| Hit@5 | 44/50 | 43/50 |
| Hit@10 | 48/50 | 48/50 |
| Hit@20 | 49/50 | 49/50 |
| Mean gold rank | 2.36 | 2.26 |
| MRR | 0.85319 | 0.85253 |

当前结果说明 B1 的 gold-session retrieval 整体基本保持，但 Hit@5 有 1 个问题的波动，MRR 有很小变化。由于 B1 50Q run 没有重新执行 QA，不能把这组结果直接解释为 QA 等价性证明。

## 7. 当前结论

B1 已经验证了一个明确现象：通过本地 temporal resolution，可以在不改变 retrieval 表示和 MemBlock 总体 schema 的前提下，将 construction total tokens 从 43.96M 降到 37.77M，节省约 14.08%。

但当前 B1 仍有两个限制：

1. B1 改变了 Pass1 输出 schema，导致 Pass1/Pass2 输出 token 增加；
2. B1 50Q 只完成 construction/retrieval 对比，完整 QA 还没有在同一批 50 个问题上重新执行。

因此当前阶段的结论是：

> B1 对 construction cost 有明显优化效果，retrieval 指标大体稳定；但在完整 QA 等价性得到验证前，不能直接将 B1 作为最终替代 B0 的方案。

下一步是继续完成行为更接近 B0 的 B2 temporal gate，并在相同 50Q 上补齐 QA、MemBlock diff 和 failure-case audit。

## 8. 代码和结果位置

- 本文：`experiments/longmemeval/B0_B1_PHASE1_PROGRESS_ZH.md`
- 实验总览：`experiments/longmemeval/README.md`
- B0 50Q baseline：`experiments/longmemeval/LONGMEMEVAL_50Q_BASELINE_REPORT.md`
- B1 8Q pilot：`experiments/longmemeval/TEMPORAL_FOLLOWUP_PILOT.md`
- B2 当前报告：`experiments/longmemeval/TEMPORAL_GATE_B2_REPORT.md`
- frozen question IDs：`experiments/longmemeval/longmemeval_s_50_ids.txt`
- 已上传的 B0 50Q 浏览 artifact：`experiments/longmemeval/50q_artifacts/`

原始 B1 50Q per-question construction/retrieval 目录保留在实验服务器：

```text
/workspace/SA-mem/longmemeval-baselines/runs/samem_2p/temporal_followup_b1_50q_retrieval/questions/<question_id>/
```

其中包含 `construction_calls.jsonl`、`final_boxes_content.jsonl`、retrieval ranking 和 question summary；这些大体积原始运行目录不复制进本次 Git commit。

本报告不包含 API key、配置文件、完整 LongMemEval 数据集、embedding cache 或 authorization headers。
