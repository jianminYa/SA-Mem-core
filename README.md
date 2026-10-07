# SA-Mem core source (audited snapshot)

This repository is a sanitized source-only snapshot extracted from the full
SA-Mem archive. It keeps the MemBlock construction, retrieval, generation,
evaluation, and LongMemEval-oriented adapter code while excluding datasets,
generated outputs, logs, caches, model files, the original Git metadata, and
web UI artifacts.

The default construction path is the existing two-pass implementation. The
active CLI imports `build_impl_graph.py`; graph support is disabled by default.
`build_impl.py` is retained as the non-graph/reference implementation. See
[`CORE_CODE_MANIFEST.md`](CORE_CODE_MANIFEST.md) for the traced pipeline and
[`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) for implementation risks and accounting
gaps.

No API key is stored here. Set credentials through environment variables or a
local untracked `.env` file based on `.env.example`.

## B1 local temporal resolution branch

本分支 `halumem-b1-local-temporal` 在 B0 two-pass construction 基础上加入 B1
local temporal resolution，`main` 分支不包含这些修改。

### B1 做了什么

```text
Pass1 LLM
  → 输出 topic / keywords / mentions / temporal_expressions
  → 本地 resolve_temporal_expression()
  → 将解析日期附加到对应 mention
  → 原有 Pass2
  → 原有 MemBlock schema
```

本地解析成功时不调用 temporal function，也不会产生 Pass1 follow-up LLM call。
如果 JSON、mention 映射、session 日期或本地日期解析不可靠，则恢复 B0 的
原始 `Pass1 → temporal tool → follow-up → Pass2` 路径，并在 construction log
中标记为 `pass1_tool_followup_fallback`。

### 启用方式

```bash
export MEMBLOCK_MERGED_EXTRACTION=0
export MEMBLOCK_LOCAL_TEMPORAL_RESOLUTION=1
export MEMBLOCK_TEMPORAL_GATE_B2=0
```

其中：

- `MEMBLOCK_MERGED_EXTRACTION=0`：保留 two-pass，不启用 B3；
- `MEMBLOCK_LOCAL_TEMPORAL_RESOLUTION=1`：启用 B1；
- `MEMBLOCK_TEMPORAL_GATE_B2=0`：避免把 B1 与 B2 gate 混用。

### 代码变更位置

- `build_impl_graph.py`：B1 分支、时间表达覆盖检查、本地解析和 fallback；
- `build_prompts.py`：B1 Pass1 输出 `temporal_expressions` 的 prompt；
- `memblock_extractor.py`：配置开关、LLM/tool 调用和 construction instrumentation；
- `temporal_resolution_tool.py`：本地日期计算函数及原始 function schema。

可以使用 GitHub 的 compare 页面直接查看本分支相对 `main` 的修改：

`main...halumem-b1-local-temporal`
