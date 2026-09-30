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
