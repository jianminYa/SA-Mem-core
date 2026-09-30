# SA-Mem source audit

This audit distinguishes behavior that is visible in the current source from
claims in surrounding documentation. It intentionally does not alter the
algorithm.

## Q1. What is the actual two-pass MemBlock construction?

The active route is:

1. `memblock_cli.py` creates a worker and a `MemoryBuilder`.
2. `memblock_extractor.py` imports the builder from `build_impl_graph.py`.
3. `MemoryBuilder._process` first performs message-level continuity checks and
   either appends to the current block or seals it.
4. `MemoryBuilder._seal` sends the accumulated block to
   `TopicClusterManager.process_new_box`.
5. With `MEMBLOCK_MERGED_EXTRACTION=0` and
   `ENABLE_EVENT_CLASSIFICATION=True`, Pass 1 calls
   `PROMPT_DIALOG_EXTRACT` and returns `topic`, `keywords`, and string
   `mentions`; Pass 2 calls `PROMPT_DIALOG_CLASSIFICATION` and returns typed,
   timed `explicit_mentions`.
6. `_seal` normalizes those event objects, writes `observedTime` and optional
   day-level start/end metadata, and writes block-level `coverage` and
   `temporal_index`.

`build_impl.py` contains a parallel non-graph/reference builder. It should not
be treated as the active route unless imports are changed explicitly.

## Q2. Does topic segmentation itself call an LLM?

Yes. It is message-continuity segmentation rather than a separate named topic
classifier. `MemoryBuilder._process` calls
`LLMWorker.check_relation` after the first two messages in a session, using
`PROMPT_MSG_CONTINUATION`. The code treats only a response containing `yes`
as `Yes`; `No` and `Partially Shifted` both seal the existing block.

For a session with `N` messages that remains in one stream, the normal upper
count is approximately `max(N-2, 0)` split-check calls. The exact number can
be lower when a session is short or an earlier boundary changes the message
grouping. The prompt is in `build_prompts.py` and is also duplicated in the
LoCoMo prompt module.

## Q3. Is event-time normalization Pass 2? Are there extra API calls?

Pass 2 is where the typed event object and start/end fields are requested. The
final normalization is local code in `_seal`, not another event-time API call:

- relative dates can be resolved by the local
  `resolve_temporal_expression` function when the provider invokes the tool;
- explicit dates are extracted/normalized locally;
- missing values fall back to the session window and `observedTime`.

Pass 1 enables tool calling. When the provider actually emits the temporal
tool, `LLMWorker.chat_completion` performs the local function execution and a
second remote chat completion to obtain the final JSON. Thus “one Pass 1
call” means one logical worker operation, but it can be two remote requests.
Pass 2 does not enable that tool in the current active graph builder.

## Q4. How many LLM calls can one session trigger?

Let `B` be the number of sealed blocks and `S` the number of eligible
continuity decisions. The ordinary lower-bound logical count is:

```text
S                         # message-level split checks
+ B                       # Pass 1, one per block
+ B                       # Pass 2 for blocks with non-empty mentions
```

`S` is approximately `N-2` for one uninterrupted session with `N` messages,
but changes with short sessions and boundaries. Pass 2 is skipped if Pass 1
returns no mentions or event classification is disabled. Each tool-using
Pass 1 request can add one remote follow-up request. The merged experiment
replaces the Pass 1+Pass 2 pair with one logical merged request per block.

## Q5. Does HTM/graph construction need an LLM?

There is no class or call site named HTM in the retained source. The actual
temporal structure is `TemporalIndex` in
`retrieval/interval_tree_index.py`, with session and event interval trees; it
is local and does not need an LLM.

The graph path is separate and disabled by default (`GraphConfig.enable_graph`
is false). When enabled, graph entity/relation extraction uses
`GraphEntitiesExtractor` with an OpenAI-compatible/Ollama tool caller, and
graph-event embeddings are also requested. Graph expansion in retrieval is
similarly opt-in. These are not required for the default vector + temporal
path.

## Q6. What LLM calls occur during retrieval?

- Baseline `retrieval_impl.py`: no LLM calls; it makes question and block
  embedding requests and cosine-ranks them.
- Enhanced retrieval: `QueryParser.parse` first runs local `FastParser`. It
  makes one direct OpenAI chat request only when the fast parser cannot
  provide a useful intent/time directive. The parser result is cached by
  date/query. The rest of temporal filtering is local interval-tree logic.
- Anchor resolution uses embeddings and local scoring; it does not call an
  LLM.
- Optional graph expansion is database/local traversal after graph data has
  been built; graph construction, not ordinary expansion, is where the graph
  LLM calls occur.

The query parser has its own OpenAI client and bypasses `TokenAnalyzer`, so
retrieval LLM usage is currently invisible to the main token stream.

## Q7. What LLM calls occur during answer generation?

`generate_impl.py:AnswerGenerator._generate_for_ranking` formats the selected
retrieval contexts with `PROMPT_QA_ANSWER` and makes one
`LLMWorker.chat_completion` call per ranking/mode/top-N generation. The
LoCoMo implementation has analogous modes and can multiply calls across those
settings. Trace construction is a separate optional stage and can issue two
additional worker calls per trace operation (`_llm_event_filter` and
`_llm_init_chain`).

The LongMemEval-specific `scripts/generate_lme.py` is a separate direct-client
path: it calls an LLM for the answer and another LLM for the judge when that
evaluation mode is enabled. Those calls are not recorded by the core worker
token logger.

## Q8. Is token usage complete?

No.

What is recorded:

- `LLMWorker` response `prompt_tokens` and `completion_tokens` are written as
  `in` and `out`; stage counters additionally maintain a `total` value when
  the provider returns usage.
- Stage counters track calls and prompt/completion/total for calls routed
  through `TokenAnalyzer`.
- QA calls using `LLMWorker` are logged under a `gen:*` stage.

What is missing or incomplete:

- no `cached_tokens`/prompt-cache details;
- no independent count for the initial tool-call response;
- no embedding call or embedding-token accounting;
- query-parser LLM calls, graph tool calls, and LME answer/judge calls bypass
  the logger;
- no reliable unified `llm_calls` total across all clients;
- no per-call latency and no unified construction/retrieval/QA/total runtime;
- `prompt_tokens_est` is an estimate, not provider usage;
- retrieval timing exists in selected records but is not normalized to the
  requested stage schema.

The central `LLMWorker.chat_completion` path catches failures and returns an
empty result; it does not provide a complete retry/backoff or call-count
ledger. The query-parser client likewise makes a single fallback request. The
separate `scripts/generate_lme.py` wrapper has its own three-attempt retry
loop, so retry behavior is inconsistent across clients. Build-level
parallelism is implemented around input files/sessions in the CLI, not as a
central API concurrency/accounting layer.

Therefore the requested LongMemEval fields cannot be reconstructed completely
from the current outputs without additional instrumentation.

## Q9. Which code is standard SA-Mem versus exploratory?

This is a code-level classification, not a claim that the surrounding paper
text is authoritative:

### Core/default path

- `build_impl_graph.py` default two-pass branch with graph disabled;
- `build_impl.py` as the retained non-graph/reference implementation;
- `build_prompts.py` and the `LLMWorker`/`EmbeddingStore` abstractions;
- `retrieval_impl.py` baseline vector retrieval;
- `retrieval_enhanced.py` temporal-query + interval filtering + vector ranking;
- `generate_impl.py`/`generate_prompts.py` QA generation;
- LoCoMo evaluators and extraction/retrieval metrics.

### Experimental, adapter, or optional paths

- `graph_*` modules and graph expansion;
- `trace_impl.py` and trace/event-chain prompts;
- `MEMBLOCK_MERGED_EXTRACTION` and `PROMPT_DIALOG_EXTRACT_MERGED`;
- `retrieval_lme.py`, `retrieve_stage_lme.py`, and
  `scripts/generate_lme.py` (LongMemEval adapter/evaluation path);
- `_locomo` variants, ablation/comparison scripts, and the one-flush retriever;
- stale/duplicate scripts not included in this snapshot.

The active import matters: the current CLI uses `build_impl_graph.py`, while
some older docs describe a copied or older builder. That mismatch is a
reproducibility risk.

## Q10. What would need to change for two-pass -> one-pass?

No change was made. The existing switch already identifies the smallest code
surface:

1. `memblock_extractor.py:Config.ENABLE_MERGED_EXTRACTION` and the runtime
   `MEMBLOCK_MERGED_EXTRACTION` setting;
2. `build_impl_graph.py:TopicClusterManager.process_new_box` (and the
   reference implementation if it is selected);
3. `build_prompts.py:PROMPT_DIALOG_EXTRACT_MERGED` (or a new prompt);
4. downstream `_normalize_event_objs`/`_seal` validation to ensure the
   one-pass event schema remains compatible.

The topic-continuity split check is a separate call and is not removed merely
by enabling merged extraction. A future one-pass experiment must therefore
state whether it means “one extraction call per block” or also changes
message-level segmentation.

## Additional risks and gaps

- `CLAUDE.md` and several markdown files describe older defaults (for example
  categories/labels or a copied builder) that no longer match the active
  source.
- `build_impl.py` and `build_impl_graph.py` duplicate substantial logic; a
  patch to one is not automatically applied to the other.
- The source archive contains many old/experimental scripts and generated
  artifacts; the core copy intentionally omits ambiguous generated-data
  managers and all outputs.
- No dedicated Time-Dialogue adapter/evaluator was found in the audited
  source; only LoCoMo, HaluMem-oriented utilities, and LongMemEval-oriented
  adapters were retained.
- Event provenance is block-level (`coverage` plus `content_text`), not a
  character-level source span for each event.
- Relative-time prompt behavior and final date normalization are split between
  provider tool calling, Pass 2, and local fallback logic; this should be
  tested explicitly before comparing LongMemEval variants.
