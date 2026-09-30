# SA-Mem core code manifest

## 1. Provenance

- Original archive: `SA-Mem.tar.xz`
- Google Drive file id: `1CXhpCuyb-ewj_K_sMBYM0-Sgdv87cb3R`
- Original archive size: `1,049,341,384` bytes (about 1.05 GB)
- SHA256: `01bc898a4a1f102cf9aa37b130f3178f980527807a131eb756c04c9349d7ee27`
- Extraction/audit date: `2026-09-30` UTC
- Source copy: `SA-Mem-core/`, made without changing the extracted `SA-Mem/`

The archive was fetched from the Drive file id with `gdown` after the Drive
connector rejected the 1.05 GB raw download because of its 256 MB connector
limit. The source archive was extracted with `tar -xJf`.

The copy is source-only. It deliberately excludes the original `.git/`,
`data/`, `out/`, `notindemoout/`, logs, caches, vector stores, checkpoints,
model files, generated JSON/JSONL/CSV outputs, and `webUI/` (which contained a
local `.env`). Two embedded API-key fallbacks and the graph password fallback
were removed from the copied source; credentials are runtime-only.

## 2. Included source inventory

### Construction and entry points

- `memblock_cli.py`: main CLI; build/trace/retrieve/generate/all stage routing,
  dataset-format selection, parallel input handling, and runtime overrides.
- `memblock_extractor.py`: `Config`, `LLMWorker`, `TokenAnalyzer`, embedding
  cache, time normalization helpers, logging, and the active imports at the
  bottom of the file.
- `build_impl_graph.py`: active current `MemoryBuilder` and
  `TopicClusterManager`; optional graph integration and LongMemEval builder.
- `build_impl.py`: non-graph/reference builder retained to make the original
  implementation split explicit; it is not the active bottom-of-module import.
- `build_prompts.py`: continuation/split prompt, Pass 1, Pass 2, merged
  extraction prompt, and trace prompt text.
- `build_prompts_locomo.py`: LoCoMo-specific extraction prompt variants.
- `build_stage.py`, `build_stage_fast.py`, `build_stage_locomo.py`,
  `build_stage_resume.py`: lightweight build entry points/resume variants.
- `temporal_resolution_tool.py`: local temporal function schema and resolver.

### Retrieval

- `retrieval/retrieval_impl.py`: baseline vector retrieval and evidence-to-box
  mapping.
- `retrieval/retrieval_enhanced.py`: temporal query parsing, interval-index
  filtering, vector ranking, optional anchor and graph expansion.
- `retrieval/query_pasing_byllm.py`: fast temporal parser plus LLM fallback
  parser (`query_pasing` spelling is retained from the source).
- `retrieval/interval_tree_index.py`: separate session-time and event-time
  interval indexes.
- `retrieval/anchor_resolver.py`: embedding-based anchor/event-window
  resolution; it does not call an LLM.
- `retrieval/retrieval_impl_locomo.py` and
  `retrieval/retrieval_enhanced_locomo.py`: LoCoMo adapters.
- `retrieval/retrieval_lme.py`: LongMemEval-specific user/session/evidence
  mapping and temporal retrieval adapter.
- `retrieval/retrieve_stage_enhanced.py`,
  `retrieval/retrieve_stage_enhanced_locomo.py`, and
  `retrieval/retrieve_stage_lme.py`: stage CLIs.

### QA, trace, graph, and evaluation

- `generate_impl.py`: generic retrieval-result-to-answer generation and
  metric/output handling.
- `generate_impl_locomo.py`: LoCoMo generation modes and context formatting.
- `generate_prompts.py`: QA answer prompt.
- `trace_impl.py`, `trace_stage.py`: optional evidence/event-chain trace
  construction and its LLM filters.
- `graph_storage.py`, `graph_entities_extractor.py`, `graph_utils.py`,
  `build_impl_graph.py`: optional graph storage/entity/relation path.
- `evaluate_locomo.py`: LoCoMo answer evaluation.
- `eval_memory_extraction.py`, `eval_evidence_recall.py`,
  `eval_retrieval_score.py`, `merged_eval.py`: extraction, evidence, and
  retrieval evaluation utilities.
- `llm_judge_eval.py`, `extract_ground_truth.py`,
  `extract_qa_for_user.py`: HaluMem-oriented helpers.
- `scripts/build_lme_reference.py`, `scripts/parse_longmemeval_queries.py`,
  `scripts/generate_lme.py`, and selected HaluMem/LME analysis scripts:
  adapter/evaluation utilities retained for the next experiment.

## 3. Actual core pipeline

The code-level path below is based on the active imports and function bodies,
not on the paper description alone:

```text
raw sessions / LoCoMo / LongMemEval haystack sessions
    |
    v
memblock_cli.py
    |
    +--> memblock_extractor.LLMWorker + Config
    |
    +--> build_impl_graph.MemoryBuilder.build_all*
              |
              +--> _process(message, session_id)
              |       |
              |       +--> after the first two messages in a session:
              |             LLMWorker.check_relation()
              |             -> build_prompts.PROMPT_MSG_CONTINUATION
              |             -> append or seal the current block
              |
              +--> _seal()
                      |
                      +--> TopicClusterManager.process_new_box()
                              |
                              +--> default: Pass 1
                              |    PROMPT_DIALOG_EXTRACT
                              |    -> topic, keywords, atomic mentions
                              |
                              +--> default when mentions exist and
                              |    ENABLE_EVENT_CLASSIFICATION=True: Pass 2
                              |    PROMPT_DIALOG_CLASSIFICATION
                              |    -> description, type, start_time, end_time
                              |
                              +--> local normalization in memblock_extractor
                                   -> event_temporal_type
                                   -> time_metadata / observedTime
                                   -> temporal_index
                                   -> coverage provenance
                                   -> features.content_text/topic_kw_text
    |
    +--> optional EmbeddingStore population
    +--> optional graph upsert/entity/relation extraction
    |
    v
final_boxes_content.jsonl + per-user vector cache
    |
    +--> baseline retrieval/retrieval_impl.py
    |       -> question/block embeddings -> cosine ranking
    |
    +--> enhanced retrieval/retrieval_enhanced.py
            -> FastParser or QueryParser LLM fallback
            -> session/event interval filtering
            -> optional anchor resolution
            -> embedding ranking
            -> optional graph expansion
    |
    v
generate_impl.py / generate_impl_locomo.py
    -> retrieval context formatting
    -> generate_prompts.PROMPT_QA_ANSWER
    -> one QA completion per generated answer/context run
    -> answer metrics and generated evaluation output
```

`memblock_extractor.py:1234` imports `TopicClusterManager` and `MemoryBuilder`
from `build_impl_graph.py`, which is why that file is the active implementation
for the current CLI. `GraphConfig.enable_graph` is false by default, so the
graph is not part of the default construction/retrieval path.

For LongMemEval, `build_impl_graph.MemoryBuilder.build_all_longmemeval`
consumes `haystack_sessions`, aligns `haystack_dates` and
`haystack_session_ids`, and preserves `question_id`/`question_type` as build
metadata. `retrieval/retrieval_lme.py` later uses `answer_session_ids` for
evidence target mapping. The question itself is not inserted into the memory
block prompt.

## 4. Two-pass extraction

### Pass 1

- Input: one topic-segmented block containing the session window, message
  timestamps/roles/text, and optional persona information.
- Implementation: `build_impl_graph.py:TopicClusterManager.process_new_box`.
- Prompt: `build_prompts.py:PROMPT_DIALOG_EXTRACT` (LoCoMo uses the variant in
  `build_prompts_locomo.py`).
- Output: JSON `{keywords, topic, mentions}` where `mentions` are grounded,
  atomic strings. It is not yet the final typed event schema.
- Calls: one `LLMWorker.chat_completion` call per block in the ordinary path,
  with `enable_functions=True`. If the provider emits
  `resolve_temporal_expression`, the worker makes an additional final chat
  completion after executing the resolver locally.

### Pass 2

- Input: the Pass 1 mention strings plus the session start/end window.
- Implementation: the same `process_new_box` method.
- Prompt: `build_prompts.py:PROMPT_DIALOG_CLASSIFICATION`.
- Output: JSON `explicit_mentions` with description, one of
  `OCCURRENCE|STATE|ATTRIBUTE|INTENTION`, and start/end times.
- Calls: one additional ordinary `LLMWorker.chat_completion` per block when
  Pass 1 returned at least one mention and
  `Config.ENABLE_EVENT_CLASSIFICATION=True`.
- Finalization: `_seal` converts these objects to `events`, fills
  `observedTime`, normalizes dates to day granularity where possible, and
  creates `temporal_index` fields. `coverage` stores block-level session and
  message indexes; there is no event-level character source span.

### Existing one-call alternative

`Config.ENABLE_MERGED_EXTRACTION`, controlled by
`MEMBLOCK_MERGED_EXTRACTION`, selects `PROMPT_DIALOG_EXTRACT_MERGED`. This is
an existing experiment, not the default path. It combines topic/keywords and
typed/timed events in one request. No one-pass redesign was made in this
cleanup.

## 5. Token and API accounting points

| Stage | File:function | Model/use | Currently recorded? |
|---|---|---|---|
| Construction split | `memblock_extractor.py:LLMWorker.check_relation` | LLM continuation/topic segmentation | Yes, through `TokenAnalyzer` under `stage=build` and a `split_check` trace; not separated as its own aggregate field |
| Construction Pass 1 | `build_impl_graph.py:TopicClusterManager.process_new_box` | structured mention/topic extraction | Yes, prompt/completion/total are written by `TokenAnalyzer` when usage is returned |
| Construction Pass 2 | same method | event type/time classification | Yes, same limitation |
| Temporal tool follow-up | `memblock_extractor.py:LLMWorker.chat_completion` | provider tool call then final response | Partially: the final response is logged; the first tool-call response is not separately logged |
| Construction merged path | same method | existing one-call experiment | Yes, through the same worker logger |
| Embeddings | `memblock_extractor.py:LLMWorker.get_embedding` | query, block, anchor, graph-event vectors | No usage/call counter; failures silently return a zero vector |
| Retrieval query parsing | `retrieval/query_pasing_byllm.py:QueryParser.parse` | LLM fallback for queries not handled by `FastParser` | No; direct OpenAI client bypasses `TokenAnalyzer` |
| Retrieval ranking/filtering | `retrieval/retrieval_enhanced.py` and `retrieval/retrieval_lme.py` | local interval filtering and embeddings | Coarse timing exists in enhanced/LME result records; no unified token/call accounting |
| Graph extraction | `graph_entities_extractor.py:OpenAIToolCaller.generate_response` / Ollama caller | entity/relation tool calls when graph is enabled | No `TokenAnalyzer` integration |
| QA generation | `generate_impl.py:AnswerGenerator._generate_for_ranking` and LoCoMo counterpart | answer completion | Yes for calls using `LLMWorker`; stage is `gen:*`, with prompt/completion/total |
| LongMemEval QA/judge | `scripts/generate_lme.py:LLMModel.call` | answer and LLM-as-judge calls | No usage/call/latency instrumentation |

The current worker log has `in` and `out` per record in `token_stream.jsonl`;
the in-memory stage counters additionally maintain `total`. It does not
natively expose the
requested named totals (`construction_input_tokens`, etc.),
`cached_tokens`, per-provider retry counts, embedding calls, all LLM calls,
per-call latency, or a unified stage/total runtime. `prompt_tokens_est` is an
estimate attached to several records and is not a substitute for provider
usage.

## 6. LongMemEval adaptation points (analysis only)

- Dataset adapter: use `build_impl_graph.py:MemoryBuilder.build_all_longmemeval`
  or the existing `scripts/parse_longmemeval_queries.py` normalization. The
  adapter should map each haystack item to one `session_id`, date, and list of
  `{role, content}` messages without changing the construction algorithm.
- Construction adapter: enter through `memblock_cli.py`'s LongMemEval format
  routing, then instrument `LLMWorker.chat_completion` and the
  `MemoryBuilder._process/_seal` boundary for split versus extraction calls.
- Retrieval adapter: reuse `retrieval/retrieval_lme.py` and
  `retrieve_stage_lme.py`; its `answer_session_ids` mapping is the existing
  evidence interface.
- QA adapter: either connect LME retrieval records to
  `generate_impl.py`/`generate_impl_locomo.py`, or use the separate
  `scripts/generate_lme.py` path. The latter currently also performs an
  LLM-as-judge call.
- Token instrumentation: the least ambiguous central hook is
  `memblock_extractor.py:LLMWorker` for worker-owned calls, augmented by a
  wrapper around `retrieval/query_pasing_byllm.py:QueryParser` and
  `scripts/generate_lme.py:LLMModel`. Embedding call counts and all stage
  runtimes need explicit counters at those boundaries.

These are future insertion points only. This repository does not implement a
LongMemEval experiment or change the extraction protocol.
