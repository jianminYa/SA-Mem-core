# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an agent memory system that extracts, stores, retrieves, and generates answers from conversational memory. The system uses temporal indexing and metadata filtering to improve retrieval accuracy through query-based time intent recognition and topic classification.

## Common Commands

### Build Stage
```bash
# Standard build (uses build_impl copy.py)
python memblock_cli.py --stage build \
    --raw-data-file data/processed_halumem/your_file.json \
    --limit-conversations 10

# Fast build mode (reduced timeouts for faster processing)
python build_stage_fast.py \
    --raw-data-file data/processed_halumem/your_file.json \
    --limit-conversations 10

# Full pipeline with all stages
python memblock_cli.py --stage all
```

### Retrieval Stage
```bash
# Enhanced retrieval (with query parsing + temporal filtering)
python memblock_cli.py --stage retrieve

# Custom run ID (outputs to out/<run_id>/)
python memblock_cli.py --stage retrieve --run-id experiment_1
```

### Generation Stage
```bash
# Generate answers from retrieved blocks
python generate_stage.py
```

### Testing
```bash
# Test enhanced retrieval system
python test_enhanced_retrieval.py
```

## Architecture

### Data Flow Pipeline

```
Raw Conversations (JSON)
    ↓
BUILD STAGE (build_impl copy.py)
  Two-Pass Extraction:
  Pass 1 (PROMPT_DIALOG_EXTRACT):
    - Extract topics and keywords
    - Extract atomic memory mentions (strings)
  Pass 2 (PROMPT_DIALOG_CLASSIFICATION):
    - Classify mentions into event types (OCCURRENCE, STATE, ATTRIBUTE, INTENTION)
    - Assign temporal metadata (start_time, end_time)

  Processing:
    - Split conversations into memory blocks
    - Generate temporal index (session time + event time)
    - Optional: Enable/disable event classification (ENABLE_EVENT_CLASSIFICATION flag)
    ↓
Memory Blocks (final_boxes_content.jsonl)
    ↓
RETRIEVAL STAGE (retrieval/retrieval_enhanced.py)
  1. Query Parsing (query_pasing_byllm.py)
     - Fast path: Regex-based pattern matching for common queries
     - LLM fallback: Complex query parsing with PROMPT_LITE
     - Extract time constraints (ABSOLUTE, RELATIVE_POINT, RELATIVE_RANGE, NONE)
     - Classify intent (WINDOW, AS_OF, STATIC, PLANNING, MISC)
     - Rewrite query (remove time expressions)

  2. Temporal Filtering (interval_tree_index.py)
     - Build interval tree index for both session_time and event_time
     - Query by time range/point/after/before
     - Fallback: Try event_time first, then session_time if no results
     - Handle anchor queries (anchor_resolver.py)

  3. Vector Similarity Ranking
     - Embed query and memory blocks (topic_kw_text + event descriptions)
     - Rank filtered blocks by cosine similarity
     - Note: Topic filtering and event type filtering are DISABLED
    ↓
Retrieved Blocks (retrieval_enhanced.jsonl)
    ↓
GENERATION STAGE (generate_impl.py)
  - Format top-K blocks as context (multiple text modes)
  - Generate answer with LLM (PROMPT_QA_ANSWER)
  - Evaluate with F1 and BLEU scores
  - Track context token usage
    ↓
Generated Answers (generation_results.jsonl)
```

### Key Components

**memblock_extractor.py**: Main entry point with Config class and utilities. All configuration parameters are defined here. Also contains helper functions for event parsing and normalization.

**memblock_cli.py**: CLI interface for running different stages (build, retrieve, generate, all).

**build_impl copy.py**: Memory building logic with TopicClusterManager and MemoryBuilder classes. Implements two-pass extraction:
- Pass 1: Extract topics, keywords, and atomic mentions
- Pass 2: Classify mentions and assign temporal metadata
- Supports both standard and LoCoMo dataset formats

**build_prompts.py**: Contains prompts for the two-pass extraction:
- PROMPT_DIALOG_EXTRACT: Extract mentions from dialog
- PROMPT_DIALOG_CLASSIFICATION: Classify mentions into event types

**build_stage_fast.py**: Fast build mode with reduced timeouts (15s vs 60s) and fewer retries for faster processing.

**retrieval/query_pasing_byllm.py**: Query parser with two-stage parsing:
- FastParser: Regex-based pattern matching for common queries
- QueryParser: LLM-based fallback for complex queries
- DateResolver: Local date parsing for relative and absolute time expressions

**retrieval/interval_tree_index.py**: Temporal index using interval trees for O(log n + k) range queries. Supports both session time and event time dimensions with automatic fallback.

**retrieval/anchor_resolver.py**: Two-stage resolution for anchor queries (e.g., "when I lived in Paris"). Stage 1 finds anchor event, Stage 2 extracts time range.

**retrieval/retrieval_enhanced.py**: Enhanced retriever that integrates query parsing with temporal filtering and vector ranking. Topic and event type filtering are currently disabled.

**generate_impl.py**: Answer generator (AnswerGenerator class) with multiple text modes and evaluation metrics (F1, BLEU).

**generate_prompts.py**: Contains PROMPT_QA_ANSWER for answer generation with step-by-step reasoning instructions.

**generate_stage.py**: Entry point for generation stage.

## Memory Schema

### Memory Block Structure
```json
{
  "user_id": "uuid",
  "block_id": 0,
  "temporal_index": {
    "sessionstart_time": "ISO-8601",
    "sessionend_time": "ISO-8601",
    "block_event_start_time": "YYYY-MM-DD",
    "block_event_end_time": "YYYY-MM-DD"
  },
  "features": {
    "topic_kw_text": "keywords"
  },
  "events_count": 3,
  "events": [...]
}
```


### Event Structure
```json
{
  "event_id": "uuid_blockid_eventid",
  "description": "Natural language description",
  "event_temporal_type": "OCCURRENCE|STATE|ATTRIBUTE|INTENTION",
  "time_metadata": {
    "observedTime": "ISO-8601",
    "startTime": "YYYY-MM-DD",
    "endTime": "YYYY-MM-DD",
    "granularity": "day"
  }
}
```

Note: The `labels` field has been removed from the schema. Event classification is now done in Pass 2 of the build stage.

## Configuration

All configuration is in `memblock_extractor.py` Config class:

```python
class Config:
    # API Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "...")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "...")

    # Data paths
    RAW_DATA_FILE = "data/processed_halumem/uuid.json"
    OUTPUT_BASE_DIR = "out"

    # Processing limits
    LIMIT_CONVERSATIONS = 1  # None for no limit
    LIMIT_SESSIONS = None

    # Retrieval parameters
    TOP_K_RETRIEVE = 20      # Max blocks to retrieve
    ANSWER_TOP_N = 20         # Blocks for answer generation

    # Models
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Build parameters
    BUILD_PREV_MSGS = 2      # Context for split decisions
    ENABLE_EVENT_CLASSIFICATION = True  # Enable/disable LLM-based event classification
    TOPIC_CLASSIFY_TIMEOUT = 60.0  # Timeout for LLM calls (15.0 in fast mode)
    TOPIC_CLASSIFY_MAX_RETRIES = 4  # Max retries (2 in fast mode)

    # Generation
    GEN_TEXT_MODES = ["content_trace_event"]  # Options: content, content_trace_event, trace_event
    TRACE_METRICS = ["content_event_topic_kw"]
```

## Key Concepts

### Temporal Dimensions
The system tracks two temporal dimensions:
- **Session time**: When the conversation occurred (ISO-8601 timestamps)
- **Event time**: When the events mentioned occurred (YYYY-MM-DD dates)

### Query Time Constraints
- **ABSOLUTE**: "in 2023", "Q1 2024"
- **RELATIVE_POINT**: "yesterday", "today"
- **RELATIVE_RANGE**: "last month", "this year"
- **ANCHOR**: "when I lived in Paris", "after I graduated"
- **NONE**: No time constraint

### Query Intent Types
- **WINDOW**: Retrieve events in a time window
- **AS_OF**: State at a specific time point
- **STATIC**: Timeless facts
- **PLANNING**: Future intentions
- **MISC**: Other intents

### Event Temporal Types
- **OCCURRENCE**: One-time events or state changes (e.g., "moved to Paris", "got married")
- **STATE**: Ongoing states or changeable status (e.g., "lived in Paris", "is married")
- **ATTRIBUTE**: Stable/timeless identity facts or traits (e.g., "MBTI: INTJ", "birthday")
- **INTENTION**: Future-oriented plans or goals (e.g., "planning to visit Paris")

### Filtering Strategy
The enhanced retriever uses temporal filtering with vector ranking:
1. **Temporal filtering** (interval tree)
   - Try event_time first for time-constrained queries
   - Fallback to session_time if event_time returns no results
   - Support for RANGE, POINT, AFTER, BEFORE query types
2. **Vector ranking** (cosine similarity)
   - Embed query and memory blocks (topic_kw_text + event descriptions)
   - Rank filtered blocks by similarity

**Note**: Topic/category filtering and event type filtering are currently DISABLED in the implementation (lines 232-245 in retrieval_enhanced.py).

## Output Files

All outputs go to `out/` (or `out/<run_id>/` with custom run ID):

**Build stage:**
- `final_boxes_content.jsonl` - Memory blocks
- `build_stats.jsonl` - Build statistics
- `trace_build_process.jsonl` - Build trace

**Retrieval stage:**
- `retrieval_enhanced.jsonl` - Enhanced retrieval results (JSON Lines)
- `retrieval_enhanced.csv` - Enhanced retrieval results (CSV)
- `retrieval_baseline.jsonl` - Baseline retrieval results

**Generation stage:**
- `generation_results.jsonl` - Generated answers
- `report_generation_qa.csv` - QA report with F1/BLEU scores
- `generation_metrics_summary.jsonl` - Summary statistics

## Development Notes

### Build Stage Architecture

The build stage uses a two-pass extraction approach:

**Pass 1 (PROMPT_DIALOG_EXTRACT)**:
- Extract topics and keywords from dialog
- Extract atomic memory mentions as strings
- Grounding: Each mention must be verbatim from dialog/persona info
- Atomicity: One fact per mention sentence

**Pass 2 (PROMPT_DIALOG_CLASSIFICATION)**:
- Classify each mention into event types (OCCURRENCE, STATE, ATTRIBUTE, INTENTION)
- Assign temporal metadata (start_time, end_time)
- Use session window as fallback when explicit dates are not available

**Event Classification**:
- Can be enabled/disabled via `ENABLE_EVENT_CLASSIFICATION` flag
- When disabled, uses heuristic fallback instead of LLM
- LLM classification provides more accurate event type assignment

### Adding New Event Types
1. Update event classification in `build_impl copy.py` (Pass 2 prompt)
2. Update `PROMPT_DIALOG_CLASSIFICATION` in `build_prompts.py`
3. Update type validation in `_construct_directive()` in `query_pasing_byllm.py`

### Modifying Query Parsing
1. Fast path patterns in `query_pasing_byllm.py` FastParser class (regex-based)
2. LLM-based parser with PROMPT_LITE for complex queries (fallback)
3. Update `SearchDirective` or `TimeConstraint` dataclass if adding new fields
4. Adjust DateResolver for new time expression patterns

### Extending Temporal Index
1. Interval tree implementation in `interval_tree_index.py`
2. Add new query methods to `TemporalIndex` class
3. Update `query_temporal()` for new query types (RANGE, POINT, AFTER, BEFORE)
4. Consider fallback strategy between event_time and session_time

### Tuning Retrieval
Key parameters to adjust:
- `TOP_K_RETRIEVE`: Number of blocks to retrieve (default: 20)
- `ANSWER_TOP_N`: Number of blocks for answer generation (default: 5)
- `LLM_TIMEOUT_SEC`: Timeout for query parsing LLM calls (default: 5.0s)
- Fast mode timeouts in `build_stage_fast.py` (15s vs 60s)

### Re-enabling Filtering
To re-enable topic or event type filtering:
1. Uncomment filtering logic in `retrieval_enhanced.py` (lines 232-245)
2. Restore `categories` field in memory block schema
3. Update embedding strategy to include category information

## Dependencies

```bash
pip install openai scikit-learn tiktoken nltk python-dateutil
```

Required environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-endpoint.com/v1"  # Optional
export LLM_TIMEOUT_SEC="5.0"  # Optional, default 5.0s for query parsing
```

每次运行Python文件前，都切换conda环境：conda activate halu_mem0

## Important Implementation Notes

1. **File naming**: The actual build implementation is in `build_impl copy.py`, not `build_impl.py`
2. **Filtering status**: Topic filtering and event type filtering are currently DISABLED in retrieval
3. **Temporal fallback**: The system tries event_time first, then falls back to session_time if no results
4. **Fast mode**: Use `build_stage_fast.py` for faster builds with reduced timeouts (15s vs 60s)
5. **Event classification**: Can be toggled via `ENABLE_EVENT_CLASSIFICATION` flag for cost/accuracy tradeoff
6. **My preference**: when meeting this: python generate_stage_locomo.py --run-id locomo --retrieval-file out/locomo/retrieval_enhanced.jsonl --answer-topn 20 --text-modes content or python retrieval/retrieve_stage_enhanced_locomo.py --mode enhanced --run-id locomo --raw-data-file  /data/locomo/data/locomo10.json. Stop work, let me run it manally.