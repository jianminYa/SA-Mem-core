# Enhanced Agent Memory Retrieval System

## Overview

This enhanced retrieval system integrates **query parsing**, **temporal filtering**, and **topic-based metadata filtering** to improve memory retrieval accuracy for agent memory systems.

## Architecture

### Components

1. **Query Parser** (`query_pasing_byllm.py`)
   - Parses natural language queries to extract:
     - **Time constraints**: ABSOLUTE, RELATIVE_POINT, RELATIVE_RANGE, ANCHOR, NONE
     - **Intent**: WINDOW, AS_OF, STATIC, PLANNING, MISC
     - **Target event types**: OCCURRENCE, STATE, ATTRIBUTE, INTENTION
   - Rewrites queries by removing time expressions for better semantic matching

2. **Temporal Index** (`interval_tree_index.py`)
   - Interval tree-based index for efficient temporal range queries
   - Supports multiple temporal dimensions:
     - **Session time**: When the conversation occurred
     - **Event time**: When the events mentioned in the conversation occurred
   - Query operations:
     - Point queries: Find blocks containing a specific time point
     - Range queries: Find blocks overlapping with a time range
     - Before/After queries: Find blocks before/after a time point

3. **Enhanced Retriever** (`retrieval_enhanced.py`)
   - Integrates query parsing with retrieval
   - Multi-stage filtering pipeline:
     1. **Temporal filtering**: Filter blocks by time constraints
     2. **Topic filtering**: Filter blocks by topic/keyword overlap
     3. **Event type filtering**: Filter blocks by event temporal types
     4. **Vector ranking**: Rank filtered blocks by semantic similarity
   - Falls back gracefully when filtering is too aggressive

4. **Retrieve Stage** (`retrieve_stage_enhanced.py`)
   - Command-line interface for running retrieval
   - Supports three modes:
     - `baseline`: Simple vector similarity (original implementation)
     - `enhanced`: Query parsing + metadata filtering
     - `both`: Run both for comparison

## Usage

### Basic Usage

```bash
# Run enhanced retrieval (recommended)
python retrieve_stage_enhanced.py --mode enhanced

# Run baseline retrieval (for comparison)
python retrieve_stage_enhanced.py --mode baseline

# Run both modes for comparison
python retrieve_stage_enhanced.py --mode both
```

### Advanced Options

```bash
python retrieve_stage_enhanced.py \
    --mode enhanced \
    --raw-data-file data/processed_halumem/your_file.json \
    --output-dir out/experiment_1 \
    --run-id exp1 \
    --limit-conversations 10 \
    --api-key your_api_key \
    --base-url https://your-api-endpoint.com/v1
```

## Query Examples

### Temporal Queries

1. **Absolute Time**
   ```
   Query: "What did I do in 2023?"
   Parsing: time_type=RANGE, start=2023-01-01, end=2023-12-31
   Filtering: Blocks with events in 2023
   ```

2. **Relative Time**
   ```
   Query: "What happened last month?"
   Parsing: time_type=RANGE, start=2024-01-01, end=2024-01-31 (if today is 2024-02-15)
   Filtering: Blocks with events in January 2024
   ```

3. **Anchor Time**
   ```
   Query: "Was I married when I lived in Paris?"
   Parsing: time_type=ANCHOR, anchor_event="when I lived in Paris"
   Filtering: Vector search for "Paris" context, then check marriage status
   ```

### Topic Queries

1. **Topic-based**
   ```
   Query: "Tell me about my trip to Japan"
   Parsing: intent=WINDOW, rewritten_query="trip to Japan"
   Filtering: Blocks with categories/keywords matching "trip", "Japan"
   ```

2. **Event Type-based**
   ```
   Query: "What are my future plans?"
   Parsing: intent=PLANNING, target_types=[INTENTION]
   Filtering: Blocks containing INTENTION events
   ```

## Memory Schema

### Block Structure

```json
{
  "user_id": "uuid",
  "block_id": 0,
  "temporal_index": {
    "sessionstart_time": "2023-05-01T10:00:00",
    "sessionend_time": "2023-05-01T11:00:00",
    "block_event_start_time": "2023-04-15",
    "block_event_end_time": "2023-05-01"
  },
  "features": {
    "categories": ["travel", "work", "family"],
    "topic_kw_text": "Paris trip vacation"
  },
  "events_count": 3,
  "events": [
    {
      "event_id": "uuid_0_e0",
      "description": "User moved to Paris",
      "event_temporal_type": "OCCURRENCE",
      "time_metadata": {
        "observedTime": "2023-05-01",
        "startTime": "2023-04-15",
        "endTime": "2023-04-15",
        "granularity": "day"
      },
      "labels": [
        {"label": "relocation", "confidence": 0.95},
        {"label": "travel", "confidence": 0.88}
      ]
    }
  ]
}
```

### Event Temporal Types

- **OCCURRENCE**: Specific events that happened (e.g., "moved to Paris", "graduated")
- **STATE**: Changeable status over time (e.g., "was married", "lived in NYC")
- **ATTRIBUTE**: Stable identity facts (e.g., "name is John", "MBTI is INTJ")
- **INTENTION**: Future plans/goals (e.g., "plans to retire", "wants to visit Japan")

## Performance Considerations

### Filtering Strategy

The enhanced retriever uses a **progressive filtering** approach:

1. Start with all blocks for the sample
2. Apply temporal filter (if time constraint exists)
3. Apply topic filter (if query has topic keywords)
4. Apply event type filter (if specific types targeted)
5. Rank remaining blocks by vector similarity

### Fallback Mechanism

If filtering reduces candidates to < 3 blocks, the system falls back to the full pool to avoid over-filtering.

### Caching

- Query parser caches parsed queries (by date + query text)
- Vector embeddings are cached in the vector store
- Topic classification results are cached

## Integration with Generation

The enhanced retrieval outputs are compatible with the existing generation pipeline:

```bash
# Full pipeline with enhanced retrieval
python memblock_extractor.py --stage all

# Or run stages separately
python build_stage.py                    # Build memory blocks
python retrieve_stage_enhanced.py --mode enhanced  # Enhanced retrieval
python generate_stage.py                 # Generate answers
```

## Configuration

Key configuration parameters in `memblock_extractor.py`:

```python
class Config:
    # Retrieval
    TOP_K_RETRIEVE = 20          # Max blocks to retrieve
    ANSWER_TOP_N = 5             # Blocks used for answer generation

    # Query parsing
    LLM_MODEL = "gpt-4o-mini"    # Model for query parsing

    # Temporal filtering
    # (automatically enabled when time constraints detected)

    # Topic filtering threshold
    # (0.2 = 20% keyword overlap required)
```

## Evaluation

Compare baseline vs enhanced retrieval:

```bash
# Run both modes
python retrieve_stage_enhanced.py --mode both

# Results saved to:
# - retrieval_baseline.jsonl / retrieval_baseline.csv
# - retrieval_enhanced.jsonl / retrieval_enhanced.csv

# Compare rankings and target overlap
```

## Troubleshooting

### Query Parsing Fails

If query parsing fails (timeout, API error), the system falls back to baseline retrieval automatically.

### Over-filtering

If temporal/topic filters are too aggressive (< 3 blocks remain), the system automatically uses the full pool.

### Missing Temporal Data

If blocks lack temporal metadata, they won't be filtered by time constraints but will still be ranked by vector similarity.

## Future Enhancements

1. **Anchor Query Resolution**: Implement two-stage retrieval for anchor queries
   - Stage 1: Find anchor event ("when I lived in Paris")
   - Stage 2: Query within anchor time range

2. **Category Expansion**: Use semantic similarity for category matching instead of exact string matching

3. **Hybrid Ranking**: Combine temporal relevance score with vector similarity score

4. **Query Rewriting**: Use LLM to expand queries with synonyms and related terms

## References

- Query Parser: `query_pasing_byllm.py`
- Memory Schema: `build_impl.py`
- Build Prompts: `build_prompts.py`
- Generation: `generate_impl.py`
