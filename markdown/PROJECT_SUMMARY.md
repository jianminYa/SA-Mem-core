# Agent Memory Project - Implementation Summary

## Project Overview

This is a complete agent memory system for extracting, storing, retrieving, and generating answers from conversational memory. The system uses a custom memory schema with temporal metadata and supports advanced query parsing with time-based and topic-based filtering.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT: Raw Conversations                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  BUILD STAGE (build_impl.py, build_prompts.py)                  │
│  - Split conversations into memory blocks                        │
│  - Extract topics, keywords, events                              │
│  - Classify events (OCCURRENCE, STATE, INTENTION, MISC)          │
│  - Generate temporal metadata                                    │
│  - Assign categories/labels                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STORAGE: Memory Blocks (final_boxes_content.jsonl)             │
│  - JSON Lines format                                             │
│  - Each block has: temporal_index, features, events              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL STAGE (retrieval_enhanced.py)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Query Parsing (query_pasing_byllm.py)                  │  │
│  │    - Extract time constraints                             │  │
│  │    - Identify intent (WINDOW, AS_OF, STATIC, etc.)        │  │
│  │    - Rewrite query (remove time expressions)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 2. Temporal Filtering (interval_tree_index.py)            │  │
│  │    - Build interval tree index                            │  │
│  │    - Query by time range/point                            │  │
│  │    - Handle ANCHOR queries (anchor_resolver.py)           │  │
│  └───────────────────────────────────────────────────────────┘  │ │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 3. Vector Similarity Ranking                              │  │
│  │    - Embed rewritten query                                │  │
│  │    - Rank filtered blocks by cosine similarity            │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  GENERATION STAGE (generate_impl.py, generate_prompts.py)       │
│  - Select top-K blocks                                           │
│  - Format as context                                             │
│  - Generate answer with LLM                                      │
│  - Evaluate (F1, BLEU)                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Generated Answers                     │
└─────────────────────────────────────────────────────────────────┘
```

## Completed Components

### ✅ Memory Schema Design (build_impl.py)

**Memory Block Structure:**
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
    "categories": ["label1", "label2"],
    "topic_kw_text": "keywords"
  },
  "events_count": 3,
  "events": [...]
}
```

**Event Structure:**
```json
{
  "event_id": "uuid_blockid_eventid",
  "description": "Natural language description",
  "event_temporal_type": "OCCURRENCE|STATE|INTENTION|MISC",
  "time_metadata": {
    "observedTime": "ISO-8601",
    "startTime": "YYYY-MM-DD",
    "endTime": "YYYY-MM-DD",
    "granularity": "day"
  },
  "labels": [
    {"label": "category", "confidence": 0.95}
  ]
}
```

### ✅ Memory Building (build_impl.py, build_prompts.py)

- **TopicClusterManager**: Extracts topics, keywords, and events from conversations
- **MemoryBuilder**: Splits conversations into coherent memory blocks
- **Event Classification**: Classifies events into temporal types
- **Label Generation**: Assigns semantic labels to events
- **Temporal Metadata**: Extracts and normalizes time information

### ✅ Query Parsing (query_pasing_byllm.py)

**Features:**
- Fast parser for common patterns (dates, relative time)
- LLM-based parser for complex queries
- Time constraint extraction:
  - ABSOLUTE: "in 2023", "Q1 2024"
  - RELATIVE_POINT: "yesterday", "today"
  - RELATIVE_RANGE: "last month", "this year"
  - ANCHOR: "when I lived in Paris", "after I graduated"
  - NONE: No time constraint
- Intent classification: WINDOW, AS_OF, STATIC, PLANNING, MISC
- Query rewriting: Removes time expressions for better semantic matching

### ✅ Enhanced Retrieval (NEW)

**1. Temporal Index (interval_tree_index.py)**
- Interval tree for efficient temporal range queries
- Supports session time and event time dimensions
- Query operations:
  - Point queries: Find blocks at specific time
  - Range queries: Find blocks overlapping time range
  - Before/After queries: Find blocks before/after time point
- Category and topic filtering

**2. Anchor Resolver (anchor_resolver.py)**
- Two-stage resolution for complex temporal queries
- Stage 1: Find anchor event using vector similarity
- Stage 2: Extract time range from anchor blocks
- Supports DURING, BEFORE, AFTER relations

**3. Enhanced Retriever (retrieval_enhanced.py)**
- Integrates query parser with retrieval
- Multi-stage filtering pipeline:
  1. Parse query → extract time/topic constraints
  2. Filter by temporal constraints (interval tree)
  3. Filter by topic/category overlap
  4. Filter by event types
  5. Rank by vector similarity
- Graceful fallback when filtering is too aggressive
- Compatible with existing generation pipeline

**4. Retrieve Stage (retrieve_stage_enhanced.py)**
- Command-line interface for retrieval
- Three modes: baseline, enhanced, both
- Configurable parameters

### ✅ Generation (generate_impl.py, generate_prompts.py)

- **AnswerGenerator**: Generates answers from retrieved blocks
- Multiple text modes: content, content_trace_event, trace_event
- Evaluation metrics: F1, BLEU
- Token usage tracking
- Summary statistics

## File Structure

```
SA-Mem/
├── Core Implementation
│   ├── memblock_extractor.py      # Main entry point, config, utilities
│   ├── build_impl.py               # Memory building logic
│   ├── build_prompts.py            # Prompts for memory extraction
│   ├── retrieval_impl.py           # Simple baseline retrieval
│   ├── retrieval_enhanced.py       # ✨ NEW: Enhanced retrieval
│   ├── generate_impl.py            # Answer generation
│   └── generate_prompts.py         # Prompts for answer generation
│
├── Query Processing (NEW)
│   ├── query_pasing_byllm.py       # Query parser
│   ├── interval_tree_index.py      # ✨ NEW: Temporal index
│   └── anchor_resolver.py          # ✨ NEW: Anchor query resolver
│
├── Stage Scripts
│   ├── build_stage.py              # Build stage entry
│   ├── retrieve_stage.py           # Simple retrieve stage
│   ├── retrieve_stage_enhanced.py  # ✨ NEW: Enhanced retrieve stage
│   └── generate_stage.py           # Generate stage entry
│
├── Testing & Documentation
│   ├── test_enhanced_retrieval.py  # ✨ NEW: Test script
│   ├── RETRIEVAL_README.md         # ✨ NEW: Retrieval documentation
│   └── PROJECT_SUMMARY.md          # ✨ NEW: This file
│
└── Data Directories
    ├── data/                       # Input data
    ├── out/                        # Output files
    ├── index/                      # Indices
    ├── retrieval/                  # Retrieval results
    └── generation/                 # Generation results
```

## Usage Examples

### 1. Build Memory Blocks

```bash
python build_stage.py \
    --raw-data-file data/processed_halumem/your_file.json \
    --limit-conversations 10
```

### 2. Run Enhanced Retrieval

```bash
# Enhanced mode (recommended)
python retrieve_stage_enhanced.py --mode enhanced

# Baseline mode (for comparison)
python retrieve_stage_enhanced.py --mode baseline

# Both modes (for evaluation)
python retrieve_stage_enhanced.py --mode both
```

### 3. Generate Answers

```bash
python generate_stage.py
```

### 4. Full Pipeline

```bash
python memblock_extractor.py --stage all
```

### 5. Test Enhanced Retrieval

```bash
python test_enhanced_retrieval.py
```

## Query Examples

### Temporal Queries

1. **Absolute Time**
   ```
   "What did I do in 2023?"
   → Filters blocks with events in 2023
   ```

2. **Relative Time**
   ```
   "What happened last month?"
   → Calculates date range, filters blocks
   ```

3. **Anchor Time**
   ```
   "Was I married when I lived in Paris?"
   → Finds "lived in Paris" blocks
   → Extracts time range
   → Queries marriage status in that range
   ```

### Topic Queries

```
"Tell me about my trip to Japan"
→ Filters blocks with "trip", "Japan" keywords
→ Ranks by semantic similarity
```

### Intent-based Queries

```
"What are my future plans?"
→ Intent: PLANNING
→ Target types: [INTENTION]
→ Filters blocks with INTENTION events
```

## Key Features

### 1. Time-based Filtering
- ✅ Absolute dates (2023, Q1 2024)
- ✅ Relative dates (yesterday, last month)
- ✅ Anchor queries (when X happened)
- ✅ Fuzzy time matching with buffer

### 2. Topic-based Filtering
- ✅ Keyword matching
- ✅ Category filtering
- ✅ Event type filtering

### 3. Query Understanding
- ✅ Intent classification
- ✅ Time constraint extraction
- ✅ Query rewriting
- ✅ Fast path for simple queries

### 4. Efficient Indexing
- ✅ Interval tree for temporal queries
- ✅ Vector embeddings for semantic search
- ✅ Category index for topic filtering

### 5. Graceful Degradation
- ✅ Fallback to baseline if parsing fails
- ✅ Fallback to full pool if filtering too aggressive
- ✅ Multiple retrieval modes for comparison

## Performance Considerations

### Filtering Strategy
- Progressive filtering: temporal → topic → event type → vector ranking
- Fallback threshold: Keep at least 3 blocks after filtering
- Caching: Query parsing, embeddings, topic classification

### Scalability
- Interval tree: O(log n + k) for range queries
- Vector similarity: O(n) but on filtered subset
- Incremental building: Checkpoint every sample

## Configuration

Key parameters in `memblock_extractor.py`:

```python
class Config:
    # Data
    RAW_DATA_FILE = "data/processed_halumem/uuid.json"
    LIMIT_CONVERSATIONS = 1  # None for no limit
    LIMIT_SESSIONS = None    # None for no limit

    # Retrieval
    TOP_K_RETRIEVE = 20      # Max blocks to retrieve
    ANSWER_TOP_N = 5         # Blocks for answer generation

    # Models
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Build
    BUILD_PREV_MSGS = 2      # Context for split decisions
    EVENT_LABEL_TOP_K = 3    # Labels per event

    # Generation
    GEN_TEXT_MODES = ["content_trace_event"]
```

## Evaluation

Compare baseline vs enhanced retrieval:

```bash
python retrieve_stage_enhanced.py --mode both
```

Metrics:
- **Ranking quality**: Compare retrieved blocks with ground truth
- **F1/BLEU scores**: Answer quality
- **Context efficiency**: Tokens used vs accuracy

## Future Enhancements

1. **Semantic Category Matching**: Use embeddings for category similarity
2. **Hybrid Ranking**: Combine temporal relevance + semantic similarity
3. **Query Expansion**: LLM-based query rewriting with synonyms
4. **Multi-hop Reasoning**: Chain multiple queries for complex questions
5. **Personalization**: Learn user-specific query patterns

## Dependencies

```
openai>=1.0.0
scikit-learn>=1.0.0
tiktoken>=0.5.0
nltk>=3.8.0
```

## License

[Your License Here]

## Contact

[Your Contact Information]

---

**Status**: ✅ Complete and ready for use

**Last Updated**: 2026-02-11
