# Quick Start Guide - Enhanced Agent Memory System

## Installation

1. **Install Dependencies**
```bash
pip install openai scikit-learn tiktoken nltk
```

2. **Set API Key**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-endpoint.com/v1"  # Optional
```

## Quick Test

### Test Query Parsing and Temporal Index

```bash
python test_enhanced_retrieval.py
```

This will test:
- Query parsing for various query types
- Temporal index with sample data
- Time range queries
- Category filtering

Expected output:
```
🧪 Enhanced Retrieval System Tests

================================================================================
QUERY PARSING TESTS
================================================================================

📝 Query: "What did I do in 2023?"
   Intent: WINDOW
   Time Type: RANGE
   Time Range: [2023-01-01, 2023-12-31]
   ...

================================================================================
TEMPORAL INDEX TESTS
================================================================================

✅ Built temporal index with 3 blocks
Point query (2023-01-12): [0]
Range query (2023-01-01 to 2023-06-30): [0, 1]
...
```

## Full Pipeline Example

### Step 1: Prepare Data

Place your conversation data in JSON format:

```json
[
  {
    "conversation": {
      "persona_info": "Name: John Doe; Age: 30; ...",
      "session_1": [
        {"speaker": "A", "text": "I moved to Paris in 2023", "time": "2024-01-15T10:00:00"},
        {"speaker": "B", "text": "How was it?", "time": "2024-01-15T10:01:00"}
      ],
      "session_1_start_time": "2024-01-15T10:00:00",
      "session_1_end_time": "2024-01-15T10:30:00"
    },
    "qa": [
      {
        "question": "When did I move to Paris?",
        "answer": "2023",
        "category": 1,
        "evidence": [{"session_id": "session_1", "start_idx": 1, "end_idx": 1}]
      }
    ]
  }
]
```

Save as `data/processed_halumem/your_file.json`

### Step 2: Build Memory Blocks

```bash
python build_stage.py \
    --raw-data-file data/processed_halumem/your_file.json \
    --limit-conversations 1
```

Output:
- `out/final_boxes_content.jsonl` - Memory blocks
- `out/build_stats.jsonl` - Build statistics
- `out/trace_build_process.jsonl` - Build trace

### Step 3: Run Enhanced Retrieval

```bash
python retrieve_stage_enhanced.py --mode enhanced
```

Output:
- `out/retrieval_enhanced.jsonl` - Retrieval results (JSON Lines)
- `out/retrieval_enhanced.csv` - Retrieval results (CSV)

### Step 4: Generate Answers

```bash
python generate_stage.py
```

Output:
- `out/generation_results.jsonl` - Generated answers
- `out/report_generation_qa.csv` - QA report with F1/BLEU scores
- `out/generation_metrics_summary.jsonl` - Summary statistics

## Compare Baseline vs Enhanced

Run both retrieval modes:

```bash
python retrieve_stage_enhanced.py --mode both
```

This creates:
- `out/retrieval_baseline.jsonl` - Baseline (simple vector similarity)
- `out/retrieval_enhanced.jsonl` - Enhanced (with query parsing + filtering)

Compare the rankings to see the improvement!

## Example Queries

### 1. Temporal Query

**Query**: "What did I do in 2023?"

**Processing**:
1. Parse: `time_type=RANGE, start=2023-01-01, end=2023-12-31`
2. Filter: Blocks with events in 2023
3. Rank: By semantic similarity to "What did I do"
4. Generate: Answer from top-5 blocks

### 2. Anchor Query

**Query**: "Was I married when I lived in Paris?"

**Processing**:
1. Parse: `time_type=ANCHOR, anchor_event="when I lived in Paris"`
2. Resolve anchor:
   - Find blocks about "lived in Paris" (vector search)
   - Extract time range (e.g., 2020-2023)
3. Filter: Blocks in 2020-2023 with marriage-related events
4. Rank: By semantic similarity to "Was I married"
5. Generate: Answer from top-5 blocks

### 3. Topic Query

**Query**: "Tell me about my trip to Japan"

**Processing**:
1. Parse: `intent=WINDOW, rewritten_query="trip to Japan"`
2. Filter: Blocks with categories ["travel", "vacation"] or keywords ["Japan", "trip"]
3. Rank: By semantic similarity to "trip to Japan"
4. Generate: Answer from top-5 blocks

## Configuration

Edit `memblock_extractor.py` to customize:

```python
class Config:
    # Input
    RAW_DATA_FILE = "data/processed_halumem/your_file.json"
    LIMIT_CONVERSATIONS = 1  # Process first N conversations

    # Retrieval
    TOP_K_RETRIEVE = 20      # Retrieve top-20 blocks
    ANSWER_TOP_N = 5         # Use top-5 for answer generation

    # Models
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
```

## Troubleshooting

### Issue: Query parsing fails

**Solution**: The system automatically falls back to baseline retrieval. Check API key and endpoint.

### Issue: No blocks retrieved

**Solution**:
1. Check if memory blocks were built: `ls out/final_boxes_content.jsonl`
2. Check if temporal metadata exists in blocks
3. Try baseline mode: `--mode baseline`

### Issue: Over-filtering (too few blocks)

**Solution**: The system automatically falls back to full pool if < 3 blocks remain after filtering.

### Issue: Poor answer quality

**Solution**:
1. Increase `ANSWER_TOP_N` to use more context
2. Try different text modes: `--text-modes content content_trace_event`
3. Check if retrieved blocks contain relevant information

## Advanced Usage

### Custom Run ID

```bash
python retrieve_stage_enhanced.py \
    --mode enhanced \
    --run-id experiment_1
```

Output goes to `out/experiment_1/`

### Process Multiple Conversations

```bash
python build_stage.py \
    --raw-data-file data/processed_halumem/your_file.json \
    --limit-conversations 100
```

### Parallel Processing

```bash
python memblock_extractor.py \
    --stage build \
    --raw-data-dir data/processed_halumem/ \
    --workers 4
```

## Next Steps

1. **Evaluate**: Compare baseline vs enhanced retrieval on your dataset
2. **Tune**: Adjust filtering thresholds and top-K parameters
3. **Extend**: Add custom event types or categories
4. **Integrate**: Use the retrieval API in your application

## Documentation

- **Full Documentation**: See `RETRIEVAL_README.md`
- **Project Summary**: See `PROJECT_SUMMARY.md`
- **Code Reference**: See inline comments in source files

## Support

For issues or questions:
1. Check the documentation files
2. Run the test script: `python test_enhanced_retrieval.py`
3. Enable debug logging in `memblock_extractor.py`

---

**Happy Memory Retrieval! 🚀**
