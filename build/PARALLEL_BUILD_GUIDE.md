# Parallel Build Guide

## Problem
Building 70 sessions takes ~5 hours (266s per session) when processing sequentially.

## Solution: Split + Workers

The `--workers` parameter only works with multiple files. By splitting your single JSON file into multiple files (one per conversation), you can process them in parallel.

### Why Splitting is Safe

Each conversation is processed independently:
- `build_impl.py:186-189` resets state for each conversation (cluster, boxes, msgs, bid)
- Sliding window (`BUILD_PREV_MSGS`) only uses `self.msgs` which is conversation-scoped
- **Splitting at conversation boundaries produces identical results**

## Step-by-Step Instructions

### 1. Split Your File

```bash
# Split into individual files (1 conversation per file)
python split_conversations.py \
    --input data/processed_halumem/1b846c59-*.json \
    --output-dir data/split \
    --convs-per-file 1
```

This creates files like:
```
data/split/1b846c59-xxx_part000.json
data/split/1b846c59-xxx_part001.json
...
data/split/1b846c59-xxx_part069.json
```

### 2. Run Build with Workers

```bash
# Use 8 workers for parallel processing
python build_stage_fast.py \
    --raw-data-dir data/split \
    --pattern "*.json" \
    --workers 8
```

### Performance Estimate

- **Sequential**: 70 sessions × 266s = 5 hours
- **8 workers**: 70 sessions ÷ 8 × 266s = ~37 minutes (8x speedup)
- **16 workers**: 70 sessions ÷ 16 × 266s = ~19 minutes (16x speedup)

**Note**: Actual speedup depends on:
- CPU cores available
- API rate limits (if using external LLM)
- I/O bottlenecks

### 3. Verify Results

After build completes, verify the output:

```bash
# Check number of blocks created
wc -l out/final_boxes_content.jsonl

# Check build stats
cat out/build_stats.jsonl | jq
```

## Alternative: Adjust Conversations Per File

If you want fewer files, group multiple conversations:

```bash
# 5 conversations per file = 14 files
python split_conversations.py \
    --input data/processed_halumem/1b846c59-*.json \
    --output-dir data/split \
    --convs-per-file 5

# Then use 4-8 workers
python build_stage_fast.py \
    --raw-data-dir data/split \
    --pattern "*.json" \
    --workers 4
```

## Checkpoint/Resume Still Works

The checkpoint mechanism works with `--raw-data-dir`:
- Progress is saved per file
- If interrupted, resume with same command
- Already-processed files are skipped

## Troubleshooting

### API Rate Limits
If you hit API rate limits with many workers:
- Reduce `--workers` to 2-4
- Add delays in the code (not recommended)
- Use `--convs-per-file` to create fewer, larger files

### Memory Issues
If running out of memory:
- Reduce `--workers`
- Process in batches (split into subdirectories)

### Verification
To verify splitting produces identical results:

```bash
# Build original file
python build_stage_fast.py \
    --raw-data-file data/processed_halumem/1b846c59-*.json \
    --output-dir out/original

# Build split files
python build_stage_fast.py \
    --raw-data-dir data/split \
    --pattern "*.json" \
    --workers 1 \
    --output-dir out/split

# Compare outputs (should be identical except for order)
diff <(sort out/original/final_boxes_content.jsonl) \
     <(sort out/split/final_boxes_content.jsonl)
```
