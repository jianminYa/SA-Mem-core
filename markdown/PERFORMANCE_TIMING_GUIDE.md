# Performance Timing Instrumentation Guide

## Overview

Timing instrumentation has been added to the enhanced retrieval pipeline to measure where time is being spent during query processing. This allows you to identify bottlenecks and optimize performance.

## What Was Added

### 1. Main Retrieval Timing (retrieval_enhanced.py)

**Location**: `_score_and_rank()` method (lines 325-414)

**Timing Components**:
- `t_parse`: Query parsing time (LLM call or fast path)
- `t_filter`: Metadata filtering time (temporal + event type)
- `t_rank`: Vector similarity ranking time
- `t_total`: Total query processing time

**Log Format**:
```
⏱️ Timing: parse=0.123s, filter=0.456s, rank=0.789s, total=1.368s
```

### 2. Metadata Filtering Timing (retrieval_enhanced.py)

**Location**: `_filter_by_metadata()` method (lines 103-295)

**Timing Components**:
- `t_temporal`: Temporal filtering time (includes anchor resolution if applicable)
- `t_anchor`: Anchor resolution time (only for ANCHOR queries)
- `t_event_type`: Event type filtering time

**Log Format**:
```
⏱️ Filter timing: temporal=0.456s (anchor=0.400s), event_type=0.001s, total=0.457s
```

### 3. Anchor Resolution Timing (anchor_resolver.py)

**Location**: `resolve_anchor()` method (lines 38-92)

**Timing Components**:
- `t_find`: Time to find anchor blocks via vector similarity
- `t_extract`: Time to extract time range from anchor blocks
- `total`: Total anchor resolution time

**Log Format**:
```
⏱️ Anchor resolution timing: find_blocks=0.400s, extract_time=0.001s, total=0.401s
```

## How to Use

### Run Enhanced Retrieval with Timing

```bash
conda activate halu_mem0
python retrieve_stage_enhanced.py --mode enhanced --run-id perf_analysis
```

**Log file location**: `out/perf_analysis/retrieve_stage_enhanced.log`

The script will automatically:
- Output logs to console (stdout)
- Write logs to file: `out/<run_id>/retrieve_stage_enhanced.log`

### Analyze the Logs

You can analyze the timing logs in two ways:

**Option 1: Use the analysis script**
```bash
# Analyze by directory (automatically finds retrieve_stage_enhanced.log)
python analyze_timing.py out/perf_analysis/

# Or specify log file directly
python analyze_timing.py out/perf_analysis/retrieve_stage_enhanced.log
```

**Option 2: Manual inspection**
```bash
# View the log file
less out/perf_analysis/retrieve_stage_enhanced.log

# Search for timing lines
grep "⏱️ Timing:" out/perf_analysis/retrieve_stage_enhanced.log

# Search for slow queries (>2s)
grep "⏱️ Timing:" out/perf_analysis/retrieve_stage_enhanced.log | awk -F'total=' '{print $2}' | awk '{if ($1 > 2.0) print}'
```

The logs will show timing for each query. Example:

```
🔍 DEBUG: Initial pool size = 273 blocks
📝 Query parsed: intent=PLANNING, time=ANCHOR, source=LLM
🔍 ANCHOR query detected: 'after retirement' AFTER
✅ Anchor resolved: 'after retirement' AFTER -> [2036-07-30, 2046-07-28]
⏱️ Anchor resolution timing: find_blocks=1.234s, extract_time=0.002s, total=1.236s
🔍 Anchor temporal(event) [2036-07-30, 2046-07-28] -> 5/273 blocks
⏱️ Filter timing: temporal=1.250s (anchor=1.236s), event_type=0.001s, total=1.251s
⏱️ Timing: parse=0.150s, filter=1.251s, rank=0.300s, total=1.701s
```

**Interpretation**:
- Query parsing took 0.150s (LLM call)
- Metadata filtering took 1.251s total:
  - Temporal filtering: 1.250s (mostly anchor resolution)
  - Anchor resolution: 1.236s (finding matching blocks)
  - Event type filtering: 0.001s (negligible)
- Vector ranking took 0.300s (embedding + similarity computation)
- Total query time: 1.701s

## Performance Analysis

### Expected Timing Breakdown

**Fast queries (no anchor, cached embeddings)**:
- Parse: 0.05-0.15s (fast path) or 0.1-0.3s (LLM)
- Filter: 0.001-0.01s (temporal index lookup)
- Rank: 0.05-0.2s (cached embeddings)
- **Total**: 0.1-0.5s

**Anchor queries (first run, uncached)**:
- Parse: 0.1-0.3s (LLM)
- Filter: 1-2s (anchor resolution with 273 blocks)
  - Anchor: 1-2s (273 embedding API calls + similarity)
- Rank: 0.2-0.5s (embedding API calls for filtered pool)
- **Total**: 1.5-3s

**Anchor queries (subsequent runs, cached)**:
- Parse: 0.1-0.3s (LLM, but query parsing is cached)
- Filter: 0.1-0.3s (anchor resolution with cached embeddings)
  - Anchor: 0.1-0.3s (cached embeddings, only similarity)
- Rank: 0.05-0.2s (cached embeddings)
- **Total**: 0.3-0.8s

### Bottleneck Identification

**If anchor resolution is slow (>1s)**:
- First run: Expected (273 embedding API calls)
- Subsequent runs: Should be faster (<0.3s) due to caching
- If still slow: Check network latency to API endpoint

**If query parsing is slow (>0.5s)**:
- LLM API latency issue
- Consider improving fast path coverage to avoid LLM calls

**If vector ranking is slow (>0.5s)**:
- Large filtered pool (many blocks to rank)
- Uncached embeddings (API calls)
- Check if embeddings are being cached properly

## Aggregate Statistics

To analyze overall performance, you can extract timing from logs:

```bash
# Extract all timing logs
grep "⏱️ Timing:" out/perf_analysis/retrieve_stage_enhanced.log > timing_stats.txt

# Calculate average times
python -c "
import re
with open('timing_stats.txt') as f:
    lines = f.readlines()

parse_times = []
filter_times = []
rank_times = []
total_times = []

for line in lines:
    match = re.search(r'parse=([\d.]+)s, filter=([\d.]+)s, rank=([\d.]+)s, total=([\d.]+)s', line)
    if match:
        parse_times.append(float(match.group(1)))
        filter_times.append(float(match.group(2)))
        rank_times.append(float(match.group(3)))
        total_times.append(float(match.group(4)))

print(f'Average parse time: {sum(parse_times)/len(parse_times):.3f}s')
print(f'Average filter time: {sum(filter_times)/len(filter_times):.3f}s')
print(f'Average rank time: {sum(rank_times)/len(rank_times):.3f}s')
print(f'Average total time: {sum(total_times)/len(total_times):.3f}s')
print(f'Total queries: {len(total_times)}')
print(f'Total time: {sum(total_times):.1f}s')
"
```

## Optimization Strategies

Based on timing analysis, you can optimize:

### 1. If Anchor Resolution is the Bottleneck

**Strategy**: Reduce pool size before anchor resolution
- Pre-filter by session time (e.g., only last 5 years)
- Expected speedup: 2-5x

**Strategy**: Use approximate nearest neighbor (ANN) search
- Replace linear scan with FAISS/Annoy
- Expected speedup: 10-100x for large pools

**Strategy**: Batch embedding API calls
- OpenAI supports batch requests
- Expected speedup: 1.5-2x

### 2. If Query Parsing is the Bottleneck

**Strategy**: Improve fast path coverage
- Add more regex patterns
- Reduce LLM fallback rate
- Expected speedup: 2-3x for common queries

**Strategy**: Verify query parsing cache
- Check if cache is working properly
- Expected: Near-instant for repeated queries

### 3. If Vector Ranking is the Bottleneck

**Strategy**: Reduce text length
- Truncate long event descriptions
- Expected speedup: 1.2-1.5x

**Strategy**: Batch embedding API calls
- Process multiple blocks in one request
- Expected speedup: 1.5-2x

## Next Steps

1. Run enhanced retrieval with timing instrumentation
2. Analyze timing logs to identify bottlenecks
3. Calculate aggregate statistics
4. Implement targeted optimizations based on findings
5. Re-run and compare before/after performance

## Notes

- Timing uses `time.perf_counter()` for high-resolution measurements
- All times are in seconds with 3 decimal places
- Caching significantly affects performance (first run vs subsequent runs)
- Network latency to API endpoint affects embedding and LLM call times
