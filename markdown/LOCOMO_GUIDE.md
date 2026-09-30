# LoCoMo Generation & Evaluation Guide

## Overview

This guide covers running generation and evaluation for LoCoMo data, keeping the HaluMem code unchanged.

## Files Created

1. **generate_stage_locomo.py** - Generation stage for LoCoMo using old prompt
2. **evaluate_locomo.py** - Evaluation with F1, BLEU, and optional LLM-as-judge
3. **run_locomo_pipeline.sh** - Complete pipeline script

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
conda activate halu_mem0
cd /data/wjl/SA-Mem

# Run generation + evaluation
./run_locomo_pipeline.sh
```

This will:
1. Generate answers using the old LoCoMo prompt
2. Evaluate with F1 and BLEU scores
3. Optionally run LLM-as-judge evaluation

### Option 2: Run Steps Manually

#### Step 1: Generate Answers

```bash
conda activate halu_mem0

python generate_stage_locomo.py \
    --run-id locomo \
    --answer-topn 5 \
    --text-modes content
```

**Parameters:**
- `--run-id`: Output directory (default: "locomo")
- `--answer-topn`: Number of top blocks to use (default: 5)
- `--text-modes`: Text modes for generation (default: ["content"])
  - Options: `content`, `content_trace_event`, `trace_event`

**Output:**
- `out/locomo/generation_results_locomo.jsonl` - Generated answers
- `out/locomo/report_generation_qa_locomo.csv` - Results with F1/BLEU

#### Step 2: Evaluate Results

**Basic Evaluation (F1 & BLEU only):**
```bash
python evaluate_locomo.py --run-id locomo
```

**With LLM-as-Judge:**
```bash
python evaluate_locomo.py \
    --run-id locomo \
    --use-llm-judge
```

**Test on Sample:**
```bash
python evaluate_locomo.py \
    --run-id locomo \
    --use-llm-judge \
    --sample-size 10
```

**Output:**
- `out/locomo/evaluation_results_locomo.jsonl` - Detailed evaluation
- `out/locomo/evaluation_summary_locomo.json` - Summary statistics

## Key Differences from HaluMem

### 1. Prompt
- **HaluMem**: Uses `generate_prompts.py` (more detailed, synthesis-focused)
- **LoCoMo**: Uses `generate_prompts_old.py` (simpler, direct answer)

### 2. Output Files
- **HaluMem**: `generation_results.jsonl`, `report_generation_qa.csv`
- **LoCoMo**: `generation_results_locomo.jsonl`, `report_generation_qa_locomo.csv`

### 3. Evaluation
- **HaluMem**: F1 and BLEU only (computed during generation)
- **LoCoMo**: F1, BLEU, + optional LLM-as-judge scoring

## Evaluation Metrics

### 1. F1 Score
- Token-level overlap between predicted and gold answers
- Range: 0.0 to 1.0 (higher is better)
- Computed automatically during generation

### 2. BLEU Score
- N-gram overlap metric (unigram only)
- Range: 0.0 to 1.0 (higher is better)
- Computed automatically during generation

### 3. LLM-as-Judge (Optional)
- **Correctness** (0-5): How correct is the answer?
- **Completeness** (0-3): Does it include all key information?
- **Precision** (0-2): Is it concise and precise?
- **Total Score** (0-10): Sum of above

## Example Workflow

```bash
# 1. Activate environment
conda activate halu_mem0
cd /data/wjl/SA-Mem

# 2. Generate answers (uses retrieval_enhanced.jsonl)
python generate_stage_locomo.py --run-id locomo --answer-topn 5

# 3. Quick evaluation (F1 & BLEU)
python evaluate_locomo.py --run-id locomo

# 4. View results
head -1 out/locomo/generation_results_locomo.jsonl | python3 -m json.tool
cat out/locomo/evaluation_summary_locomo.json

# 5. Optional: LLM judge on sample
python evaluate_locomo.py --run-id locomo --use-llm-judge --sample-size 20
```

## Output Structure

### generation_results_locomo.jsonl
```json
{
  "user_id": "locomo10",
  "qa_idx": 0,
  "ranking_strategy": "baseline",
  "question": "When did Caroline go to the LGBTQ support group?",
  "gold": "7 May 2023",
  "pred": "7 May 2023",
  "f1": 1.0,
  "bleu": 1.0,
  "metric": "content_event_topic_kw",
  "topk": [31, 1, 19, 12, 7],
  "target_boxes": [31],
  "category": 2,
  "context_tokens": 1234
}
```

### evaluation_summary_locomo.json
```json
{
  "total_samples": 233,
  "overall": {
    "avg_f1": 0.7234,
    "avg_bleu": 0.6891,
    "avg_judge_score": 7.45
  },
  "by_category": {
    "1": {
      "count": 89,
      "avg_f1": 0.7123,
      "avg_bleu": 0.6745,
      "avg_judge_score": 7.23
    },
    "2": {
      "count": 144,
      "avg_f1": 0.7312,
      "avg_bleu": 0.6989,
      "avg_judge_score": 7.58
    }
  }
}
```

## Troubleshooting

### Generation fails with "Retrieval file not found"
```bash
# Check if retrieval was run
ls -lh out/locomo/retrieval_enhanced.jsonl

# If missing, run retrieval first
python retrieval/retrieve_stage_enhanced.py \
    --mode enhanced \
    --run-id locomo \
    --raw-data-file /data/locomo/data/locomo10.json \
    --limit-conversations 2
```

### No answers generated
- Check that memory blocks exist: `ls -lh out/locomo/final_boxes_content.jsonl`
- Check retrieval results: `wc -l out/locomo/retrieval_enhanced.jsonl`
- Check logs: `tail -50 out/locomo/generate_stage_locomo.log`

### LLM judge evaluation is slow
- Use `--sample-size` to test on subset first
- Each evaluation makes 1 LLM API call per QA pair
- For 233 QA pairs, expect ~5-10 minutes depending on API speed

## Cost Estimation

### Generation Stage
- **LLM calls**: 1 per QA pair × number of text modes
- **Example**: 233 QA pairs × 1 mode = 233 calls
- **Tokens**: ~1000-2000 tokens per call (depends on context size)

### LLM-as-Judge Evaluation
- **LLM calls**: 1 per QA pair
- **Example**: 233 QA pairs = 233 calls
- **Tokens**: ~200-500 tokens per call (smaller prompts)

### Total for 233 QA pairs
- **Generation**: ~233 calls, ~300K tokens
- **Evaluation**: ~233 calls, ~100K tokens
- **Total**: ~466 calls, ~400K tokens
- **Estimated cost** (gpt-4o-mini): ~$0.10-0.20

## Next Steps

After running evaluation, you can:

1. **Analyze results by category**:
   ```bash
   python3 << 'EOF'
   import json
   with open('out/locomo/evaluation_summary_locomo.json') as f:
       summary = json.load(f)
   for cat, metrics in summary['by_category'].items():
       print(f"Category {cat}: F1={metrics['avg_f1']:.3f}, BLEU={metrics['avg_bleu']:.3f}")
   EOF
   ```

2. **Find low-scoring examples**:
   ```bash
   python3 << 'EOF'
   import json
   with open('out/locomo/generation_results_locomo.jsonl') as f:
       results = [json.loads(line) for line in f]
   low_f1 = [r for r in results if r['f1'] < 0.3]
   print(f"Found {len(low_f1)} examples with F1 < 0.3")
   for r in low_f1[:5]:
       print(f"Q: {r['question']}")
       print(f"Gold: {r['gold']}, Pred: {r['pred']}, F1: {r['f1']:.3f}")
       print()
   EOF
   ```

3. **Compare with baseline**: Run with different `--answer-topn` values to see impact

## References

- **Generation prompt**: `generate_prompts_old.py`
- **Evaluation prompt**: `evaluate_locomo.py` (JUDGE_PROMPT)
- **Main implementation**: `generate_impl.py` (AnswerGenerator class)
