#!/usr/bin/env python3
"""
generate_lme.py  —  SA-Mem retrieval → LongMemEval 答案生成 + LLM-as-Judge 评测

参照 LightMem 的做法：
  1. 用 retrieval 排序取 Top-K block 的原文作为 memories
  2. 直接 prompt: "Question time:... and question:...\n Please answer based on memories: ..."
  3. 内联 LLM-as-Judge 评测（与 LongMemEval evaluate_qa.py 相同的 prompt）

用法:
    python scripts/generate_lme.py \
        --retrieval-jsonl  out/longmemeval_s_merged/retrieval_enhanced.jsonl \
        --boxes-jsonl      out/longmemeval_s_merged/final_boxes_content.jsonl \
        --output-dir       out/longmemeval_s_merged/generation \
        --topk 10 \
        --model gpt-4o-mini \
        --judge-model gpt-4o-mini
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LLM-as-Judge prompts (identical to LongMemEval evaluate_qa.py)
# ---------------------------------------------------------------------------

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unknown task: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        return template.format(question, answer, response)


def true_or_false(response):
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace('.', '').replace('!', '').replace(':', '').replace(';', '').split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

class LLMModel:
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        self.name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens = 2000
        self.temperature = 0.0
        self.top_p = 0.8

    def call(self, messages: list, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(3)
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_boxes(boxes_jsonl: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    user_boxes: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    with open(boxes_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            uid = b.get("user_id")
            bid = b.get("block_id")
            if uid is not None and bid is not None:
                user_boxes[uid][bid] = b
    return dict(user_boxes)


def get_block_text(box: Dict[str, Any]) -> str:
    """从 block 中提取 content_text（包含原始对话）。"""
    features = box.get("features", {})
    content_text = features.get("content_text", "")
    if content_text:
        return content_text
    # Fallback
    parts = []
    topic_kw = features.get("topic_kw_text", "")
    if topic_kw:
        parts.append(f"Topics: {topic_kw}")
    events = box.get("events", [])
    descs = [e.get("description", "") for e in events if e.get("description")]
    if descs:
        parts.append("Events:\n" + "\n".join(f"- {d}" for d in descs))
    return "\n\n".join(parts)


def build_memories_string(
    user_boxes: Dict[int, Dict[str, Any]],
    ranking: List[int],
    topk: int,
) -> str:
    """取 Top-K blocks 的 content_text，按时间排序后拼接。"""
    selected_ids = ranking[:topk]
    items = []
    for bid in selected_ids:
        box = user_boxes.get(bid)
        if box is None:
            continue
        text = get_block_text(box)
        if not text:
            continue
        ti = box.get("temporal_index", {})
        sort_key = ti.get("start", "") or ""
        items.append((sort_key, bid, text))

    items.sort(key=lambda x: x[0])
    parts = []
    for sort_key, bid, text in items:
        parts.append(text)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SA-Mem → LongMemEval generation + evaluation")
    p.add_argument("--retrieval-jsonl", required=True, help="SA-Mem retrieval result JSONL")
    p.add_argument("--boxes-jsonl", required=True, help="SA-Mem final_boxes_content.jsonl")
    p.add_argument("--output-dir", required=True, help="Output directory for results")
    p.add_argument("--topk", type=int, default=10, help="Top-K blocks as context (default: 10)")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="Generation LLM")
    p.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="Judge LLM for evaluation")
    p.add_argument("--api-key", type=str, default=None, help="API key")
    p.add_argument("--base-url", type=str, default=None, help="API base URL")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    return p.parse_args()


def main():
    args = parse_args()

    # Setup API
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", None)
    if not api_key:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from memblock_extractor import Config
            api_key = Config.API_KEY
            base_url = base_url or Config.BASE_URL
        except Exception:
            pass
    if not api_key:
        print("ERROR: No API key. Set OPENAI_API_KEY or use --api-key.")
        sys.exit(1)

    llm = LLMModel(args.model, api_key, base_url)
    llm_judge = LLMModel(args.judge_model, api_key, base_url)

    # Load data
    print(f"Loading boxes from {args.boxes_jsonl} ...")
    all_boxes = load_boxes(args.boxes_jsonl)
    total_blocks = sum(len(v) for v in all_boxes.values())
    print(f"  {total_blocks} blocks, {len(all_boxes)} users")

    print(f"Loading retrieval results from {args.retrieval_jsonl} ...")
    with open(args.retrieval_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(entries)} questions")

    # Output setup
    os.makedirs(args.output_dir, exist_ok=True)
    hypothesis_file = os.path.join(args.output_dir, "generation_hypothesis.jsonl")
    results_file = os.path.join(args.output_dir, "generation_results.json")

    # Resume: load already-done question_ids
    done_qids = set()
    existing_results = []
    if os.path.exists(results_file) and not args.overwrite:
        with open(results_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        done_qids = {r["question_id"] for r in existing_results}
        print(f"  Resuming: {len(done_qids)} already done")

    results = list(existing_results)
    type2correct = defaultdict(list)

    # Count existing correct stats
    for r in existing_results:
        type2correct[r.get("question_type", "unknown")].append(r.get("correct", 0))

    # Generate + Judge
    for entry in tqdm(entries, desc="Generate & Judge"):
        qid = entry["question_id"]
        if qid in done_qids:
            continue

        uid = entry.get("user_id", qid)
        question = entry["question"]
        question_date = entry.get("question_date", "")
        question_type = entry.get("question_type", "")
        answer = entry.get("answer", "")
        is_abstention = "_abs" in qid

        # Get ranking
        rankings = entry.get("rankings", {})
        ranking = rankings.get("content_event_topic_kw", [])
        if not ranking:
            print(f"  WARNING: No ranking for {qid}, skipping")
            continue

        user_boxes = all_boxes.get(uid, {})
        if not user_boxes:
            print(f"  WARNING: No blocks for user {uid}, skipping")
            continue

        # Build memories context
        memories_str = build_memories_string(user_boxes, ranking, args.topk)
        if not memories_str:
            print(f"  WARNING: Empty context for {qid}, skipping")
            continue

        # --- Step 1: Generate answer (LightMem style) ---
        gen_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question time:{question_date} and question:{question}\nPlease answer the question based on the following memories: {memories_str}"},
        ]
        generated_answer = llm.call(gen_messages) or ""

        # --- Step 2: LLM-as-Judge ---
        judge_prompt = get_anscheck_prompt(
            question_type, question, answer, generated_answer, abstention=is_abstention
        )
        judge_response = llm_judge.call([{"role": "user", "content": judge_prompt}])
        correct = 1 if true_or_false(judge_response) else 0

        result = {
            "question_id": qid,
            "question_type": question_type,
            "question": question,
            "ground_truth": answer,
            "hypothesis": generated_answer,
            "judge_response": judge_response,
            "correct": correct,
        }
        results.append(result)
        type2correct[question_type].append(correct)

        tqdm.write(
            f"  [{qid}] type={question_type} correct={correct}\n"
            f"    Q: {question[:80]}\n"
            f"    Gold: {str(answer)[:80]}\n"
            f"    Pred: {generated_answer[:120]}"
        )

    # --- Save results ---
    # 1. Full results JSON
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 2. Hypothesis JSONL (LongMemEval evaluate_qa.py compatible)
    with open(hypothesis_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "question_id": r["question_id"],
                "hypothesis": r["hypothesis"],
            }, ensure_ascii=False) + "\n")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"Results: {len(results)} questions")
    print(f"\nAccuracy by question type:")
    all_correct = []
    task_accs = []
    for qt in sorted(type2correct.keys()):
        vals = type2correct[qt]
        acc = sum(vals) / len(vals) if vals else 0.0
        print(f"  {qt}: {acc:.4f} ({sum(vals)}/{len(vals)})")
        all_correct.extend(vals)
        task_accs.append(acc)

    if all_correct:
        print(f"\nTask-averaged Accuracy: {sum(task_accs)/len(task_accs):.4f}")
        print(f"Overall Accuracy: {sum(all_correct)/len(all_correct):.4f}")

    abstention_correct = [r["correct"] for r in results if "_abs" in r["question_id"]]
    if abstention_correct:
        print(f"Abstention Accuracy: {sum(abstention_correct)/len(abstention_correct):.4f} ({len(abstention_correct)})")

    print(f"\nOutput files:")
    print(f"  {results_file}")
    print(f"  {hypothesis_file}")


if __name__ == "__main__":
    main()
