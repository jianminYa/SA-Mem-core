#!/usr/bin/env python3
"""
LLM as Judge Evaluation for SA-MEM Generation Results
Evaluates generated answers using LLM-based classification: Correct, Hallucination, Omission
"""

import os
import json
import re
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import config from memblock_extractor
from memblock_extractor import Config


EVALUATION_PROMPT_FOR_QUESTION = """You are an **evaluation expert for AI memory system question answering**.
Based **only** on the provided **"Question"**, **"Reference Answer"**, and **"Key Memory Points"** (the essential facts needed to derive the reference answer), strictly evaluate the **accuracy** of the **"Memory System Response."** Classify it as one of **"Correct"**, **"Hallucination"**, or **"Omission."** Do **not** use any external knowledge or subjective inference. Finally, output your judgment **strictly** in the specified JSON format.

# Evaluation Criteria

## Answer Type Classification

### 1. Correct

* The "Memory System Response" accurately answers the "Question," and its content is **semantically equivalent** to the "Reference Answer."
* It contains **no contradictions** with the "Key Memory Points" or "Reference Answer."
* It introduces **no unsupported details** beyond the "Key Memory Points" that could alter the conclusion.
* Synonyms, paraphrasing, and reasonable summarization are acceptable.

### 2. Hallucination

* The "Memory System Response" includes information or facts that **contradict or are inconsistent** with the "Reference Answer" or the "Key Memory Points."
* When the "Reference Answer" is labeled as *unknown/uncertain*, yet the response provides a specific verifiable fact or conclusion.
* Extra irrelevant information that does **not change** the conclusion is **not** considered hallucination by itself; however, if it **changes or misleads** the conclusion, or **contradicts** the "Key Memory Points," it should be judged as a **Hallucination**.

### 3. Omission

* The response is **incomplete** compared to the "Reference Answer."
* It explicitly states "don't know," "can't remember," or "no related memory," even though relevant information exists in the "Key Memory Points."
* For multi-element questions, **all elements must be correct and present**; omission of **any** element is considered an **Omission**.

## Priority Rules (Conflict Handling)

* If the response contains **both missing necessary information** and **fabricated/contradictory information**, classify it as **Hallucination**.
* If there is **no fabrication/contradiction** but some necessary information is missing, classify it as **Omission**.
* Only when the meaning is **fully equivalent** to the reference answer should it be classified as **Correct**.

## Detailed Guidelines and Tolerance

* Equivalent expressions of numbers, times, and units are acceptable, but the **numerical values themselves must not differ**.
* For multi-element questions, **all elements must be complete and accurate**; missing any element counts as **Omission**.
* If the reference answer is *"unknown / cannot be determined"* and the system provides a definite fact, that is a **Hallucination**.
  If the system also answers *"unknown"* (without guessing), it may be **Correct**.
* The evaluation must rely **only** on the *Reference Answer*, *Key Memory Points*, and *System Response* — no external context, world knowledge, or speculative reasoning is allowed.

# Information for Evaluation

* **Question:**
  {question}

* **Reference Answer:**
  {reference_answer}

* **Key Memory Points:**
  {key_memory_points}

* **Memory System Response:**
  {response}

# Output Requirements

Please provide your evaluation result **strictly** in the JSON format below.
Do **not** add any extra explanation or comments outside the JSON block.

```json
{{
  "reasoning": "Provide a concise and traceable evaluation rationale: first compare the system's response with the Key Memory Points (which were correctly used, which were missing, and whether there was any fabrication/contradiction), then assess its consistency with the Reference Answer, and finally state the classification basis.",
  "evaluation_result": "Correct | Hallucination | Omission"
}}
```
"""


class LLMJudgeEvaluator:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or Config.API_KEY
        self.base_url = base_url or Config.BASE_URL
        self.model = model or Config.LLM_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def evaluate_answer(self, question, reference_answer, key_memory_points, response):
        """
        Evaluate a single answer using LLM as Judge

        Args:
            question: The question string
            reference_answer: The gold-standard answer
            key_memory_points: The memory points (evidence) as string
            response: The system-generated answer

        Returns:
            dict with 'reasoning' and 'evaluation_result' (Correct/Hallucination/Omission)
        """
        prompt = EVALUATION_PROMPT_FOR_QUESTION.format(
            question=question,
            reference_answer=reference_answer,
            key_memory_points=key_memory_points,
            response=response
        )

        response_obj = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0
        )

        content = response_obj.choices[0].message.content or ""

        # Extract JSON block
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON block found in model output: {content}")

        json_str = match.group(1).strip()
        result = json.loads(json_str)

        return result


def load_qa_data(halumem_file, user_id):
    """Load QA data with evidence from HaluMem file"""
    qa_list = []

    with open(halumem_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['uuid'] == user_id:
                for session in data['sessions']:
                    if 'questions' in session:
                        for q in session['questions']:
                            # Extract evidence as formatted string
                            evidence_list = q.get('evidence', [])
                            if evidence_list:
                                evidence_str = "\n".join([
                                    mem['memory_content']
                                    for mem in evidence_list
                                ])
                            else:
                                evidence_str = "No specific memory points provided."

                            qa_list.append({
                                'question': q['question'],
                                'answer': q['answer'],
                                'evidence': evidence_str,
                                'difficulty': q.get('difficulty', 'unknown'),
                                'category': q.get('question_type', 'unknown')
                            })
                break

    return qa_list


def evaluate_generation_results(
    generation_file,
    halumem_file,
    output_file,
    max_workers=5
):
    """
    Evaluate generation results using LLM as Judge

    Args:
        generation_file: Path to generation_results.jsonl
        halumem_file: Path to HaluMem-Medium.jsonl
        output_file: Path to save evaluation results
        max_workers: Number of parallel workers
    """
    # Load generation results
    print(f"Loading generation results from {generation_file}...")
    gen_results = []
    with open(generation_file, 'r', encoding='utf-8') as f:
        for line in f:
            gen_results.append(json.loads(line))

    print(f"Loaded {len(gen_results)} generation results")

    # Get user_id from first result
    user_id = gen_results[0]['user_id']

    # Load QA data with evidence
    print(f"Loading QA data for user {user_id}...")
    qa_data = load_qa_data(halumem_file, user_id)
    print(f"Loaded {len(qa_data)} questions with evidence")

    # Create evaluator
    evaluator = LLMJudgeEvaluator()

    # Evaluate each result
    print(f"\nEvaluating with LLM as Judge (using {max_workers} workers)...")

    eval_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for gen_result in gen_results:
            qa_idx = gen_result['qa_idx']

            # Get corresponding QA data
            if qa_idx >= len(qa_data):
                print(f"Warning: qa_idx {qa_idx} out of range, skipping")
                continue

            qa = qa_data[qa_idx]

            # Submit evaluation task
            future = executor.submit(
                evaluator.evaluate_answer,
                qa['question'],
                qa['answer'],
                qa['evidence'],
                gen_result['pred']
            )
            futures[future] = (gen_result, qa)

        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            gen_result, qa = futures[future]

            try:
                llm_eval = future.result()
                eval_type = llm_eval.get('evaluation_result', 'Unknown')
                reasoning = llm_eval.get('reasoning', '')
            except Exception as e:
                print(f"\nError evaluating qa_idx {gen_result['qa_idx']}: {e}")
                eval_type = 'Error'
                reasoning = str(e)

            # Combine results
            result = {
                **gen_result,
                'llm_judge_result': eval_type,
                'llm_judge_reasoning': reasoning,
                'evidence': qa['evidence']
            }
            eval_results.append(result)

    # Sort by qa_idx
    eval_results.sort(key=lambda x: x['qa_idx'])

    # Save results
    print(f"\nSaving evaluation results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in eval_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Calculate statistics
    total = len(eval_results)
    correct = sum(1 for r in eval_results if r['llm_judge_result'] == 'Correct')
    hallucination = sum(1 for r in eval_results if r['llm_judge_result'] == 'Hallucination')
    omission = sum(1 for r in eval_results if r['llm_judge_result'] == 'Omission')
    error = sum(1 for r in eval_results if r['llm_judge_result'] == 'Error')

    stats = {
        'total': total,
        'correct': correct,
        'hallucination': hallucination,
        'omission': omission,
        'error': error,
        'correct_ratio': correct / total if total > 0 else 0,
        'hallucination_ratio': hallucination / total if total > 0 else 0,
        'omission_ratio': omission / total if total > 0 else 0,
        'error_ratio': error / total if total > 0 else 0
    }

    # Save statistics
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("LLM as Judge Evaluation Summary")
    print("="*60)
    print(f"Total: {total}")
    print(f"Correct: {correct} ({stats['correct_ratio']:.2%})")
    print(f"Hallucination: {hallucination} ({stats['hallucination_ratio']:.2%})")
    print(f"Omission: {omission} ({stats['omission_ratio']:.2%})")
    if error > 0:
        print(f"Error: {error} ({stats['error_ratio']:.2%})")
    print("="*60)
    print(f"\nResults saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SA-MEM generation results using LLM as Judge"
    )
    parser.add_argument(
        '--generation-file',
        type=str,
        default='out/data-1b846c59/generation_results.jsonl',
        help='Path to generation results file'
    )
    parser.add_argument(
        '--halumem-file',
        type=str,
        default='/data/HaluMem/yjm/data/HaluMem-Medium.jsonl',
        help='Path to HaluMem data file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save evaluation results (default: same dir as generation file)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Number of parallel workers for evaluation'
    )

    args = parser.parse_args()

    # Set default output file
    if args.output_file is None:
        base_dir = os.path.dirname(args.generation_file)
        args.output_file = os.path.join(base_dir, 'llm_judge_evaluation.jsonl')

    evaluate_generation_results(
        args.generation_file,
        args.halumem_file,
        args.output_file,
        args.max_workers
    )


if __name__ == '__main__':
    main()
