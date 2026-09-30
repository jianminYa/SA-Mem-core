#!/usr/bin/env python3
"""
Generation stage specifically for LoCoMo data.
Uses the old prompt format from generate_prompts_old.py
LoCoMo-specific version with global qa_idx support for multi-conversation files.
"""
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import memblock_extractor as mx
from generate_impl_locomo import AnswerGenerator
from generate_prompts_old import PROMPT_QA_ANSWER as PROMPT_QA_ANSWER_LOCOMO


def main():
    parser = argparse.ArgumentParser(description="Generate answers for LoCoMo data")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=mx.Config.LLM_PROVIDER,
        help="LLM provider selector: openai (default) or ollama (local).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name (applies to selected provider).",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="Override Ollama OpenAI-compatible base URL (default: http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--ollama-api-key",
        type=str,
        default=None,
        help="Override Ollama API key (default: ollama).",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=None,
        help="Override Ollama context length (num_ctx).",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=None,
        help="Override Ollama max tokens to generate (num_predict).",
    )
    parser.add_argument("--run-id", type=str, default="locomo", help="Run ID for output directory")
    parser.add_argument("--retrieval-file", type=str, default=None,
                       help="Path to retrieval results JSONL file")
    parser.add_argument("--answer-topn", type=int, default=5,
                       help="Number of top blocks to use for answer generation")
    parser.add_argument("--text-modes", type=str, nargs="+", default=["content"],
                       help="Text modes for generation: content (Membox style, original text only), event (SA-Mem style, events with temporal metadata), content_trace_event, trace_event")
    parser.add_argument("--trace-event-topk", type=int, default=50,
                       help="Per-trace events truncation cap when building trace contexts (default: 50, 0 means no truncation)")
    parser.add_argument("--use-graph-context", action="store_true",
                       help="Append graph expansion context to the prompt (requires retrieval JSONL with graph field).")
    parser.add_argument(
        "--graph-context-categories",
        type=str,
        default=None,
        help=(
            "Comma-separated LoCoMo QA categories to enable graph context injection for (e.g. '3,4'). "
            "Only takes effect when --use-graph-context is set. "
            "If omitted, graph context is injected for all categories."
        ),
    )
    parser.add_argument("--limit-conversations", type=int, default=None,
                       help="Limit number of conversations to process (must match build/retrieval)")
    parser.add_argument("--raw-data-file", type=str, default=None,
                       help="Path to raw data file (needed for loading QA pairs)")
    parser.add_argument("--final-content-file", type=str, default=None,
                       help="Path to final_boxes_content.jsonl (memory blocks)")
    # 添加一个用于区分不同生成实验的后缀参数
    parser.add_argument("--output-suffix", type=str, default="",
                       help="Suffix for output files (e.g. 'top20')")
    
    args = parser.parse_args()

    # Apply run_id configuration
    mx.Config.apply_run_id(args.run_id)

    # LLM provider/model overrides
    mx.Config.LLM_PROVIDER = (args.provider or mx.Config.LLM_PROVIDER or "openai").strip().lower()
    if args.ollama_base_url is not None:
        mx.Config.OLLAMA_BASE_URL = args.ollama_base_url
    if args.ollama_api_key is not None:
        mx.Config.OLLAMA_API_KEY = args.ollama_api_key
    if args.llm_model:
        if mx.Config.LLM_PROVIDER == "ollama":
            mx.Config.OLLAMA_LLM_MODEL = args.llm_model
        else:
            mx.Config.LLM_MODEL = args.llm_model
    if args.ollama_num_ctx is not None:
        mx.Config.OLLAMA_NUM_CTX = int(args.ollama_num_ctx)
    if args.ollama_num_predict is not None:
        mx.Config.OLLAMA_NUM_PREDICT = int(args.ollama_num_predict)

    # Set limit conversations if provided
    if args.limit_conversations is not None:
        mx.Config.LIMIT_CONVERSATIONS = (
            None if args.limit_conversations == -1
            else max(0, args.limit_conversations)
        )
        mx.logger.info(f"ℹ️ LIMIT_CONVERSATIONS set to {mx.Config.LIMIT_CONVERSATIONS}")

    # Set raw data file if provided
    if args.raw_data_file:
        mx.Config.RAW_DATA_FILE = args.raw_data_file

    # Set final content file if provided
    if args.final_content_file:
        mx.Config.FINAL_CONTENT_FILE = args.final_content_file

    # Override prompt with LoCoMo-specific prompt
    mx.Config.PROMPT_QA_ANSWER = PROMPT_QA_ANSWER_LOCOMO

    # Set answer topn
    if args.answer_topn:
        mx.Config.ANSWER_TOP_N = args.answer_topn

    # Set text modes
    if args.text_modes:
        mx.Config.GEN_TEXT_MODES = args.text_modes

    # Trace event topk for trace context building
    mx.Config.TRACE_EVENT_TOPK = int(args.trace_event_topk) if args.trace_event_topk is not None else 50

    # Determine retrieval file path
    if args.retrieval_file:
        retrieval_jsonl = args.retrieval_file
    else:
        retrieval_jsonl = os.path.join(mx.Config.OUTPUT_DIR, "retrieval_enhanced.jsonl")

    mx.Config.USE_GRAPH_CONTEXT = bool(args.use_graph_context)

    # Optional: only inject graph context for selected LoCoMo QA categories.
    # Categories are integer IDs in raw qa entries (e.g., 3=Temporal, 4=Open Domain in official setup).
    if args.graph_context_categories is not None:
        raw = str(args.graph_context_categories).strip()
        if raw:
            cats = set()
            for part in raw.replace(" ", "").split(","):
                if not part:
                    continue
                try:
                    cats.add(int(part))
                except Exception:
                    continue
            mx.Config.GRAPH_CONTEXT_CATEGORIES = cats
        else:
            mx.Config.GRAPH_CONTEXT_CATEGORIES = set()
        mx.logger.info("ℹ️ GRAPH_CONTEXT_CATEGORIES=%s", sorted(list(mx.Config.GRAPH_CONTEXT_CATEGORIES)))

    # # Output files
    # output_jsonl = os.path.join(mx.Config.OUTPUT_DIR, "generation_results_locomo.jsonl")
    # output_csv = os.path.join(mx.Config.OUTPUT_DIR, "report_generation_qa_locomo.csv")

    # Output files (avoid long suffixes; optional custom suffix only)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_jsonl = os.path.join(
        mx.Config.OUTPUT_DIR,
        f"generation_results_locomo{suffix}.jsonl",
    )
    output_csv = os.path.join(
        mx.Config.OUTPUT_DIR,
        f"report_generation_qa_locomo{suffix}.csv",
    )


    mx.logger.info("=" * 60)
    mx.logger.info("🚀 LoCoMo Generation Stage")
    mx.logger.info("=" * 60)
    mx.logger.info(f"Run ID: {args.run_id}")
    mx.logger.info(f"Output directory: {mx.Config.OUTPUT_DIR}")
    mx.logger.info(f"Retrieval file: {retrieval_jsonl}")
    mx.logger.info(f"Answer TopN: {mx.Config.ANSWER_TOP_N}")
    mx.logger.info(f"Text modes: {mx.Config.GEN_TEXT_MODES}")
    mx.logger.info(f"Trace event topk: {getattr(mx.Config, 'TRACE_EVENT_TOPK', 50)}")
    mx.logger.info(f"Limit conversations: {mx.Config.LIMIT_CONVERSATIONS}")
    mx.logger.info(f"LLM provider: {mx.Config.LLM_PROVIDER}")
    mx.logger.info(f"LLM model: {mx.Config.effective_llm_model()}")
    mx.logger.info(f"LLM base_url: {mx.Config.effective_base_url()}")
    if mx.Config.LLM_PROVIDER == "ollama":
        mx.logger.info(f"Ollama num_ctx: {getattr(mx.Config, 'OLLAMA_NUM_CTX', None)}")
        mx.logger.info(f"Ollama num_predict: {getattr(mx.Config, 'OLLAMA_NUM_PREDICT', None)}")
    mx.logger.info(f"Using LoCoMo prompt from generate_prompts_old.py")
    mx.logger.info("=" * 60)

    # Check if retrieval file exists
    if not os.path.exists(retrieval_jsonl):
        mx.logger.error(f"❌ Retrieval file not found: {retrieval_jsonl}")
        mx.logger.error("Please run retrieval stage first!")
        sys.exit(1)

    # Initialize worker (reads from Config)
    worker = mx.LLMWorker()

    # Initialize generator
    generator = AnswerGenerator(
        worker=worker,
        answer_topn=mx.Config.ANSWER_TOP_N,
        text_modes=mx.Config.GEN_TEXT_MODES,
        stage_label="gen_locomo"
    )

    # Run generation
    mx.logger.info("🔄 Starting generation...")
    generator.run(
        retrieval_jsonl=retrieval_jsonl,
        base_out_jsonl=output_jsonl,
        base_out_csv=output_csv
    )

    mx.logger.info("=" * 60)
    mx.logger.info("✅ LoCoMo Generation Complete!")
    mx.logger.info(f"📄 Results saved to:")
    mx.logger.info(f"   - {output_jsonl}")
    mx.logger.info(f"   - {output_csv}")
    mx.logger.info("=" * 60)


if __name__ == "__main__":
    main()
