#!/usr/bin/env python3
"""
Resume-capable build stage that can continue from last processed session.
"""
import sys
import json
import os

from memblock_cli import main
from memblock_extractor import Config, logger

def load_resume_state(output_dir):
    """Load the last processed state from existing output files."""
    final_boxes_file = os.path.join(output_dir, "final_boxes_content.jsonl")

    if not os.path.exists(final_boxes_file):
        logger.info("ℹ️ No existing output found, starting from beginning")
        return None

    # Read last block to get resume info
    last_block = None
    with open(final_boxes_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                last_block = json.loads(line)

    if not last_block:
        return None

    last_block_id = last_block.get('block_id', -1)
    last_session = last_block.get('coverage', {}).get('session_id', '')

    # Extract session number
    session_num = None
    if last_session.startswith('session_'):
        try:
            session_num = int(last_session.split('_')[1])
        except:
            pass

    resume_info = {
        'last_block_id': last_block_id,
        'last_session': last_session,
        'last_session_num': session_num,
        'next_block_id': last_block_id + 1,
        'next_session_num': session_num + 1 if session_num is not None else None,
    }

    logger.info("📍 Resume state: last_block=%d, last_session=%s, next_session=%d",
                resume_info['last_block_id'],
                resume_info['last_session'],
                resume_info['next_session_num'] or 0)

    return resume_info


# Override configuration for faster builds
Config.TOPIC_CLASSIFY_TIMEOUT = 15.0
Config.TOPIC_CLASSIFY_MAX_RETRIES = 2
Config.EVENT_LABEL_TOP_K = 1

print("🚀 Fast Build Mode with Resume Support")
print(f"   TOPIC_CLASSIFY_TIMEOUT: {Config.TOPIC_CLASSIFY_TIMEOUT}s")
print(f"   TOPIC_CLASSIFY_MAX_RETRIES: {Config.TOPIC_CLASSIFY_MAX_RETRIES}")
print(f"   EVENT_LABEL_TOP_K: {Config.EVENT_LABEL_TOP_K}")
print()

if __name__ == "__main__":
    argv = sys.argv[1:]

    # Check if we should resume
    run_id = None
    for i, arg in enumerate(argv):
        if arg == "--run-id" and i + 1 < len(argv):
            run_id = argv[i + 1]
            break
        if arg.startswith("--run-id="):
            run_id = arg.split("=", 1)[1]
            break

    if not run_id:
        run_id = Config.RUN_ID

    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, run_id)
    resume_info = load_resume_state(output_dir)

    if resume_info:
        # Inject session skip logic into Config
        Config.RESUME_FROM_SESSION = resume_info['next_session_num']
        Config.RESUME_NEXT_BLOCK_ID = resume_info['next_block_id']
        logger.info("✅ Will resume from session_%d, starting at block_id=%d",
                    Config.RESUME_FROM_SESSION, Config.RESUME_NEXT_BLOCK_ID)
    else:
        Config.RESUME_FROM_SESSION = None
        Config.RESUME_NEXT_BLOCK_ID = None

    def _guess_dataset_format(args: list[str]) -> str:
        for i, token in enumerate(args):
            if token == "--raw-data-file" and i + 1 < len(args):
                p = (args[i + 1] or "").lower()
                if "longmemeval" in p:
                    return "longmemeval"
                if "locomo" in p:
                    return "locomo"
                return "default"
            if token.startswith("--raw-data-file="):
                p = token.split("=", 1)[1].lower()
                if "longmemeval" in p:
                    return "longmemeval"
                if "locomo" in p:
                    return "locomo"
                return "default"
        return "default"

    if "--dataset-format" not in argv:
        argv = ["--dataset-format", _guess_dataset_format(argv), *argv]
    main(argv=argv, default_stage="build")
