#!/usr/bin/env python3
"""
Fast build stage with optimized timeout and retry settings.
Use this for faster builds when API is slow.
"""
import sys

from memblock_cli import main
from memblock_extractor import Config

# Override configuration for faster builds
Config.TOPIC_CLASSIFY_TIMEOUT = 15.0  # Reduce from 60s to 15s
Config.TOPIC_CLASSIFY_MAX_RETRIES = 2  # Reduce from 4 to 2
Config.EVENT_LABEL_TOP_K = 1  # Reduce from 3 to 1 (fewer labels per event)

print("🚀 Fast Build Mode")
print(f"   TOPIC_CLASSIFY_TIMEOUT: {Config.TOPIC_CLASSIFY_TIMEOUT}s")
print(f"   TOPIC_CLASSIFY_MAX_RETRIES: {Config.TOPIC_CLASSIFY_MAX_RETRIES}")
print(f"   EVENT_LABEL_TOP_K: {Config.EVENT_LABEL_TOP_K}")
print("   Tip: add --graph-from-jsonl to build graph from memblock JSONL")
print()

if __name__ == "__main__":
    argv = sys.argv[1:]

    def _guess_dataset_format(args: list[str]) -> str:
        # Collect paths from --raw-data-file, --raw-data-dir, and --run-id
        paths: list[str] = []
        for i, token in enumerate(args):
            for flag in ("--raw-data-file", "--raw-data-dir", "--run-id"):
                if token == flag and i + 1 < len(args):
                    paths.append((args[i + 1] or "").lower())
                elif token.startswith(f"{flag}="):
                    paths.append(token.split("=", 1)[1].lower())
        for p in paths:
            if "longmemeval" in p or "lme_preprocessed" in p:
                return "longmemeval"
            if "locomo" in p:
                return "locomo"
        return "default"

    if "--dataset-format" not in argv:
        argv = ["--dataset-format", _guess_dataset_format(argv), *argv]
    main(argv=argv, default_stage="build")
