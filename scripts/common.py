#!/usr/bin/env python3
"""Small, shared utilities for the isolated LongMemEval phase-1 runners."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
import tiktoken


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "").rstrip("/")
    return value if value.endswith("/v1") else value + "/v1"


def load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE lines without printing or sourcing the file."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    os.environ.update(values)
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # A few upstream writers append JSONL directly to a file.  Give a writer
    # that has just closed its handle a short settling window, but never turn
    # malformed output into a silently accepted partial result.
    last_error: json.JSONDecodeError | None = None
    for attempt in range(61):
        try:
            rows = []
            # Iterate the file so only the JSONL newline byte ("\n")
            # separates records.  str.splitlines() also treats Unicode line
            # separators inside valid JSON string content as record breaks.
            with path.open("r", encoding="utf-8") as handle:
                lines = handle
                for line in lines:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == 60:
                raise
            time.sleep(1.0)
    assert last_error is not None
    raise last_error


def load_questions(path: Path, ids_path: Path) -> list[dict[str, Any]]:
    ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(ids)
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["question_id"]: item for item in data}
    missing = [qid for qid in ids if qid not in by_id]
    if missing:
        raise KeyError(f"missing question IDs: {missing}")
    return [by_id[qid] for qid in ids]


def encoder_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def history_stats(item: dict[str, Any], encoding) -> dict[str, int]:
    sessions = item.get("haystack_sessions") or []
    messages = 0
    chars = 0
    token_text: list[str] = []
    for session in sessions:
        for message in session or []:
            if not isinstance(message, dict):
                continue
            messages += 1
            text = str(message.get("content", "") or "")
            chars += len(text)
            token_text.append(f"{message.get('role', '')}: {text}")
    return {
        "sessions": len(sessions),
        "messages": messages,
        "characters": chars,
        "estimated_tokens": len(encoding.encode("\n".join(token_text))),
    }


def build_qa_prompt(item: dict[str, Any], memories: str) -> str:
    return (
        "You are a memory assistant. Answer the question using only the retrieved memories below. "
        "If the retrieved memories do not contain enough evidence, say that the information is unavailable. "
        "Keep the answer concise and directly answer the question.\n\n"
        f"Question time: {item.get('question_date', '')}\n"
        f"Question: {item.get('question', '')}\n\n"
        f"Retrieved memories:\n{memories}\n\n"
        "Answer:"
    )


class QAClient:
    def __init__(self, *, model: str, base_url: str, api_key: str, log_path: Path):
        self.model = model
        self.log_path = log_path
        self.client = OpenAI(api_key=api_key, base_url=normalize_base_url(base_url), timeout=90.0, max_retries=0)

    def call(self, item: dict[str, Any], memories: str) -> tuple[str, dict[str, Any]]:
        prompt = build_qa_prompt(item, memories)
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            usage = getattr(response, "usage", None)
            usage_available = usage is not None and getattr(usage, "prompt_tokens", None) is not None
            row = {
                "system": item.get("_system"),
                "question_id": item.get("question_id"),
                "stage": "qa",
                "model": self.model,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage_available else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage_available else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage_available else None,
                "latency_sec": round(time.perf_counter() - start, 6),
                "success": True,
                "provider_usage_available": bool(usage_available),
            }
            append_jsonl(self.log_path, row)
            return (response.choices[0].message.content or "").strip(), row
        except Exception as exc:
            row = {
                "system": item.get("_system"),
                "question_id": item.get("question_id"),
                "stage": "qa",
                "model": self.model,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency_sec": round(time.perf_counter() - start, 6),
                "success": False,
                "provider_usage_available": False,
                "error_type": type(exc).__name__,
            }
            append_jsonl(self.log_path, row)
            return "", row


def memories_from_boxes(boxes: dict[int, dict[str, Any]], ranking: list[int], top_k: int) -> str:
    rows = []
    for bid in ranking[:top_k]:
        box = boxes.get(int(bid))
        if not box:
            continue
        features = box.get("features", {}) or {}
        text = str(features.get("content_text", "") or "").strip()
        if not text:
            continue
        start = str(box.get("start_time") or box.get("temporal_index", {}).get("start", ""))
        rows.append((start, int(bid), text))
    rows.sort(key=lambda x: (x[0], x[1]))
    return "\n\n".join(row[2] for row in rows)


def evidence_hit(retrieved: Iterable[str], gold: Iterable[str]) -> bool:
    gold_set = {str(x) for x in gold}
    if not gold_set:
        return True
    return bool({str(x) for x in retrieved} & gold_set)
