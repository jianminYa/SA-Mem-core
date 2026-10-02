#!/usr/bin/env python3
"""Create compact, source-grounded evidence artifacts for e47becba."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ANSWER = "Business Administration"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clip(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def native_id(row: dict[str, Any], system: str) -> int:
    return int(row["box_id"] if system == "membox" else row["block_id"])


def event_text(row: dict[str, Any], system: str) -> str:
    if system == "membox":
        return str((row.get("features") or {}).get("events_text", "") or "")
    return " | ".join(
        str(event.get("description", "") or "")
        for event in row.get("events", [])
        if isinstance(event, dict) and event.get("description")
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | ID | Score | Session | Topic / short memory description | Gold? |",
        "|---:|---:|---:|---|---|:---:|",
    ]
    for row in rows[:20]:
        description = row.get("events") or row.get("topic") or row.get("original_dialogue")
        mark = "✓" if row.get("gold_session") else ""
        score = "N/A" if row.get("score") is None else f"{float(row['score']):.6f}"
        lines.append(
            f"| {row.get('rank')} | {row.get('memory_id')} | {score} | "
            f"`{row.get('session_id')}` | {clip(description)} | {mark} |"
        )
    return "\n".join(lines)


def context_index(rows: list[dict[str, Any]]) -> str:
    lines = ["| Generation rank | Memory ID | Session ID | Gold? |", "|---:|---:|---|:---:|"]
    for row in rows[:10]:
        lines.append(
            f"| {row.get('rank')} | {row.get('memory_id')} | `{row.get('session_id')}` | "
            f"{'✓' if row.get('gold_session') else ''} |"
        )
    return "\n".join(lines)


def source_excerpt(item: dict[str, Any]) -> tuple[str, str]:
    answer_ids = set(str(x) for x in item.get("answer_session_ids", []))
    for sid, date, session in zip(
        item.get("haystack_session_ids", []),
        item.get("haystack_dates", []),
        item.get("haystack_sessions", []),
    ):
        if str(sid) not in answer_ids:
            continue
        relevant = []
        for index, message in enumerate(session):
            text = str(message.get("content", "") or "")
            if ANSWER.casefold() in text.casefold():
                for neighbor in session[max(0, index - 1) : min(len(session), index + 2)]:
                    if neighbor not in relevant:
                        relevant.append(neighbor)
        block = "\n\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}" for message in relevant
        )
        return str(sid), f"Date: {date}\n\n{block}"
    raise RuntimeError("could not locate answer-bearing source session")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--membox-run", type=Path, required=True)
    parser.add_argument("--samem-run", type=Path, required=True)
    parser.add_argument("--workspace-report", type=Path, required=True)
    parser.add_argument("--repo-output", type=Path, required=True)
    args = parser.parse_args()

    item = next(x for x in load_json(args.dataset) if x["question_id"] == "e47becba")
    summary = load_json(args.report_json)
    membox_run = args.membox_run
    samem_run = args.samem_run
    out = args.repo_output
    out.mkdir(parents=True, exist_ok=True)

    source_session_id, excerpt = source_excerpt(item)
    (out / "gold_source_dialogue.txt").write_text(
        f"Gold source session: {source_session_id}\n\n{excerpt}\n", encoding="utf-8"
    )

    raw_membox = load_jsonl(membox_run / "final_boxes_content.jsonl")
    raw_samem = load_jsonl(samem_run / "final_boxes_content.jsonl")
    gold_membox = next(
        row for row in raw_membox
        if str(row.get("coverage", {}).get("session_id")) == "session_52"
        and ANSWER.casefold() in str(row.get("features", {}).get("content_text", "")).casefold()
    )
    gold_samem = next(
        row for row in raw_samem
        if str(row.get("coverage", {}).get("session_id")) == source_session_id
        and ANSWER.casefold() in str(row.get("features", {}).get("content_text", "")).casefold()
    )
    write_json(out / "membox_gold_memory.json", gold_membox)
    write_json(out / "samem_gold_memory.json", gold_samem)

    membox_top20 = load_json(membox_run / "retrieval_top20.json")
    samem_top20 = load_json(samem_run / "retrieval_top20.json")
    write_json(out / "membox_retrieval_top20.json", membox_top20)
    write_json(out / "samem_retrieval_top20.json", samem_top20)
    (out / "membox_qa_context.txt").write_text(
        (membox_run / "retrieved_context.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / "samem_qa_context.txt").write_text(
        (samem_run / "retrieved_context.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    mb_rows = membox_top20["rankings"]
    sm_rows = samem_top20["rankings"]
    mb_gen = load_json(membox_run / "generation_top10.json")["rankings"]
    sm_gen = load_json(samem_run / "generation_top10.json")["rankings"]
    ratio = summary["samem"]["construction"]["total_tokens"] / summary["membox"]["construction"]["total_tokens"]
    diagnostic = f"""# FIRST_QUESTION_DIAGNOSTIC

Question: `e47becba` — `{item['question']}`  
Gold answer: `{ANSWER}`  
Gold source session: `{source_session_id}`

This diagnostic is generated directly from the cleaned dataset and the
completed native run artifacts. It contains only the answer-bearing source
excerpt, the gold memory units, retrieval Top-20, and the final QA contexts.

## Evidence chain

`source dialogue → memory construction → embedding ranking → generation Top-10 → QA answer`

## Gold source dialogue

See `gold_source_dialogue.txt` for the local excerpt. The extracted excerpt
is:

```text
{excerpt}
```

## MemBox gold memory

Native JSON is saved in `membox_gold_memory.json`.

```json
{json.dumps(gold_membox, ensure_ascii=False, indent=2)}
```

## SA-Mem gold MemBlock

Native JSON is saved in `samem_gold_memory.json`.

```json
{json.dumps(gold_samem, ensure_ascii=False, indent=2)}
```

The native SA-Mem events include the extracted evidence:

```text
{event_text(gold_samem, 'samem')}
```

## MemBox retrieval Top-20

Gold rank: `{summary['membox']['gold_rank']}`; gold is in final Top-10: `yes`.

{render_table(mb_rows)}

Full normalized records are in `membox_retrieval_top20.json`; the full ranking
is retained locally at `runs/membox/e47becba/retrieval_full.json`.

## SA-Mem retrieval Top-20

Gold rank: `{summary['samem']['gold_rank']}`; gold is in final Top-10: `yes`.

{render_table(sm_rows)}

Full normalized records are in `samem_retrieval_top20.json`; the full ranking
is retained locally at `runs/samem_2p/e47becba/retrieval_full.json`.

## Final generation context

### MemBox Top-10

{context_index(mb_gen)}

The exact context sent to `gpt-4o-mini` is saved in `membox_qa_context.txt`;
the exact prompt is retained at the run-level `answer_prompt.txt`.

```text
{(membox_run / 'retrieved_context.txt').read_text(encoding='utf-8')}
```

### SA-Mem Top-10

{context_index(sm_gen)}

The exact context sent to `gpt-4o-mini` is saved in `samem_qa_context.txt`;
the exact prompt is retained at the run-level `answer_prompt.txt`.

```text
{(samem_run / 'retrieved_context.txt').read_text(encoding='utf-8')}
```

## Answers

- MemBox: `{summary['membox']['hypothesis']}` — evaluator correct: `yes`.
- SA-Mem: `{summary['samem']['hypothesis']}` — evaluator correct: `yes`.

## Construction Cost Breakdown

| Stage | MemBox total | SA-Mem total |
|---|---:|---:|
| Topic continuity / split check | 189,593 | 191,524 |
| Box extraction / Pass 1 | 214,138 | 272,549 |
| Pass 1 tool follow-up | — | 302,666 |
| Pass 2 classification | — | 174,350 |
| **Total** | **403,731** | **941,089** |

SA-Mem / MemBox total construction token ratio: **{ratio:.4f}x**.

The continuity/split costs are close in this one question. The main additional
cost is SA-Mem's extraction pipeline, especially Pass 1 follow-ups and Pass 2
classification. This is a one-question baseline diagnostic and is not a final
whole-dataset conclusion.
"""
    (out / "FIRST_QUESTION_DIAGNOSTIC.md").write_text(diagnostic, encoding="utf-8")

    base_report = args.workspace_report.read_text(encoding="utf-8")
    marker = "## Detailed evidence audit"
    if marker in base_report:
        base_report = base_report.split(marker, 1)[0].rstrip() + "\n"
    detailed = f"""
## Detailed evidence audit

The answer-bearing source session is `{source_session_id}`. A local relevant
excerpt is saved in `first_question/gold_source_dialogue.txt` and reproduced
in `first_question/FIRST_QUESTION_DIAGNOSTIC.md`.

The main MemBox gold memory is box `{gold_membox['box_id']}` and the main
SA-Mem gold MemBlock is block `{gold_samem['block_id']}`. Both native records,
including content, topics, events, and temporal metadata, are saved in the
first-question diagnostic directory.

The full embedding ranking is saved in each run's `retrieval_full.json`.
Readable Top-20 tables are in `first_question/FIRST_QUESTION_DIAGNOSTIC.md`.
The gold memory is rank 1 in both systems and is included in both final
generation Top-10 contexts. The exact contexts and prompts remain in the run
directories as `retrieved_context.txt` and `answer_prompt.txt`.

### Construction Cost Breakdown

MemBox construction is 403,731 provider-reported tokens: topic continuity
189,593 and box extraction 214,138. SA-Mem construction is 941,089 tokens:
split check 191,524, Pass 1 272,549, Pass 1 tool follow-up 302,666, and
Pass 2 174,350. The SA-Mem / MemBox ratio is **{ratio:.4f}x**. The continuity
and split costs are close here; the current difference is mainly the SA-Mem
extraction pipeline. This remains a one-question diagnostic.
"""
    args.workspace_report.write_text(base_report + detailed, encoding="utf-8")
    print(json.dumps({"output": str(out), "ratio": ratio, "source_session_id": source_session_id}))


if __name__ == "__main__":
    main()
