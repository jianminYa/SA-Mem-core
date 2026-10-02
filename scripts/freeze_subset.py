#!/usr/bin/env python3
"""Freeze a deterministic, type/abstention-stratified LongMemEval subset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_counts(strata: dict[tuple[str, bool], list[dict]], total: int) -> dict[tuple[str, bool], int]:
    if total < 1 or total > sum(len(v) for v in strata.values()):
        raise ValueError("requested total is outside the dataset size")
    raw = {key: total * len(items) / sum(map(len, strata.values())) for key, items in strata.items()}
    counts = {key: min(len(strata[key]), int(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        strata,
        key=lambda key: (raw[key] - int(raw[key]), len(strata[key]), key[0], key[1]),
        reverse=True,
    )
    while remaining:
        changed = False
        for key in order:
            if counts[key] < len(strata[key]):
                counts[key] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise RuntimeError("could not allocate requested subset size")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=50)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    strata: dict[tuple[str, bool], list[dict]] = collections.defaultdict(list)
    for item in data:
        qid = str(item["question_id"])
        key = (str(item["question_type"]), qid.endswith("_abs"))
        strata[key].append(item)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    target_counts = allocate_counts(strata, args.total)
    for key in sorted(strata):
        items = list(strata[key])
        rng.shuffle(items)
        selected.extend(items[: target_counts[key]])

    # Keep the original dataset order in the frozen file; selection itself is seeded.
    selected_ids = {item["question_id"] for item in selected}
    frozen = [item for item in data if item["question_id"] in selected_ids]
    if len(frozen) != args.total or len({item["question_id"] for item in frozen}) != args.total:
        raise RuntimeError("subset is not unique or does not have the requested size")

    question_type_counts = collections.Counter(item["question_type"] for item in frozen)
    abstention_count = sum(str(item["question_id"]).endswith("_abs") for item in frozen)
    strata_counts = collections.Counter(
        (item["question_type"], str(item["question_id"]).endswith("_abs")) for item in frozen
    )
    manifest = {
        "source_dataset": args.input.name,
        "source_sha256": sha256(args.input),
        "seed": args.seed,
        "total": len(frozen),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "abstention_count": abstention_count,
        "stratum_counts": {
            f"{question_type}|abstention={abstention}": count
            for (question_type, abstention), count in sorted(strata_counts.items())
        },
        "question_ids": [item["question_id"] for item in frozen],
    }
    for path in (args.output, args.ids_output, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.ids_output.write_text("\n".join(manifest["question_ids"]) + "\n", encoding="utf-8")
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_dataset", "seed", "total", "question_type_counts", "abstention_count", "stratum_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
