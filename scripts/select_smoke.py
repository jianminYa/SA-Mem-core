#!/usr/bin/env python3
"""Select fixed smoke IDs from the already frozen subset, independent of model output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.subset.read_text(encoding="utf-8"))

    selected = []
    for wanted in ("temporal-reasoning", "multi-session", "knowledge-update", "abstention"):
        match = next(
            (
                item
                for item in data
                if (str(item["question_id"]).endswith("_abs") if wanted == "abstention" else item["question_type"] == wanted and not str(item["question_id"]).endswith("_abs"))
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"frozen subset does not contain {wanted}")
        selected.append(match)
    if len(selected) != 4:
        raise RuntimeError("frozen subset does not contain the requested smoke strata")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(item["question_id"] for item in selected) + "\n", encoding="utf-8")
    print(json.dumps([{"question_id": x["question_id"], "question_type": x["question_type"], "abstention": str(x["question_id"]).endswith("_abs")} for x in selected], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
