#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ["halu-2e36c193", "halu-348492a7", "halucp-2f1f897e"]
DEFAULT_TOPNS = [1, 5, 10, 20]
DEFAULT_JUDGE_REF = "/data/HaluMem/yjm/data/HaluMem-Medium.jsonl"


@dataclass
class TargetResult:
    run_id: str
    method: str
    topn: int
    generation_file: str
    judge_file: str
    stats_file: str
    status: str
    generation_action: str
    judge_action: str
    retrieval_action: str
    total: int | None = None
    correct: int | None = None
    hallucination: int | None = None
    omission: int | None = None
    error: int | None = None
    accuracy: float | None = None
    hallucination_ratio: float | None = None
    omission_ratio: float | None = None
    error_ratio: float | None = None
    message: str = ""


def parse_int_list(raw: str) -> List[int]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    if not vals:
        raise ValueError("Empty topn list")
    # Keep order but deduplicate
    uniq = []
    seen = set()
    for v in vals:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def default_raw_file_for_run(run_id: str) -> Path:
    suffix = run_id.split("-")[-1]
    return PROJECT_ROOT / "data" / f"qa_{suffix}.json"


def run_cmd(cmd: List[str], dry_run: bool, cwd: Path) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def get_path_candidates(out_dir: Path, method: str, topn: int) -> Tuple[List[Path], List[Path], List[Path]]:
    if topn == 20:
        generation_candidates = [out_dir / f"generation_{method}.jsonl"]
        judge_candidates = [out_dir / f"judge_{method}.jsonl"]
        stats_candidates = [out_dir / f"judge_{method}_stats.json"]
    else:
        generation_candidates = [
            out_dir / f"generation_{method}_top{topn}.jsonl",
            out_dir / f"generation_{method}-top{topn}.jsonl",
        ]
        judge_candidates = [
            out_dir / f"judge_{method}_top{topn}.jsonl",
            out_dir / f"judge_{method}-top{topn}.jsonl",
        ]
        stats_candidates = [
            out_dir / f"judge_{method}_top{topn}_stats.json",
            out_dir / f"judge_{method}-top{topn}_stats.json",
        ]
    return generation_candidates, judge_candidates, stats_candidates

def pick_existing_or_default(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def resolve_paths(out_dir: Path, method: str, topn: int) -> Tuple[Path, Path, Path, List[Path], List[Path], List[Path]]:
    gen_cands, judge_cands, stats_cands = get_path_candidates(out_dir, method, topn)
    generation = pick_existing_or_default(gen_cands)
    judge = pick_existing_or_default(judge_cands)
    stats = pick_existing_or_default(stats_cands)
    return generation, judge, stats, gen_cands, judge_cands, stats_cands


def load_stats(stats_file: Path) -> Dict[str, float | int] | None:
    if not stats_file.exists():
        return None
    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_retrieval(
    run_id: str,
    raw_data_file: Path,
    out_dir: Path,
    dry_run: bool,
    force_rerun: bool,
) -> str:
    baseline = out_dir / "retrieval_baseline.jsonl"
    enhanced = out_dir / "retrieval_enhanced.jsonl"
    if not force_rerun and baseline.exists() and enhanced.exists():
        return "reused"

    cmd = [
        sys.executable,
        "retrieval/retrieve_stage_enhanced.py",
        "--mode",
        "both",
        "--run-id",
        run_id,
        "--raw-data-file",
        str(raw_data_file),
    ]
    if force_rerun:
        cmd.append("--overwrite")

    run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)
    return "recomputed"


def run_generation(
    run_id: str,
    raw_data_file: Path,
    retrieval_file: Path,
    generation_file: Path,
    topn: int,
    text_modes: str,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "generate_stage.py",
        "--run-id",
        run_id,
        "--raw-data-file",
        str(raw_data_file),
        "--retrieval-jsonl",
        str(retrieval_file),
        "--gen-out-jsonl",
        str(generation_file),
        "--text-modes",
        text_modes,
        "--answer-topn",
        str(topn),
    ]
    run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def run_judge(
    generation_file: Path,
    judge_ref_file: Path,
    judge_file: Path,
    max_workers: int,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "llm_judge_eval.py",
        "--generation-file",
        str(generation_file),
        "--halumem-file",
        str(judge_ref_file),
        "--output-file",
        str(judge_file),
        "--max-workers",
        str(max_workers),
    ]
    run_cmd(cmd, dry_run=dry_run, cwd=PROJECT_ROOT)


def aggregate_delta_rows(rows: List[TargetResult]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], Dict[str, TargetResult]] = {}
    for row in rows:
        grouped.setdefault((row.run_id, row.topn), {})[row.method] = row

    out: List[Dict[str, object]] = []
    for (run_id, topn), pair in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        b = pair.get("baseline")
        e = pair.get("enhanced")
        b_acc = b.accuracy if b else None
        e_acc = e.accuracy if e else None
        delta = None
        if b_acc is not None and e_acc is not None:
            delta = e_acc - b_acc
        out.append(
            {
                "run_id": run_id,
                "topn": topn,
                "baseline_accuracy": b_acc,
                "enhanced_accuracy": e_acc,
                "delta_enhanced_minus_baseline": delta,
                "baseline_status": b.status if b else "missing",
                "enhanced_status": e.status if e else "missing",
            }
        )
    return out


def write_reports(rows: List[TargetResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    full_json = out_dir / "generation_accuracy_summary.json"
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    full_csv = out_dir / "generation_accuracy_summary.csv"
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(full_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    delta_rows = aggregate_delta_rows(rows)
    delta_json = out_dir / "generation_accuracy_delta.json"
    with open(delta_json, "w", encoding="utf-8") as f:
        json.dump(delta_rows, f, ensure_ascii=False, indent=2)

    md = out_dir / "generation_accuracy_report.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Generation Accuracy Report\n\n")
        f.write("## Per Method\n\n")
        f.write("| run_id | method | topn | accuracy | hallucination | omission | status |\n")
        f.write("|---|---|---:|---:|---:|---:|---|\n")
        for r in sorted(rows, key=lambda x: (x.run_id, x.topn, x.method)):
            acc = "-" if r.accuracy is None else f"{r.accuracy:.4f}"
            hal = "-" if r.hallucination_ratio is None else f"{r.hallucination_ratio:.4f}"
            omi = "-" if r.omission_ratio is None else f"{r.omission_ratio:.4f}"
            f.write(f"| {r.run_id} | {r.method} | {r.topn} | {acc} | {hal} | {omi} | {r.status} |\n")

        f.write("\n## Enhanced - Baseline Delta\n\n")
        f.write("| run_id | topn | baseline_acc | enhanced_acc | delta |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for d in delta_rows:
            b = "-" if d["baseline_accuracy"] is None else f"{float(d['baseline_accuracy']):.4f}"
            e = "-" if d["enhanced_accuracy"] is None else f"{float(d['enhanced_accuracy']):.4f}"
            x = "-" if d["delta_enhanced_minus_baseline"] is None else f"{float(d['delta_enhanced_minus_baseline']):.4f}"
            f.write(f"| {d['run_id']} | {d['topn']} | {b} | {e} | {x} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation for generation accuracy (baseline vs enhanced, multi-topN)")
    parser.add_argument("--runs", type=str, default=",".join(DEFAULT_RUNS), help="Comma-separated run ids")
    parser.add_argument("--topns", type=str, default=",".join(str(x) for x in DEFAULT_TOPNS), help="Comma-separated topN values")
    parser.add_argument("--judge-ref-file", type=str, default=DEFAULT_JUDGE_REF, help="Reference HaluMem file for llm_judge_eval")
    parser.add_argument("--text-modes", type=str, default="content", help="Generation text mode")
    parser.add_argument("--prefer-existing", dest="prefer_existing", action="store_true", default=True, help="Reuse existing artifacts when available")
    parser.add_argument("--no-prefer-existing", dest="prefer_existing", action="store_false", help="Do not trust existing stats; rerun judge/generation when needed")
    parser.add_argument("--force-rerun", action="store_true", help="Recompute retrieval/generation/judge regardless of existing files")
    parser.add_argument("--max-workers", type=int, default=5, help="Judge max workers")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--summary-dir", type=str, default=str(PROJECT_ROOT / "out" / "multi_eval_summary"), help="Directory for summary outputs")
    args = parser.parse_args()

    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    topns = parse_int_list(args.topns)
    methods = ["baseline", "enhanced"]
    judge_ref_file = Path(args.judge_ref_file)

    if not judge_ref_file.exists() and not args.dry_run:
        raise FileNotFoundError(f"Judge reference file not found: {judge_ref_file}")

    results: List[TargetResult] = []

    for run_id in runs:
        raw_data_file = default_raw_file_for_run(run_id)
        out_dir = PROJECT_ROOT / "out" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if not raw_data_file.exists() and not args.dry_run:
            for method in methods:
                for topn in topns:
                    generation_file, judge_file, stats_file, _, _, _ = resolve_paths(out_dir, method, topn)
                    results.append(
                        TargetResult(
                            run_id=run_id,
                            method=method,
                            topn=topn,
                            generation_file=str(generation_file),
                            judge_file=str(judge_file),
                            stats_file=str(stats_file),
                            status="failed",
                            generation_action="skipped",
                            judge_action="skipped",
                            retrieval_action="skipped",
                            message=f"raw data file missing: {raw_data_file}",
                        )
                    )
            continue

        # Decide which targets need generation and/or judge
        target_needs: Dict[Tuple[str, int], Dict[str, object]] = {}
        for method in methods:
            for topn in topns:
                generation_file, judge_file, stats_file, gen_cands, judge_cands, stats_cands = resolve_paths(out_dir, method, topn)
                stats_exists = any(p.exists() for p in stats_cands)
                gen_exists = any(p.exists() for p in gen_cands)

                need_judge = args.force_rerun or (not stats_exists) or (not args.prefer_existing)
                need_generation = args.force_rerun or (not gen_exists)

                if args.prefer_existing and (not args.force_rerun) and stats_exists:
                    need_judge = False
                    need_generation = False

                target_needs[(method, topn)] = {
                    "need_generation": need_generation,
                    "need_judge": need_judge,
                    "stats_exists": stats_exists,
                    "gen_exists": gen_exists,
                    "generation_file": generation_file,
                    "judge_file": judge_file,
                    "stats_file": stats_file,
                }

        any_generation_needed = any(v["need_generation"] for v in target_needs.values())
        retrieval_action = "reused"
        if any_generation_needed:
            retrieval_action = ensure_retrieval(
                run_id=run_id,
                raw_data_file=raw_data_file,
                out_dir=out_dir,
                dry_run=args.dry_run,
                force_rerun=args.force_rerun,
            )

        for method in methods:
            retrieval_file = out_dir / f"retrieval_{method}.jsonl"
            for topn in topns:
                flags = target_needs[(method, topn)]
                generation_file = flags["generation_file"]
                judge_file = flags["judge_file"]
                stats_file = flags["stats_file"]
                generation_action = "reused"
                judge_action = "reused"
                status = "ok"
                message = ""

                try:
                    if flags["need_generation"]:
                        run_generation(
                            run_id=run_id,
                            raw_data_file=raw_data_file,
                            retrieval_file=retrieval_file,
                            generation_file=generation_file,
                            topn=topn,
                            text_modes=args.text_modes,
                            dry_run=args.dry_run,
                        )
                        generation_action = "recomputed"

                    if flags["need_judge"]:
                        run_judge(
                            generation_file=generation_file,
                            judge_ref_file=judge_ref_file,
                            judge_file=judge_file,
                            max_workers=args.max_workers,
                            dry_run=args.dry_run,
                        )
                        judge_action = "recomputed"
                except subprocess.CalledProcessError as e:
                    status = "failed"
                    message = f"command failed: {e}"
                except Exception as e:
                    status = "failed"
                    message = str(e)

                stats = load_stats(stats_file)
                total = correct = hallucination = omission = error = None
                accuracy = hallucination_ratio = omission_ratio = error_ratio = None

                if stats is not None:
                    total = int(stats.get("total", 0))
                    correct = int(stats.get("correct", 0))
                    hallucination = int(stats.get("hallucination", 0))
                    omission = int(stats.get("omission", 0))
                    error = int(stats.get("error", 0))
                    accuracy = float(stats.get("correct_ratio", 0.0))
                    hallucination_ratio = float(stats.get("hallucination_ratio", 0.0))
                    omission_ratio = float(stats.get("omission_ratio", 0.0))
                    error_ratio = float(stats.get("error_ratio", 0.0))
                elif args.dry_run and status != "failed":
                    status = "planned"
                elif status != "failed":
                    status = "missing"
                    if not message:
                        message = "stats file missing after execution"

                results.append(
                    TargetResult(
                        run_id=run_id,
                        method=method,
                        topn=topn,
                        generation_file=str(generation_file),
                        judge_file=str(judge_file),
                        stats_file=str(stats_file),
                        status=status,
                        generation_action=generation_action,
                        judge_action=judge_action,
                        retrieval_action=retrieval_action if any_generation_needed else "reused",
                        total=total,
                        correct=correct,
                        hallucination=hallucination,
                        omission=omission,
                        error=error,
                        accuracy=accuracy,
                        hallucination_ratio=hallucination_ratio,
                        omission_ratio=omission_ratio,
                        error_ratio=error_ratio,
                        message=message,
                    )
                )

    write_reports(results, Path(args.summary_dir))

    print("\nDone. Summary files:")
    print(f"- {Path(args.summary_dir) / 'generation_accuracy_summary.json'}")
    print(f"- {Path(args.summary_dir) / 'generation_accuracy_summary.csv'}")
    print(f"- {Path(args.summary_dir) / 'generation_accuracy_delta.json'}")
    print(f"- {Path(args.summary_dir) / 'generation_accuracy_report.md'}")


if __name__ == "__main__":
    main()
