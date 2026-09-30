"""Aggregate halumem judge_*_stats.json across multiple user runs.

For each experiment group (baseline, enhanced, baseline_top{1,5,10},
enhanced_top{1,5,10}), we collect per-user stats from each user output
directory and produce two aggregates:

    - micro: sum the raw counts (correct/hallucination/omission/error/total)
             across users, then recompute ratios. Reflects overall accuracy
             weighted by sample size.
    - macro: average the per-user ratios with equal weight. Reflects "average
             user behavior" regardless of per-user sample count.

Outputs:
    - Pretty-printed table to stdout
    - <out_dir>/halumem_summary.json  (nested structure)
    - <out_dir>/halumem_summary.csv   (flat table; AGGREGATE-{micro,macro} rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("aggregate_halumem")

# (user_label, user_dir)
DEFAULT_USERS: List[Tuple[str, str]] = [
    ("halucp-2f1f897e", "/data/lyc/SA-Mem/out/halucp-2f1f897e"),
    ("halu-2e36c193", "/data/lyc/SA-Mem/out/halu-2e36c193"),
    ("halu-348492a7", "/data/lyc/SA-Mem/out/halu-348492a7"),
    ("halu-1b846c59-categories", "/data/lyc/SA-Mem/out/halu-1b846c59-categories"),
]

# group_name -> ordered list of candidate filenames (first hit wins per user)
DEFAULT_GROUPS: Dict[str, List[str]] = {
    "baseline":        ["judge_baseline_stats.json"],
    "enhanced":        ["judge_enhanced_stats.json"],
    "baseline_top1":   ["judge_baseline_top1_stats.json"],
    "baseline_top5":   ["judge_baseline_top5_stats.json"],
    "baseline_top10":  ["judge_baseline_top10_stats.json"],
    "enhanced_top1":   ["judge_enhanced_top1_stats.json", "judge_enhanced-top1_stats.json"],
    "enhanced_top5":   ["judge_enhanced_top5_stats.json", "judge_enhanced-top5_stats.json"],
    "enhanced_top10":  ["judge_enhanced_top10_stats.json", "judge_enhanced-top10_stats.json"],
}

COUNT_FIELDS = ("total", "correct", "hallucination", "omission", "error")
RATIO_FIELDS = ("correct_ratio", "hallucination_ratio", "omission_ratio", "error_ratio")


def _safe_load(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("failed to read %s: %s", path, e)
        return None


def _find_stats(user_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = user_dir / name
        if p.exists():
            return p
    return None


def _compute_micro(per_user: Dict[str, Dict]) -> Dict[str, float]:
    sums = {k: 0 for k in COUNT_FIELDS}
    for stats in per_user.values():
        for k in COUNT_FIELDS:
            sums[k] += int(stats.get(k, 0) or 0)
    total = sums["total"] or 1  # avoid div-by-zero; reflects empty group
    return {
        "total": sums["total"],
        "correct": sums["correct"],
        "hallucination": sums["hallucination"],
        "omission": sums["omission"],
        "error": sums["error"],
        "correct_ratio": round(sums["correct"] / total, 4),
        "hallucination_ratio": round(sums["hallucination"] / total, 4),
        "omission_ratio": round(sums["omission"] / total, 4),
        "error_ratio": round(sums["error"] / total, 4),
    }


def _compute_macro(per_user: Dict[str, Dict]) -> Dict[str, float]:
    if not per_user:
        return {k: 0.0 for k in RATIO_FIELDS} | {"total_avg": 0.0, "total_sum": 0}
    out: Dict[str, float] = {}
    for k in RATIO_FIELDS:
        vals = [float(s.get(k, 0.0) or 0.0) for s in per_user.values()]
        out[k] = round(mean(vals), 4)
    totals = [int(s.get("total", 0) or 0) for s in per_user.values()]
    out["total_avg"] = round(mean(totals), 2)
    out["total_sum"] = sum(totals)
    return out


def collect(users: List[Tuple[str, str]], groups: Dict[str, List[str]]) -> Dict:
    summary: Dict[str, Dict] = {}
    for group, candidates in groups.items():
        per_user: Dict[str, Dict] = {}
        per_user_files: Dict[str, str] = {}
        for label, d in users:
            udir = Path(d)
            if not udir.exists():
                logger.warning("[%s] user dir missing: %s", group, udir)
                continue
            stats_path = _find_stats(udir, candidates)
            if not stats_path:
                logger.info("[%s/%s] no stats file found (tried %s)", group, label, candidates)
                continue
            data = _safe_load(stats_path)
            if not isinstance(data, dict):
                continue
            per_user[label] = data
            per_user_files[label] = str(stats_path)
        summary[group] = {
            "per_user": per_user,
            "per_user_files": per_user_files,
            "aggregate": {
                "micro": _compute_micro(per_user),
                "macro": _compute_macro(per_user),
                "user_count": len(per_user),
                "user_ids": list(per_user.keys()),
            },
        }
    return summary


def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "-"


def print_report(summary: Dict) -> None:
    line = "=" * 110
    for group, info in summary.items():
        per_user = info["per_user"]
        agg = info["aggregate"]
        print(line)
        print(f"GROUP: {group}   (user_count={agg['user_count']})")
        print(f"{'user':<30} {'total':>7} {'correct':>9} {'hallu':>7} {'omiss':>7} {'err':>5} "
              f"{'corr%':>8} {'hallu%':>8} {'omiss%':>8} {'err%':>6}")
        for label, s in per_user.items():
            print(f"{label:<30} {s.get('total',0):>7} {s.get('correct',0):>9} "
                  f"{s.get('hallucination',0):>7} {s.get('omission',0):>7} {s.get('error',0):>5} "
                  f"{_fmt_pct(s.get('correct_ratio')):>8} {_fmt_pct(s.get('hallucination_ratio')):>8} "
                  f"{_fmt_pct(s.get('omission_ratio')):>8} {_fmt_pct(s.get('error_ratio')):>6}")
        m = agg["micro"]
        ma = agg["macro"]
        print(f"{'AGGREGATE-micro':<30} {m['total']:>7} {m['correct']:>9} {m['hallucination']:>7} "
              f"{m['omission']:>7} {m['error']:>5} {_fmt_pct(m['correct_ratio']):>8} "
              f"{_fmt_pct(m['hallucination_ratio']):>8} {_fmt_pct(m['omission_ratio']):>8} "
              f"{_fmt_pct(m['error_ratio']):>6}")
        print(f"{'AGGREGATE-macro':<30} {ma['total_sum']:>7} {'-':>9} {'-':>7} {'-':>7} {'-':>5} "
              f"{_fmt_pct(ma['correct_ratio']):>8} {_fmt_pct(ma['hallucination_ratio']):>8} "
              f"{_fmt_pct(ma['omission_ratio']):>8} {_fmt_pct(ma['error_ratio']):>6}   "
              f"(total_avg={ma['total_avg']})")
    print(line)


def write_json(summary: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", out_path)


def write_csv(summary: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["user", "group", "total", "correct", "hallucination", "omission", "error",
            "correct_ratio", "hallucination_ratio", "omission_ratio", "error_ratio"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for group, info in summary.items():
            for label, s in info["per_user"].items():
                w.writerow([
                    label, group,
                    s.get("total", 0), s.get("correct", 0),
                    s.get("hallucination", 0), s.get("omission", 0), s.get("error", 0),
                    s.get("correct_ratio", ""), s.get("hallucination_ratio", ""),
                    s.get("omission_ratio", ""), s.get("error_ratio", ""),
                ])
            m = info["aggregate"]["micro"]
            ma = info["aggregate"]["macro"]
            w.writerow([
                "AGGREGATE-micro", group,
                m["total"], m["correct"], m["hallucination"], m["omission"], m["error"],
                m["correct_ratio"], m["hallucination_ratio"], m["omission_ratio"], m["error_ratio"],
            ])
            w.writerow([
                "AGGREGATE-macro", group,
                ma["total_sum"], "", "", "", "",
                ma["correct_ratio"], ma["hallucination_ratio"],
                ma["omission_ratio"], ma["error_ratio"],
            ])
    logger.info("wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("/data/lyc/SA-Mem/out"))
    p.add_argument("--json-name", default="halumem_summary.json")
    p.add_argument("--csv-name", default="halumem_summary.csv")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    summary = collect(DEFAULT_USERS, DEFAULT_GROUPS)
    print_report(summary)
    write_json(summary, args.out_dir / args.json_name)
    write_csv(summary, args.out_dir / args.csv_name)


if __name__ == "__main__":
    main()
