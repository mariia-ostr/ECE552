#!/usr/bin/env python3
"""
Analyze ChampSim benchmark outputs by comparing L1D prefetches, total ROI cycles,
and speedup against the `nopred` baseline across all benchmarks, with optional
Discord webhook reporting.
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_BENCHMARKS = [
    "charlie",
    "k_clique",
    "llama2",
    "mcf",
    "merced",
    "roms",
    "sssp",
    "whisper",
]


def parse_champsim_log(filepath: Path) -> Optional[Dict[str, float]]:
    """Extract ROI cycle count, IPC, L1D prefetch stats, and L1D load misses."""
    if not filepath.is_file():
        return None

    text = filepath.read_text(encoding="utf-8", errors="replace")
    roi_marker = "Region of Interest Statistics"
    roi_idx = text.find(roi_marker)
    if roi_idx == -1:
        return None

    roi_text = text[roi_idx:]

    ipc_match = re.search(
        r"CPU\s+\d+\s+cumulative\s+IPC:\s+([0-9.eE+-]+)\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)",
        roi_text,
    )
    if not ipc_match:
        return None

    ipc = float(ipc_match.group(1))
    instructions = int(ipc_match.group(2))
    cycles = int(ipc_match.group(3))

    l1d_pf_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+PREFETCH\s+REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)",
        roi_text,
    )
    if l1d_pf_match:
        pf_requested = int(l1d_pf_match.group(1))
        pf_issued = int(l1d_pf_match.group(2))
        pf_useful = int(l1d_pf_match.group(3))
        pf_useless = int(l1d_pf_match.group(4))
    else:
        pf_requested = pf_issued = pf_useful = pf_useless = 0

    l1d_load_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+LOAD\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)",
        roi_text,
    )
    if l1d_load_match:
        l1d_load_access = int(l1d_load_match.group(1))
        l1d_load_miss = int(l1d_load_match.group(3))
    else:
        l1d_load_access = l1d_load_miss = 0

    return {
        "ipc": ipc,
        "instructions": instructions,
        "cycles": cycles,
        "pf_requested": pf_requested,
        "pf_issued": pf_issued,
        "pf_useful": pf_useful,
        "pf_useless": pf_useless,
        "l1d_load_access": l1d_load_access,
        "l1d_load_miss": l1d_load_miss,
    }


def analyze_runs(
    output_dir: Path,
    pred: str,
    baseline: str,
    benchmarks: List[str],
) -> List[Dict]:
    """Compare predictor metrics against baseline across all benchmarks."""
    rows = []
    for bench in benchmarks:
        base_stats = parse_champsim_log(output_dir / f"{baseline}.{bench}")
        pred_stats = parse_champsim_log(output_dir / f"{pred}.{bench}")

        row = {
            "benchmark": bench,
            "base": base_stats,
            "pred": pred_stats,
            "speedup": None,
            "cycle_delta_pct": None,
            "useful_pct": None,
            "useless_pct": None,
            "load_miss_delta": None,
        }

        if base_stats and pred_stats and pred_stats["cycles"] > 0:
            row["speedup"] = base_stats["cycles"] / pred_stats["cycles"]
            row["cycle_delta_pct"] = (
                (pred_stats["cycles"] - base_stats["cycles"]) / base_stats["cycles"]
            ) * 100.0
            row["load_miss_delta"] = (
                pred_stats["l1d_load_miss"] - base_stats["l1d_load_miss"]
            )

        if pred_stats and pred_stats["pf_issued"] > 0:
            row["useful_pct"] = (
                pred_stats["pf_useful"] / pred_stats["pf_issued"]
            ) * 100.0
            row["useless_pct"] = (
                pred_stats["pf_useless"] / pred_stats["pf_issued"]
            ) * 100.0

        rows.append(row)
    return rows


def format_report(pred: str, baseline: str, rows: List[Dict]) -> str:
    """Build a formatted fixed-width comparison table with summary statistics."""
    header = (
        f"{'BENCH':<10} {'NOPRED_CYC':>13} {'PRED_CYC':>13} {'SPEEDUP':>9} "
        f"{'L1D_ISSUED':>11} {'L1D_USEFUL':>11} {'USEFUL%':>8} {'L1D_USELESS':>12} {'LOAD_MISS_Δ':>12}"
    )
    sep = "-" * len(header)
    lines = [
        f"ChampSim ROI Comparison: {pred} vs {baseline}",
        sep,
        header,
        sep,
    ]

    speedups = []
    total_base_cyc = 0
    total_pred_cyc = 0
    total_issued = 0
    total_useful = 0
    total_useless = 0

    for r in rows:
        b_cyc = f"{r['base']['cycles']:,}" if r["base"] else "N/A"
        p_cyc = f"{r['pred']['cycles']:,}" if r["pred"] else "N/A"
        sp = f"{r['speedup']:.4f}x" if r["speedup"] is not None else "N/A"

        if r["pred"]:
            iss = f"{r['pred']['pf_issued']:,}"
            use = f"{r['pred']['pf_useful']:,}"
            usl = f"{r['pred']['pf_useless']:,}"
            use_pct = f"{r['useful_pct']:.1f}%" if r["useful_pct"] is not None else "0.0%"
            total_issued += r["pred"]["pf_issued"]
            total_useful += r["pred"]["pf_useful"]
            total_useless += r["pred"]["pf_useless"]
        else:
            iss = use = usl = use_pct = "N/A"

        if r["load_miss_delta"] is not None:
            miss_delta = f"{r['load_miss_delta']:+,}"
        else:
            miss_delta = "N/A"

        if r["speedup"] is not None:
            speedups.append(r["speedup"])
            total_base_cyc += r["base"]["cycles"]
            total_pred_cyc += r["pred"]["cycles"]

        lines.append(
            f"{r['benchmark']:<10} {b_cyc:>13} {p_cyc:>13} {sp:>9} "
            f"{iss:>11} {use:>11} {use_pct:>8} {usl:>12} {miss_delta:>12}"
        )

    lines.append(sep)
    if speedups:
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        agg_speedup = total_base_cyc / total_pred_cyc if total_pred_cyc > 0 else 0.0
        agg_useful_pct = (total_useful / total_issued * 100.0) if total_issued > 0 else 0.0
        lines.append(
            f"Geomean Speedup: {geomean:.4f}x | Aggregate Cycle Speedup: {agg_speedup:.4f}x | "
            f"Total L1D Useful Rate: {agg_useful_pct:.2f}% ({total_useful:,}/{total_issued:,})"
        )
    else:
        lines.append("No complete benchmark pairs found to compute aggregate speedup.")

    return "\n".join(lines)


def send_discord_notification(webhook_url: str, report: str) -> None:
    """Send the formatted report to Discord, automatically chunking under 2000 chars."""
    max_chunk = 1900
    for i in range(0, len(report), max_chunk):
        payload = json.dumps({"content": f"```\n{report[i:i + max_chunk]}\n```"}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ChampSimAnalyzer/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ChampSim ROI cycles and L1D prefetch metrics against a nopred baseline."
    )
    parser.add_argument(
        "predictor",
        help="Predictor prefix name used in output/<predictor>.<benchmark> (e.g., stride, next_line)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing ChampSim output logs (default: output)",
    )
    parser.add_argument(
        "--baseline",
        default="nopred",
        help="Baseline prefix name (default: nopred)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="List of benchmark names to analyze",
    )
    parser.add_argument(
        "--discord",
        action="store_true",
        help="Post the comparison table to DISCORD_WEBHOOK_URL",
    )
    parser.add_argument(
        "--webhook-url",
        default=os.environ.get("DISCORD_WEBHOOK_URL", ""),
        help="Discord webhook URL (defaults to $DISCORD_WEBHOOK_URL)",
    )
    args = parser.parse_args()

    rows = analyze_runs(args.output_dir, args.predictor, args.baseline, args.benchmarks)
    report = format_report(args.predictor, args.baseline, rows)
    print(report)

    if args.discord:
        if not args.webhook_url:
            print("Error: --discord specified but DISCORD_WEBHOOK_URL is not set.", file=sys.stderr)
            sys.exit(1)
        send_discord_notification(args.webhook_url, report)
        print("Posted results to Discord.", file=sys.stderr)


if __name__ == "__main__":
    main()