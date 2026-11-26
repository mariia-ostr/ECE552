#!/usr/bin/env python3
"""
Parse ChampSim benchmark logs across predictors (nopred, nextline, stride, openpref)
and benchmarks, report per-benchmark and cumulative L1D hit rates, plot only
the per-benchmark L1D hit rate chart, and optionally post summaries to a
Discord webhook.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PREDICTORS = ["nopred", "nextline", "stride", "openpref"]
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


def _parse_optional_float(value: Optional[str], default: float = 0.0) -> float:
    """Convert a ChampSim numeric string to float, treating '-' or missing values as default."""
    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned or cleaned == "-":
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def parse_champsim_log(filepath: Path) -> Optional[Dict[str, float]]:
    """Extract ROI metrics and L1D hit rate statistics from a ChampSim log."""
    if not filepath.is_file():
        return None

    text = filepath.read_text(encoding="utf-8", errors="replace")
    roi_idx = text.find("Region of Interest Statistics")
    if roi_idx == -1:
        return None

    roi_text = text[roi_idx:]

    ipc_match = re.search(
        r"CPU\s+\d+\s+cumulative\s+IPC:\s+([0-9.eE+-]+)\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)",
        roi_text,
    )
    if not ipc_match:
        return None

    l1d_tot_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)",
        roi_text,
    )
    l1i_tot_match = re.search(
        r"cpu\d+->cpu\d+_L1I\s+TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)",
        roi_text,
    )

    l1d_lat_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+AVERAGE\s+MISS\s+LATENCY:\s*([0-9.eE+-]+|-)",
        roi_text,
    )
    l1i_lat_match = re.search(
        r"cpu\d+->cpu\d+_L1I\s+AVERAGE\s+MISS\s+LATENCY:\s*([0-9.eE+-]+|-)",
        roi_text,
    )

    pf_acc_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+PREFETCH\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)",
        roi_text,
    )
    pf_use_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+PREFETCH\s+REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)",
        roi_text,
    )

    l1d_load_match = re.search(
        r"cpu\d+->cpu\d+_L1D\s+LOAD\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)",
        roi_text,
    )

    l1d_access = int(l1d_tot_match.group(1)) if l1d_tot_match else 0
    l1d_hit = int(l1d_tot_match.group(2)) if l1d_tot_match else 0
    l1d_miss = int(l1d_tot_match.group(3)) if l1d_tot_match else 0

    l1i_access = int(l1i_tot_match.group(1)) if l1i_tot_match else 0
    l1i_hit = int(l1i_tot_match.group(2)) if l1i_tot_match else 0
    l1i_miss = int(l1i_tot_match.group(3)) if l1i_tot_match else 0

    l1_comb_access = l1d_access + l1i_access
    l1_comb_hit = l1d_hit + l1i_hit
    l1_comb_miss = l1d_miss + l1i_miss

    pf_access = int(pf_acc_match.group(1)) if pf_acc_match else 0
    pf_miss = int(pf_acc_match.group(3)) if pf_acc_match else 0
    pf_requested = int(pf_use_match.group(1)) if pf_use_match else 0
    pf_issued = int(pf_use_match.group(2)) if pf_use_match else 0
    pf_useful = int(pf_use_match.group(3)) if pf_use_match else 0
    pf_useless = int(pf_use_match.group(4)) if pf_use_match else 0

    return {
        "ipc": _parse_optional_float(ipc_match.group(1), default=0.0),
        "instructions": int(ipc_match.group(2)),
        "cycles": int(ipc_match.group(3)),
        "l1d_access": l1d_access,
        "l1d_hit": l1d_hit,
        "l1d_miss": l1d_miss,
        "l1d_hit_rate_pct": (l1d_hit / l1d_access * 100.0) if l1d_access > 0 else 0.0,
        "l1i_access": l1i_access,
        "l1i_hit": l1i_hit,
        "l1i_miss": l1i_miss,
        "l1i_hit_rate_pct": (l1i_hit / l1i_access * 100.0) if l1i_access > 0 else 0.0,
        "l1_comb_access": l1_comb_access,
        "l1_comb_hit": l1_comb_hit,
        "l1_comb_miss": l1_comb_miss,
        "l1_comb_hit_rate_pct": (l1_comb_hit / l1_comb_access * 100.0)
        if l1_comb_access > 0
        else 0.0,
        "l1d_avg_miss_latency": _parse_optional_float(
            l1d_lat_match.group(1) if l1d_lat_match else None, default=0.0
        ),
        "l1i_avg_miss_latency": _parse_optional_float(
            l1i_lat_match.group(1) if l1i_lat_match else None, default=0.0
        ),
        "pf_access": pf_access,
        "pf_miss": pf_miss,
        "pf_miss_rate_pct": (pf_miss / pf_access * 100.0) if pf_access > 0 else 0.0,
        "pf_requested": pf_requested,
        "pf_issued": pf_issued,
        "pf_useful": pf_useful,
        "pf_useless": pf_useless,
        "pf_useful_rate_pct": (pf_useful / pf_issued * 100.0) if pf_issued > 0 else 0.0,
        "l1d_load_access": int(l1d_load_match.group(1)) if l1d_load_match else 0,
        "l1d_load_miss": int(l1d_load_match.group(3)) if l1d_load_match else 0,
    }


def collect_dataframe(
    output_dir: Path,
    predictors: List[str],
    benchmarks: List[str],
    pattern: str = "{predictor}.{benchmark}",
) -> pd.DataFrame:
    """Load parsed log metrics for all predictor/benchmark combinations into a DataFrame."""
    records: List[Dict] = []
    for bench in benchmarks:
        for pred in predictors:
            log_path = output_dir / pattern.format(predictor=pred, benchmark=bench)
            parsed = parse_champsim_log(log_path)
            if parsed is not None:
                records.append({"benchmark": bench, "predictor": pred, **parsed})
    return pd.DataFrame(records)


def compute_cumulative_l1_hit_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative L1D, L1I, and combined L1 hit rates across benchmarks per predictor."""
    cum = df.groupby("predictor", as_index=False).agg(
        benchmarks_count=("benchmark", "nunique"),
        l1d_hit=("l1d_hit", "sum"),
        l1d_access=("l1d_access", "sum"),
        l1i_hit=("l1i_hit", "sum"),
        l1i_access=("l1i_access", "sum"),
        l1_comb_hit=("l1_comb_hit", "sum"),
        l1_comb_access=("l1_comb_access", "sum"),
        total_instructions=("instructions", "sum"),
        total_cycles=("cycles", "sum"),
    )
    cum["cum_l1d_hit_rate_pct"] = np.where(
        cum["l1d_access"] > 0, cum["l1d_hit"] / cum["l1d_access"] * 100.0, 0.0
    )
    cum["cum_l1i_hit_rate_pct"] = np.where(
        cum["l1i_access"] > 0, cum["l1i_hit"] / cum["l1i_access"] * 100.0, 0.0
    )
    cum["cum_l1_comb_hit_rate_pct"] = np.where(
        cum["l1_comb_access"] > 0, cum["l1_comb_hit"] / cum["l1_comb_access"] * 100.0, 0.0
    )
    cum["cum_ipc"] = np.where(
        cum["total_cycles"] > 0, cum["total_instructions"] / cum["total_cycles"], 0.0
    )
    return cum


def plot_l1d_hit_rate_per_benchmark(
    df: pd.DataFrame,
    predictors: List[str],
    benchmarks: List[str],
    out_path: Path,
    fmt: str = "%.1f%%",
) -> None:
    """Render a grouped bar chart comparing predictors per benchmark for L1D hit rate only."""
    piv = df.pivot(index="benchmark", columns="predictor", values="l1d_hit_rate_pct")
    ordered_benchmarks = [b for b in benchmarks if b in piv.index]
    piv = piv.reindex(ordered_benchmarks)

    present_predictors = [p for p in predictors if p in piv.columns]
    if not present_predictors or piv.empty:
        print("Skipping 'l1d_hit_rate_pct': no matching data to plot.", file=sys.stderr)
        return

    x = np.arange(len(piv.index))
    width = 0.82 / len(present_predictors)

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, pred in enumerate(present_predictors):
        offset = (idx - (len(present_predictors) - 1) / 2) * width
        vals = piv[pred].to_numpy(dtype=float)
        bars = ax.bar(x + offset, vals, width=width, label=pred)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_ylabel("L1D Hit Rate (%)")
    ax.set_xlabel("Benchmark")
    ax.set_title("Per-Benchmark L1D Hit Rate")
    ax.legend(title="Predictor")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.margins(y=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def format_text_summary(df: pd.DataFrame, cum_df: pd.DataFrame) -> str:
    """Return per-benchmark L1D hit rates plus cumulative L1D hit rates across benchmarks."""
    per_bench_cols = [
        "benchmark",
        "predictor",
        "l1d_hit_rate_pct",
        "l1d_hit",
        "l1d_access",
        "ipc",
    ]
    per_bench_view = df[per_bench_cols].sort_values(["benchmark", "predictor"]).copy()
    per_bench_view["l1d_hit_rate_pct"] = per_bench_view["l1d_hit_rate_pct"].map(
        lambda x: f"{x:.2f}%"
    )
    per_bench_view["ipc"] = per_bench_view["ipc"].map(lambda x: f"{x:.4f}")

    cum_view = cum_df[
        [
            "predictor",
            "benchmarks_count",
            "cum_l1d_hit_rate_pct",
            "l1d_hit",
            "l1d_access",
            "cum_ipc",
        ]
    ].copy()
    cum_view["cum_l1d_hit_rate_pct"] = cum_view["cum_l1d_hit_rate_pct"].map(
        lambda x: f"{x:.2f}%"
    )
    cum_view["cum_ipc"] = cum_view["cum_ipc"].map(lambda x: f"{x:.4f}")

    return (
        "=== Per-Benchmark L1D Hit Rates ===\n"
        + per_bench_view.to_string(index=False)
        + "\n\n=== Cumulative L1D Hit Rates Across Benchmarks ===\n"
        + cum_view.to_string(index=False)
    )


def send_discord_notification(webhook_url: str, message: str) -> None:
    """Post a message to a Discord webhook, splitting into chunks under 2000 chars."""
    max_chunk = 1900
    for start in range(0, len(message), max_chunk):
        chunk = message[start : start + max_chunk]
        payload = json.dumps({"content": f"```\n{chunk}\n```"}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ChampSimPerBenchmarkPlotter/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report L1D hit rates and plot only the per-benchmark L1D hit rate chart."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing ChampSim log files (default: output)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where generated PNG chart is saved (default: plots)",
    )
    parser.add_argument(
        "--pattern",
        default="{predictor}.{benchmark}",
        help="Log filename pattern inside --output-dir (default: '{predictor}.{benchmark}')",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=DEFAULT_PREDICTORS,
        help="Predictors to compare (default: nopred nextline stride openpref)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="Benchmarks to include (default: charlie k_clique llama2 mcf merced roms sssp whisper)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to export parsed per-benchmark metrics as CSV",
    )
    parser.add_argument(
        "--cum-csv",
        type=Path,
        default=None,
        help="Optional path to export cumulative L1 hit rates as CSV",
    )
    parser.add_argument(
        "--discord",
        action="store_true",
        help="Send text summary to DISCORD_WEBHOOK_URL",
    )
    parser.add_argument(
        "--webhook-url",
        default=os.environ.get("DISCORD_WEBHOOK_URL", ""),
        help="Discord webhook URL (defaults to $DISCORD_WEBHOOK_URL)",
    )
    args = parser.parse_args()

    df = collect_dataframe(
        output_dir=args.output_dir,
        predictors=args.predictors,
        benchmarks=args.benchmarks,
        pattern=args.pattern,
    )
    if df.empty:
        print(
            f"No valid ChampSim logs found in '{args.output_dir}' matching pattern '{args.pattern}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    cum_df = compute_cumulative_l1_hit_rates(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Saved CSV to {args.csv}")
    if args.cum_csv:
        args.cum_csv.parent.mkdir(parents=True, exist_ok=True)
        cum_df.to_csv(args.cum_csv, index=False)
        print(f"Saved cumulative CSV to {args.cum_csv}")

    summary = format_text_summary(df, cum_df)
    print(summary)

    plot_l1d_hit_rate_per_benchmark(
        df=df,
        predictors=args.predictors,
        benchmarks=args.benchmarks,
        out_path=args.out_dir / "per_benchmark_l1d_hit_rate.png",
        fmt="%.1f%%",
    )

    if args.discord:
        if not args.webhook_url:
            print("Error: --discord set but DISCORD_WEBHOOK_URL is empty.", file=sys.stderr)
            sys.exit(1)
        send_discord_notification(args.webhook_url, summary)
        print("Sent summary to Discord.")


if __name__ == "__main__":
    main()