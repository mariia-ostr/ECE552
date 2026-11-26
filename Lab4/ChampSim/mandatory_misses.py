#!/usr/bin/env python3
"""
Count the mandatory (compulsory / cold) misses in a memory-access trace.

A mandatory miss is the first access ever made to a cache line: no cache of any
size or associativity can avoid it, so it equals the number of distinct lines
the trace touches.

Input CSV (no header needed), three columns per row:
    pc, address, hit/miss

Reports:
  * mandatory misses (= unique lines touched) and their share of all accesses
  * how they compare with the misses recorded in the trace:
      - recorded misses that are mandatory
      - recorded misses that are NOT mandatory (capacity / conflict misses)
      - first touches the trace recorded as hits (e.g. warm cache or prefetched)
  * a per-PC CSV and a plot of mandatory misses over time

Usage:
    python mandatory_misses.py trace.csv
    python mandatory_misses.py trace.csv --line-size 64 --outdir mandatory_out
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HIT_STRINGS = {"1", "hit", "h", "true", "t", "yes", "y"}


def parse_int(s):
    """Parse decimal, 0x-prefixed hex, or bare hex strings."""
    s = str(s).strip()
    try:
        return int(s, 0)
    except ValueError:
        return int(s, 16)


def parse_column(col):
    mapping = {v: parse_int(v) for v in col.unique()}
    return col.map(mapping).to_numpy(dtype=np.int64)


def load_trace(path):
    df = pd.read_csv(path, header=None, names=["pc", "addr", "hit"],
                     usecols=[0, 1, 2], dtype=str, skipinitialspace=True)
    try:  # skip a header row if there is one
        parse_int(df.iloc[0]["pc"])
    except ValueError:
        df = df.iloc[1:].reset_index(drop=True)
    return pd.DataFrame({
        "pc": parse_column(df["pc"]),
        "addr": parse_column(df["addr"]),
        "hit": df["hit"].str.strip().str.lower().isin(HIT_STRINGS).to_numpy(),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="input trace: pc,address,hit")
    ap.add_argument("--line-size", type=int, default=64,
                    help="cache line size in bytes, a power of two (default 64)")
    ap.add_argument("--outdir", default="mandatory_misses_out", help="output directory")
    args = ap.parse_args()

    if args.line_size & (args.line_size - 1) or args.line_size < 1:
        ap.error("--line-size must be a power of two")
    shift = args.line_size.bit_length() - 1
    os.makedirs(args.outdir, exist_ok=True)

    df = load_trace(args.csv)
    n = len(df)
    lines = df["addr"].to_numpy() >> shift

    # First access to each line in the whole trace = mandatory miss
    is_first = ~pd.Series(lines).duplicated().to_numpy()
    hit = df["hit"].to_numpy()
    miss = ~hit

    mandatory = int(is_first.sum())
    recorded_misses = int(miss.sum())
    mandatory_recorded_miss = int((is_first & miss).sum())
    mandatory_recorded_hit = int((is_first & hit).sum())
    non_mandatory_misses = recorded_misses - mandatory_recorded_miss

    # ---- console report ----
    def pct(a, b):
        return f"{a / b:.1%}" if b else "n/a"

    print(f"Trace accesses            : {n:,}")
    print(f"Line size                 : {args.line_size} B")
    print(f"Mandatory misses          : {mandatory:,}  ({pct(mandatory, n)} of accesses)")
    print(f"Recorded misses in trace  : {recorded_misses:,}  ({pct(recorded_misses, n)} of accesses)")
    print(f"  mandatory               : {mandatory_recorded_miss:,}  "
          f"({pct(mandatory_recorded_miss, recorded_misses)} of recorded misses)")
    print(f"  non-mandatory (capacity/conflict): {non_mandatory_misses:,}  "
          f"({pct(non_mandatory_misses, recorded_misses)} of recorded misses)")
    print(f"First touches recorded as hits    : {mandatory_recorded_hit:,}  "
          f"(warm cache or prefetch; these are mandatory but not counted as misses)")

    # ---- per-PC CSV: a mandatory miss is credited to the PC that first touches the line ----
    per_pc = pd.DataFrame({"pc": df["pc"], "first": is_first, "miss": miss,
                           "first_miss": is_first & miss})
    g = per_pc.groupby("pc", sort=False)
    out = pd.DataFrame({
        "accesses": g.size(),
        "mandatory_misses": g["first"].sum(),
        "recorded_misses": g["miss"].sum(),
        "mandatory_recorded_misses": g["first_miss"].sum(),
    })
    out["non_mandatory_misses"] = out["recorded_misses"] - out["mandatory_recorded_misses"]
    out["mandatory_share_of_misses"] = (out["mandatory_recorded_misses"] /
                                        out["recorded_misses"].replace(0, np.nan))
    out = out.sort_values("mandatory_misses", ascending=False)
    out.index = [hex(pc) for pc in out.index]
    out.index.name = "pc"
    out.to_csv(os.path.join(args.outdir, "mandatory_misses_by_pc.csv"))

    print("\nTop 10 PCs by mandatory misses:")
    print(out.head(10)[["accesses", "mandatory_misses", "recorded_misses",
                        "mandatory_share_of_misses"]].to_string())

    # ---- plot: cumulative mandatory vs. recorded misses over the trace ----
    step = max(n // 2000, 1)
    xs = np.arange(0, n, step)
    cum_mand = np.cumsum(is_first)[xs]
    cum_miss = np.cumsum(miss)[xs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, cum_miss, label="recorded misses", color="#E45756")
    ax.plot(xs, cum_mand, label="mandatory misses (unique lines)", color="#4C78A8")
    ax.set_xlabel("Access index")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Mandatory vs. recorded misses over the trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "mandatory_misses.png"), dpi=150)
    plt.close(fig)

    print(f"\nOutputs written to: {args.outdir}/")


if __name__ == "__main__":
    main()