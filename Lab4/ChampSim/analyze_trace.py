#!/usr/bin/env python3
"""
Analyze a memory-access trace CSV with three columns per row:
    pc, address, hit/miss

For each unique PC it:
  * collects every address that PC accessed (in program order)
  * classifies the access pattern as constant / strided / patterned_random / random
    (patterned_random = the PC replays roughly the same address sequence over and
    over; the pattern length is reported)
  * finds "across-PC patterns": different PCs that request the same run of
    addresses in the same order (see analyze_across_pc)
  * records the PC's lifespan (last occurrence index - first occurrence index)

"Time" is the row index in the file, since the trace has no timestamps.

Usage:
    python analyze_trace.py trace.csv
    python analyze_trace.py trace.csv --outdir results --stride-threshold 0.9
"""
import argparse
import csv
import os
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = ["constant", "strided", "patterned_random", "random"]
COLORS = {"constant": "#4C78A8", "strided": "#F58518",
          "patterned_random": "#B279A2", "random": "#54A24B"}
HIT_STRINGS = {"1", "hit", "h", "true", "t", "yes", "y"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_int(s):
    """Parse decimal, 0x-prefixed hex, or bare hex strings."""
    s = str(s).strip()
    try:
        return int(s, 0)
    except ValueError:
        return int(s, 16)


def parse_column(col):
    """Parse a column of ints, only converting each unique string once."""
    mapping = {v: parse_int(v) for v in col.unique()}
    return col.map(mapping).to_numpy(dtype=np.int64)


def load_trace(path):
    df = pd.read_csv(path, header=None, names=["pc", "addr", "hit"],
                     usecols=[0, 1, 2], dtype=str, skipinitialspace=True)
    # Skip a header row if present (first PC isn't parseable as a number)
    try:
        parse_int(df.iloc[0]["pc"])
    except ValueError:
        df = df.iloc[1:].reset_index(drop=True)

    out = pd.DataFrame({
        "pc": parse_column(df["pc"]),
        "addr": parse_column(df["addr"]),
        "hit": df["hit"].str.strip().str.lower().isin(HIT_STRINGS).to_numpy(),
    })
    out.index.name = "time"  # row index acts as the timestamp
    return out


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def find_pattern(addrs, match_threshold, min_repeats, max_candidates):
    """
    Look for a repeating address sequence with period p: addrs[i] ~= addrs[i+p].

    Candidate periods come from the most common distances between repeat
    visits to the same address (cheap, O(n log n)). For each candidate, the
    match fraction is mean(addrs[p:] == addrs[:-p]). The smallest period whose
    match fraction >= match_threshold wins, so noise/substitutions are tolerated.

    Returns (period, match_fraction) or None.
    """
    n = len(addrs)
    order = np.argsort(addrs, kind="stable")  # stable: positions ascend per address
    sorted_addrs = addrs[order]
    same = sorted_addrs[1:] == sorted_addrs[:-1]
    dists = (order[1:] - order[:-1])[same]
    if len(dists) == 0:
        return None

    vals, counts = np.unique(dists, return_counts=True)
    candidates = np.sort(vals[np.argsort(-counts)[:max_candidates]])
    for p in candidates:
        if p < 2 or n < min_repeats * p:
            continue
        match = float(np.mean(addrs[p:] == addrs[:-p]))
        if match >= match_threshold:
            return int(p), match
    return None


def classify(addrs, stride_threshold, pattern_threshold=0.8,
             min_repeats=3, max_candidates=16):
    """
    Returns (category, stride, pattern_length, pattern_match).
      constant        : every access hits the same address (or >= stride_threshold
                        of the deltas are zero)
      strided         : one nonzero delta between consecutive accesses accounts
                        for >= stride_threshold of all deltas (tolerates
                        loop-wraparound jumps)
      patterned_random: not constant/strided, but the address sequence repeats
                        with some period p (>= pattern_threshold of accesses
                        match the access p steps earlier, and the sequence
                        covers at least min_repeats periods)
      random          : everything else
    """
    if len(addrs) < 2 or np.all(addrs == addrs[0]):
        return "constant", 0, 0, np.nan

    deltas = np.diff(addrs)
    vals, counts = np.unique(deltas, return_counts=True)
    top = vals[counts.argmax()]
    frac = counts.max() / len(deltas)

    if frac >= stride_threshold:
        if top == 0:
            return "constant", 0, 0, np.nan
        return "strided", int(top), 0, np.nan

    found = find_pattern(addrs, pattern_threshold, min_repeats, max_candidates)
    if found is not None:
        return "patterned_random", 0, found[0], found[1]
    return "random", 0, 0, np.nan


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze(df, stride_threshold, pattern_threshold=0.8, min_repeats=3):
    pc_addrs = {}  # pc -> numpy array of every address it requested, in order
    pc_gaps = {}   # pc -> memory instructions between consecutive uses of that PC
    gaps_by_cat = {c: [] for c in CATEGORIES}
    rows = []

    for pc, g in df.groupby("pc", sort=False):
        addrs = g["addr"].to_numpy()
        times = g.index.to_numpy()
        pc_addrs[pc] = addrs

        category, stride, pat_len, pat_match = classify(
            addrs, stride_threshold, pattern_threshold, min_repeats)

        # Gap = number of memory instructions strictly between two consecutive
        # executions of this PC (adjacent rows -> gap of 0).
        gaps = np.diff(times) - 1
        pc_gaps[pc] = gaps
        if len(gaps):
            gaps_by_cat[category].append(gaps)

        rows.append({
            "pc": hex(pc),
            "n_accesses": len(addrs),
            "n_unique_addrs": len(np.unique(addrs)),
            "category": category,
            "stride": stride,
            "addr_min": int(addrs.min()),
            "addr_max": int(addrs.max()),
            "addr_range": int(addrs.max() - addrs.min()),
            "pattern_length": pat_len,
            "pattern_match": pat_match,
            "first_seen": int(times[0]),
            "last_seen": int(times[-1]),
            "lifespan": int(times[-1] - times[0]),
            "mean_gap": float(gaps.mean()) if len(gaps) else np.nan,
            "median_gap": float(np.median(gaps)) if len(gaps) else np.nan,
            "hit_rate": g["hit"].mean(),
        })

    summary = pd.DataFrame(rows)
    gaps_by_cat = {c: (np.concatenate(v) if v else np.array([], dtype=np.int64))
                   for c, v in gaps_by_cat.items()}
    return pc_addrs, pc_gaps, gaps_by_cat, summary


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(summary, total_rows, outdir):
    # 1. Category counts: by PC and by access
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    by_pc = summary["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    by_acc = summary.groupby("category")["n_accesses"].sum().reindex(CATEGORIES, fill_value=0)
    for ax, data, title in [(axes[0], by_pc, "Unique PCs per category"),
                            (axes[1], by_acc, "Memory accesses per category")]:
        ax.bar(data.index, data.values, color=[COLORS[c] for c in data.index])
        ax.set_title(title)
        for i, v in enumerate(data.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "category_counts.png"), dpi=150)
    plt.close(fig)

    # 2. Lifespan histogram per category (as fraction of trace length)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(0, np.log10(max(total_rows, 10)), 40)
    for c in CATEGORIES:
        sub = summary[summary["category"] == c]["lifespan"].clip(lower=1)
        if len(sub):
            ax.hist(sub, bins=bins, alpha=0.6, label=f"{c} ({len(sub)})", color=COLORS[c])
    ax.set_xscale("log")
    ax.set_xlabel("PC lifespan (accesses between first and last use)")
    ax.set_ylabel("Number of PCs")
    ax.set_title("PC lifespan distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lifespan_hist.png"), dpi=150)
    plt.close(fig)

    # 3. Timeline of the most-accessed PCs (Gantt-style lifespan bars)
    top = summary.nlargest(30, "n_accesses").sort_values("first_seen")
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(top) + 1.5))
    for i, (_, r) in enumerate(top.iterrows()):
        ax.barh(i, max(r["lifespan"], total_rows * 0.001), left=r["first_seen"],
                color=COLORS[r["category"]])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["pc"], fontsize=7)
    ax.set_xlabel("Access index")
    ax.set_title("Lifespan of the 30 most-accessed PCs")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CATEGORIES]
    ax.legend(handles, CATEGORIES, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pc_timeline.png"), dpi=150)
    plt.close(fig)

    # 4. Hit rate per category (access-weighted)
    fig, ax = plt.subplots(figsize=(5, 4))
    hits = (summary["hit_rate"] * summary["n_accesses"]).groupby(summary["category"]).sum()
    accs = summary.groupby("category")["n_accesses"].sum()
    rate = (hits / accs).reindex(CATEGORIES).fillna(0)
    ax.bar(rate.index, rate.values, color=[COLORS[c] for c in rate.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit rate by access pattern")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hit_rate_by_category.png"), dpi=150)
    plt.close(fig)


def make_gap_plots(gaps_by_cat, outdir):
    """Distribution of memory instructions between consecutive uses of a PC."""
    all_gaps = np.concatenate([g for g in gaps_by_cat.values() if len(g)]
                              or [np.array([], dtype=np.int64)])
    if len(all_gaps) == 0:
        print("No PC executed more than once; skipping gap plots.")
        return

    # Power-of-two bins; the first bin is exactly 0 (back-to-back accesses).
    max_gap = max(int(all_gaps.max()), 1)
    edges = [0, 1] + [2 ** k for k in range(1, int(np.ceil(np.log2(max_gap + 1))) + 2)]
    labels = ["0"] + [f"{lo}-{hi - 1}" if hi - 1 > lo else f"{lo}"
                      for lo, hi in zip(edges[1:-1], edges[2:])]

    counts = {c: np.histogram(g, bins=edges)[0] for c, g in gaps_by_cat.items()}

    # Save the raw histogram
    hist = pd.DataFrame({"gap_range": labels, **{c: counts[c] for c in CATEGORIES}})
    hist["total"] = hist[CATEGORIES].sum(axis=1)
    hist.to_csv(os.path.join(outdir, "gap_histogram.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: stacked histogram
    bottom = np.zeros(len(labels))
    for c in CATEGORIES:
        axes[0].bar(range(len(labels)), counts[c], bottom=bottom,
                    color=COLORS[c], label=c)
        bottom += counts[c]
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Memory instructions between two accesses of the same PC")
    axes[0].set_ylabel("Number of PC re-executions")
    axes[0].set_title("Reuse-gap distribution")
    axes[0].legend()

    # Right: CDF per category
    xs = np.unique(np.concatenate([[0], all_gaps]))
    for c in CATEGORIES:
        g = np.sort(gaps_by_cat[c])
        if len(g):
            cdf = np.searchsorted(g, xs, side="right") / len(g)
            axes[1].plot(xs + 1, cdf, color=COLORS[c], label=c)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Gap + 1 (log scale)")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("Reuse-gap CDF by access pattern")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "gap_distribution.png"), dpi=150)
    plt.close(fig)

    return all_gaps


def make_pattern_plot(summary, outdir):
    """Distribution of detected pattern lengths for patterned_random PCs."""
    pat = summary[summary["category"] == "patterned_random"]
    if pat.empty:
        print("No patterned_random PCs found; skipping pattern-length plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lengths = pat["pattern_length"].to_numpy()
    bins = np.logspace(np.log10(max(lengths.min(), 2)) - 0.05,
                       np.log10(lengths.max()) + 0.05, 25) if lengths.max() > lengths.min() \
        else np.array([lengths.min() - 0.5, lengths.min() + 0.5])
    axes[0].hist(lengths, bins=bins, color=COLORS["patterned_random"])
    if lengths.max() > lengths.min():
        axes[0].set_xscale("log")
    axes[0].set_xlabel("Pattern length (accesses per repetition)")
    axes[0].set_ylabel("Number of PCs")
    axes[0].set_title("Pattern length distribution")

    axes[1].scatter(pat["pattern_length"], pat["pattern_match"],
                    s=np.clip(pat["n_accesses"] / pat["n_accesses"].max() * 200, 10, 200),
                    color=COLORS["patterned_random"], alpha=0.7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Pattern length")
    axes[1].set_ylabel("Match fraction (1.0 = exact repeat)")
    axes[1].set_title("Pattern fidelity (size = access count)")
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pattern_lengths.png"), dpi=150)
    plt.close(fig)


RANDOM_LIKE = ["patterned_random", "random"]


def make_range_plot(summary, outdir):
    """Address range (max - min) of the two random-like categories."""
    sub = summary[summary["category"].isin(RANDOM_LIKE)]
    if sub.empty:
        print("No random / patterned_random PCs; skipping address-range plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of per-PC address range
    ranges = sub["addr_range"].clip(lower=1)
    bins = np.logspace(0, np.log10(max(ranges.max(), 10)) + 0.1, 40)
    for c in RANDOM_LIKE:
        r = sub[sub["category"] == c]["addr_range"].clip(lower=1)
        if len(r):
            axes[0].hist(r, bins=bins, alpha=0.6, color=COLORS[c], label=f"{c} ({len(r)})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Address range per PC (max - min, bytes)")
    axes[0].set_ylabel("Number of PCs")
    axes[0].set_title("Address range of random-like PCs")
    axes[0].legend()

    # Right: [min, max] span of the busiest PCs (log address axis)
    top = sub.nlargest(25, "n_accesses").sort_values("addr_min")
    for i, (_, r) in enumerate(top.iterrows()):
        left = max(r["addr_min"], 1)                 # log axis needs > 0
        right = max(r["addr_max"], left * 1.03)      # keep tiny spans visible
        axes[1].barh(i, right - left, left=left, color=COLORS[r["category"]])
    axes[1].set_xscale("log", base=16)  # ticks land on 0x10, 0x100, 0x1000, ...
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top["pc"], fontsize=7)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: hex(int(x))))
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=8)
    axes[1].set_xlabel("Address (log scale)")
    axes[1].set_title("Address span of the 25 busiest random-like PCs")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in RANDOM_LIKE]
    axes[1].legend(handles, RANDOM_LIKE, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "address_ranges.png"), dpi=150)
    plt.close(fig)


def write_pc_addresses(pc_addrs, summary, path, max_addrs=None):
    """One row per PC: the PC followed by every address it requested, in
    program order (hex). Rows are ragged; PCs are ordered by access count."""
    order = [int(pc, 16) for pc in summary["pc"]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for pc in order:
            addrs = pc_addrs[pc]
            if max_addrs is not None:
                addrs = addrs[:max_addrs]
            w.writerow([hex(pc)] + [hex(a) for a in addrs])


# --------------------------------------------------------------------------- #
# Across-PC patterns
# --------------------------------------------------------------------------- #
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MULT = [np.uint64((0x9E3779B97F4A7C15 * (2 * j + 1)) & _MASK64) for j in range(64)]


def kgram_hashes(addrs, k):
    """64-bit hash of every window of k consecutive addresses (order-sensitive)."""
    n = len(addrs)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    a = addrs.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n - k + 1, dtype=np.uint64)
    for j in range(k):
        h += a[j:n - k + 1 + j] * _MULT[j]
    return h


def analyze_across_pc(pc_addrs, summary, k, threshold, include_constant, max_run):
    """
    Find PCs that request the same address sequences.

    Every PC's stream is cut into overlapping windows of k consecutive addresses
    (k-grams). Two PCs share an across-PC pattern when a large fraction of the
    smaller PC's distinct k-grams also appear in the other PC:
        containment = shared_kgrams / min(distinct_kgrams_a, distinct_kgrams_b)
    Pairs with containment >= threshold are linked, and linked PCs are merged
    into groups. Because matching is on k-grams, the sequences don't have to be
    identical end to end, only to share runs of >= k addresses.

    Returns (pairs_df, groups_df, pc_group, pc_frac_shared, skipped_runs).
      pc_group       : {pc: group_id}  (ids start at 1; PCs not in a group are absent)
      pc_frac_shared : {pc: fraction of that PC's distinct k-grams seen in another PC}
    """
    cat_of = {int(r.pc, 16): r.category for r in summary.itertuples()}
    total_of = {int(r.pc, 16): r.n_accesses for r in summary.itertuples()}
    pcs = [pc for pc in pc_addrs if include_constant or cat_of[pc] != "constant"]

    H, P, POS = [], [], []
    uniq = np.zeros(len(pcs), dtype=np.int64)
    for i, pc in enumerate(pcs):
        h = kgram_hashes(pc_addrs[pc], k)
        if len(h) == 0:
            continue
        uh, first = np.unique(h, return_index=True)
        H.append(uh)
        P.append(np.full(len(uh), i, dtype=np.int64))
        POS.append(first)
        uniq[i] = len(uh)

    empty = (pd.DataFrame(), pd.DataFrame(), {}, {}, 0)
    if not H:
        return empty
    H, P, POS = np.concatenate(H), np.concatenate(P), np.concatenate(POS)
    order = np.argsort(H, kind="stable")
    H, P, POS = H[order], P[order], POS[order]

    # Runs of equal hash = the same k-gram seen in several PCs
    bounds = np.concatenate(([0], np.flatnonzero(H[1:] != H[:-1]) + 1, [len(H)]))
    lens = np.diff(bounds)
    shared_count = np.zeros(len(pcs), dtype=np.int64)
    pair_counts, pair_example, skipped = Counter(), {}, 0
    for r in np.flatnonzero(lens >= 2):
        st, en = bounds[r], bounds[r + 1]
        members = P[st:en]
        shared_count[members] += 1
        if len(members) > max_run:   # avoid pair explosion for very common k-grams
            skipped += 1
            continue
        for a, b in combinations(sorted(members.tolist()), 2):
            pair_counts[(a, b)] += 1
            pair_example.setdefault((a, b), (int(P[st]), int(POS[st])))

    def example(pc_idx, pos):
        return "->".join(hex(x) for x in pc_addrs[pcs[pc_idx]][pos:pos + k])

    # Pairs that pass the containment threshold, merged with union-find
    parent = list(range(len(pcs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pair_rows = []
    for (a, b), cnt in pair_counts.items():
        contain = cnt / min(uniq[a], uniq[b])
        if contain < threshold:
            continue
        parent[find(a)] = find(b)
        ex = pair_example[(a, b)]
        pair_rows.append({
            "pc_a": hex(pcs[a]), "pc_b": hex(pcs[b]),
            "category_a": cat_of[pcs[a]], "category_b": cat_of[pcs[b]],
            "shared_kgrams": cnt,
            "containment": contain,
            "jaccard": cnt / (uniq[a] + uniq[b] - cnt),
            "example_sequence": example(*ex),
            "_a": a, "_b": b,
        })
    if not pair_rows:
        return (pd.DataFrame(), pd.DataFrame(), {}, {}, skipped)
    pairs = pd.DataFrame(pair_rows).sort_values("shared_kgrams", ascending=False)

    comps = {}
    for i in {x for r in pair_rows for x in (r["_a"], r["_b"])}:
        comps.setdefault(find(i), []).append(i)
    comps = sorted(comps.values(),
                   key=lambda m: (-len(m), -sum(total_of[pcs[i]] for i in m)))

    pc_group, group_rows = {}, []
    for gid, members in enumerate(comps, start=1):
        mset = set(members)
        for i in members:
            pc_group[pcs[i]] = gid
        best = pairs[pairs["_a"].isin(mset)].iloc[0]   # strongest pair in the group
        group_rows.append({
            "group_id": gid,
            "n_pcs": len(members),
            "pcs": " ".join(hex(pcs[i]) for i in sorted(members, key=lambda i: pcs[i])),
            "categories": " ".join(sorted({cat_of[pcs[i]] for i in members})),
            "total_accesses": sum(total_of[pcs[i]] for i in members),
            "example_sequence": best["example_sequence"],
        })
    groups = pd.DataFrame(group_rows)
    pairs = pairs.drop(columns=["_a", "_b"])

    frac = {pcs[i]: shared_count[i] / uniq[i] for i in range(len(pcs)) if uniq[i]}
    return pairs, groups, pc_group, frac, skipped


def make_across_plot(summary, groups, outdir):
    if groups.empty:
        print("No across-PC patterns found; skipping across-PC plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sizes = groups["n_pcs"].value_counts().sort_index()
    axes[0].bar(sizes.index.astype(str), sizes.values, color="#E45756")
    axes[0].set_xlabel("PCs in group")
    axes[0].set_ylabel("Number of groups")
    axes[0].set_title("Across-PC group sizes")

    in_group = summary[summary["across_pc_group"] > 0]
    by_cat = in_group["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    axes[1].bar(by_cat.index, by_cat.values, color=[COLORS[c] for c in by_cat.index])
    axes[1].set_title("PCs in an across-PC group, by own category")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(by_cat.values):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "across_pc_groups.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="input trace: pc,address,hit")
    ap.add_argument("--outdir", default="trace_analysis", help="output directory")
    ap.add_argument("--stride-threshold", type=float, default=0.9,
                    help="fraction of deltas that must share one value to call a "
                         "PC constant/strided (default 0.9)")
    ap.add_argument("--pattern-threshold", type=float, default=0.8,
                    help="fraction of accesses that must equal the access one "
                         "period earlier to call a PC patterned_random (default 0.8)")
    ap.add_argument("--min-repeats", type=int, default=3,
                    help="a pattern must repeat at least this many times (default 3)")
    ap.add_argument("--max-addrs-per-pc", type=int, default=None,
                    help="truncate each row of pc_addresses.csv to the first N "
                         "addresses (default: write all)")
    ap.add_argument("--across-k", type=int, default=4,
                    help="window length (consecutive addresses) used to match "
                         "sequences across PCs (default 4, max 64)")
    ap.add_argument("--across-threshold", type=float, default=0.5,
                    help="fraction of the smaller PC's k-grams that must also "
                         "appear in the other PC (default 0.5)")
    ap.add_argument("--include-constant", action="store_true",
                    help="also compare constant PCs across PCs (off by default: "
                         "two PCs hitting one address trivially match)")
    ap.add_argument("--across-max-run", type=int, default=64,
                    help="ignore k-grams shared by more than this many PCs when "
                         "pairing PCs (default 64)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_trace(args.csv)
    pc_addrs, pc_gaps, gaps_by_cat, summary = analyze(
        df, args.stride_threshold, args.pattern_threshold, args.min_repeats)

    summary = summary.sort_values("n_accesses", ascending=False)

    pairs, groups, pc_group, pc_frac, skipped = analyze_across_pc(
        pc_addrs, summary, min(args.across_k, 64), args.across_threshold,
        args.include_constant, args.across_max_run)
    pcs_int = summary["pc"].map(lambda x: int(x, 16))
    summary["across_pc_group"] = pcs_int.map(lambda x: pc_group.get(x, 0))
    summary["across_pc_pattern"] = summary["across_pc_group"] > 0
    summary["frac_kgrams_shared"] = pcs_int.map(lambda x: pc_frac.get(x, np.nan))
    pairs.to_csv(os.path.join(args.outdir, "across_pc_pairs.csv"), index=False)
    groups.to_csv(os.path.join(args.outdir, "across_pc_groups.csv"), index=False)
    write_pc_addresses(pc_addrs, summary, os.path.join(args.outdir, "pc_addresses.csv"),
                       args.max_addrs_per_pc)
    csv_out = summary.copy()
    csv_out["addr_min"] = csv_out["addr_min"].map(hex)
    csv_out["addr_max"] = csv_out["addr_max"].map(hex)
    csv_out.to_csv(os.path.join(args.outdir, "pc_summary.csv"), index=False)
    make_plots(summary, len(df), args.outdir)
    all_gaps = make_gap_plots(gaps_by_cat, args.outdir)
    make_pattern_plot(summary, args.outdir)
    make_range_plot(summary, args.outdir)
    make_across_plot(summary, groups, args.outdir)

    # Console report
    print(f"Total accesses : {len(df):,}")
    print(f"Unique PCs     : {len(summary):,}")
    print(f"Overall hit rate: {df['hit'].mean():.3f}\n")
    print("Category breakdown:")
    for c in CATEGORIES:
        sub = summary[summary["category"] == c]
        print(f"  {c:16s} {len(sub):6d} PCs  {sub['n_accesses'].sum():12,d} accesses")
    print(f"\nLifespan (accesses): mean={summary['lifespan'].mean():,.0f}  "
          f"median={summary['lifespan'].median():,.0f}  "
          f"max={summary['lifespan'].max():,}")
    n_across = int(summary["across_pc_pattern"].sum())
    print(f"\nAcross-PC patterns (k={args.across_k}, containment >= {args.across_threshold}): "
          f"{len(groups)} groups, {n_across} of {len(summary)} PCs involved")
    for r in groups.head(5).itertuples():
        print(f"  group {r.group_id}: {r.n_pcs} PCs [{r.categories}]  e.g. {r.example_sequence}")
    if skipped:
        print(f"  (skipped {skipped} k-grams shared by > {args.across_max_run} PCs)")

    print("\nAddress ranges (max - min, bytes) for random-like PCs:")
    for c in RANDOM_LIKE:
        sub = summary[summary["category"] == c]
        if sub.empty:
            print(f"  {c:16s} (none)")
            continue
        print(f"  {c:16s} {len(sub):5d} PCs  overall {hex(sub['addr_min'].min())} - "
              f"{hex(sub['addr_max'].max())}  |  per-PC range: "
              f"min={sub['addr_range'].min():,}  median={sub['addr_range'].median():,.0f}  "
              f"max={sub['addr_range'].max():,}")
    print()
    pat = summary[summary["category"] == "patterned_random"]
    if len(pat):
        print(f"Pattern length (patterned_random PCs): min={pat['pattern_length'].min()}  "
              f"median={pat['pattern_length'].median():.0f}  "
              f"max={pat['pattern_length'].max()}")
    if all_gaps is not None:
        p = np.percentile(all_gaps, [50, 90, 99])
        print(f"Reuse gap (memory instrs between same-PC accesses): "
              f"mean={all_gaps.mean():,.1f}  median={p[0]:,.0f}  "
              f"p90={p[1]:,.0f}  p99={p[2]:,.0f}  max={all_gaps.max():,}")
    print(f"\nTop 10 PCs by access count:")
    print(summary.head(10).assign(addr_min=lambda d: d["addr_min"].map(hex),
                                  addr_max=lambda d: d["addr_max"].map(hex))[["pc", "n_accesses", "n_unique_addrs", "category",
                            "pattern_length", "addr_min", "addr_max", "addr_range"]].to_string(index=False))
    print(f"\nOutputs written to: {args.outdir}/")


if __name__ == "__main__":
    main()