// Triangel (Ainsworth & Mukhanov, ISCA 2024, arXiv:2406.10627) behind the
// openpref interface: openpref::prefetcher_initialize() and
// openpref::access(pc, addr, hit), issuing prefetches via prefetch(addr).
//
// Implemented from the paper:
//   - PC-indexed Training Table (LastAddr[0..1], Timestamp, ReuseConf,
//     2x PatternConf, SampleRate, Lookahead)
//   - Markov table (single successor + 1 confidence bit, SRRIP replacement)
//   - History Sampler (2-way) with per-PC adaptive sampling rate
//   - Second-Chance Sampler (FIFO)
//   - Confidence-gated storage/prefetch, lookahead-2 and degree-4 when confident
//
// Adaptations (the paper assumes a gem5 L2 + partitioned L3 which openpref lacks):
//   - Markov table is a fixed-size standalone structure, so the Set Dueller
//     (cache/Markov partition sizing) and the Metadata Reuse Buffer (L3 access
//     filter) have nothing to act on and are omitted.
//   - "L2 miss or tagged prefetch hit" becomes "access with hit == false, or an
//     access to a block this prefetcher previously prefetched".
//   - A per-PC stride prefetcher is kept alongside, matching the paper's
//     baseline system (stride at L1 + Triangel).

#include "openpref.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {

constexpr uint64_t BLOCK_OFFSET_BITS = 6;

// ---- Training table ----
constexpr size_t TT_ENTRIES = 512;

// ---- Markov table (paper: 1MiB partition; here a fixed 64K-entry table) ----
constexpr size_t MK_SETS = 4096;
constexpr size_t MK_WAYS = 16;
constexpr uint64_t MAX_SIZE = MK_SETS * MK_WAYS; // reuse-distance limit

// ---- History sampler (paper: 512 entries, 2-way) ----
constexpr size_t HS_SETS = 256;
constexpr size_t HS_WAYS = 2;
constexpr uint64_t HS_ENTRIES = HS_SETS * HS_WAYS;

// ---- Second-chance sampler (paper: 64 entries, 512-access window) ----
constexpr size_t SCS_ENTRIES = 64;
constexpr uint64_t SCS_WINDOW = 512;

// ---- Confidence counters (4-bit saturating, initial value 8) ----
constexpr int CONF_MAX = 15;
constexpr int CONF_INIT = 8;
constexpr int SAMPLE_RATE_INIT = 8;
// BasePatternConf: +1 / -2 (needs > 66% accuracy to exceed 8)
// HighPatternConf: +1 / -5 (needs > 83% accuracy to exceed 8)
constexpr int BASE_UP = 1, BASE_DOWN = 2;
constexpr int HIGH_UP = 1, HIGH_DOWN = 5;

constexpr size_t MAX_DEGREE = 4;

// ---- Outstanding prefetch tracker (for "tagged prefetch hit" detection) ----
constexpr size_t PT_ENTRIES = 4096;

// ---- Stride prefetcher (paper's L1 baseline) ----
constexpr size_t STRIDE_ENTRIES = 512;
constexpr int64_t MAX_TRACKED_STRIDE = 128;
constexpr size_t STRIDE_MAX_STEPS = 4;

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

inline void conf_up(uint8_t& v, int d) {
    v = static_cast<uint8_t>(std::min<int>(CONF_MAX, static_cast<int>(v) + d));
}
inline void conf_down(uint8_t& v, int d) {
    v = static_cast<uint8_t>(std::max<int>(0, static_cast<int>(v) - d));
}

struct TrainEntry {
    bool valid = false;
    uint16_t tag = 0;
    uint64_t last[2] = {0, 0}; // LastAddr[0] (newest), LastAddr[1]
    int n_last = 0;            // how many of last[] are valid
    uint64_t timestamp = 0;    // per-PC access counter
    uint8_t reuse = CONF_INIT;
    uint8_t base = CONF_INIT;  // BasePatternConf
    uint8_t high = CONF_INIT;  // HighPatternConf
    uint8_t sample_rate = SAMPLE_RATE_INIT;
    bool lookahead = false;    // false: lookahead 1, true: lookahead 2
};

struct MarkovEntry {
    bool valid = false;
    uint16_t tag = 0;
    uint64_t target = 0; // block address
    bool conf = false;
    uint8_t rrpv = 3;
};

struct SamplerEntry {
    bool valid = false;
    uint64_t addr = 0;   // x (LastAddr[0] at sample time)
    uint64_t target = 0; // y (address that followed x)
    uint32_t train_idx = 0;
    uint64_t ts = 0;     // per-PC timestamp at sample time
    uint64_t stamp = 0;  // global recency, for victim selection
    bool used = false;
};

struct ScsEntry {
    bool valid = false;
    uint64_t addr = 0;
    uint32_t train_idx = 0;
    uint16_t pc_tag = 0;
    uint64_t ts = 0; // global training-event count at insertion
};

struct PrefetchTrack {
    bool valid = false;
    bool used = false;
    uint64_t block = 0;
};

struct StrideEntry {
    bool valid = false;
    uint64_t pc = 0;
    uint64_t last_block = 0;
    int64_t last_stride = 0;
    uint8_t confidence = 0;
};

struct PfBuf {
    uint64_t a[32];
    size_t n = 0;
    void push(uint64_t x) {
        for (size_t i = 0; i < n; ++i) {
            if (a[i] == x) {
                return;
            }
        }
        if (n < 32) {
            a[n++] = x;
        }
    }
};

struct TriangelState {
    uint64_t tick = 0;         // global recency counter
    uint64_t train_count = 0;  // global training-event counter
    uint64_t rng = 0x2545F4914F6CDD1DULL;

    std::array<TrainEntry, TT_ENTRIES> train{};
    std::array<std::array<MarkovEntry, MK_WAYS>, MK_SETS> markov{};
    std::array<std::array<SamplerEntry, HS_WAYS>, HS_SETS> hist{};
    std::array<ScsEntry, SCS_ENTRIES> scs{};
    size_t scs_head = 0;
    std::array<PrefetchTrack, PT_ENTRIES> pt{};
    std::array<StrideEntry, STRIDE_ENTRIES> stride{};
};

TriangelState g;

inline uint64_t next_rand() {
    g.rng ^= g.rng << 13;
    g.rng ^= g.rng >> 7;
    g.rng ^= g.rng << 17;
    return g.rng;
}

// ---------------------------------------------------------------------------
// Pattern-confidence helpers
// ---------------------------------------------------------------------------

inline void pattern_up(TrainEntry& te) {
    conf_up(te.base, BASE_UP);
    conf_up(te.high, HIGH_UP);
}

inline void pattern_down(TrainEntry& te) {
    conf_down(te.base, BASE_DOWN);
    conf_down(te.high, HIGH_DOWN);
}

// Decrement a PC's pattern confidence by training-table slot, if still resident
inline void pattern_down_at(uint32_t idx, uint16_t tag) {
    TrainEntry& te = g.train[idx];
    if (te.valid && te.tag == tag) {
        pattern_down(te);
    }
}

// ---------------------------------------------------------------------------
// Prefetch tracker
// ---------------------------------------------------------------------------

// Records an outstanding prefetch. Returns false if it is already outstanding.
bool pt_issue(uint64_t block) {
    PrefetchTrack& e = g.pt[mix64(block) % PT_ENTRIES];
    if (e.valid && e.block == block && !e.used) {
        return false;
    }
    e.valid = true;
    e.used = false;
    e.block = block;
    return true;
}

// True the first time a prefetched block is demanded ("tagged prefetch hit")
bool pt_consume(uint64_t block) {
    PrefetchTrack& e = g.pt[mix64(block) % PT_ENTRIES];
    if (e.valid && e.block == block && !e.used) {
        e.used = true;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Markov table (single successor, 1 confidence bit, SRRIP)
// ---------------------------------------------------------------------------

inline void markov_locate(uint64_t key, size_t& set, uint16_t& tag) {
    uint64_t h = mix64(key);
    set = static_cast<size_t>(h % MK_SETS);
    tag = static_cast<uint16_t>((h >> 40) & 0xFFFF);
}

bool markov_lookup(uint64_t key, uint64_t& target) {
    size_t set;
    uint16_t tag;
    markov_locate(key, set, tag);
    for (auto& e : g.markov[set]) {
        if (e.valid && e.tag == tag) {
            e.rrpv = 0;
            target = e.target;
            return true;
        }
    }
    return false;
}

void markov_train(uint64_t key, uint64_t target) {
    size_t set;
    uint16_t tag;
    markov_locate(key, set, tag);
    auto& ways = g.markov[set];

    for (auto& e : ways) {
        if (e.valid && e.tag == tag) {
            e.rrpv = 0;
            if (e.target == target) {
                e.conf = true;
            } else if (!e.conf) {
                e.target = target; // replace only when confidence bit is clear
            } else {
                e.conf = false;
            }
            return;
        }
    }

    // Miss: invalid way first, otherwise SRRIP victim (age until rrpv == 3)
    MarkovEntry* victim = nullptr;
    for (auto& e : ways) {
        if (!e.valid) {
            victim = &e;
            break;
        }
    }
    while (!victim) {
        for (auto& e : ways) {
            if (e.rrpv >= 3) {
                victim = &e;
                break;
            }
        }
        if (!victim) {
            for (auto& e : ways) {
                ++e.rrpv;
            }
        }
    }
    victim->valid = true;
    victim->tag = tag;
    victim->target = target;
    victim->conf = false;
    victim->rrpv = 2; // SRRIP long re-reference insertion
}

// ---------------------------------------------------------------------------
// Second-Chance Sampler
// ---------------------------------------------------------------------------

void scs_insert(uint64_t addr, uint32_t idx, uint16_t tag) {
    ScsEntry& slot = g.scs[g.scs_head];
    if (slot.valid) {
        // Left the SCS without being accessed: the prefetch would have been wasted
        pattern_down_at(slot.train_idx, slot.pc_tag);
    }
    slot.valid = true;
    slot.addr = addr;
    slot.train_idx = idx;
    slot.pc_tag = tag;
    slot.ts = g.train_count;
    g.scs_head = (g.scs_head + 1) % SCS_ENTRIES;
}

void scs_check(uint32_t idx, TrainEntry& te, uint64_t cur) {
    for (auto& e : g.scs) {
        if (e.valid && e.addr == cur && e.train_idx == idx && e.pc_tag == te.tag) {
            if (g.train_count - e.ts <= SCS_WINDOW) {
                pattern_up(te);
            } else {
                pattern_down(te);
            }
            e.valid = false;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// History Sampler
// ---------------------------------------------------------------------------

SamplerEntry* hs_find(uint64_t addr, uint32_t idx) {
    auto& set = g.hist[mix64(addr) % HS_SETS];
    for (auto& e : set) {
        if (e.valid && e.addr == addr && e.train_idx == idx) {
            return &e;
        }
    }
    return nullptr;
}

// probability = SamplerSize / MaxSize * 2^(SampleRate - 8)
bool should_sample(uint8_t sample_rate) {
    double prob = std::ldexp(static_cast<double>(HS_ENTRIES) / static_cast<double>(MAX_SIZE),
                             static_cast<int>(sample_rate) - SAMPLE_RATE_INIT);
    if (prob >= 1.0) {
        return true;
    }
    double r = static_cast<double>(next_rand() >> 11) * (1.0 / 9007199254740992.0);
    return r < prob;
}

void hs_insert(uint32_t idx, TrainEntry& te, uint64_t x, uint64_t y) {
    auto& set = g.hist[mix64(x) % HS_SETS];

    SamplerEntry* victim = nullptr;
    for (auto& e : set) {
        if (!e.valid) {
            victim = &e;
            break;
        }
        if (!victim || e.stamp < victim->stamp) {
            victim = &e;
        }
    }

    if (victim->valid) {
        TrainEntry& vte = g.train[victim->train_idx];
        bool resident = vte.valid && victim->train_idx < TT_ENTRIES;
        bool stale = !resident || (vte.timestamp - victim->ts > MAX_SIZE);
        if (stale) {
            // Only replacing an old sample: its PC never showed reuse in range
            if (!victim->used && resident) {
                conf_down(vte.reuse, 1);
            }
            conf_up(te.sample_rate, 1);
        } else if (!victim->used) {
            // Evicting a still-useful unobserved sample: sample less often
            conf_down(te.sample_rate, 1);
        }
    }

    victim->valid = true;
    victim->addr = x;
    victim->target = y;
    victim->train_idx = idx;
    victim->ts = te.timestamp;
    victim->stamp = ++g.tick;
    victim->used = false;
}

void history_sample(uint32_t idx, TrainEntry& te, uint64_t cur) {
    if (te.n_last == 0) {
        return;
    }
    uint64_t x = te.last[0];

    if (SamplerEntry* s = hs_find(x, idx)) {
        s->used = true;
        if (te.timestamp - s->ts < MAX_SIZE) {
            conf_up(te.reuse, 1);
        } else {
            conf_down(te.reuse, 1);
        }
        if (s->target == cur) {
            pattern_up(te);
        } else {
            // Not the exact successor; see whether it still shows up soon
            scs_insert(s->target, idx, te.tag);
        }
        s->target = cur;
        s->ts = te.timestamp;
        s->stamp = ++g.tick;
        return;
    }

    if (should_sample(te.sample_rate)) {
        hs_insert(idx, te, x, cur);
    }
}

// ---------------------------------------------------------------------------
// Triangel core: runs on each training event
// ---------------------------------------------------------------------------

void triangel_event(uint64_t pc, uint64_t cur, PfBuf& out) {
    uint64_t h = mix64(pc);
    uint32_t idx = static_cast<uint32_t>(h % TT_ENTRIES);
    uint16_t tag = static_cast<uint16_t>((h >> 48) & 0xFFFF);

    TrainEntry& te = g.train[idx];
    if (!te.valid || te.tag != tag) {
        te = TrainEntry{};
        te.valid = true;
        te.tag = tag;
    }

    ++te.timestamp;
    ++g.train_count;

    // Sampling-based classification
    scs_check(idx, te, cur);
    history_sample(idx, te, cur);

    // Lookahead control: 2 when HighPatternConf saturates; back to 1 only when
    // BasePatternConf falls below its initial value (avoids rapid switching)
    if (te.high >= CONF_MAX) {
        te.lookahead = true;
    } else if (te.lookahead && te.base < CONF_INIT) {
        te.lookahead = false;
    }

    // Store/prefetch only for PCs with proven reuse and > 66% pattern accuracy
    bool confident = te.reuse > CONF_INIT && te.base > CONF_INIT;

    if (confident) {
        int need = te.lookahead ? 2 : 1;
        if (te.n_last >= need) {
            uint64_t index_addr = te.last[need - 1];
            if (index_addr != cur) {
                markov_train(index_addr, cur);
            }
        }

        // Degree 4 only when HighPatternConf (> 83%) is above its initial value
        size_t degree = (te.high > CONF_INIT) ? MAX_DEGREE : 1;
        uint64_t key = cur;
        for (size_t d = 0; d < degree; ++d) {
            uint64_t target;
            if (!markov_lookup(key, target)) {
                break;
            }
            if (target != cur && pt_issue(target)) {
                out.push(target << BLOCK_OFFSET_BITS);
            }
            key = target;
        }
    }

    // Shift register
    te.last[1] = te.last[0];
    te.last[0] = cur;
    if (te.n_last < 2) {
        ++te.n_last;
    }
}

// ---------------------------------------------------------------------------
// Stride prefetcher
// ---------------------------------------------------------------------------

void stride_event(uint64_t pc, uint64_t addr, PfBuf& out) {
    uint64_t curr_block = addr >> BLOCK_OFFSET_BITS;
    StrideEntry& e = g.stride[mix64(pc) % STRIDE_ENTRIES];

    if (!e.valid || e.pc != pc) {
        e = StrideEntry{};
        e.valid = true;
        e.pc = pc;
        e.last_block = curr_block;
        return;
    }

    int64_t delta = static_cast<int64_t>(curr_block) - static_cast<int64_t>(e.last_block);
    e.last_block = curr_block;
    if (delta == 0) {
        return;
    }
    if (delta > MAX_TRACKED_STRIDE || delta < -MAX_TRACKED_STRIDE) {
        if (e.confidence > 0) {
            --e.confidence;
        }
        return;
    }

    if (delta == e.last_stride) {
        if (e.confidence < 3) {
            ++e.confidence;
        }
    } else {
        e.last_stride = delta;
        e.confidence = (e.confidence > 0) ? e.confidence - 1 : 1;
    }

    if (e.confidence < 2 || e.last_stride == 0) {
        return;
    }

    size_t steps = (e.confidence == 3) ? STRIDE_MAX_STEPS : 2;
    for (size_t s = 1; s <= steps; ++s) {
        int64_t target = static_cast<int64_t>(curr_block) + e.last_stride * static_cast<int64_t>(s);
        if (target > 0) {
            out.push(static_cast<uint64_t>(target) << BLOCK_OFFSET_BITS);
        }
    }
}

} // namespace

void openpref::prefetcher_initialize()
{
    g.tick = 0;
    g.train_count = 0;
    g.rng = 0x2545F4914F6CDD1DULL;
    g.scs_head = 0;

    for (auto& e : g.train) {
        e = TrainEntry{};
    }
    for (auto& set : g.markov) {
        for (auto& e : set) {
            e = MarkovEntry{};
        }
    }
    for (auto& set : g.hist) {
        for (auto& e : set) {
            e = SamplerEntry{};
        }
    }
    for (auto& e : g.scs) {
        e = ScsEntry{};
    }
    for (auto& e : g.pt) {
        e = PrefetchTrack{};
    }
    for (auto& e : g.stride) {
        e = StrideEntry{};
    }
}

void openpref::access(uint64_t pc, uint64_t addr, bool hit)
{
    uint64_t block = addr >> BLOCK_OFFSET_BITS;
    PfBuf out;

    // Training events: a miss, or the first use of a block we prefetched
    bool tagged_hit = pt_consume(block);
    if (!hit || tagged_hit) {
        triangel_event(pc, block, out);
    }

    stride_event(pc, addr, out);

    for (size_t i = 0; i < out.n; ++i) {
        prefetch(out.a[i]);
    }
}