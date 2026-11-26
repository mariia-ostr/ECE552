// Triangel temporal prefetcher (arXiv:2406.10627) extended with ideas adapted
// from LINKEY (Maruszewski, arXiv:2505.21669), plus an accuracy-gated stride engine.
//
// v2 changes (targeting llama2 and sssp):
//  * Stride engine keeps a per-PC "prefetch frontier", so it can run much
//    deeper (up to 12 blocks ahead) while issuing only 1-2 new prefetches per
//    access. This targets long streaming scans (llama2 weight matrices,
//    sssp edge arrays), where degree 4 was too shallow to hide latency.
//  * A confident stride survives a single outlier delta instead of being
//    overwritten (row wraps in llama2, vertex changes in sssp).
//  * Larger outstanding-prefetch tracker for the deeper stride window.
//
// Interface is unchanged: openpref::prefetcher_initialize(),
// openpref::access(pc, addr, hit), and prefetch(addr).

#include "openpref.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {

constexpr uint64_t BLOCK_OFFSET_BITS = 6;

// ---- Training table ----
constexpr size_t TT_ENTRIES = 512;

// ---- Child-set (Markov) table ----
constexpr size_t MK_SETS = 4096;
constexpr size_t MK_WAYS = 16;
constexpr size_t MK_KIDS = 3;
constexpr uint8_t KCONF_MAX = 3;
constexpr uint64_t MAX_SIZE = MK_SETS * MK_WAYS;

// ---- History sampler / second-chance sampler ----
constexpr size_t HS_SETS = 256;
constexpr size_t HS_WAYS = 2;
constexpr uint64_t HS_ENTRIES = HS_SETS * HS_WAYS;
constexpr size_t SCS_ENTRIES = 64;
constexpr uint64_t SCS_WINDOW = 512;

// ---- Confidence counters ----
constexpr int CONF_MAX = 15;
constexpr int CONF_INIT = 8;
constexpr int SAMPLE_RATE_INIT = 8;
constexpr int BASE_UP = 1, BASE_DOWN = 2;
constexpr int HIGH_UP = 1, HIGH_DOWN = 5;

// ---- Issue limits ----
constexpr size_t OUT_BUDGET_MAX = 8;
constexpr size_t OUT_BUDGET_MIN = 2;
constexpr size_t MAX_BFS_DEPTH = 3;

// ---- Accuracy feedback for temporal & stride engines ----
constexpr uint64_t ACC_EPOCH = 256;
constexpr double TEMP_ACC_HIGH = 0.45;
constexpr double TEMP_ACC_LOW = 0.18;
constexpr double STRIDE_ACC_HIGH = 0.50;
constexpr double STRIDE_ACC_LOW = 0.25;

// ---- Outstanding prefetch tracker ----
constexpr size_t PT_ENTRIES = 8192;

enum class PfSource : uint8_t {
    NONE = 0,
    TEMPORAL = 1,
    STRIDE = 2
};

// ---- Stride engine ----
constexpr size_t STRIDE_ENTRIES = 512;
constexpr int64_t MAX_TRACKED_STRIDE = 4096; // blocks (256 KiB)
constexpr size_t STRIDE_DEG_MED = 2;         // conf == 2
constexpr size_t STRIDE_DEG_HIGH = 8;        // conf == 3, upper bound of adaptive cap
constexpr size_t STRIDE_DEG_INIT = 6;
// Keep fully-covered stride streams out of the temporal engine.
constexpr bool STRIDE_FILTERS_TEMPORAL = false;

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
    uint64_t last[2] = {0, 0};
    int n_last = 0;
    uint64_t timestamp = 0;
    uint8_t reuse = CONF_INIT;
    uint8_t base = CONF_INIT;
    uint8_t high = CONF_INIT;
    uint8_t sample_rate = SAMPLE_RATE_INIT;
    bool lookahead = false;
};

struct MarkovEntry {
    bool valid = false;
    bool just_built = false;
    uint16_t tag = 0;
    uint8_t rrpv = 3;
    uint64_t kid[MK_KIDS] = {0, 0, 0};
    uint8_t kconf[MK_KIDS] = {0, 0, 0};
};

struct SamplerEntry {
    bool valid = false;
    uint64_t addr = 0;
    uint64_t target = 0;
    uint32_t train_idx = 0;
    uint64_t ts = 0;
    uint64_t stamp = 0;
    bool used = false;
};

struct ScsEntry {
    bool valid = false;
    uint64_t addr = 0;
    uint32_t train_idx = 0;
    uint16_t pc_tag = 0;
    uint64_t ts = 0;
};

struct PrefetchTrack {
    bool valid = false;
    bool used = false;
    PfSource src = PfSource::NONE;
    uint64_t block = 0;
};

struct StrideEntry {
    bool valid = false;
    uint64_t pc = 0;
    uint64_t last_block = 0;
    int64_t last_stride = 0;
    uint8_t confidence = 0;
    // Furthest block already prefetched for the current stride run.
    bool pf_valid = false;
    int64_t pf_stride = 0;
    int64_t pf_head = 0;
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

struct State {
    uint64_t tick = 0;
    uint64_t train_count = 0;
    uint64_t rng = 0x2545F4914F6CDD1DULL;

    uint64_t temp_issued = 0;
    uint64_t temp_useful = 0;
    size_t temporal_budget = OUT_BUDGET_MAX;

    uint64_t stride_issued = 0;
    uint64_t stride_useful = 0;
    size_t stride_deg_cap = STRIDE_DEG_INIT;

    std::array<TrainEntry, TT_ENTRIES> train{};
    std::array<std::array<MarkovEntry, MK_WAYS>, MK_SETS> markov{};
    std::array<std::array<SamplerEntry, HS_WAYS>, HS_SETS> hist{};
    std::array<ScsEntry, SCS_ENTRIES> scs{};
    size_t scs_head = 0;
    std::array<PrefetchTrack, PT_ENTRIES> pt{};
    std::array<StrideEntry, STRIDE_ENTRIES> stride{};
};

State g;

inline uint64_t next_rand() {
    g.rng ^= g.rng << 13;
    g.rng ^= g.rng >> 7;
    g.rng ^= g.rng << 17;
    return g.rng;
}

inline void pattern_up(TrainEntry& te) {
    conf_up(te.base, BASE_UP);
    conf_up(te.high, HIGH_UP);
}
inline void pattern_down(TrainEntry& te) {
    conf_down(te.base, BASE_DOWN);
    conf_down(te.high, HIGH_DOWN);
}
inline void pattern_down_at(uint32_t idx, uint16_t tag) {
    TrainEntry& te = g.train[idx];
    if (te.valid && te.tag == tag) {
        pattern_down(te);
    }
}

void adapt_temporal_budget() {
    if (g.temp_issued < ACC_EPOCH) {
        return;
    }
    double acc = static_cast<double>(g.temp_useful) / static_cast<double>(g.temp_issued);
    if (acc < TEMP_ACC_LOW) {
        g.temporal_budget = std::max(OUT_BUDGET_MIN, g.temporal_budget - 2);
    } else if (acc > TEMP_ACC_HIGH) {
        g.temporal_budget = std::min(OUT_BUDGET_MAX, g.temporal_budget + 2);
    }
    g.temp_issued = 0;
    g.temp_useful = 0;
}

void adapt_stride_degree() {
    if (g.stride_issued < ACC_EPOCH) {
        return;
    }
    double acc = static_cast<double>(g.stride_useful) / static_cast<double>(g.stride_issued);
    if (acc < STRIDE_ACC_LOW) {
        g.stride_deg_cap = (g.stride_deg_cap >= STRIDE_DEG_MED + 2) ? (g.stride_deg_cap - 2) : STRIDE_DEG_MED;
    } else if (acc > STRIDE_ACC_HIGH) {
        g.stride_deg_cap = std::min<size_t>(STRIDE_DEG_HIGH, g.stride_deg_cap + 2);
    }
    g.stride_issued = 0;
    g.stride_useful = 0;
}

bool pt_issue(uint64_t block, PfSource src) {
    PrefetchTrack& e = g.pt[mix64(block) % PT_ENTRIES];
    if (e.valid && e.block == block && !e.used) {
        return false;
    }
    e.valid = true;
    e.used = false;
    e.src = src;
    e.block = block;
    if (src == PfSource::TEMPORAL) {
        ++g.temp_issued;
    } else if (src == PfSource::STRIDE) {
        ++g.stride_issued;
    }
    return true;
}

PfSource pt_consume(uint64_t block) {
    PrefetchTrack& e = g.pt[mix64(block) % PT_ENTRIES];
    if (e.valid && e.block == block && !e.used) {
        e.used = true;
        PfSource s = e.src;
        if (s == PfSource::TEMPORAL) {
            ++g.temp_useful;
        } else if (s == PfSource::STRIDE) {
            ++g.stride_useful;
        }
        return s;
    }
    return PfSource::NONE;
}

inline void markov_locate(uint64_t key, size_t& set, uint16_t& tag) {
    uint64_t h = mix64(key);
    set = static_cast<size_t>(h % MK_SETS);
    tag = static_cast<uint16_t>((h >> 40) & 0xFFFF);
}

MarkovEntry* markov_find(uint64_t key) {
    size_t set;
    uint16_t tag;
    markov_locate(key, set, tag);
    for (auto& e : g.markov[set]) {
        if (e.valid && e.tag == tag) {
            return &e;
        }
    }
    return nullptr;
}

MarkovEntry* markov_use(uint64_t key) {
    MarkovEntry* e = markov_find(key);
    if (e) {
        e->rrpv = 0;
        e->just_built = false;
    }
    return e;
}

void learn_kid(MarkovEntry& e, uint64_t target) {
    for (size_t i = 0; i < MK_KIDS; ++i) {
        if (e.kconf[i] && e.kid[i] == target) {
            if (e.kconf[i] < KCONF_MAX) {
                ++e.kconf[i];
            }
            return;
        }
    }
    for (size_t i = 0; i < MK_KIDS; ++i) {
        if (!e.kconf[i]) {
            e.kid[i] = target;
            e.kconf[i] = 1;
            return;
        }
    }
    size_t w = 0;
    for (size_t i = 1; i < MK_KIDS; ++i) {
        if (e.kconf[i] < e.kconf[w]) {
            w = i;
        }
    }
    if (--e.kconf[w] == 0) {
        e.kid[w] = target;
        e.kconf[w] = 1;
    }
}

void markov_train(uint64_t key, uint64_t target) {
    if (MarkovEntry* e = markov_find(key)) {
        e->rrpv = 0;
        learn_kid(*e, target);
        return;
    }

    size_t set;
    uint16_t tag;
    markov_locate(key, set, tag);
    auto& ways = g.markov[set];

    MarkovEntry* victim = nullptr;
    for (auto& e : ways) {
        if (!e.valid) {
            victim = &e;
            break;
        }
    }
    for (int round = 0; !victim && round < 4; ++round) {
        for (auto& e : ways) {
            if (e.rrpv >= 3 && !e.just_built) {
                victim = &e;
                break;
            }
        }
        if (!victim) {
            for (auto& e : ways) {
                if (e.rrpv < 3) {
                    ++e.rrpv;
                }
            }
        }
    }
    if (!victim) {
        for (auto& e : ways) {
            e.just_built = false;
        }
        return;
    }

    *victim = MarkovEntry{};
    victim->valid = true;
    victim->just_built = true;
    victim->tag = tag;
    victim->rrpv = 2;
    victim->kid[0] = target;
    victim->kconf[0] = 1;
}

void issue_children(uint64_t cur, size_t depth_max, size_t budget, uint8_t min_conf,
                    PfBuf& out) {
    struct QItem {
        uint64_t key;
        size_t depth;
    };
    QItem q[16];
    size_t qh = 0, qt = 0;
    uint64_t seen[16];
    size_t ns = 0;

    q[qt++] = {cur, 0};
    seen[ns++] = cur;
    size_t issued = 0;

    while (qh < qt && issued < budget) {
        QItem it = q[qh++];
        MarkovEntry* e = markov_use(it.key);
        if (!e) {
            continue;
        }
        uint8_t eff_min = (it.depth == 0) ? min_conf : static_cast<uint8_t>(std::max<int>(2, min_conf));
        for (int c = KCONF_MAX; c >= static_cast<int>(eff_min) && issued < budget; --c) {
            for (size_t i = 0; i < MK_KIDS && issued < budget; ++i) {
                if (e->kconf[i] != c) {
                    continue;
                }
                uint64_t t = e->kid[i];
                if (t == cur) {
                    continue;
                }
                bool dup = false;
                for (size_t s = 0; s < ns; ++s) {
                    if (seen[s] == t) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    continue;
                }
                if (ns < 16) {
                    seen[ns++] = t;
                }
                if (pt_issue(t, PfSource::TEMPORAL)) {
                    out.push(t << BLOCK_OFFSET_BITS);
                    ++issued;
                }
                if (it.depth + 1 < depth_max && qt < 16) {
                    q[qt++] = {t, it.depth + 1};
                }
            }
        }
    }
}

void scs_insert(uint64_t addr, uint32_t idx, uint16_t tag) {
    ScsEntry& slot = g.scs[g.scs_head];
    if (slot.valid) {
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

SamplerEntry* hs_find(uint64_t addr, uint32_t idx) {
    auto& set = g.hist[mix64(addr) % HS_SETS];
    for (auto& e : set) {
        if (e.valid && e.addr == addr && e.train_idx == idx) {
            return &e;
        }
    }
    return nullptr;
}

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
        bool resident = vte.valid;
        bool stale = !resident || (vte.timestamp - victim->ts > MAX_SIZE);
        if (stale) {
            if (!victim->used && resident) {
                conf_down(vte.reuse, 1);
            }
            conf_up(te.sample_rate, 1);
        } else if (!victim->used) {
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

void temporal_event(uint64_t pc, uint64_t cur, PfBuf& out) {
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

    scs_check(idx, te, cur);
    history_sample(idx, te, cur);

    if (te.high >= CONF_MAX) {
        te.lookahead = true;
    } else if (te.lookahead && te.base < CONF_INIT) {
        te.lookahead = false;
    }

    bool confident = te.reuse > CONF_INIT && te.base > CONF_INIT;

    if (confident) {
        int need = te.lookahead ? 2 : 1;
        if (te.n_last >= need) {
            uint64_t index_addr = te.last[need - 1];
            if (index_addr != cur) {
                markov_train(index_addr, cur);
            }
        }

        bool strong = te.high > CONF_INIT;
        size_t depth = strong ? MAX_BFS_DEPTH : 2;
        uint8_t min_conf = strong ? 1 : 2;
        issue_children(cur, depth, g.temporal_budget, min_conf, out);
        adapt_temporal_budget();
    }

    te.last[1] = te.last[0];
    te.last[0] = cur;
    if (te.n_last < 2) {
        ++te.n_last;
    }
}

// Returns true when this access is part of a fully-confident stride stream.
bool stride_event(uint64_t pc, uint64_t addr, PfBuf& out) {
    uint64_t curr_block = addr >> BLOCK_OFFSET_BITS;
    StrideEntry& e = g.stride[mix64(pc) % STRIDE_ENTRIES];

    if (!e.valid || e.pc != pc) {
        e = StrideEntry{};
        e.valid = true;
        e.pc = pc;
        e.last_block = curr_block;
        return false;
    }

    int64_t delta = static_cast<int64_t>(curr_block) - static_cast<int64_t>(e.last_block);
    e.last_block = curr_block;
    if (delta == 0) {
        return false;
    }
    if (delta > MAX_TRACKED_STRIDE || delta < -MAX_TRACKED_STRIDE) {
        if (e.confidence > 0) {
            --e.confidence;
        }
        e.pf_valid = false;
        return false;
    }

    bool matched = (delta == e.last_stride);
    if (matched) {
        if (e.confidence < 3) {
            ++e.confidence;
        }
    } else if (e.confidence > 1) {
        // Established stream hit an outlier (row wrap, new vertex's edge list):
        // keep the stride, just lose confidence.
        --e.confidence;
    } else {
        e.last_stride = delta;
        e.confidence = 1;
        e.pf_valid = false;
    }

    if (e.confidence < 2 || e.last_stride == 0) {
        return false;
    }

    const int64_t stride = e.last_stride;
    const int64_t cur = static_cast<int64_t>(curr_block);
    size_t raw_steps = (e.confidence == 3) ? STRIDE_DEG_HIGH : STRIDE_DEG_MED;
    size_t deg = std::min(raw_steps, g.stride_deg_cap);

    // Resume from the frontier if it is still ahead of us on the same stride.
    int64_t first = 1;
    if (e.pf_valid && e.pf_stride == stride) {
        int64_t ahead = (e.pf_head - cur) / stride;
        if (ahead > 0 && ahead <= static_cast<int64_t>(STRIDE_DEG_HIGH)) {
            first = ahead + 1;
        }
    }

    for (int64_t s = first; s <= static_cast<int64_t>(deg); ++s) {
        int64_t target = cur + stride * s;
        if (target <= 0) {
            break;
        }
        uint64_t target_block = static_cast<uint64_t>(target);
        if (pt_issue(target_block, PfSource::STRIDE)) {
            out.push(target_block << BLOCK_OFFSET_BITS);
        }
        e.pf_valid = true;
        e.pf_stride = stride;
        e.pf_head = target;
    }
    adapt_stride_degree();

    return STRIDE_FILTERS_TEMPORAL && matched && e.confidence == 3;
}

} // namespace

void openpref::prefetcher_initialize()
{
    g.tick = 0;
    g.train_count = 0;
    g.rng = 0x2545F4914F6CDD1DULL;
    g.scs_head = 0;
    g.temp_issued = 0;
    g.temp_useful = 0;
    g.temporal_budget = OUT_BUDGET_MAX;
    g.stride_issued = 0;
    g.stride_useful = 0;
    g.stride_deg_cap = STRIDE_DEG_INIT;

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

    PfSource consumed = pt_consume(block);

    bool covered = false;
    if (!hit || consumed == PfSource::STRIDE) {
        covered = stride_event(pc, addr, out);
    }
    if ((!hit || consumed == PfSource::TEMPORAL) && !covered) {
        temporal_event(pc, block, out);
    }

    for (size_t i = 0; i < out.n; ++i) {
        prefetch(out.a[i]);
    }
}