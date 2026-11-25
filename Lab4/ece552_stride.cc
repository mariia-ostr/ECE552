#include "stride.h"
#include <cstdint>

enum rpt_state_e {
	rps_init = 0,
	rps_transient = 1,
	rps_steady = 2,
	rps_disable = 3,
	rps_count
};

struct referencePredictionEntry {
	uint64_t tag;
	uint64_t prev_addr;
	uint64_t stride;
	rpt_state_e state;
};

typedef referencePredictionEntry RPE;

constexpr uint64_t RPT_ENTRY_COUNT = 16;
constexpr uint64_t RPT_INDEX_MASK = RPT_ENTRY_COUNT - 1;

inline uint64_t pc_index(const uint64_t PC) {
	return PC & RPT_INDEX_MASK;
}

inline uint64_t pc_tag(const uint64_t PC) {
	return PC & (~RPT_INDEX_MASK);
}

referencePredictionEntry RPT[RPT_ENTRY_COUNT];

rpt_state_e RPT_FSM_LUT[rps_count][2];

void rpt_fsm_update(RPE* entry, uint64_t new_stride)
{
	const bool same_stride = entry->stride == new_stride;
	if (!same_stride && !(entry->state == rps_steady)) {
		entry->stride = new_stride;
	}
	entry->state = RPT_FSM_LUT[entry->state][same_stride];
}

void stride::prefetcher_initialize()
{
	// [old_state][same_stride?]
	RPT_FSM_LUT[rps_init][true] = rps_steady;
	RPT_FSM_LUT[rps_init][false] = rps_transient;

	RPT_FSM_LUT[rps_transient][true] = rps_steady;
	RPT_FSM_LUT[rps_transient][false] = rps_disable;

	RPT_FSM_LUT[rps_steady][true] = rps_steady;
	RPT_FSM_LUT[rps_steady][false] = rps_init;

	RPT_FSM_LUT[rps_disable][true] = rps_transient;
	RPT_FSM_LUT[rps_disable][false] = rps_disable;

	for (int i = 0; i < RPT_ENTRY_COUNT; ++i) {
		RPE* e = &RPT[i];
		e->prev_addr = 0;
		e->state = rps_init;
		e->stride = 0;
		e->tag = 0;
	}
}

void stride::access(uint64_t pc, uint64_t addr, bool hit)
{
	RPE* entry = &RPT[pc_index(pc)];
	const uint64_t tag = pc_tag(pc);

	if (entry->tag != tag) {

		entry->stride = 0;
		entry->state = rps_init;
	} else {
		
		const uint64_t stride = addr - entry->prev_addr;
		rpt_fsm_update(entry, stride);

		if (!(entry->state == rps_disable)) {
			prefetch(addr + stride);
		}
	}
	entry->prev_addr = addr;
}

