#include "twobitsat.h"

#define PREDICTOR_BITS 8192
#define BRANCH_BITS 2
constexpr int TABLE_ENTRY_COUNT = PREDICTOR_BITS / BRANCH_BITS;
constexpr int INDEX_BITS = 12;
constexpr unsigned int INDEX_MASK = (1 << (INDEX_BITS)) - 1;

enum branch_state {
    BRANCH_STATE_NT = -1, 
    BRANCH_STATE_nt = 0, 
    BRANCH_STATE_t  = 1, 
    BRANCH_STATE_T  = 2
};

branch_state table[TABLE_ENTRY_COUNT];

void twobitsat::initialize_branch_predictor() {
    for (int i = 0; i < TABLE_ENTRY_COUNT; ++i) {
        table[i] = BRANCH_STATE_nt;
    }
}

bool twobitsat::predict(uint64_t ip) {
    const branch_state state = table[ip & INDEX_MASK];
    if (state > 0) {
        return true;
    } else {
        return false;
    }
}

void twobitsat::update(uint64_t ip, bool resolveDir) {
    const uint64_t index = ip & INDEX_MASK;
    const branch_state old_state = table[index];

    branch_state new_state = static_cast<branch_state>(old_state + (resolveDir ? 1 : -1));

    if (new_state < BRANCH_STATE_NT) {
        new_state = BRANCH_STATE_NT;
    }
    if (new_state > BRANCH_STATE_T) {
        new_state = BRANCH_STATE_T;
    }

    table[index] = new_state; 
}


