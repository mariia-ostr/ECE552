#include "openbranch.h"

// reusing local prediction logic from two bit counter
constexpr int LOCAL_BHT_BITS = 11;
constexpr int LOCAL_BHT_ENTRIES = 2048;
constexpr int LOCAL_BHT_MASK = LOCAL_BHT_ENTRIES - 1;
constexpr int LOCAL_PHT_ENTRIES = 2048;
constexpr int LOCAL_PHT_MASK = LOCAL_PHT_ENTRIES - 1;

// global and choice predictor logic
constexpr int GLOBAL_BITS = 14;
constexpr int GLOBAL_ENTRIES = 16384;
constexpr int GLOBAL_MASK = GLOBAL_ENTRIES - 1;

// HW arrays
uint16_t local_bht[LOCAL_BHT_ENTRIES];
int8_t local_pht[LOCAL_PHT_ENTRIES];
int8_t global_pht[GLOBAL_ENTRIES];
int8_t choice_pht[GLOBAL_ENTRIES];
uint16_t global_history_reg;

void openbranch::initialize_branch_predictor() {
    global_history_reg = 0;
    for (int i = 0; i < LOCAL_BHT_ENTRIES; ++i) local_bht[i] = 0;
    for (int i = 0; i < LOCAL_PHT_ENTRIES; ++i) local_pht[i] = 0;
    for (int i = 0; i < GLOBAL_ENTRIES; ++i) {
        global_pht[i] = 0;
        choice_pht[i] = 1; // weakly prefer local to start
    }
}

bool openbranch::predict(uint64_t ip) {
    // local prediction (reused logic from twolvl code)
    uint16_t bht_index = ip & LOCAL_BHT_MASK;
    uint16_t local_history = local_bht[bht_index];
    int8_t local_state = local_pht[local_history];
    bool local_pred = (local_state > 0);

    // global prediction
    int8_t global_state = global_pht[global_history_reg];
    bool global_pred = (global_state > 0);

    // choice prediction
    int8_t choice_state = choice_pht[global_history_reg];
    
    // if choice state >= 2, use global. Otherwise, use local.
    return (choice_state >= 2) ? global_pred : local_pred;
}

void openbranch::update(uint64_t ip, bool resolveDir) {
    // recalculate indices
    uint16_t bht_index = ip & LOCAL_BHT_MASK;
    uint16_t local_history = local_bht[bht_index];
    
    bool local_pred = (local_pht[local_history] > 0);
    bool global_pred = (global_pht[global_history_reg] > 0);

    // update choice predictor (train it if the predictors disagreed)
    if (local_pred != global_pred) {
        int8_t choice = choice_pht[global_history_reg];
        if (global_pred == resolveDir) {
            if (choice < 3) choice_pht[global_history_reg]++; // move toward Global
        } else {
            if (choice > 0) choice_pht[global_history_reg]--; // move toward Local
        }
    }

    // update local predictor
    int8_t local_state = local_pht[local_history];
    local_state += (resolveDir ? 1 : -1);
    if (local_state < -1) local_state = -1;
    if (local_state > 2) local_state = 2;
    local_pht[local_history] = local_state;
    local_bht[bht_index] = ((local_history << 1) | resolveDir) & LOCAL_PHT_MASK;

    // update global predictor
    int8_t global_state = global_pht[global_history_reg];
    global_state += (resolveDir ? 1 : -1);
    if (global_state < -1) global_state = -1;
    if (global_state > 2) global_state = 2;
    global_pht[global_history_reg] = global_state;

    // update global history register
    global_history_reg = ((global_history_reg << 1) | resolveDir) & GLOBAL_MASK;
}
