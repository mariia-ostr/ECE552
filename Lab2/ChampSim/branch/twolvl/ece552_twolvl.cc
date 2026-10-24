#include "twolvl.h"

#define BHT_LEN 512
#define HISTORY_BITS 6
#define BHT_BITS 9

#define PHT_BITS 4
#define PHT_LEN 16

constexpr uint8_t PHT_ENTRIES = (1 << HISTORY_BITS);

constexpr uint8_t PHT_MASK = (1 << PHT_BITS) - 1;
constexpr uint8_t BHT_MASK = (1 << BHT_BITS) - 1;

constexpr uint8_t HISTORY_MASK = (1 << HISTORY_BITS) - 1;

uint8_t branch_history_table[BHT_LEN];
uint8_t prediction_history_table[PHT_LEN][PHT_ENTRIES];

void twolvl::initialize_branch_predictor() {
    for (int i = 0; i < BHT_LEN; ++i) {
        branch_history_table[i] = 0;
    }

    for (int i = 0; i < PHT_LEN; ++i) {
        for (int j = 0; j < PHT_ENTRIES; ++j) {
            prediction_history_table[i][j] = 0;
        }
    }
}

bool twolvl::predict(uint64_t ip) {
    const uint8_t pattern_index = ip & PHT_MASK;
    ip >>= PHT_BITS;
    const uint8_t bht_index = ip & BHT_MASK;

    const uint8_t history = branch_history_table[bht_index];
    const int8_t state = prediction_history_table[pattern_index][history];

    if (state > 0) {
        return true;
    }
    
    return false;
}

void twolvl::update(uint64_t ip, bool resolveDir) {
    const uint8_t pattern_index = ip & PHT_MASK;
    ip >>= PHT_BITS;
    const uint8_t bht_index = ip & BHT_MASK;
    
    const uint8_t history = branch_history_table[bht_index];
    const uint8_t new_history = (history << 1 | resolveDir) & HISTORY_MASK;
    branch_history_table[bht_index] = new_history;
    
    const int8_t old_state = prediction_history_table[pattern_index][new_history];

    int8_t new_state = old_state + (resolveDir ? 1 : -1);
    if (new_state < -1) {
        new_state = -1;
    }
    if (new_state > 2) {
        new_state = 2;
    }

    prediction_history_table[pattern_index][history] = new_state;
}


