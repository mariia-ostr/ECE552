/*
 * mbq1.c - microbenchmark to verify the Next-Line Prefetcher (Question 1)
 *
 * Structure/logic:
 * - Sequentially traverse an array larger than the L1D cache,
 *   touching one element per cache line (BLOCK_SIZE = 64 bytes).
 * - A small delay loop is between accesses to give the prefetched
 *   cache line enough time to get to L1D before it's demanded.
 * - Expected Result (Baseline / nopred): Almost every array access is an L1D
 *   miss (~100% miss rate on the traversal) because each line is touched once.
 * - Expected Result (Next-Line Prefetcher): Accessing line i prefetches line
 *   i+1. Because of the delay loop, line i+1 is already in L1D when accessed,
 *   dropping the L1D miss rate dramatically (approaching ~0% on the traversal).
 */

#include <stdio.h>
#include <stdint.h>

#define BLOCK_SIZE 64
#define NUM_LINES  8192                 // 8192 * 64B = 512 KB (larger than typical L1D)
#define INTS_PER_LINE (BLOCK_SIZE / sizeof(uint32_t))

// Align to 64-byte cache line boundaries, to start on an address divisible by 64
static uint32_t array[NUM_LINES * INTS_PER_LINE] __attribute__((aligned(64)));

int main(void) {
    volatile uint64_t sum = 0;

    // Sequential traversal: touch 1 element per cache line
    for (int i = 0; i < NUM_LINES; i++) {
        sum += array[i * INTS_PER_LINE];

        // Small delay so the prefetch for (i + 1) finishes before it's needed
        // volatile means the compiler can't just skip this empty loop
        for (volatile int d = 0; d < 20; d++) {
            // busy wait
        }
    }

    printf("Done: %lu\n", (unsigned long)sum);
    return 0;
}
