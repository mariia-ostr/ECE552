/*
 * mbq2.c - Microbenchmark to verify the Stride Prefetcher (Question 2)
 *
 * Structure/logic:
 *  * - Sequentially traverse an array larger than the L1D cache, touching
 *   4096 distinct lines (1 MB total span, much larger than the 48 KB L1D cache).
 * - A small delay loop runs between accesses so prefetched lines arrive in
 *   L1D before they are demanded.
 * - Expected Result (Baseline / nopred): Almost every array load misses in L1D.
 * - Expected Result (Next-Line): Prefetches line (i + 1) when the loop jumps
 *   to line (i + 4), so L1D load misses remain high and prefetches are useless.
 * - Expected Result (Stride): The RPT learns the constant 256-byte stride for
 *   the load PC (init -> transient -> steady) and prefetches line (i + 4)
 *   ahead of time, sharply reducing L1D load misses with high useful prefetches.
 */

#include <stdio.h>
#include <stdint.h>

#define BLOCK_SIZE    64
#define STRIDE_LINES  4 // 4 cache lines = 256 bytes
#define NUM_ACCESSES  4096 // 4096 * 256B = 1 MB (> 48 KB L1D)
#define INTS_PER_LINE (BLOCK_SIZE / sizeof(uint32_t)) // 16 uint32_t per 64B line
#define STRIDE_INTS   (STRIDE_LINES * INTS_PER_LINE) // 64 uint32_t per step

static volatile uint32_t array[NUM_ACCESSES * STRIDE_INTS] __attribute__((aligned(64)));

int main(void) {
    volatile uint64_t sum = 0;

    // Constant 256-byte (4-line) stride traversal
    for (int i = 0; i < NUM_ACCESSES; i++) {
        sum += array[i * STRIDE_INTS];

        // Delay so the stride prefetch finishes before the next demand load
        for (volatile int d = 0; d < 20; d++) {
            // busy wait
        }
    }

    printf("Done: %lu\n", (unsigned long)sum);
    return 0;
}
