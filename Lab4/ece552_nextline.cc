#include "nextline.h"

void nextline::prefetcher_initialize()
{
}

void nextline::access(uint64_t pc, uint64_t addr, bool hit)
{
	prefetch(addr + BLOCK_SIZE);
}

