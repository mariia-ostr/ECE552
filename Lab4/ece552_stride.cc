#include "stride.h"

void stride::prefetcher_initialize() 
{
}

void stride::access(uint64_t pc, uint64_t addr, bool hit)
{
	prefetch(addr);
}

