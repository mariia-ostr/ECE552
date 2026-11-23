#include "openpref.h"

void openpref::prefetcher_initialize() 
{
}

void openpref::access(uint64_t pc, uint64_t addr, bool hit)
{
	prefetch(addr);
}

