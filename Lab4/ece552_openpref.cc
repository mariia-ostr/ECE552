#include "openpref.h"

#include <stdlib.h>
#include <stdio.h>

FILE* f;

void openpref::prefetcher_initialize()
{
	f = fopen("mtrace", "w");
	if (f == NULL) {
		perror("ERROR OPENING FILE!");
	}
}

void openpref::access(uint64_t pc, uint64_t addr, bool hit)
{
	prefetch(addr);
	fprintf(f, "%lu, %lu,%d\n", pc, addr, hit);
}

