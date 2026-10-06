#include <stdlib.h>
#include <stdio.h>

// __asm__ __volatile__ ("nop");

int main(int argc, char **argv)
{
  const int iterations = atoi(argv[1]);

  register int a = 1;
  register int b = 1;
  int d = 10;

  for (int i = 0; i < iterations; ++i) {
    register int e = a + b; // Write E
    e += a + b; // Use E -> Double cycle stall
    e += a + b; // Use E -> Double cycle stall
    a = a + b;  // Write A
    d = e + d; // E stalls one cycle
    // Two double cycle stalls, One single cycle stall
  }

  return 0;
}
