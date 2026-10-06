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
    register int e = a + b;
    e += a + b;
    e += a + b;
    a = a + b;
    d = e + d;
  }

  return 0;
}
