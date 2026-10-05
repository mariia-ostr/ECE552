#include <stdlib.h>
#include <stdio.h>

// __asm__ __volatile__ ("nop");

int main(int argc, char **argv)
{
  const int iterations = atoi(argv[1]);

  register int a = 1;
  register int b = 1;

  for (int i = 0; i < iterations; ++i) {
    const register int c = a + b;
    a = b - c;
    b = a + c;
  }

  return 0;
}
