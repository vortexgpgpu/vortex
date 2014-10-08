#include "include/harpfloat.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace Harp;
using namespace std;

static void try_val(double d, Size sz) {
  Float f(d, sz);
  Word w(f);
  double e(Float(w, sz));
  printf("%uB: %g -> 0x%llx -> %g\n", sz, d, (unsigned long long)w, e);
}

int main() {
  /* Loop over available integer sizes. */
  unsigned randseed(time(NULL));
  for (unsigned sz = 1; sz <= 8; sz++) {
    srand(randseed);

    /* First test some random ordinary numbers and their conversions. */
    for (unsigned i = 0; i < 2; i++) {
      int n = rand() - RAND_MAX/2;
      double d = n * 0.0000001;

      // Sometimes do negative numbers.
      if (rand() & 1) d = -d;

      try_val(d, sz);
    }

    /* Next, let's try +/- infinity. */
    for (unsigned i = 0; i < 2; i++) {
      double d(i?HUGE_VAL:-HUGE_VAL);
      try_val(d, sz);
    }

    /* Last, let's try some random subnormal numbers and their conversions. */
    double mote;
    if (sz < 4) mote = pow(2, int(-14 - (sz*8 - 6)));
    else if (sz < 8) mote = pow(2, int(-126 - (sz*8 - 9)));
    else mote = pow(2, int(-1022 - (sz*8 - 12)));
    for (unsigned i = 0; i < 2; i++) {
      int n = rand()%256;
      double d = n * mote;
      try_val(d, sz);
    }
  }

  return 0;
}
