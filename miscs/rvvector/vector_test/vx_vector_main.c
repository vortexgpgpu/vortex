#include <vx_intrinsics.h>
#include "vx_vec.h"

int main() {
  vx_tmc(1);

  int n = 32;
  int *a = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
  int *b = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
  int *c = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};

  for(int i = 0; i < n; ++i) {
    a[i] = 1;
    b[i] = 1;
    c[i] = 1;
  }

  vx_vec_test(n, a, b, c);
  for (int i = 0; i < n; ++i) {
    vx_printf("%d", c[i]);
  }

  vx_tmc(0);
}
