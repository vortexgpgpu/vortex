#include "tests.h"
#include <vx_print.h>

int main()
{
  int errors = 0;
  errors += test_thread_kernel();
  if (0 == errors) {
    vx_printf("Passed!\n");
  } else {
    vx_printf("Failed!\n");
  }

  return errors;
}
