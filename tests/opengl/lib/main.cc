#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>
#include "GLSC2/binary.c"
#include "test/test.h"


static bool almost_equal(float a, float b, int ulp = 4) {
  union fi_t { int i; float f; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return std::abs(fa.i - fb.i) <= ulp;
}


// TESTS
int test_color_kernel();
int test_color_kernel_discard_true();

cl_device_id device_id = NULL;
cl_context context = NULL;

int main (int argc, char **argv) {
  
  cl_platform_id platform_id;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  uint32_t errors = 0;
  TEST(test_perspective_div);
  TEST(test_color_kernel);
  TEST(test_color_kernel_discard_true);
  // CLEANUP
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  printf("Total errors %d.\n", errors);
  return errors;
}