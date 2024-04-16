#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>
#include "GLSC2/binary.c"

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })


#define TEST(_func) 
  ({
    printf("Running %s.\n",#_func);
    uint32_t result = _func();
    if (!result) printf("PASSED.\n");
    else printf("FAILED with %d errors.\n", result);
    errors += result;
  })


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
  size_t kernel_size;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  uint32_t errors = 0;
  TEST(test_color_kernel);
  TEST(test_color_kernel_discard_true);

  // CLEANUP
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  printf("Total errors %d.\n", errors);
  return errors;
}

int test_color_kernel() {
  const char KERNEL_NAME[] = "gl_rgba4";

  cl_program program = NULL;
  cl_kernel kernel = NULL;

  unsigned int width = 600, height = 400;
  cl_mem colorBuffer, fragCoord, discard, fragColor;

  size_t kernel_size = sizeof(GLSC2_kernel_color_pocl);

  uint16_t color_init[width*height];
  uint16_t color_out[width*height];
  uint8_t discard_init[width*height];
  float fragColor_init[width*height][4];

  for (uint32_t i=0; i<width*height; ++i) {
    color_init[i] = 0x0000;
    discard_init[i] = 0x00;
    color_init[i][0] = 1.f;
    color_init[i][1] = 1.f;
    color_init[i][2] = 1.f;
    color_init[i][3] = 1.f;
  }

  colorBuffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*2, color_init, &_err));
  fragCoord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[4])*width*height, NULL, &_err));
  discard = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t)*width*height, discard_init, &_err));
  fragColor = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[4])*width*height, NULL, &_err));

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, (const uint8_t**)&GLSC2_kernel_color_pocl, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(width), &width));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(height), &height));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(colorBuffer), &colorBuffer));	
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(fragCoord), &fragCoord));	
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(discard), &discard));	
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(fragColor), &fragColor));	

  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = width*height;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, colorBuffer, CL_TRUE, 0, sizeof(uint16_t)*width*height, color_out, 0, NULL, NULL));

  int errors = 0;
  for (int i = 0; i < width*height; ++i) {
    unsigned short ref = 0xFFFF;
    if (color_out[i] != ref) {
      if (errors < 1) 
        printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, rgba8[i]);
      ++errors;
    }
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (colorBuffer) clReleaseMemObject(colorBuffer);
  if (fragCoord) clReleaseMemObject(fragCoord);
  if (discard) clReleaseMemObject(discard);  
  if (fragColor) clReleaseMemObject(fragColor);  
  
  return errors;
}

int test_color_kernel_discard_true() {
  const char KERNEL_NAME[] = "gl_rgba4";

  cl_program program = NULL;
  cl_kernel kernel = NULL;

  unsigned int width = 600, height = 400;
  cl_mem colorBuffer, fragCoord, discard, fragColor;

  size_t kernel_size = sizeof(GLSC2_kernel_color_pocl);

  uint16_t color_init[width*height];
  uint16_t color_out[width*height];
  uint8_t discard_init[width*height];
  float fragColor_init[width*height][4];

  for (uint32_t i=0; i<width*height; ++i) {
    color_init[i] = 0x0000;
    discard_init[i] = 0x01;
    color_init[i][0] = 1.f;
    color_init[i][1] = 1.f;
    color_init[i][2] = 1.f;
    color_init[i][3] = 1.f;
  }

  colorBuffer = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*2, color_init, &_err));
  fragCoord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[4])*width*height, NULL, &_err));
  discard = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t)*width*height, discard_init, &_err));
  fragColor = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[4])*width*height, NULL, &_err));

  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, (const uint8_t**)&GLSC2_kernel_color_pocl, NULL, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(width), &width));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(height), &height));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(colorBuffer), &colorBuffer));	
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(fragCoord), &fragCoord));	
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(discard), &discard));	
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(fragColor), &fragColor));	

  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
  size_t global_work_size = width*height;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, colorBuffer, CL_TRUE, 0, sizeof(uint16_t)*width*height, color_out, 0, NULL, NULL));

  int errors = 0;
  for (int i = 0; i < width*height; ++i) {
    unsigned short ref = 0x0000;
    if (color_out[i] != ref) {
      if (errors < 1) 
        printf("*** error: [%d] expected=%08x, actual=%08x\n", i, ref, rgba8[i]);
      ++errors;
    }
  }

  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (colorBuffer) clReleaseMemObject(colorBuffer);
  if (fragCoord) clReleaseMemObject(fragCoord);
  if (discard) clReleaseMemObject(discard);  
  if (fragColor) clReleaseMemObject(fragColor);

  return errors;
}