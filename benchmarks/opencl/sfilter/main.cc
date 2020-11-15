/*
 *  Simple OpenCL demo program
 *
 *  Copyright (C) 2009  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  gcc -o cldemo -std=gnu99 -Wall -I/usr/include/nvidia-current cldemo.c
 * -lOpenCL
 *
 */

#include <CL/cl.h>

#include <errno.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NUM_DATA 16

#define CL_CHECK(_expr)                                                        \
  do {                                                                         \
    cl_int _err = _expr;                                                       \
    if (_err == CL_SUCCESS)                                                    \
      break;                                                                   \
    fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
    abort();                                                                   \
  } while (0)

#define CL_CHECK_ERR(_expr)                                                    \
  ({                                                                           \
    cl_int _err = CL_INVALID_VALUE;                                            \
    decltype(_expr) _ret = _expr;                                                \
    if (_err != CL_SUCCESS) {                                                  \
      fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
      abort();                                                                 \
    }                                                                          \
    _ret;                                                                      \
  })

void pfn_notify(const char *errinfo, const void *private_info, size_t cb,
                void *user_data) {
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}

uint8_t *kernel_bin = NULL;

// inlcude pocl float to half conversions
typedef union {
  int32_t i;
  float f;
} FloatConvUnion;
cl_half poclu_float_to_cl_half(float value) {
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  cl_half fraction =
      (u.i >> 12) & 0x007ff;             // fraction with extra bit for rounding
  cl_half exponent = (u.i >> 23) & 0xff; // exponent

  if (exponent < 0x0067) // Return signed zero if zero or value is too small for
                         // denormal half
    return half;

  if (exponent > 0x008e) { // value was NaN or Inf
    half |= 0x7c00u;       // Make into inf
    half |= exponent == 255 &&
            (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if (exponent < 0x0071) { // Denormal
    fraction |= 0x0800u;

    // rounding
    half |= (fraction >> (0x0072 - exponent)) +
            ((fraction >> (0x0071 - exponent)) & 1);
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | (fraction >> 1);
  half += fraction & 1; // rounding
  return half;
}
#ifndef INFINITY
#define INFINITY 1.0 / 0.0
#endif

#ifndef NAN
#define NAN 0.0 / 0.0
#endif

float poclu_cl_half_to_float(cl_half value) {
  if (value == 0xFC00) {
    return -INFINITY;
  }
  if (value == 0x7C00) {
    return INFINITY;
  }

  int sgn = ((value & 0x8000) >> 15);
  int exp = (value & 0x7C00) >> 10;
  int mant = value & 0x03FF;

  if (exp == 0x1F && mant != 0) {
    return NAN;
  }

  float v = (exp == 0) ? mant : mant | 0x0400; // 1.x if not denormal
  v /= 0x400;
  float mul = exp2((float)exp - 15);
  v *= mul;
  if (sgn) {
    v *= -1;
  }
  return v;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3]) {
  for (int i = 0; i < 3; i++) {
    if (memObjects[i] != 0)
      clReleaseMemObject(memObjects[i]);
  }
  if (commandQueue != 0)
    clReleaseCommandQueue(commandQueue);

  if (kernel != 0)
    clReleaseKernel(kernel);

  if (program != 0)
    clReleaseProgram(program);

  if (context != 0)
    clReleaseContext(context);

  if (kernel_bin) free(kernel_bin);
}

int main(int argc, char **argv) {
  printf("enter demo main\n");

  cl_platform_id platform_id;
  cl_device_id device_id;
  size_t kernel_size;
  cl_int binary_status = 0;
  int i;

  // read kernel binary from file  
  if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
    return -1;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  cl_context context;
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, &pfn_notify, NULL, &_err));

  cl_command_queue queue;
  queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &_err));

  cl_kernel kernel = 0;
  cl_mem memObjects[2] = {0, 0};

  // Create OpenCL program - first attempt to load cached binary.
  //  If that is not available, then create the program from source
  //  and store the binary for future use.
  std::cout << "Attempting to create program from binary..." << std::endl;
  cl_program program = CL_CHECK_ERR(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, &binary_status, &_err));
  if (program == NULL) {
    std::cerr << "Failed to write program binary" << std::endl;
    Cleanup(context, queue, program, kernel, memObjects);
    return 1;
  } else {
    std::cout << "Read program from binary." << std::endl;
  }

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  printf("attempting to create input buffer\n");
  fflush(stdout);
  cl_mem input_buffer;
  input_buffer = CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_ONLY,
                     sizeof(float) * NUM_DATA * NUM_DATA, NULL, &_err));

  printf("attempting to create output buffer\n");
  fflush(stdout);
  cl_mem output_buffer;
  output_buffer = CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                     sizeof(float) * NUM_DATA * NUM_DATA, NULL, &_err));

  memObjects[0] = input_buffer;
  memObjects[1] = output_buffer;

  long long ldc = NUM_DATA;

  float m0 = 1.0;
  float m1 = 1.0;
  float m2 = 1.0;
  float m3 = 1.0;
  float m4 = 1.0;
  float m5 = 1.0;
  float m6 = 1.0;
  float m7 = 1.0;
  float m8 = 1.0;

  printf("attempting to create kernel\n");
  fflush(stdout);
  kernel = CL_CHECK_ERR(clCreateKernel(program, "sfilter", &_err));
  printf("setting up kernel args cl_mem:%lx \n", input_buffer);
  fflush(stdout);
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(ldc), (&ldc)));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(m0), (&m0)));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(m1), (&m1)));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(m2), (&m2)));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(m3), (&m3)));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(m4), (&m4)));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(m5), (&m5)));
  CL_CHECK(clSetKernelArg(kernel, 9, sizeof(m6), (&m6)));
  CL_CHECK(clSetKernelArg(kernel, 10, sizeof(m7), (&m7)));
  CL_CHECK(clSetKernelArg(kernel, 11, sizeof(m8), (&m8)));

  printf("attempting to enqueue write buffer\n");
  fflush(stdout);
  for (int i = 0; i < NUM_DATA * NUM_DATA; i++) {
    float in = ((float)rand() / (float)(RAND_MAX)) * 100.0;
    CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE,
                                  i * sizeof(float), 4, &in, 0, NULL, NULL));
  }

  cl_event kernel_completion;
  size_t global_offset[2] = {1, 1};
  size_t global_work_size[2] = {NUM_DATA - 2, NUM_DATA - 2}; // avoid the edges
  const size_t local_work_size[2] = {64, 1};
  printf("attempting to enqueue kernel\n");
  fflush(stdout);
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, global_offset,
                                  global_work_size, local_work_size, 0, NULL,
                                  &kernel_completion));
  printf("Enqueue'd kernel\n");
  fflush(stdout);
  cl_ulong time_start, time_end;
  CL_CHECK(clWaitForEvents(1, &kernel_completion));
  CL_CHECK(clGetEventProfilingInfo(kernel_completion,
                                   CL_PROFILING_COMMAND_START,
                                   sizeof(time_start), &time_start, NULL));
  CL_CHECK(clGetEventProfilingInfo(kernel_completion, CL_PROFILING_COMMAND_END,
                                   sizeof(time_end), &time_end, NULL));
  double elapsed = time_end - time_start;
  printf("time(ns):%lg\n", elapsed);
  CL_CHECK(clReleaseEvent(kernel_completion));

  printf("Result:");
  for (int i = 0; i < NUM_DATA * NUM_DATA; i++) {
    float data;
    CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                                 i * sizeof(float), 4, &data, 0, NULL, NULL));
    // printf(" %f", data);
  }
  printf("\n");
  printf("Passed!\n");
  CL_CHECK(clReleaseMemObject(memObjects[0]));
  CL_CHECK(clReleaseMemObject(memObjects[1]));

  CL_CHECK(clReleaseKernel(kernel));
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseContext(context));

  return 0;
}
