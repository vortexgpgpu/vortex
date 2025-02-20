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
#include <chrono>
#include <vector>

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
    decltype(_expr) _ret = _expr;                                              \
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

static bool almost_equal(float a, float b, int ulp = 4) {
  union fi_t { int i; float f; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return std::abs(fa.i - fb.i) <= ulp;
}

void Cleanup(uint8_t *kernel_bin, cl_device_id device_id, cl_context context,
             cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[2]) {
  if (kernel_bin != NULL)
    free(kernel_bin);

  if (commandQueue != 0)
    clReleaseCommandQueue(commandQueue);

  for (int i = 0; i < 2; i++) {
    if (memObjects[i] != 0)
      clReleaseMemObject(memObjects[i]);
  }

  if (kernel != 0)
    clReleaseKernel(kernel);

  if (program != 0)
    clReleaseProgram(program);

  if (context != 0)
    clReleaseContext(context);

  if (device_id != 0)
    clReleaseDevice(device_id);
}

static void show_usage() {
  printf("Usage: [-n size] [-h: help]\n");
}

int size = 16;

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

int main(int argc, char **argv) {
  // parse command arguments
  parse_args(argc, argv);

  printf("input size=%d\n", size);
  if (size < 3) {
    printf("Error: input size must be >= 3\n");
    return -1;
  }

  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_program program;
  size_t kernel_size;
  uint8_t *kernel_bin = NULL;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  cl_context context;
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, &pfn_notify, NULL, &_err));

  cl_command_queue queue;
  queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &_err));

  cl_kernel kernel = 0;
  cl_mem memObjects[2] = {0, 0};

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK_ERR(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  size_t nbytes = sizeof(float) * size * size;

  printf("create input buffer\n");
  cl_mem input_buffer;
  input_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));

  printf("create output buffer\n");
  cl_mem output_buffer;
  output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

  memObjects[0] = input_buffer;
  memObjects[1] = output_buffer;

  long long ldc = size;
  float m0 = 1.0;
  float m1 = 1.0;
  float m2 = 1.0;
  float m3 = 1.0;
  float m4 = 1.0;
  float m5 = 1.0;
  float m6 = 1.0;
  float m7 = 1.0;
  float m8 = 1.0;

  printf("create kernel\n");
  kernel = CL_CHECK_ERR(clCreateKernel(program, "sfilter", &_err));

  printf("setting up kernel args\n");
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

  size_t global_offset[2]    = {1, 1};
  size_t global_work_size[2] = {size - 2, size - 2};
  size_t local_work_size[2]  = {1, 1}; // {size-2,1}

  printf("enqueue write buffer\n");
  std::vector<float> ref_vec(size * size);
  {
    std::vector<float> src_vec(size * size);
    std::vector<float> dst_vec(size * size, 0.0f);

    for (int i = 0; i < size * size; ++i) {
      src_vec[i] = ((float)rand() / (float)(RAND_MAX)) * 100.0;
    }

    CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, nbytes, src_vec.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, nbytes, dst_vec.data(), 0, NULL, NULL));

    // reference generation
    size_t num_groups_y = global_work_size[1] / local_work_size[1];
    size_t num_groups_x = global_work_size[0] / local_work_size[0];
    for (size_t workgroup_id_y = 0; workgroup_id_y < num_groups_y; ++workgroup_id_y) {
      for (size_t workgroup_id_x = 0; workgroup_id_x < num_groups_x; ++workgroup_id_x) {
        for (size_t local_id_y = 0; local_id_y < local_work_size[1]; ++local_id_y) {
          for (size_t local_id_x = 0; local_id_x < local_work_size[0]; ++local_id_x) {
            // calculate global ID for the work-item
            int global_id_x = global_offset[0] + local_work_size[0] * workgroup_id_x + local_id_x;
            int global_id_y = global_offset[1] + local_work_size[1] * workgroup_id_y + local_id_y;
            // kernel operation
            int x = global_id_x;
            int y = global_id_y;
            float i0 = src_vec.at((x-1) + (y-1) * ldc) * m0;
            float i1 = src_vec.at((x+0) + (y-1) * ldc) * m1;
            float i2 = src_vec.at((x+1) + (y-1) * ldc) * m2;
            float i3 = src_vec.at((x-1) + (y+0) * ldc) * m3;
            float i4 = src_vec.at((x+0) + (y+0) * ldc) * m4;
            float i5 = src_vec.at((x+1) + (y+0) * ldc) * m5;
            float i6 = src_vec.at((x-1) + (y+1) * ldc) * m6;
            float i7 = src_vec.at((x+0) + (y+1) * ldc) * m7;
            float i8 = src_vec.at((x+1) + (y+1) * ldc) * m8;
            float v = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
            //printf("*** x=%d, y=%d, v=%f\n", x, y, v);
            ref_vec.at(x + y * ldc) = v;
          }
        }
      }
    }
  }

  printf("enqueue kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, global_offset, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Verify result\n");
  int errors = 0;
  {
    std::vector<float> dst_vec(size * size);
    CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, nbytes, dst_vec.data(), 0, NULL, NULL));

    for (int i = 0; i < size * size; ++i) {
      if (!almost_equal(dst_vec[i], ref_vec[i])) {
        if (errors < 100)
          printf("*** error: [%d] expected=%f, actual=%f\n", i, ref_vec[i], dst_vec[i]);
        ++errors;
      }
    }

    if (0 == errors) {
      printf("PASSED!\n");
    } else {
      printf("FAILED! - %d errors\n", errors);
    }
  }

  Cleanup(kernel_bin, device_id, context, queue, program, kernel, memObjects);

  return errors;
}
