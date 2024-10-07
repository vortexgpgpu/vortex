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
#include <fstream>
#include <iostream>
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

static bool almost_equal(float a, float b, int ulp = 4) {
  union fi_t { int i; float f; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return std::abs(fa.i - fb.i) <= ulp;
}

uint8_t *kernel_bin = NULL;

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_device_id device_id, cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[2]) {
  if (kernel_bin)
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

int size = 1024;

static void show_usage() {
  printf("Usage: [-n size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'h':{
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }

  printf("Workload size=%d\n", size);
}

int main(int argc, char **argv) {
  // parse command arguments
  parse_args(argc, argv);

  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_program program;
  cl_mem input_buffer;
  cl_mem output_buffer;
  size_t kernel_size;
  cl_context context;
  cl_command_queue queue;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, &pfn_notify, NULL, &_err));
  queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &_err));

  cl_kernel kernel = 0;
  cl_mem memObjects[2] = {0, 0};

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK_ERR(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  size_t nbytes = sizeof(float) * size;

  printf("create input buffer\n");
  input_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));

  printf("create output buffer\n");
  output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

  memObjects[0] = input_buffer;
  memObjects[1] = output_buffer;

  float factor = ((float)rand() / (float)(RAND_MAX)) * 100.0;

  printf("create kernel\n");
  kernel = CL_CHECK_ERR(clCreateKernel(program, "saxpy", &_err));

  printf("setting up kernel args\n");
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(factor), &factor));

  size_t global_offset[1] = {0};
  size_t global_work_size[1] = {size};
  size_t local_work_size[1] = {1};

  printf("initialize buffers\n");
  std::vector<float> ref_vec(size, 0.0f);
  {
    std::vector<float> dst_vec(size, 0.0f);
    std::vector<float> src_vec(size);

    for (int i = 0; i < size; i++) {
      src_vec[i] = ((float)rand() / (float)(RAND_MAX)) * 100.0;
    }

    CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, nbytes, src_vec.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, nbytes, dst_vec.data(), 0, NULL, NULL));

    size_t num_groups_x = global_work_size[0] / local_work_size[0];
    for (size_t workgroup_id_x = 0; workgroup_id_x < num_groups_x; ++workgroup_id_x) {
      for (size_t local_id_x = 0; local_id_x < local_work_size[0]; ++local_id_x) {
        // Calculate global ID for the work-item
        int global_id_x = global_offset[0] + local_work_size[0] * workgroup_id_x + local_id_x;
        // kernel operation
        int i = global_id_x;
        ref_vec[i] += src_vec[i] * factor;
      }
    }
  }

  printf("enqueue kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, global_offset, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Verify result\n");
  int errors = 0;
  {
    std::vector<float> dst_vec(size);
    CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, nbytes, dst_vec.data(), 0, NULL, NULL));

    for (int i = 0; i < size; ++i) {
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

  Cleanup(device_id, context, queue, program, kernel, memObjects);

  return errors;
}
