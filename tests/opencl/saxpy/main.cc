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
  while ((c = getopt(argc, argv, "n:h?")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'h':
    case '?': {
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
  size_t kernel_size;
  cl_int binary_status = 0;

  // read kernel binary from file  
  if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
    return -1;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  cl_context context;
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, &pfn_notify, NULL, &_err));

  cl_command_queue queue;
  queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &_err));

  cl_kernel kernel = 0;
  cl_mem memObjects[2] = {0, 0};

  // Create OpenCL program - first attempt to load cached binary.
  //  If that is not available, then create the program from source
  //  and store the binary for future use.
  std::cout << "Attempting to create program from binary..." << std::endl;
  cl_program program = CL_CHECK_ERR(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &_err));
  if (program == NULL) {
    std::cerr << "Failed to write program binary" << std::endl;
    Cleanup(device_id, context, queue, program, kernel, memObjects);
    return 1;
  } else {
    std::cout << "Read program from binary." << std::endl;
  }

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  size_t nbytes = sizeof(float) * size;

  printf("attempting to create input buffer\n");
  cl_mem input_buffer;
  input_buffer = CL_CHECK_ERR(clCreateBuffer(
      context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));

  printf("attempting to create output buffer\n");
  cl_mem output_buffer;
  output_buffer = CL_CHECK_ERR(clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

  memObjects[0] = input_buffer;
  memObjects[1] = output_buffer;

  float factor = ((float)rand() / (float)(RAND_MAX)) * 100.0;

  printf("attempting to create kernel\n");
  kernel = CL_CHECK_ERR(clCreateKernel(program, "saxpy", &_err));

  printf("setting up kernel args\n");
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(factor), &factor));

  printf("attempting to enqueue write buffer\n");
  float* h_src = (float*)malloc(nbytes);
  for (int i = 0; i < size; i++) {
    h_src[i] = ((float)rand() / (float)(RAND_MAX)) * 100.0;
  }
  CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, nbytes, h_src, 0, NULL, NULL));
  free(h_src);

  size_t global_work_size[] = {size/2, size/2};
  printf("attempting to enqueue kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                  NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  float* h_dst = (float*)malloc(nbytes);
  CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, nbytes, h_dst, 0, NULL, NULL));

  /*printf("Result:");
  for (int i = 0; i < size; i++) {
    float data = h_dst[i];
    printf(" %f", data);
  }*/
  free(h_dst);

  Cleanup(device_id, context, queue, program, kernel, memObjects);

  return 0;
}
