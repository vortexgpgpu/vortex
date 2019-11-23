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

//#define NUM_DATA 65536
#define NUM_DATA 4096

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
    typeof(_expr) _ret = _expr;                                                \
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
}

int main(int argc, char **argv) {
  printf("enter demo main\n");
  
  cl_platform_id platform_id;
  cl_device_id device_id;
  size_t binary_size;
  int i;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  cl_context context;
  context = CL_CHECK_ERR(
      clCreateContext(NULL, 1, &device_id, &pfn_notify, NULL, &_err));

  cl_command_queue queue;
  queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_id,
                                            CL_QUEUE_PROFILING_ENABLE, &_err));

  cl_kernel kernel = 0;
  cl_mem memObjects[2] = {0, 0};

  // Create OpenCL program - first attempt to load cached binary.
  //  If that is not available, then create the program from source
  //  and store the binary for future use.
  std::cout << "Attempting to create program from binary..." << std::endl;
  cl_program program =
      clCreateProgramWithBuiltInKernels(context, 1, &device_id, "saxpy", NULL);
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
  input_buffer = CL_CHECK_ERR(clCreateBuffer(
      context, CL_MEM_READ_ONLY, sizeof(float) * NUM_DATA, NULL, &_err));

  printf("attempting to create output buffer\n");
  fflush(stdout);
  cl_mem output_buffer;
  output_buffer = CL_CHECK_ERR(clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, sizeof(float) * NUM_DATA, NULL, &_err));

  memObjects[0] = input_buffer;
  memObjects[1] = output_buffer;

  float factor = ((float)rand() / (float)(RAND_MAX)) * 100.0;

  printf("attempting to create kernel\n");
  fflush(stdout);
  kernel = CL_CHECK_ERR(clCreateKernel(program, "saxpy", &_err));
  printf("setting up kernel args cl_mem:%lx \n", input_buffer);
  fflush(stdout);
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(factor), &factor));

  printf("attempting to enqueue write buffer\n");
  fflush(stdout);
  for (int i = 0; i < NUM_DATA; i++) {
    float in = ((float)rand() / (float)(RAND_MAX)) * 100.0;
    CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE,
                                  i * sizeof(float), 4, &in, 0, NULL, NULL));
  }

  cl_event kernel_completion;
  size_t global_work_size[1] = {NUM_DATA};
  printf("attempting to enqueue kernel\n");
  fflush(stdout);
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                  NULL, 0, NULL, &kernel_completion));
  printf("Enqueue'd kerenel\n");
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
  for (int i = 0; i < NUM_DATA; i++) {
    float data;
    CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE,
                                 i * sizeof(float), 4, &data, 0, NULL, NULL));
    // printf(" %f", data);
  }
  printf("\n");

  CL_CHECK(clReleaseMemObject(memObjects[0]));
  CL_CHECK(clReleaseMemObject(memObjects[1]));

  CL_CHECK(clReleaseKernel(kernel));
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseContext(context));

  return 0;
}
