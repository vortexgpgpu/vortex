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
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NUM_DATA 64

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
	 cleanup();			                                                     \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
	   cleanup();			                                                   \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
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

cl_device_id device_id = NULL;
uint8_t *kernel_bin = NULL;
cl_context context = 0;
cl_kernel kernel = 0;
cl_command_queue queue = 0;
cl_program program = 0;
cl_mem memObjects[3] = {0, 0, 0};

///
//  Cleanup any created OpenCL resources
//
void cleanup() {
  for (int i = 0; i < 3; i++) {
    if (memObjects[i]) clReleaseMemObject(memObjects[i]);
  }
  if (queue) clReleaseCommandQueue(queue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id); 
  
  if (kernel_bin) free(kernel_bin);
}

int main(int argc, char **argv) {
  printf("enter demo main\n");
  
  cl_platform_id platform_id;  
  size_t kernel_size;
  cl_int binary_status = 0;
  int i;

  // read kernel binary from file  
  if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
    return -1;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, &pfn_notify, NULL, &_err));
  
  queue = CL_CHECK2(clCreateCommandQueue(context, device_id,
                                            CL_QUEUE_PROFILING_ENABLE, &_err));  

  // Create OpenCL program - first attempt to load cached binary.
  //  If that is not available, then create the program from source
  //  and store the binary for future use.
  std::cout << "Attempting to create program from binary..." << std::endl;
  // cl_program program = CreateProgramFromBinary(context, device_id,
  // "kernel.cl.bin");
  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, &binary_status, &_err));
  if (program == NULL) {
    printf("clCreateProgramWithBinary() failed\n");
    cleanup();
    return -1;
  } 

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  printf("attempting to create input buffer\n");
  fflush(stdout);
  cl_mem input_bufferA;
  input_bufferA = CL_CHECK2(
      clCreateBuffer(context, CL_MEM_READ_ONLY,
                     sizeof(float) * NUM_DATA * NUM_DATA, NULL, &_err));

  cl_mem input_bufferB;
  input_bufferB = CL_CHECK2(
      clCreateBuffer(context, CL_MEM_READ_ONLY,
                     sizeof(float) * NUM_DATA * NUM_DATA, NULL, &_err));

  printf("attempting to create output buffer\n");
  fflush(stdout);
  cl_mem output_buffer;
  output_buffer = CL_CHECK2(
      clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                     sizeof(float) * NUM_DATA * NUM_DATA, NULL, &_err));

  memObjects[0] = input_bufferA;
  memObjects[1] = input_bufferB;
  memObjects[2] = output_buffer;

  int width = NUM_DATA;

  printf("attempting to create kernel\n");
  fflush(stdout);
  kernel = CL_CHECK2(clCreateKernel(program, "sgemm", &_err));
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_bufferA), &input_bufferA));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(input_bufferB), &input_bufferB));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(width), &width));
  
  printf("attempting to enqueue write buffer\n");
  fflush(stdout);
  for (int i = 0; i < NUM_DATA * NUM_DATA; i++) {

    float in = ((float)rand() / (float)(RAND_MAX)) * 100.0;
    CL_CHECK(clEnqueueWriteBuffer(queue, input_bufferA, CL_TRUE,
                                  i * sizeof(float), 4, &in, 0, NULL, NULL));
    in = ((float)rand() / (float)(RAND_MAX)) * 100.0;
    CL_CHECK(clEnqueueWriteBuffer(queue, input_bufferB, CL_TRUE,
                                  i * sizeof(float), 4, &in, 0, NULL, NULL));
  }

  printf("Done enqueueing\n");

  cl_event kernel_completion;
  const size_t local_work_size[3] = {1, 1, 1};
  //                             a_offset
  size_t global_work_size[3] = {NUM_DATA, NUM_DATA, NUM_DATA};
  printf("attempting to enqueue kernel\n");
  fflush(stdout);
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size,
                                  local_work_size, 0, NULL,
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
  // Clean up		
  cleanup();  

  return 0;
}
