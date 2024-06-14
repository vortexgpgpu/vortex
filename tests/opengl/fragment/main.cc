#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <GLSC2/glsc2.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>

#include "../debug.cc"
#include "../common.c"

#define KERNEL_NAME "gl_main_fs"
#define WIDTH 700
#define HEIGHT 700

cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_kernel kernel;
cl_program program;
cl_mem m_image, gl_FragCoord, gl_Rasterization, gl_Discard, gl_FragColor;
cl_int _err;
size_t kernel_size;

int main (int argc, char **argv) {

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  PPMImage* image = readPPM("dog.ppm");
  uint8_t* data = (uint8_t*) malloc(image->x*image->y*sizeof(uint8_t[4]));
  for(uint32_t i=0; i<image->x*image->y; ++i) {
    data[i*4+0] = image->data[i].red;
    data[i*4+1] = image->data[i].green;
    data[i*4+2] = image->data[i].blue;
    data[i*4+3] = 0xFFu;
  }
  m_image = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image->x*image->y*sizeof(uint8_t[4]), data, &_err));
  
  float* data_coord = (float*)malloc(WIDTH*HEIGHT*sizeof(float[4]));
  for(int h=0; h<HEIGHT; ++h) {
    for(int w=0; w<WIDTH; ++w) {
      float *coord = data_coord + (h*WIDTH+w)*sizeof(float[4]);
      coord[0] = w;
      coord[1] = h;
    }
  }
  gl_Rasterization = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, WIDTH*HEIGHT*sizeof(float[4]), data_coord, &_err));
  gl_FragColor = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, WIDTH*HEIGHT*sizeof(float[4]), NULL, &_err));
  // unused buffers
  gl_FragCoord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH*HEIGHT*sizeof(float[4]), NULL, &_err));
  gl_Discard = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH*HEIGHT*sizeof(uint8_t), NULL, &_err));

  uint8_t* kernel_bin;
  printf("Create program from kernel source\n");
#ifdef HOSTGPU
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));  
#else
  if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, (const uint8_t**)&kernel_bin, NULL, &_err));
#endif

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  
  // Create kernel
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  // Set kernel arguments
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int[2]), &image));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&m_image));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&gl_FragCoord));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&gl_Rasterization));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&gl_Discard));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&gl_FragColor));

  // Creating command queue
  cl_command_queue commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));  

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {WIDTH*HEIGHT};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  float result[WIDTH][HEIGHT][4];
  CL_CHECK(clEnqueueReadBuffer(commandQueue, gl_FragColor, CL_TRUE, 0, sizeof(result), &result, 0, NULL, NULL));
  // Draw

  return 0; 
}
