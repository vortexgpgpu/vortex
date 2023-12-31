#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <time.h>
#include <unistd.h> 
#include <chrono>
#include <vector>
#include <cmath>
#include "common.h"

#define KERNEL_NAME "fft_radix4"

#define FLOAT_ULP 6

struct float2 {
    float x;
    float y;

    float2(float real = 0.0f, float imag = 0.0f) : x(real), y(imag) {}

    float2 operator+(const float2& other) const {
        return {x + other.x, y + other.y};
    }

    float2 operator-(const float2& other) const {
        return {x - other.x, y - other.y};
    }

    float2 operator*(const float2& other) const {
        return {x * other.x - y * other.y, x * other.y + y * other.x};
    }
};

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

static std::vector<float2> referenceDFT(const std::vector<float2>& input) {
  std::vector<float2> output(input.size());
  for (unsigned int k = 0; k < input.size(); ++k) { // For each output element
    output[k] = {0, 0}; // Initialize to zero
    for (unsigned int n = 0; n < input.size(); ++n) { // For each input element
      float angle = -2 * M_PI * k * n / input.size();
      float2 twiddle = {cos(angle), sin(angle)};
      output[k].x += input[n].x * twiddle.x - input[n].y * twiddle.y;
      output[k].y += input[n].x * twiddle.y + input[n].y * twiddle.x;
    }
  }
  return output;
}

static int verifyOutput(const std::vector<float2>& output, 
                         const std::vector<float2>& reference, 
                         unsigned int N) {
  int errors = 0;
  for (unsigned int i = 0; i < N; ++i) {
    float2 diff = {output[i].x - reference[i].x, output[i].y - reference[i].y};
    float error = sqrt(diff.x * diff.x + diff.y * diff.y);
    if (error > 1e-5) {
      printf("*** error: [%d] expected=(%f,%f), actual=(%f,%f)\n", i, reference[i].x, reference[i].y, output[i].x, output[i].y);
      ++errors;
    }
  }
  return errors;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem i_memobj = NULL;
cl_mem o_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (i_memobj) clReleaseMemObject(i_memobj);
  if (o_memobj) clReleaseMemObject(o_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);  
  if (kernel_bin) free(kernel_bin);
}

int size = 64;

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

int main (int argc, char **argv) {
  // parse command arguments
  parse_args(argc, argv);
  
  cl_platform_id platform_id;
  size_t kernel_size;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  size_t nbytes = size * sizeof(float2);
  i_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  o_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

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
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&i_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&o_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void *)&size));

  // Allocate memories for input arrays and output arrays.
  std::vector<float2> h_i(size);
  std::vector<float2> h_o(size);
	
  // Generate input values
  for (int i = 0; i < size; ++i) {
    h_i[i].x = sin(2 * M_PI * i / size); // Sine wave as an example
    h_i[i].y = 0.0f; // Zero imaginary part
  }

  // Creating command queue
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));  

	printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, i_memobj, CL_TRUE, 0, nbytes, h_i.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {size};
  size_t local_work_size[1] = {LOCAL_SIZE};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, o_memobj, CL_TRUE, 0, nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<float2> reference = referenceDFT(h_i);
  auto errors = verifyOutput(h_o, reference, size);
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);    
  }

  // Clean up		
  cleanup();  

  return errors;
}
