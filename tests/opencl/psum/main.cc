#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#define FLOAT_ULP 6

#define KERNEL_NAME "parallelSum"

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

static bool compare_equal(float a, float b) {
  union fi_t { float f; int32_t i; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  auto d = std::abs(fa.i - fb.i);
  return d <= FLOAT_ULP;
}

static float computeParallelSumCPU(float *A, int N) {
  float sum = 0;
  for (int i = 0; i < N; ++i) {
      sum += A[i];
  }
  return sum;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem c_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

size_t size = 16;
size_t local_size = 8;

static void show_usage() {
  printf("Usage: [-n size] [-l local size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:l:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'l':
      local_size = atoi(optarg);
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

int main (int argc, char **argv) {
  // parse command arguments
  parse_args(argc, argv);

  printf("input size=%ld, local size=%ld\n", size, local_size);
  if ((size / local_size) * local_size != size) {
    printf("Error: input size must be a multiple of %ld\n", local_size);
    return -1;
  }

  uint32_t num_inputs = size;
  uint32_t num_outputs = size / local_size;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  size_t i_nbytes = num_inputs * sizeof(float);
  size_t o_nbytes = num_outputs * sizeof(float);
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, i_nbytes, NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, o_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  // Create kernel
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  // Set kernel arguments
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&c_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(uint32_t), &size));
  CL_CHECK(clSetKernelArg(kernel, 3, local_size*sizeof(float), NULL));

 // Allocate memories for input arrays and output arrays.
 std::vector<float> h_a(num_inputs);
 std::vector<float> h_c(num_outputs);

  // Generate input values
  for (uint32_t i = 0; i < num_inputs; ++i) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Creating command queue
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

	printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, i_nbytes, h_a.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &size, &local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, o_nbytes, h_c.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  auto result = computeParallelSumCPU(h_c.data(), num_outputs);
  auto gold = computeParallelSumCPU(h_a.data(), num_inputs);
  if (!compare_equal(result, gold)) {
    printf("*** error: expected=%f, actual=%f", gold, result);
    for (uint32_t i = 0; i < num_outputs; ++i) {
        printf(", output[%d]=%f", i, h_c[i]);
    }
    printf("\n");
    errors = 1;
  }
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
  } else {
    printf("PASSED!\n");
  }

  // Clean up
  cleanup();

  return errors;
}
