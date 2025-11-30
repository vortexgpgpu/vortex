#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>

#define KERNEL0_NAME "psorti"
#define KERNEL1_NAME "psortf"

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

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem c_memobj = NULL;
int *h_a = NULL;
int *h_c = NULL;
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
  if (h_a) free(h_a);
  if (h_c) free(h_c);
}

int size = 64;
bool float_enable = false;

static void show_usage() {
  printf("Usage: [-f] [-n size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "fn:h")) != -1) {
    switch (c) {
    case 'f':
      float_enable = 1;
      break;
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
  size_t nbytes = size * sizeof(int);
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  // Create kernel
  kernel = CL_CHECK2(clCreateKernel(program, (float_enable ? KERNEL1_NAME : KERNEL0_NAME), &_err));

  // Set kernel arguments
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&c_memobj));

  // Allocate memories for input arrays and output arrays.
  h_a = (int*)malloc(nbytes);
  h_c = (int*)malloc(nbytes);

  // Generate input values
  for (int i = 0; i < size; ++i) {
    if (float_enable) {
      float value = sinf(i)*sinf(i);
      ((float*)h_a)[i] = value;
      printf("*** [%d]: %f\n", i, value);
    } else {
      int value = size*sinf(i);
      h_a[i] = value;
      printf("*** [%d]: %d\n", i, value);
    }
  }

  // Creating command queue
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

	printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a, 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {size};
  size_t local_work_size[1] = {1};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, nbytes, h_c, 0, NULL, NULL));

  printf("Verify result\n");
  for (int i = 0; i < size; ++i) {
    if (float_enable) {
      float value = ((float*)h_c)[i];
      printf("*** [%d]: %f\n", i, value);
    } else {
      int value = h_c[i];
      printf("*** [%d]: %d\n", i, value);
    }
  }
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    int pos = 0;
    if (float_enable) {
      float ref = ((float*)h_a)[i];
      for (int j = 0; j < size; ++j) {
        float cur = ((float*)h_a)[j];
        pos += (cur < ref) || (cur == ref && j < i);
      }
      float value = ((float*)h_c)[pos];
      if (value != ref) {
        if (errors < 100) {
          printf("*** error: [%d] expected=%f, actual=%f\n", pos, ref, value);
        }
        ++errors;
      }
    } else {
      int ref = h_a[i];
      for (int j = 0; j < size; ++j) {
        int cur = h_a[j];
        pos += (cur < ref) || (cur == ref && j < i);
      }
      int value = h_c[pos];
      if (value != ref) {
        if (errors < 100) {
          printf("*** error: [%d] expected=%d, actual=%d\n", pos, ref, value);
        }
        ++errors;
      }
    }
  }
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  // Clean up
  cleanup();

  return errors;
}
