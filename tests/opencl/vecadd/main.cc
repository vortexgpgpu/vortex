#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>

#define KERNEL_NAME "vecadd"

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

static bool almost_equal(float a, float b, int ulp = 4) {
  union fi_t { int i; float f; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return std::abs(fa.i - fb.i) <= ulp;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem c_memobj = NULL;  
float *h_a = NULL;
float *h_b = NULL;
float *h_c = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);  
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  
  if (kernel_bin) free(kernel_bin);
  if (h_a) free(h_a);
  if (h_b) free(h_b);
  if (h_c) free(h_c);
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
  size_t nbytes = size * sizeof(float);
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

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
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_memobj));

  // Allocate memories for input arrays and output arrays.    
  h_a = (float*)malloc(nbytes);
  h_b = (float*)malloc(nbytes);
  h_c = (float*)malloc(nbytes);	
	
  // Generate input values
  for (int i = 0; i < size; ++i) {
    h_a[i] = sinf(i)*sinf(i);
    h_b[i] = cosf(i)*cosf(i);
  }

  // Creating command queue
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));  

	printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, nbytes, h_b, 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {size};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, nbytes, h_c, 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float ref = h_a[i] + h_b[i];
    if (!almost_equal(h_c[i], ref)) {
      if (errors < 100) 
        printf("*** error: [%d] expected=%f, actual=%f, a=%f, b=%f\n", i, ref, h_c[i], h_a[i], h_b[i]);
      ++errors;
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
