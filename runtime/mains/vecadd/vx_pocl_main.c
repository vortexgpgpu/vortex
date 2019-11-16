
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "../../tests/tests.h"
#include "../../vx_api/vx_api.h"
#include "../../fileio/fileio.h"
#include <CL/opencl.h>

// Newlib
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_KERNELS 1
#define KERNEL_NAME "vecadd"
#define KERNEL_FILE_NAME "vecadd.pocl"
#define SIZE 4
#define NUM_WORK_GROUPS 2

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
     typeof(_expr) _ret = _expr;                                       \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
	   cleanup();			                                                   \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

typedef struct {
  const char* name;
  const void* pfn;
  uint32_t num_args;
  uint32_t num_locals;
  const uint8_t* arg_types;
  const uint32_t* local_sizes;
} kernel_info_t;

static int g_num_kernels = 0;
static kernel_info_t g_kernels [MAX_KERNELS];

#ifdef __cplusplus
extern "C" {
#endif

int _pocl_register_kernel(const char* name, const void* pfn, uint32_t num_args, uint32_t num_locals, const uint8_t* arg_types, const uint32_t* local_sizes) {
  printf("******** _pocl_register_kernel\n");
  printf("Name to register: %s\n", name);
  printf("PTR of name: %x\n", name);
  if (g_num_kernels == MAX_KERNELS)
  {
  	printf("ERROR: REACHED MAX KERNELS\n");
    return -1;	
  }

  printf("Going to register at index: %d\n", g_num_kernels);

  kernel_info_t* kernel = g_kernels + g_num_kernels++;
  kernel->name = name;
  kernel->pfn = pfn;
  kernel->num_args = num_args;
  kernel->num_locals = num_locals;
  kernel->arg_types = arg_types;
  kernel->local_sizes = local_sizes;
  printf("New kernel name: %s\n", kernel->name);
  return 0;
}

int _pocl_query_kernel(const char* name, const void** p_pfn, uint32_t* p_num_args, uint32_t* p_num_locals, const uint8_t** p_arg_types, const uint32_t** p_local_sizes) {
  printf("********* Inside _pocl_query_kernel\n");
  printf("name: %s\n", name);
  printf("g_num_kernels: %d\n", g_num_kernels);
  for (int i = 0; i < g_num_kernels; ++i) {
  	printf("Currently quering index %d\n", i);
    kernel_info_t* kernel = g_kernels + i;
    if (strcmp(kernel->name, name) != 0)
    {
      printf("STR CMP failed! kernel->name = %s \t name: %s\n", kernel->name, name);
      continue;
    }
    printf("!!!!!!!!!STR CMP PASSED\n");
    if (p_pfn) *p_pfn = kernel->pfn;
    if (p_num_args) *p_num_args = kernel->num_args;
    if (p_num_locals) *p_num_locals = kernel->num_locals;
    if (p_arg_types) *p_arg_types = kernel->arg_types;
    if (p_local_sizes) *p_local_sizes = kernel->local_sizes;
    return 0;
  }
  return -1;
}

#ifdef __cplusplus
}
#endif

int exitcode = 0;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem c_memobj = NULL;  
cl_int *A = NULL;
cl_int *B = NULL;
cl_int *C = NULL;
char *binary = NULL;

void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);  
  if (context) clReleaseContext(context);
  if (binary) free(binary);
  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
}

int main (int argc, char **argv) {  
  vx_tmc(1);

  vx_print_str("Hello from vecadd\n");
  printf("New cleared vecadd running\n");

  cl_platform_id platform_id;
  cl_device_id device_id;
  size_t binary_size;
  int i;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));

  printf("Got platform id: %x\n", platform_id);

  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));


  printf("Got platformID and deviceID\n");

  // Creating context.
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));


  printf("Got context\n");

  // Memory buffers for each array
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(cl_int), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(cl_int), NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(cl_int), NULL, &_err));


  printf("Created buffers\n");

  // Allocate memories for input arrays and output arrays.  
  A = (cl_int*)malloc(sizeof(cl_int)*SIZE);
  B = (cl_int*)malloc(sizeof(cl_int)*SIZE);
  C = (cl_int*)malloc(sizeof(cl_int)*SIZE);

  printf("Allocated memory: A=%x\tB=%x\tC=%x\n", A, B, C);	
	
  // Initialize values for array members.  
  for (i=0; i<SIZE; ++i) {
    A[i] = i*2+0;
    B[i] = i*2+1;
  }

  printf("About to call clCreateProgramWithBuiltInKernels\n");

  // Create program from kernel source
  program = CL_CHECK2(clCreateProgramWithBuiltInKernels(context, 1, &device_id, KERNEL_NAME, &_err));	

  printf("Returned from clCreateProgramWithBuiltInKernels\n");

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  
  // Create kernel
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  // Set arguments for kernel
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_memobj));

  // Creating command queue
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

	// Copy lists to memory buffers
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, SIZE * sizeof(float), A, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, SIZE * sizeof(float), B, 0, NULL, NULL));

  // Execute the kernel
  size_t globalItemSize = SIZE;
  size_t localItemSize = SIZE/NUM_WORK_GROUPS;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  // Read from device back to host.
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, SIZE * sizeof(float), C, 0, NULL, NULL));

  // Test if correct answer
  int exitcode = 0;
  for (i=0; i<SIZE; ++i) {
    if (C[i] != (A[i] + B[i])) {
      printf("Failed!\n");
      exitcode = 1;
      break;
    }
  }
  if (i == SIZE) {
    printf("Ok!\n");
  }

  // Clean up		
  cleanup();  

  return exitcode;
}


// int main()
// {
// 	// Main is called with all threads active of warp 0
// 	vx_tmc(1);


// 	printf("printf: including pocl Main!\n");


// 	return 0;
// }





