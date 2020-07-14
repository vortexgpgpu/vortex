#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>

#define SIZE 4096
#define NUM_WORK_GROUPS 2
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
uint8_t *kernel_bin = NULL;

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

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);  
  if (context) clReleaseContext(context);
  if (kernel_bin) free(kernel_bin);
  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
}

static int find_device(char* name, cl_platform_id platform_id, cl_device_id *device_id) {
  cl_device_id device_ids[64];
  cl_uint num_devices = 0;

  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 64, device_ids, &num_devices));

  for (int i=0; i<num_devices; i++) 	{
		char buffer[1024];
		cl_uint buf_uint;
		cl_ulong buf_ulong;

		CL_CHECK(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		
    if (0 == strncmp(buffer, name, strlen(name))) {
      *device_id = device_ids[i];
      return 0;
    }
	}

  return 1;
}

int main (int argc, char **argv) {
  printf("enter demo main\n");

  cl_platform_id platform_id;
  cl_device_id device_id;
  size_t kernel_size;
  cl_int binary_status = 0;
  int i;

  // read kernel binary from file  
  if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
    return -1;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  // Creating context.
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  // Memory buffers for each array
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(cl_int), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(cl_int), NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(cl_int), NULL, &_err));

  // Allocate memories for input arrays and output arrays.  
  A = (cl_int*)malloc(sizeof(cl_int)*SIZE);
  B = (cl_int*)malloc(sizeof(cl_int)*SIZE);
  C = (cl_int*)malloc(sizeof(cl_int)*SIZE);	
	
  // Initialize values for array members.  
  for (i=0; i<SIZE; ++i) {
    A[i] = i*2+0;
    B[i] = i*2+1;
  }

  // Create program from kernel source
  program = CL_CHECK2(clCreateProgramWithBinary(
    context, 1, &device_id, &kernel_size, &kernel_bin, &binary_status, &_err));

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
