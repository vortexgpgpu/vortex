#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>
#include "common.h"

#define KERNEL_NAME "rangelock"

// Work-items per work-group. Must be >= the hardware warp size so that several
// work-items land in the same warp and contend overlapping lock windows.
#define LOCAL_SIZE 32  // preferred work-group size; clamped to device max at runtime

static size_t g_local = LOCAL_SIZE;

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     cleanup();                                                        \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       cleanup();                                                      \
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
cl_mem locks_memobj = NULL;
cl_mem cell_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (locks_memobj) clReleaseMemObject(locks_memobj);
  if (cell_memobj) clReleaseMemObject(cell_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

uint32_t size = 1024;

static void show_usage() {
  printf("Usage: [-n size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
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
  // Work-group size clamped to the device max, and `size` rounded, in main().
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  cl_platform_id platform_id;
  size_t kernel_size;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  // Clamp the work-group size to the device max so several work-items share a
  // warp (needed for intra-warp lock contention) on any configuration.
  size_t max_wg = 0;
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL));
  if (g_local > max_wg) g_local = max_wg;
  if (g_local < 1) g_local = 1;
  size = (uint32_t)(((size + g_local - 1) / g_local) * g_local);
  printf("Workload size=%d (local=%zu, window=%d)\n", size, g_local, WINDOW);

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  // Lock/cell arrays span every window: lane g uses [g, g+WINDOW).
  int nlocks = (int)size + WINDOW - 1;
  std::vector<int> h_locks(nlocks, 0);
  std::vector<int> h_cell(nlocks, 0);

  printf("Allocate device buffers\n");
  size_t bytes = nlocks * sizeof(int);
  locks_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &_err));
  cell_memobj  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&locks_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cell_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void *)&nlocks));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, locks_memobj, CL_TRUE, 0, bytes, h_locks.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, cell_memobj, CL_TRUE, 0, bytes, h_cell.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  printf("NOTE: each lane holds %d nested locks; a SCS table of depth K<%d needs\n"
         "      the memory-backed split spill. On the current IPDOM stack this\n"
         "      DEADLOCKS at the first contended lock -- clFinish() never returns.\n",
         WINDOW, WINDOW);
  size_t global_work_size[1] = {size};
  size_t local_work_size[1] = {g_local};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffers\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, cell_memobj, CL_TRUE, 0, bytes, h_cell.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, locks_memobj, CL_TRUE, 0, bytes, h_locks.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Each cell j must have been bumped once per lane whose window covers it; under
  // correct mutual exclusion across overlapping windows that count is exact.
  // Coverage(j) = #{ g in [0,size) : g <= j < g+WINDOW }.
  int errors = 0;
  for (int j = 0; j < nlocks; ++j) {
    int lo = j - (WINDOW - 1); if (lo < 0) lo = 0;
    int hi = j; if (hi > (int)size - 1) hi = (int)size - 1;
    int expected = (hi >= lo) ? (hi - lo + 1) : 0;
    if (h_cell[j] != expected) {
      if (errors < 8)
        printf("*** error: cell[%d]=%d, expected=%d (lost mutual exclusion)\n", j, h_cell[j], expected);
      ++errors;
    }
    if (h_locks[j] != 0) {
      if (errors < 8)
        printf("*** error: lock[%d] left held (=%d)\n", j, h_locks[j]);
      ++errors;
    }
  }

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}
