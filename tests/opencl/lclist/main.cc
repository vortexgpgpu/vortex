#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>
#include "common.h"

#define KERNEL_NAME "lclist"

// Work-items per work-group. Must be >= the hardware warp size so that several
// work-items land in the SAME warp and contend for the head lock -- the
// deterministic IPDOM deadlock trigger.
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
cl_mem node_key_memobj = NULL;
cl_mem node_next_memobj = NULL;
cl_mem node_lock_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (node_key_memobj) clReleaseMemObject(node_key_memobj);
  if (node_next_memobj) clReleaseMemObject(node_next_memobj);
  if (node_lock_memobj) clReleaseMemObject(node_lock_memobj);
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
  printf("Workload size=%d (local=%zu)\n", size, g_local);

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  // Node pool: 2 sentinels + one node per work-item.
  uint32_t num_nodes = size + FIRST_DATA_NODE;
  std::vector<int> h_node_key(num_nodes);
  std::vector<int> h_node_next(num_nodes, -1);
  std::vector<int> h_node_lock(num_nodes, 0);

  h_node_key[HEAD_NODE] = INT_MIN;
  h_node_key[TAIL_NODE] = INT_MAX;
  h_node_next[HEAD_NODE] = TAIL_NODE;   // empty list: head -> tail
  h_node_next[TAIL_NODE] = -1;
  for (uint32_t i = 0; i < size; ++i)
    h_node_key[FIRST_DATA_NODE + i] = rand() % 100000;

  printf("Allocate device buffers\n");
  size_t pool_bytes = num_nodes * sizeof(int);
  node_key_memobj  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, pool_bytes, NULL, &_err));
  node_next_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, pool_bytes, NULL, &_err));
  node_lock_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, pool_bytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  {
    cl_int berr = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (berr != CL_SUCCESS) {
      size_t logsz = 0;
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
      std::vector<char> log(logsz + 1, 0);
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logsz, log.data(), NULL);
      printf("clBuildProgram failed (%d):\n%s\n", (int)berr, log.data());
      cleanup();
      exit(-1);
    }
  }

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&node_key_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&node_next_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&node_lock_memobj));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, node_key_memobj, CL_TRUE, 0, pool_bytes, h_node_key.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, node_next_memobj, CL_TRUE, 0, pool_bytes, h_node_next.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, node_lock_memobj, CL_TRUE, 0, pool_bytes, h_node_lock.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  printf("NOTE: on a SIMT reconvergence-stack (IPDOM) device this kernel\n"
         "      DEADLOCKS at the first head lock -- clFinish() never returns.\n");
  size_t global_work_size[1] = {size};
  size_t local_work_size[1] = {g_local};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffers\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, node_key_memobj, CL_TRUE, 0, pool_bytes, h_node_key.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, node_next_memobj, CL_TRUE, 0, pool_bytes, h_node_next.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Walk head -> tail: the list must be sorted, contain every data node exactly
  // once, and have no cycles. A lost update shows up as a missing node; a
  // deadlock never reaches this point.
  int errors = 0;
  std::vector<int> seen(num_nodes, 0);
  uint32_t count = 0;
  int prev_key = INT_MIN;
  int node = h_node_next[HEAD_NODE];
  uint32_t guard = 0;
  while (node != TAIL_NODE) {
    if (node < 0 || (uint32_t)node >= num_nodes) {
      printf("*** error: out-of-range link %d\n", node); ++errors; break;
    }
    if (seen[node]++) {
      printf("*** error: node %d visited twice (cycle/corruption)\n", node); ++errors; break;
    }
    if (h_node_key[node] < prev_key) {
      printf("*** error: out-of-order key %d after %d\n", h_node_key[node], prev_key); ++errors;
    }
    prev_key = h_node_key[node];
    ++count;
    node = h_node_next[node];
    if (++guard > num_nodes) { printf("*** error: list does not terminate\n"); ++errors; break; }
  }
  if (count != size) {
    printf("*** error: list has %u data nodes, expected %u (lost inserts)\n", count, size);
    ++errors;
  }

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}
