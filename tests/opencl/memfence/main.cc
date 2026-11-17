#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include "common.h"

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
  if (nullptr == filename || nullptr == data || 0 == size) {
    return -1;
  }

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
cl_mem buf_memobj = NULL;
cl_mem tmp_memobj = NULL;
cl_mem out_memobj = NULL;
cl_mem count_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) {
    clReleaseCommandQueue(commandQueue);
  }
  if (kernel) {
    clReleaseKernel(kernel);
  }
  if (program) {
    clReleaseProgram(program);
  }
  if (buf_memobj) {
    clReleaseMemObject(buf_memobj);
  }
  if (tmp_memobj) {
    clReleaseMemObject(tmp_memobj);
  }
  if (out_memobj) {
    clReleaseMemObject(out_memobj);
  }
  if (count_memobj) {
    clReleaseMemObject(count_memobj);
  }
  if (context) {
    clReleaseContext(context);
  }
  if (device_id) {
    clReleaseDevice(device_id);
  }
  if (kernel_bin) {
    free(kernel_bin);
  }
}

uint32_t size = 1024;
uint32_t local_size = 0;  // 0: the device's maximum work-group size
int rounds = 4;

static void show_usage() {
  printf("Usage: [-n size] [-l local size] [-r rounds] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:l:r:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'l':
      local_size = atoi(optarg);
      break;
    case 'r':
      rounds = atoi(optarg);
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

static void build_kernel(const char* name) {
  if (kernel) {
    clReleaseKernel(kernel);
  }
  kernel = CL_CHECK2(clCreateKernel(program, name, &_err));
}

static double run_kernel(size_t global, size_t local) {
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
}

// What work-item gid of the exchange kernels accumulates.
static uint32_t expect_exchange(uint32_t gid) {
  uint32_t lid = gid % local_size;
  uint32_t base = gid - lid;
  uint32_t acc = 0;
  for (int r = 0; r < rounds; ++r) {
    uint32_t peer = FENCE_PEER(lid, local_size, (uint32_t)r);
    acc = acc * 31u + fence_value(base + peer, peer, r);
  }
  return acc;
}

static int check_out(const char* tag, const std::vector<uint32_t>& actual,
                     const std::vector<uint32_t>& expected) {
  int errors = 0;
  for (uint32_t i = 0; i < size; ++i) {
    if (actual[i] != expected[i]) {
      if (errors < 20) {
        printf("*** error: %s [%d]: actual=0x%x, expected=0x%x\n", tag, i, actual[i], expected[i]);
      }
      ++errors;
    }
  }
  return errors;
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  cl_platform_id platform_id;
  size_t kernel_size;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  if (local_size == 0) {
    size_t max_wg;
    CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL));
    local_size = (uint32_t)std::min<size_t>(max_wg, size);
  }
  // lastgroup reduces a work-group by halving.
  if (size % local_size != 0 || (local_size & (local_size - 1)) != 0) {
    printf("Error: the local size must be a power of two dividing the size\n");
    cleanup();
    return -1;
  }
  printf("Workload size=%d, local size=%d, rounds=%d\n", size, local_size, rounds);

  size_t ext_size;
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_size));
  std::string extensions(ext_size, '\0');
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, ext_size, &extensions[0], NULL));
  bool has_atomics = extensions.find("cl_khr_global_int32_base_atomics") != std::string::npos;

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Allocate device buffers\n");
  size_t bytes = size * sizeof(uint32_t);
  buf_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &_err));
  tmp_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &_err));
  out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &_err));
  count_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size)) {
    return -1;
  }
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, has_atomics ? "-DHAS_ATOMICS" : NULL, NULL, NULL));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  int errors = 0;
  double elapsed = 0;
  std::vector<uint32_t> h_out(size);
  std::vector<uint32_t> h_ref(size);

  for (uint32_t i = 0; i < size; ++i) {
    h_ref[i] = expect_exchange(i);
  }

  printf("Execute gbarrier\n");
  build_kernel("gbarrier");
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &rounds));
  elapsed += run_kernel(size, local_size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, bytes, h_out.data(), 0, NULL, NULL));
  errors += check_out("gbarrier", h_out, h_ref);

  printf("Execute lbarrier\n");
  build_kernel("lbarrier");
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, local_size * sizeof(uint32_t), NULL));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &rounds));
  elapsed += run_kernel(size, local_size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, bytes, h_out.data(), 0, NULL, NULL));
  errors += check_out("lbarrier", h_out, h_ref);

  printf("Execute async_copy\n");
  {
    std::vector<uint32_t> h_in(size);
    for (uint32_t i = 0; i < size; ++i) {
      h_in[i] = (uint32_t)rand();
    }
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, buf_memobj, CL_TRUE, 0, bytes, h_in.data(), 0, NULL, NULL));
    for (uint32_t i = 0; i < size; ++i) {
      uint32_t lid = i % local_size;
      uint32_t base = i - lid;
      uint32_t p = FENCE_PEER(lid, local_size, 0u);
      h_ref[i] = h_in[base + (p + 1) % local_size] * 3u + p;
    }
    build_kernel("async_copy");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_memobj));
    CL_CHECK(clSetKernelArg(kernel, 3, local_size * sizeof(uint32_t), NULL));
    elapsed += run_kernel(size, local_size);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, bytes, h_out.data(), 0, NULL, NULL));
    errors += check_out("async_copy", h_out, h_ref);
  }

  if (has_atomics) {
    uint32_t num_groups = size / local_size;
    std::vector<int> h_data(size);
    int sum = 0;
    for (uint32_t i = 0; i < size; ++i) {
      h_data[i] = rand() % 2000 - 1000;
      sum += h_data[i];
    }
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, buf_memobj, CL_TRUE, 0, bytes, h_data.data(), 0, NULL, NULL));
    build_kernel("lastgroup");
    for (int mode = 0; mode < 2; ++mode) {
      printf("Execute lastgroup (%s)\n", mode ? "mem_fence" : "read/write_mem_fence");
      std::vector<int> zeros(num_groups, 0);
      uint32_t tickets = 0;
      int sentinel = 0x7fffffff;
      CL_CHECK(clEnqueueWriteBuffer(commandQueue, out_memobj, CL_TRUE, 0, sizeof(int), &sentinel, 0, NULL, NULL));
      CL_CHECK(clEnqueueWriteBuffer(commandQueue, tmp_memobj, CL_TRUE, 0, num_groups * sizeof(int), zeros.data(), 0, NULL, NULL));
      CL_CHECK(clEnqueueWriteBuffer(commandQueue, count_memobj, CL_TRUE, 0, sizeof(tickets), &tickets, 0, NULL, NULL));
      CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_memobj));
      CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp_memobj));
      CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &count_memobj));
      CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_memobj));
      CL_CHECK(clSetKernelArg(kernel, 4, local_size * sizeof(int), NULL));
      CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &mode));
      elapsed += run_kernel(size, local_size);
      int result;
      CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, sizeof(int), &result, 0, NULL, NULL));
      CL_CHECK(clEnqueueReadBuffer(commandQueue, count_memobj, CL_TRUE, 0, sizeof(tickets), &tickets, 0, NULL, NULL));
      if (result != sum || tickets != num_groups) {
        printf("*** error: lastgroup mode %d: sum=%d (expected %d), tickets=%u (expected %u)\n",
               mode, result, sum, tickets, num_groups);
        ++errors;
      }
    }
  } else {
    printf("Skip lastgroup: the device has no global atomics\n");
  }

  printf("Elapsed time: %lg ms\n", elapsed);

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors != 0;
}
