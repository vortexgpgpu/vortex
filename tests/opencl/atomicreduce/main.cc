#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <chrono>
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
cl_mem data_memobj = NULL;
cl_mem result_memobj = NULL;
cl_mem rets_memobj = NULL;
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
  if (data_memobj) {
    clReleaseMemObject(data_memobj);
  }
  if (result_memobj) {
    clReleaseMemObject(result_memobj);
  }
  if (rets_memobj) {
    clReleaseMemObject(rets_memobj);
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

static const char* op_names[NUM_OPS] = {
  "add", "sub", "inc", "dec", "min", "max", "umin", "umax",
  "and", "or", "xor", "xchg", "cmpxchg", "fxchg"
};

static int bits_of(float f) {
  int i;
  memcpy(&i, &f, sizeof(i));
  return i;
}

static float float_of(int i) {
  float f;
  memcpy(&f, &i, sizeof(f));
  return f;
}

// Counter values before the kernel runs. A block of unsigned counters starts
// min/max at the unsigned extremes.
static void init_counters(int* c, bool is_unsigned) {
  for (int i = 0; i < NUM_OPS; ++i) {
    c[i] = 0;
  }
  c[OP_MIN]   = is_unsigned ? -1 : INT_MAX;
  c[OP_MAX]   = is_unsigned ? 0 : INT_MIN;
  c[OP_UMIN]  = -1;
  c[OP_AND]   = -1;
  c[OP_XCHG]  = XCHG_INIT;
  c[OP_FXCHG] = bits_of((float)XCHG_INIT);
}

// Counter values after every work-item in [lo, hi) applied its operations.
// The exchanges have no single expected value and are checked by the caller.
static void expect_counters(const std::vector<int>& data, uint32_t lo, uint32_t hi,
                            int* c, bool is_unsigned) {
  init_counters(c, is_unsigned);
  for (uint32_t gid = lo; gid < hi; ++gid) {
    int v = data[gid];
    c[OP_ADD] += v;
    c[OP_SUB] -= v;
    c[OP_INC] += 1;
    c[OP_DEC] -= 1;
    if (is_unsigned) {
      c[OP_MIN] = (int)std::min((uint32_t)c[OP_MIN], (uint32_t)v);
      c[OP_MAX] = (int)std::max((uint32_t)c[OP_MAX], (uint32_t)v);
    } else {
      c[OP_MIN] = std::min(c[OP_MIN], v);
      c[OP_MAX] = std::max(c[OP_MAX], v);
    }
    c[OP_UMIN] = (int)std::min((uint32_t)c[OP_UMIN], (uint32_t)v);
    c[OP_UMAX] = (int)std::max((uint32_t)c[OP_UMAX], (uint32_t)v);
    c[OP_AND] &= ~(1 << (gid & 31));
    c[OP_OR] |= v;
    c[OP_XOR] ^= v;
    c[OP_CMPXCHG] += v;
  }
}

// An exchange chain leaves one of the exchanged values in the counter.
static bool is_member(const std::vector<int>& data, uint32_t lo, uint32_t hi, int value) {
  for (uint32_t gid = lo; gid < hi; ++gid) {
    if (data[gid] == value) {
      return true;
    }
  }
  return false;
}

// has_fxchg: the kernel exchanged the float counter; otherwise it must still
// hold its start value.
static int check_counters(const char* tag, const int* actual, const int* expected,
                          const std::vector<int>& data, uint32_t lo, uint32_t hi,
                          bool has_fxchg) {
  int errors = 0;
  for (int i = 0; i < NUM_OPS; ++i) {
    bool ok;
    if (i == OP_XCHG) {
      ok = is_member(data, lo, hi, actual[i]);
    } else if (i == OP_FXCHG && has_fxchg) {
      float f = float_of(actual[i]);
      ok = (f == (float)(int)f) && is_member(data, lo, hi, (int)f);
    } else {
      ok = (actual[i] == expected[i]);
    }
    if (!ok) {
      if (errors < 20) {
        printf("*** error: %s %s: actual=%d, expected=%d\n", tag, op_names[i], actual[i], expected[i]);
      }
      ++errors;
    }
  }
  return errors;
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
  if (size % local_size != 0) {
    printf("Error: size must be a multiple of the local size\n");
    cleanup();
    return -1;
  }
  printf("Workload size=%d, local size=%d\n", size, local_size);

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  uint32_t num_groups = size / local_size;
  size_t result_count = (size_t)num_groups * 2 * NUM_OPS;

  printf("Allocate device buffers\n");
  size_t data_bytes = size * sizeof(int);
  data_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, data_bytes, NULL, &_err));
  result_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, result_count * sizeof(int), NULL, &_err));
  rets_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, size * NUM_RETS * sizeof(int), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size)) {
    return -1;
  }
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Values in [-1000, 1000): negative ones make the signed and unsigned
  // min/max differ.
  std::vector<int> h_data(size);
  int h_ref = 0;
  for (uint32_t i = 0; i < size; ++i) {
    h_data[i] = rand() % 2000 - 1000;
    h_ref += h_data[i];
  }

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_memobj, CL_TRUE, 0, data_bytes, h_data.data(), 0, NULL, NULL));

  int errors = 0;
  double elapsed = 0;
  std::vector<int> h_result(result_count);
  int expected[NUM_OPS];

  printf("Execute atomicreduce\n");
  {
    int zero = 0;
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, result_memobj, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL));
    build_kernel("atomicreduce");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_memobj));
    elapsed += run_kernel(size, 1);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, result_memobj, CL_TRUE, 0, sizeof(int), h_result.data(), 0, NULL, NULL));
    if (h_result[0] != h_ref) {
      printf("*** error: atomicreduce: actual=%d, expected=%d\n", h_result[0], h_ref);
      ++errors;
    }
  }

  printf("Execute atomic_global\n");
  {
    int init[NUM_OPS];
    init_counters(init, false);
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, result_memobj, CL_TRUE, 0, sizeof(init), init, 0, NULL, NULL));
    build_kernel("atomic_global");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &rets_memobj));
    elapsed += run_kernel(size, local_size);
    std::vector<int> rets(size * NUM_RETS);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, result_memobj, CL_TRUE, 0, sizeof(init), h_result.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, rets_memobj, CL_TRUE, 0, rets.size() * sizeof(int), rets.data(), 0, NULL, NULL));

    expect_counters(h_data, 0, size, expected, false);
    errors += check_counters("global", h_result.data(), expected, h_data, 0, size, true);

    // inc hands out every ticket exactly once.
    std::vector<int> tickets(size);
    for (uint32_t i = 0; i < size; ++i) {
      tickets[i] = rets[i * NUM_RETS + RET_INC];
    }
    std::sort(tickets.begin(), tickets.end());
    for (uint32_t i = 0; i < size; ++i) {
      if (tickets[i] != (int)i) {
        printf("*** error: global inc: ticket %d handed out %s\n", i, tickets[i] < (int)i ? "twice" : "never");
        ++errors;
        break;
      }
    }

    // An exchange chain returns every value it replaced: the returned values
    // and the final one are the initial value and all exchanged ones.
    long long xchg_sum = h_result[OP_XCHG];
    long long fxchg_sum = (long long)float_of(h_result[OP_FXCHG]);
    for (uint32_t i = 0; i < size; ++i) {
      xchg_sum += rets[i * NUM_RETS + RET_XCHG];
      fxchg_sum += rets[i * NUM_RETS + RET_FXCHG];
    }
    long long chain_sum = (long long)XCHG_INIT + h_ref;
    if (xchg_sum != chain_sum) {
      printf("*** error: global xchg chain: sum=%lld, expected=%lld\n", xchg_sum, chain_sum);
      ++errors;
    }
    if (fxchg_sum != chain_sum) {
      printf("*** error: global float xchg chain: sum=%lld, expected=%lld\n", fxchg_sum, chain_sum);
      ++errors;
    }
  }

  printf("Execute atomic_local\n");
  {
    build_kernel("atomic_local");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, NUM_OPS * sizeof(int), NULL));
    elapsed += run_kernel(size, local_size);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, result_memobj, CL_TRUE, 0, num_groups * NUM_OPS * sizeof(int), h_result.data(), 0, NULL, NULL));
    for (uint32_t g = 0; g < num_groups; ++g) {
      uint32_t lo = g * local_size, hi = lo + local_size;
      char tag[32];
      snprintf(tag, sizeof(tag), "local[%u]", g);
      expect_counters(h_data, lo, hi, expected, false);
      errors += check_counters(tag, &h_result[g * NUM_OPS], expected, h_data, lo, hi, true);
    }
  }

  // atom_*: an int block and a uint block of counters, on global memory and
  // per work-group on local memory.
  printf("Execute atom_global\n");
  {
    int init[2 * NUM_OPS];
    init_counters(init, false);
    init_counters(init + NUM_OPS, true);
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, result_memobj, CL_TRUE, 0, sizeof(init), init, 0, NULL, NULL));
    build_kernel("atom_global");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_memobj));
    elapsed += run_kernel(size, local_size);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, result_memobj, CL_TRUE, 0, sizeof(init), h_result.data(), 0, NULL, NULL));
    for (int u = 0; u < 2; ++u) {
      expect_counters(h_data, 0, size, expected, u != 0);
      errors += check_counters(u ? "atom global uint" : "atom global int", &h_result[u * NUM_OPS],
                               expected, h_data, 0, size, false);
    }
  }

  printf("Execute atom_local\n");
  {
    build_kernel("atom_local");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, 2 * NUM_OPS * sizeof(int), NULL));
    elapsed += run_kernel(size, local_size);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, result_memobj, CL_TRUE, 0, result_count * sizeof(int), h_result.data(), 0, NULL, NULL));
    for (uint32_t g = 0; g < num_groups; ++g) {
      uint32_t lo = g * local_size, hi = lo + local_size;
      for (int u = 0; u < 2; ++u) {
        char tag[48];
        snprintf(tag, sizeof(tag), "atom local[%u] %s", g, u ? "uint" : "int");
        expect_counters(h_data, lo, hi, expected, u != 0);
        errors += check_counters(tag, &h_result[(g * 2 + u) * NUM_OPS], expected, h_data, lo, hi, false);
      }
    }
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
