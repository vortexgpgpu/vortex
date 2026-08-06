// StreamCluster (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// StreamCluster clusters N-dimensional points. Its OpenCL-accelerated core is
// pgain_kernel: given a candidate new center at point x, each point computes
// the cost gain of reassigning to x versus its current center. The result is
// the work_mem array (per-point cost deltas) and switch_membership flags.
//
// This test exercises a single pgain invocation over a fixed, seeded set of
// points + open centers, and verifies the kernel's output (work_mem +
// switch_membership) against a serial CPU recomputation of the SAME math.
// Points are generated in-host from a fixed seed (the Rodinia 'none' input
// mode), so the test is self-contained and deterministic — no external files.
//
// pgain_kernel uses NO atomics: each work-item writes only into its own
// base = tid*(K+1) slice of work_mem, so no cross-thread contention.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

// Host mirror of the device Point_Struct. `assign` is int (not long) so the
// struct layout matches the 32-bit RISC-V device ABI (fix (a)).
typedef struct {
  float weight;
  int assign;
  float cost;
} Point_Struct;

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                               \
    if (_err == CL_SUCCESS)                                            \
      break;                                                           \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);    \
    cleanup();                                                         \
    exit(-1);                                                          \
  } while (0)

#define CL_CHECK2(_expr)                                               \
  ({                                                                   \
    cl_int _err = CL_INVALID_VALUE;                                    \
    decltype(_expr) _ret = _expr;                                      \
    if (_err != CL_SUCCESS) {                                          \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);  \
      cleanup();                                                       \
      exit(-1);                                                        \
    }                                                                  \
    _ret;                                                              \
  })

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;
  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }
  fseek(fp, 0, SEEK_END);
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
cl_kernel pgain_kernel = NULL;
cl_kernel memset_kernel = NULL;
cl_mem d_p = NULL;
cl_mem d_coord = NULL;
cl_mem d_work_mem = NULL;
cl_mem d_center_table = NULL;
cl_mem d_switch = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (pgain_kernel) clReleaseKernel(pgain_kernel);
  if (memset_kernel) clReleaseKernel(memset_kernel);
  if (program) clReleaseProgram(program);
  if (d_p) clReleaseMemObject(d_p);
  if (d_coord) clReleaseMemObject(d_coord);
  if (d_work_mem) clReleaseMemObject(d_work_mem);
  if (d_center_table) clReleaseMemObject(d_center_table);
  if (d_switch) clReleaseMemObject(d_switch);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (tiny by default so RTL simulation stays fast).
static int num = 128;   // number of points
static int dim = 32;    // dimensionality
static int local_size = 16;  // work-group size; must be <= NUM_WARPS*NUM_THREADS (16)

static void show_usage() {
  printf("Usage: [-n num_points] [-d dim] [-l local_size] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:d:l:h")) != -1) {
    switch (c) {
    case 'n': num = atoi(optarg); break;
    case 'd': dim = atoi(optarg); break;
    case 'l': local_size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (num < 2 || dim < 1 || local_size < 1 || local_size > 16) {
    printf("Error: invalid parameters (num>=2, dim>=1, 1<=local_size<=16)\n");
    exit(-1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // Pick the open centers: every stride-th point is a center (>=1 center).
  int stride = 8;
  int K = 0;                                  // number of open centers
  std::vector<int> center_table(num, -1);     // compact center index per point
  std::vector<char> is_center(num, 0);
  for (int i = 0; i < num; i++) {
    if (i % stride == 0) {
      is_center[i] = 1;
      center_table[i] = K++;
    }
  }
  if (K == 0) { is_center[0] = 1; center_table[0] = K++; }

  int kplus1 = K + 1;
  long x = num / 2;  // candidate new center: a fixed point index

  printf("StreamCluster pgain: num=%d dim=%d K=%d local_size=%d x=%ld\n",
         num, dim, K, local_size, x);

  // Deterministic point generation (Rodinia 'none' mode: in-host, seeded).
  // coord layout matches the kernel: coord[i*num + j] = dim-i of point j.
  srand(9);
  std::vector<float> coord((size_t)num * dim);
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < num; j++)
      coord[(size_t)i * num + j] = (float)(rand() % 1000) / 1000.0f;

  // Per-point weight/assign/cost. Each point is assigned to its NEAREST open
  // center; cost = weight * squared-distance to that center (as streamcluster
  // initializes it). weight = 1.0 as in the synthetic SimStream.
  std::vector<Point_Struct> p(num);
  for (int j = 0; j < num; j++) {
    p[j].weight = 1.0f;
    int best_c = 0;
    float best_d = 1e30f;
    for (int cc = 0; cc < num; cc++) {
      if (!is_center[cc]) continue;
      float d = 0.0f;
      for (int i = 0; i < dim; i++) {
        float diff = coord[(size_t)i * num + j] - coord[(size_t)i * num + cc];
        d += diff * diff;
      }
      if (d < best_d) { best_d = d; best_c = cc; }
    }
    p[j].assign = best_c;             // index of a center point (center_table valid)
    p[j].cost = p[j].weight * best_d; // current assignment cost
  }

  // ---- OpenCL setup ----
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  pgain_kernel = CL_CHECK2(clCreateKernel(program, "pgain_kernel", &_err));
  memset_kernel = CL_CHECK2(clCreateKernel(program, "memset_kernel", &_err));

  size_t work_mem_bytes = (size_t)kplus1 * num * sizeof(float);
  d_p = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, num * sizeof(Point_Struct), NULL, &_err));
  d_coord = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, (size_t)num * dim * sizeof(float), NULL, &_err));
  d_work_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, work_mem_bytes, NULL, &_err));
  d_center_table = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, num * sizeof(int), NULL, &_err));
  d_switch = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, num * sizeof(char), NULL, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_p, CL_TRUE, 0, num * sizeof(Point_Struct), p.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_coord, CL_TRUE, 0, (size_t)num * dim * sizeof(float), coord.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_center_table, CL_TRUE, 0, num * sizeof(int), center_table.data(), 0, NULL, NULL));

  auto round_up = [](size_t n, size_t l) { return ((n + l - 1) / l) * l; };
  size_t local = (size_t)local_size;

  // Zero work_mem and switch_membership via memset_kernel (exercises its guard).
  {
    short zero = 0;
    int nbytes = (int)work_mem_bytes;
    CL_CHECK(clSetKernelArg(memset_kernel, 0, sizeof(cl_mem), &d_work_mem));
    CL_CHECK(clSetKernelArg(memset_kernel, 1, sizeof(short), &zero));
    CL_CHECK(clSetKernelArg(memset_kernel, 2, sizeof(int), &nbytes));
    size_t g = round_up((size_t)nbytes, local);
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, memset_kernel, 1, NULL, &g, &local, 0, NULL, NULL));

    nbytes = num;  // switch_membership bytes
    CL_CHECK(clSetKernelArg(memset_kernel, 0, sizeof(cl_mem), &d_switch));
    CL_CHECK(clSetKernelArg(memset_kernel, 1, sizeof(short), &zero));
    CL_CHECK(clSetKernelArg(memset_kernel, 2, sizeof(int), &nbytes));
    g = round_up((size_t)nbytes, local);
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, memset_kernel, 1, NULL, &g, &local, 0, NULL, NULL));
  }

  // pgain kernel. Local shared-mem for the x-coordinate is dim floats.
  {
    int arg = 0;
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(cl_mem), &d_p));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(cl_mem), &d_coord));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(cl_mem), &d_work_mem));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(cl_mem), &d_center_table));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(cl_mem), &d_switch));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, dim * sizeof(float), NULL));  // __local coord_s
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(int), &num));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(int), &dim));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(long), &x));
    CL_CHECK(clSetKernelArg(pgain_kernel, arg++, sizeof(int), &K));

    size_t g = round_up((size_t)num, local);  // multiple of local; kernel guards tail
    auto t0 = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, pgain_kernel, 1, NULL, &g, &local, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Elapsed time: %lg ms\n",
           (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  }

  // Read back device results.
  std::vector<float> gpu_work(kplus1 * (size_t)num);
  std::vector<char> gpu_switch(num);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_work_mem, CL_TRUE, 0, work_mem_bytes, gpu_work.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_switch, CL_TRUE, 0, num * sizeof(char), gpu_switch.data(), 0, NULL, NULL));

  // ---- Serial CPU golden reference: same pgain math, per point ----
  std::vector<float> ref_work(kplus1 * (size_t)num, 0.0f);
  std::vector<char> ref_switch(num, 0);
  std::vector<float> coord_s(dim);
  for (int i = 0; i < dim; i++)
    coord_s[i] = coord[(size_t)i * num + x];
  for (int tid = 0; tid < num; tid++) {
    float x_cost = 0.0f;
    for (int i = 0; i < dim; i++) {
      float d = coord[(size_t)i * num + tid] - coord_s[i];
      x_cost += d * d;
    }
    x_cost *= p[tid].weight;
    float current_cost = p[tid].cost;
    int base = tid * kplus1;
    if (x_cost < current_cost) {
      ref_switch[tid] = '1';
      ref_work[base + K] = x_cost - current_cost;
    } else {
      ref_work[base + center_table[p[tid].assign]] += current_cost - x_cost;
    }
  }

  // ---- Compare ----
  const float tol = 1e-3f;  // relative tolerance for float accumulation
  int errors = 0;
  for (size_t i = 0; i < ref_work.size(); i++) {
    float a = ref_work[i], b = gpu_work[i];
    float denom = fabsf(a) > 1.0f ? fabsf(a) : 1.0f;
    if (fabsf(a - b) / denom > tol) {
      if (errors < 20)
        printf("*** work_mem error: [%zu] expected=%f, actual=%f\n", i, a, b);
      ++errors;
    }
  }
  for (int i = 0; i < num; i++) {
    if (gpu_switch[i] != ref_switch[i]) {
      if (errors < 20)
        printf("*** switch error: [%d] expected=%d, actual=%d\n", i, ref_switch[i], gpu_switch[i]);
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}
