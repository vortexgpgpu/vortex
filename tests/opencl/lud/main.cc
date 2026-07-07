// LUD (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Blocked LU decomposition (no pivoting) of a dense float matrix, factored
// in place into a unit-lower L and upper U (Doolittle layout). The device
// runs the classic three-phase blocked loop (diagonal -> perimeter ->
// internal per block step) exactly like Rodinia's host. Correctness is
// checked against a serial CPU Doolittle LU over the identical (seeded,
// diagonally-dominant) input with a relative floating tolerance.
//
// Device constraint: max work-group size = NUM_WARPS*NUM_THREADS = 16.
// lud_internal uses a BLOCK_SIZE x BLOCK_SIZE 2-D group and lud_perimeter a
// BLOCK_SIZE*2 group, so BLOCK_SIZE=4 keeps every group within 16 work-items.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

// Compile-time block size shared by host launch geometry and the kernels.
// 4 keeps lud_internal (BLOCK_SIZE^2 = 16) and lud_perimeter (BLOCK_SIZE*2 = 8)
// within the device's 16 work-item group limit.
#define BLOCK_SIZE 4

// Relative tolerance for comparing GPU vs CPU factors; the blocked and serial
// eliminations sum in different orders, so exact equality is not expected.
#define REL_TOL 1e-3f
#define ABS_TOL 1e-4f

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
cl_kernel diagonal = NULL;
cl_kernel perimeter = NULL;
cl_kernel internal = NULL;
cl_mem d_m = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (diagonal) clReleaseKernel(diagonal);
  if (perimeter) clReleaseKernel(perimeter);
  if (internal) clReleaseKernel(internal);
  if (program) clReleaseProgram(program);
  if (d_m) clReleaseMemObject(d_m);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Matrix dimension (must be a multiple of BLOCK_SIZE). Kept tiny by default so
// RTL simulation stays within budget; override with -s.
static int matrix_dim = 16;

static void show_usage() {
  printf("Usage: [-s matrix_dim (multiple of %d)] [-h]\n", BLOCK_SIZE);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "s:h")) != -1) {
    switch (c) {
    case 's': matrix_dim = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (matrix_dim < BLOCK_SIZE || (matrix_dim % BLOCK_SIZE) != 0) {
    printf("Error: matrix_dim must be a positive multiple of BLOCK_SIZE (%d)\n", BLOCK_SIZE);
    exit(-1);
  }
}

// Serial CPU Doolittle LU (in place, no pivoting). Produces the same layout the
// device does: unit-diagonal L below the diagonal, U on/above it.
static void lud_cpu(std::vector<float>& a, int n) {
  for (int k = 0; k < n; ++k) {
    for (int j = k; j < n; ++j) {       // U row k
      float sum = a[k * n + j];
      for (int p = 0; p < k; ++p)
        sum -= a[k * n + p] * a[p * n + j];
      a[k * n + j] = sum;
    }
    for (int i = k + 1; i < n; ++i) {   // L column k
      float sum = a[i * n + k];
      for (int p = 0; p < k; ++p)
        sum -= a[i * n + p] * a[p * n + k];
      a[i * n + k] = sum / a[k * n + k];
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("LUD: matrix_dim=%d block_size=%d\n", matrix_dim, BLOCK_SIZE);

  int n = matrix_dim;
  int size = n * n;
  size_t nbytes = size * sizeof(float);

  // Deterministic, diagonally-dominant input: random in [0,1) plus n on the
  // diagonal. Diagonal dominance keeps LU (which uses no pivoting) stable.
  std::vector<float> h_input(size);
  srand(50);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      h_input[i * n + j] = static_cast<float>(rand()) / RAND_MAX;
  for (int i = 0; i < n; ++i)
    h_input[i * n + i] += static_cast<float>(n);

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_m = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // BLOCK_SIZE is a compile-time macro inside the kernels.
  char options[64];
  snprintf(options, sizeof(options), "-D BLOCK_SIZE=%d", BLOCK_SIZE);
  CL_CHECK(clBuildProgram(program, 1, &device_id, options, NULL, NULL));

  diagonal  = CL_CHECK2(clCreateKernel(program, "lud_diagonal", &_err));
  perimeter = CL_CHECK2(clCreateKernel(program, "lud_perimeter", &_err));
  internal  = CL_CHECK2(clCreateKernel(program, "lud_internal", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_m, CL_TRUE, 0, nbytes,
                                h_input.data(), 0, NULL, NULL));

  // Blocked three-phase loop, identical to Rodinia's host driver.
  auto time_start = std::chrono::high_resolution_clock::now();
  int i = 0;
  for (i = 0; i < n - BLOCK_SIZE; i += BLOCK_SIZE) {
    CL_CHECK(clSetKernelArg(diagonal, 0, sizeof(cl_mem), &d_m));
    CL_CHECK(clSetKernelArg(diagonal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(diagonal, 2, sizeof(cl_int), &matrix_dim));
    CL_CHECK(clSetKernelArg(diagonal, 3, sizeof(cl_int), &i));
    {
      size_t global_work[1] = {BLOCK_SIZE};
      size_t local_work[1]  = {BLOCK_SIZE};
      CL_CHECK(clEnqueueNDRangeKernel(commandQueue, diagonal, 1, NULL,
                                      global_work, local_work, 0, NULL, NULL));
    }

    CL_CHECK(clSetKernelArg(perimeter, 0, sizeof(cl_mem), &d_m));
    CL_CHECK(clSetKernelArg(perimeter, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(perimeter, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(perimeter, 3, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(perimeter, 4, sizeof(cl_int), &matrix_dim));
    CL_CHECK(clSetKernelArg(perimeter, 5, sizeof(cl_int), &i));
    {
      size_t global_work[1] = {(size_t)(BLOCK_SIZE * 2 * ((n - i) / BLOCK_SIZE - 1))};
      size_t local_work[1]  = {BLOCK_SIZE * 2};
      if (global_work[0] > 0) {
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, perimeter, 1, NULL,
                                        global_work, local_work, 0, NULL, NULL));
      }
    }

    CL_CHECK(clSetKernelArg(internal, 0, sizeof(cl_mem), &d_m));
    CL_CHECK(clSetKernelArg(internal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(internal, 2, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(internal, 3, sizeof(cl_int), &matrix_dim));
    CL_CHECK(clSetKernelArg(internal, 4, sizeof(cl_int), &i));
    {
      size_t blocks = (n - i) / BLOCK_SIZE - 1;
      size_t global_work[2] = {(size_t)BLOCK_SIZE * blocks, (size_t)BLOCK_SIZE * blocks};
      size_t local_work[2]  = {BLOCK_SIZE, BLOCK_SIZE};
      if (global_work[0] > 0) {
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, internal, 2, NULL,
                                        global_work, local_work, 0, NULL, NULL));
      }
    }
  }

  // Final diagonal block.
  CL_CHECK(clSetKernelArg(diagonal, 0, sizeof(cl_mem), &d_m));
  CL_CHECK(clSetKernelArg(diagonal, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, NULL));
  CL_CHECK(clSetKernelArg(diagonal, 2, sizeof(cl_int), &matrix_dim));
  CL_CHECK(clSetKernelArg(diagonal, 3, sizeof(cl_int), &i));
  {
    size_t global_work[1] = {BLOCK_SIZE};
    size_t local_work[1]  = {BLOCK_SIZE};
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, diagonal, 1, NULL,
                                    global_work, local_work, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<float> h_gpu(size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_m, CL_TRUE, 0, nbytes,
                               h_gpu.data(), 0, NULL, NULL));

  // CPU reference: serial Doolittle LU over the same input.
  std::vector<float> h_ref(h_input);
  lud_cpu(h_ref, n);

  int errors = 0;
  for (int idx = 0; idx < size; ++idx) {
    float ref = h_ref[idx];
    float got = h_gpu[idx];
    float tol = fmaxf(ABS_TOL, REL_TOL * fabsf(ref));
    if (fabsf(got - ref) > tol) {
      if (errors < 20)
        printf("*** error: [%d,%d] expected=%f, actual=%f\n",
               idx / n, idx % n, ref, got);
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
