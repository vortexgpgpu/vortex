// Needleman-Wunsch (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Global sequence alignment via dynamic programming. The score matrix is filled
// with the recurrence max(diagonal + substitution, up - penalty, left - penalty)
// along anti-diagonal wavefronts, one block at a time (nw_kernel1 sweeps the
// upper-left triangle of blocks, nw_kernel2 the lower-right). Correctness is
// checked against a serial CPU fill over the identical (seeded) inputs.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

// OpenCL local work-group size == kernel BLOCK_SIZE. Must not exceed the device
// max work-group size (NUM_WARPS*NUM_THREADS = 16 in the default CI config).
#define BLOCK_SIZE 16
#define LIMIT -999

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
cl_kernel kernel1 = NULL;
cl_kernel kernel2 = NULL;
cl_mem reference_d = NULL;
cl_mem input_itemsets_d = NULL;
cl_mem output_itemsets_d = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (program) clReleaseProgram(program);
  if (reference_d) clReleaseMemObject(reference_d);
  if (input_itemsets_d) clReleaseMemObject(input_itemsets_d);
  if (output_itemsets_d) clReleaseMemObject(output_itemsets_d);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// BLOSUM62-like substitution table (indices are the generated sequence symbols).
static const int blosum62[24][24] = {
  { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
  {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
  {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
  {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
  {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
  {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
  {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
  {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
  {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
  {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
  {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
  {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
  {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
  { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
  { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
  {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
  {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
  { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
  {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
  {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

static inline int maximum(int a, int b, int c) {
  int k = (a <= b) ? b : a;
  return (k <= c) ? c : k;
}

// Workload parameters (small by default so RTL simulation stays under budget).
// dimension must be a multiple of BLOCK_SIZE.
static int dimension = 2 * BLOCK_SIZE;
static int penalty = 10;

static void show_usage() {
  printf("Usage: [-n dimension] [-p penalty] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:p:h")) != -1) {
    switch (c) {
    case 'n': dimension = atoi(optarg); break;
    case 'p': penalty = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (dimension < BLOCK_SIZE || (dimension % BLOCK_SIZE) != 0 || penalty < 1) {
    printf("Error: dimension must be a positive multiple of BLOCK_SIZE (%d)\n", BLOCK_SIZE);
    exit(-1);
  }
}

// Serial CPU reference fill over the same border-seeded matrix.
static void nw_cpu(std::vector<int>& itemsets, const std::vector<int>& reference,
                   int max_cols, int max_rows) {
  for (int i = 1; i < max_rows; ++i) {
    for (int j = 1; j < max_cols; ++j) {
      itemsets[i * max_cols + j] = maximum(
          itemsets[(i - 1) * max_cols + (j - 1)] + reference[i * max_cols + j],
          itemsets[i * max_cols + (j - 1)] - penalty,
          itemsets[(i - 1) * max_cols + j] - penalty);
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("Needleman-Wunsch: dimension=%d penalty=%d block_size=%d\n",
         dimension, penalty, BLOCK_SIZE);

  int max_rows = dimension + 1;
  int max_cols = dimension + 1;
  int matrix_size = max_rows * max_cols;

  // Deterministic input generation (matches Rodinia nw.c ordering).
  std::vector<int> reference(matrix_size, 0);
  std::vector<int> input_itemsets(matrix_size, 0);

  srand(7);
  for (int i = 1; i < max_rows; ++i)          // seed first column
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  for (int j = 1; j < max_cols; ++j)          // seed first row
    input_itemsets[j] = rand() % 10 + 1;

  // Substitution scores from the seeded border symbols.
  for (int i = 1; i < max_cols; ++i)
    for (int j = 1; j < max_rows; ++j)
      reference[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];

  // Overwrite borders with gap penalties (the DP boundary conditions).
  for (int i = 1; i < max_rows; ++i)
    input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; ++j)
    input_itemsets[j] = -j * penalty;

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  size_t nbytes = matrix_size * sizeof(int);
  reference_d = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  input_itemsets_d = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  output_itemsets_d = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  char options[64];
  snprintf(options, sizeof(options), "-D BLOCK_SIZE=%d", BLOCK_SIZE);
  CL_CHECK(clBuildProgram(program, 1, &device_id, options, NULL, NULL));

  kernel1 = CL_CHECK2(clCreateKernel(program, "nw_kernel1", &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, "nw_kernel2", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_itemsets_d, CL_TRUE, 0,
                                nbytes, input_itemsets.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, reference_d, CL_TRUE, 0,
                                nbytes, reference.data(), 0, NULL, NULL));

  int worksize = max_cols - 1;
  int offset_r = 0, offset_c = 0;
  int block_width = worksize / BLOCK_SIZE;

  auto set_common_args = [&](cl_kernel k) {
    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &reference_d));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &input_itemsets_d));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &output_itemsets_d));
    CL_CHECK(clSetKernelArg(k, 3, sizeof(int) * (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1), NULL));
    CL_CHECK(clSetKernelArg(k, 4, sizeof(int) * BLOCK_SIZE * BLOCK_SIZE, NULL));
    CL_CHECK(clSetKernelArg(k, 5, sizeof(int), &max_cols));
    CL_CHECK(clSetKernelArg(k, 6, sizeof(int), &penalty));
    CL_CHECK(clSetKernelArg(k, 8, sizeof(int), &block_width));
    CL_CHECK(clSetKernelArg(k, 9, sizeof(int), &worksize));
    CL_CHECK(clSetKernelArg(k, 10, sizeof(int), &offset_r));
    CL_CHECK(clSetKernelArg(k, 11, sizeof(int), &offset_c));
  };
  set_common_args(kernel1);
  set_common_args(kernel2);

  size_t local_work_size = BLOCK_SIZE;

  auto time_start = std::chrono::high_resolution_clock::now();

  // Upper-left triangle of blocks (growing anti-diagonals).
  for (int blk = 1; blk <= block_width; ++blk) {
    size_t global_work_size = (size_t)BLOCK_SIZE * blk;
    CL_CHECK(clSetKernelArg(kernel1, 7, sizeof(int), &blk));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }

  // Lower-right triangle of blocks (shrinking anti-diagonals).
  for (int blk = block_width - 1; blk >= 1; --blk) {
    size_t global_work_size = (size_t)BLOCK_SIZE * blk;
    CL_CHECK(clSetKernelArg(kernel2, 7, sizeof(int), &blk));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<int> h_gpu(matrix_size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, input_itemsets_d, CL_TRUE, 0,
                               nbytes, h_gpu.data(), 0, NULL, NULL));

  // CPU reference over the same seeded matrix.
  std::vector<int> h_ref(input_itemsets);
  nw_cpu(h_ref, reference, max_cols, max_rows);

  int errors = 0;
  for (int i = 0; i < matrix_size; ++i) {
    if (h_gpu[i] != h_ref[i]) {
      if (errors < 20)
        printf("*** error: [%d] expected=%d, actual=%d\n", i, h_ref[i], h_gpu[i]);
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
