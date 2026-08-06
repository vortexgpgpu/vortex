// PathFinder (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Dynamic-programming shortest accumulated-weight path from the bottom row to
// the top row of a 2-D integer wall, moving straight or diagonally ahead. The
// GPU version uses pyramid/ghost-zone blocking; correctness is checked against
// a serial CPU dynamic-programming reference over the identical (seeded) wall.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

#define HALO 1
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                              \
    if (_err == CL_SUCCESS)                                           \
      break;                                                          \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
    cleanup();                                                        \
    exit(-1);                                                         \
  } while (0)

#define CL_CHECK2(_expr)                                              \
  ({                                                                  \
    cl_int _err = CL_INVALID_VALUE;                                   \
    decltype(_expr) _ret = _expr;                                    \
    if (_err != CL_SUCCESS) {                                        \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);\
      cleanup();                                                     \
      exit(-1);                                                      \
    }                                                                 \
    _ret;                                                             \
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
cl_kernel kernel = NULL;
cl_mem d_gpuWall = NULL;
cl_mem d_gpuResult[2] = {NULL, NULL};
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (d_gpuWall) clReleaseMemObject(d_gpuWall);
  if (d_gpuResult[0]) clReleaseMemObject(d_gpuResult[0]);
  if (d_gpuResult[1]) clReleaseMemObject(d_gpuResult[1]);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (small by default so RTL simulation stays under budget).
// block_size is the OpenCL local work-group size; it must not exceed the
// device's max work-group size (NUM_WARPS*NUM_THREADS = 16 in the default CI
// config), hence the modest defaults below.
static int rows = 16;
static int cols = 64;
static int pyramid_height = 2;
static int block_size = 16;

static void show_usage() {
  printf("Usage: [-r rows] [-c cols] [-y pyramid_height] [-b block_size] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:y:b:h")) != -1) {
    switch (c) {
    case 'r': rows = atoi(optarg); break;
    case 'c': cols = atoi(optarg); break;
    case 'y': pyramid_height = atoi(optarg); break;
    case 'b': block_size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (rows < 2 || cols < 2 || pyramid_height < 1 || block_size < 2 * pyramid_height * HALO + 2) {
    printf("Error: invalid parameters (block_size must exceed 2*pyramid_height*HALO)\n");
    exit(-1);
  }
}

// Serial CPU dynamic-programming reference over the same wall.
static void pathfinder_cpu(std::vector<int>& result, const std::vector<int>& wall) {
  std::vector<int> src(result);  // row 0 already in result
  std::vector<int> dst(cols);
  for (int t = 0; t < rows - 1; ++t) {
    for (int n = 0; n < cols; ++n) {
      int up = src[n];
      int left = (n > 0) ? src[n - 1] : up;
      int right = (n < cols - 1) ? src[n + 1] : up;
      int shortest = MIN(up, MIN(left, right));
      dst[n] = shortest + wall[(t + 1) * cols + n];
    }
    src.swap(dst);
  }
  result.swap(src);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("PathFinder: rows=%d cols=%d pyramid_height=%d block_size=%d\n",
         rows, cols, pyramid_height, block_size);

  int size = rows * cols;

  // Generate the wall (deterministic seed so host and device see the same data).
  std::vector<int> wall(size);
  srand(9);
  for (int i = 0; i < size; ++i)
    wall[i] = rand() % 10;

  // Row 0 is the DP seed.
  std::vector<int> h_result(wall.begin(), wall.begin() + cols);

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_gpuWall = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       sizeof(int) * (size - cols), NULL, &_err));
  d_gpuResult[0] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(int) * cols, NULL, &_err));
  d_gpuResult[1] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(int) * cols, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, "dynproc_kernel", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Upload wall rows 1.. and the row-0 seed.
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_gpuWall, CL_TRUE, 0,
                                sizeof(int) * (size - cols), wall.data() + cols, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_gpuResult[0], CL_TRUE, 0,
                                sizeof(int) * cols, h_result.data(), 0, NULL, NULL));

  int borderCols = pyramid_height * HALO;
  int smallBlockCol = block_size - pyramid_height * HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  size_t local_work_size = block_size;
  size_t global_work_size = (size_t)blockCols * block_size;

  auto time_start = std::chrono::high_resolution_clock::now();
  int src = 1, final_ret = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    int temp = src;
    src = final_ret;
    final_ret = temp;
    int iteration = MIN(pyramid_height, rows - t - 1);
    int theHalo = HALO;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), &iteration));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_gpuWall));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_gpuResult[src]));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_gpuResult[final_ret]));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &cols));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &rows));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &t));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &borderCols));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &theHalo));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int) * block_size, NULL));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int) * block_size, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<int> h_gpu(cols);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_gpuResult[final_ret], CL_TRUE, 0,
                               sizeof(int) * cols, h_gpu.data(), 0, NULL, NULL));

  // CPU reference.
  std::vector<int> h_ref(wall.begin(), wall.begin() + cols);
  pathfinder_cpu(h_ref, wall);

  int errors = 0;
  for (int i = 0; i < cols; ++i) {
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
