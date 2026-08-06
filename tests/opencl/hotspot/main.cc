// HotSpot (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// 2-D transient thermal stencil. Each cell's temperature evolves from its four
// neighbours (Neumann/clamped boundaries), the local power dissipation and an
// ambient-coupling term. The GPU version uses pyramid/ghost-zone blocking; the
// result is checked against a serial CPU transient-thermal reference running the
// identical update math over the same (seeded) temp/power grids.
//
// Unlike stock HotSpot, the temp/power inputs are generated deterministically
// in-host (no external data files).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

#define MIN(a, b) ((a) <= (b) ? (a) : (b))

// HotSpot chip/model constants (from Rodinia hotspot.h).
#define EXPAND_RATE  2        // pyramid base grows by 2 per borderline per iter
#define MAX_PD       (3.0e6f) // max power density (W)
#define PRECISION    0.001f   // required precision (degrees)
#define SPEC_HEAT_SI 1.75e6f
#define K_SI         100.0f
#define FACTOR_CHIP  0.5f     // capacitance fitting factor
static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;  // ambient temperature (matches kernel)

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
cl_kernel kernel = NULL;
cl_mem d_power = NULL;
cl_mem d_temp[2] = {NULL, NULL};
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (d_power) clReleaseMemObject(d_power);
  if (d_temp[0]) clReleaseMemObject(d_temp[0]);
  if (d_temp[1]) clReleaseMemObject(d_temp[1]);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (small by default so RTL simulation stays under budget).
// block_size is the OpenCL local work-group edge; the kernel launches a 2-D
// block_size x block_size work-group, which must not exceed the device max
// work-group size (NUM_WARPS*NUM_THREADS = 16 in the default CI config). Hence
// block_size = 4 -> 4x4 = 16. With EXPAND_RATE=2 the usable (non-halo) block is
// block_size - pyramid_height*EXPAND_RATE, so pyramid_height must stay below
// block_size/2; at block_size=4 that forces pyramid_height=1.
static int grid_rows = 32;
static int grid_cols = 32;
static int pyramid_height = 1;
static int total_iterations = 4;
static int block_size = 4;

static void show_usage() {
  printf("Usage: [-r rows] [-c cols] [-y pyramid_height] [-i iterations] [-b block_size] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:y:i:b:h")) != -1) {
    switch (c) {
    case 'r': grid_rows = atoi(optarg); break;
    case 'c': grid_cols = atoi(optarg); break;
    case 'y': pyramid_height = atoi(optarg); break;
    case 'i': total_iterations = atoi(optarg); break;
    case 'b': block_size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (grid_rows < 2 || grid_cols < 2 || pyramid_height < 1 ||
      total_iterations < 1 || block_size < 2 ||
      block_size - pyramid_height * EXPAND_RATE < 1) {
    printf("Error: invalid parameters (need block_size > pyramid_height*EXPAND_RATE)\n");
    exit(-1);
  }
}

// Serial CPU transient-thermal reference. Applies the same per-cell stencil as
// the kernel for `iters` full-grid time steps, using clamped (Neumann) boundary
// neighbours -- identical to the kernel's valid-range neighbour clamping.
static void hotspot_cpu(std::vector<float>& out,
                        const std::vector<float>& temp_in,
                        const std::vector<float>& power,
                        int row, int col, int iters,
                        float Cap, float Rx, float Ry, float Rz, float step) {
  float step_div_Cap = step / Cap;
  float Rx_1 = 1.0f / Rx, Ry_1 = 1.0f / Ry, Rz_1 = 1.0f / Rz;
  std::vector<float> src(temp_in), dst(row * col);
  for (int it = 0; it < iters; ++it) {
    for (int r = 0; r < row; ++r) {
      for (int c = 0; c < col; ++c) {
        int N = (r > 0) ? r - 1 : 0;
        int S = (r < row - 1) ? r + 1 : row - 1;
        int W = (c > 0) ? c - 1 : 0;
        int E = (c < col - 1) ? c + 1 : col - 1;
        float cur = src[r * col + c];
        dst[r * col + c] = cur + step_div_Cap * (power[r * col + c] +
            (src[S * col + c] + src[N * col + c] - 2.0f * cur) * Ry_1 +
            (src[r * col + E] + src[r * col + W] - 2.0f * cur) * Rx_1 +
            (amb_temp - cur) * Rz_1);
      }
    }
    src.swap(dst);
  }
  out.swap(src);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("HotSpot: rows=%d cols=%d pyramid_height=%d iterations=%d block_size=%d\n",
         grid_rows, grid_cols, pyramid_height, total_iterations, block_size);

  int size = grid_rows * grid_cols;

  // Generate deterministic temp/power grids (fixed seed so host and device
  // see identical inputs). temp in ~300-324 K, power in ~0-1.
  std::vector<float> h_temp(size), h_power(size);
  srand(7);
  for (int i = 0; i < size; ++i) {
    h_temp[i] = 300.0f + (rand() % 250) * 0.1f;  // 300.0 .. 324.9
    h_power[i] = (rand() % 1000) * 0.001f;        // 0.000 .. 0.999
  }

  // Thermal model coefficients (computed once, shared by device and reference).
  float grid_height = chip_height / grid_rows;
  float grid_width = chip_width / grid_cols;
  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_power   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * size, NULL, &_err));
  d_temp[0] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &_err));
  d_temp[1] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // BLOCK_SIZE is a compile-time constant in the kernel; pass it via -D so the
  // local-memory tiles match the host's launch geometry.
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-DBLOCK_SIZE=%d", block_size);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, "hotspot", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Upload inputs; d_temp[0] is the initial temperature, d_power is constant.
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_temp[0], CL_TRUE, 0,
                                sizeof(float) * size, h_temp.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_power, CL_TRUE, 0,
                                sizeof(float) * size, h_power.data(), 0, NULL, NULL));

  // Pyramid tiling parameters.
  int borderCols = pyramid_height * EXPAND_RATE / 2;
  int borderRows = pyramid_height * EXPAND_RATE / 2;
  int smallBlockCol = block_size - pyramid_height * EXPAND_RATE;
  int smallBlockRow = block_size - pyramid_height * EXPAND_RATE;
  int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
  int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

  size_t local_work_size[2]  = {(size_t)block_size, (size_t)block_size};
  size_t global_work_size[2] = {(size_t)block_size * blockCols,
                                (size_t)block_size * blockRows};

  auto time_start = std::chrono::high_resolution_clock::now();
  int src = 0, dst = 1;
  for (int t = 0; t < total_iterations; t += pyramid_height) {
    int iter = MIN(pyramid_height, total_iterations - t);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), &iter));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_power));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_temp[src]));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_temp[dst]));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &grid_cols));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &grid_rows));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &borderCols));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &borderRows));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float), &Cap));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(float), &Rx));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(float), &Ry));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float), &Rz));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float), &step));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                    global_work_size, local_work_size, 0, NULL, NULL));
    src = 1 - src;
    dst = 1 - dst;
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Final result lives in d_temp[src] after the ping-pong.
  std::vector<float> h_gpu(size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_temp[src], CL_TRUE, 0,
                               sizeof(float) * size, h_gpu.data(), 0, NULL, NULL));

  // CPU reference over the same inputs and iteration count.
  std::vector<float> h_ref;
  hotspot_cpu(h_ref, h_temp, h_power, grid_rows, grid_cols, total_iterations,
              Cap, Rx, Ry, Rz, step);

  const float tol = 1e-3f;
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    if (fabsf(h_gpu[i] - h_ref[i]) > tol) {
      if (errors < 20)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, h_ref[i], h_gpu[i]);
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
