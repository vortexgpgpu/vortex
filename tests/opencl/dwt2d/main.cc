// dwt2d (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Forward 5/3 reversible (integer) discrete wavelet transform of a synthetic
// single-component image. The Rodinia benchmark reads a .bmp and runs a tiled
// 256-thread sliding-window kernel; here the image is generated in-host from a
// fixed seed (no external files) and the transform uses a simple non-tiled
// row/column 5/3 lifting kernel that fits the device work-group limit of 16.
//
// The 5/3 lifting scheme is exact integer arithmetic, so the device output is
// compared BIT-EXACTLY against a serial CPU reference that runs the identical
// lifting. One DWT level = one horizontal (row) pass + one vertical (column)
// pass; both use symmetric (mirror) boundary extension.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

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
cl_kernel kernel_h = NULL;
cl_kernel kernel_v = NULL;
cl_mem d_data = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel_h) clReleaseKernel(kernel_h);
  if (kernel_v) clReleaseKernel(kernel_v);
  if (program) clReleaseProgram(program);
  if (d_data) clReleaseMemObject(d_data);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Image size (N x N) and DWT levels. Kept tiny by default so RTL simulation
// stays under budget. block_size is the OpenCL local work-group size and must
// not exceed the device max (NUM_WARPS*NUM_THREADS = 16 in the default config).
static int N = 16;
static int levels = 1;

static void show_usage() {
  printf("Usage: [-n size] [-l levels] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:l:h")) != -1) {
    switch (c) {
    case 'n': N = atoi(optarg); break;
    case 'l': levels = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  // 5/3 separable lifting needs an even, non-trivial dimension.
  if (N < 2 || (N & 1) || levels < 1) {
    printf("Error: size must be even and >= 2, levels >= 1\n");
    exit(-1);
  }
}

// Serial 1-D forward 5/3 lifting, identical to the device lift53().
static void lift53_cpu(int* a, int n, int stride) {
  for (int i = 1; i < n; i += 2) {
    int prev = a[(i - 1) * stride];
    int next = (i + 1 < n) ? a[(i + 1) * stride] : a[(i - 1) * stride];
    a[i * stride] -= (prev + next) / 2;
  }
  for (int i = 0; i < n; i += 2) {
    int prev = (i - 1 >= 0) ? a[(i - 1) * stride] : a[(i + 1) * stride];
    int next = (i + 1 < n) ? a[(i + 1) * stride] : a[(i - 1) * stride];
    a[i * stride] += (prev + next + 2) / 4;
  }
}

// CPU golden: one level = horizontal (rows) then vertical (columns).
static void fdwt53_cpu(std::vector<int>& img, int n) {
  for (int r = 0; r < n; ++r)
    lift53_cpu(img.data() + r * n, n, 1);
  for (int c = 0; c < n; ++c)
    lift53_cpu(img.data() + c, n, n);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // Work-group size = image dimension, capped at the device limit (16).
  int block_size = (N <= 16) ? N : 16;
  if (N % block_size != 0) {
    printf("Error: size must be a multiple of the work-group size (%d)\n", block_size);
    exit(-1);
  }
  printf("dwt2d: image=%dx%d levels=%d block_size=%d\n", N, N, levels, block_size);

  int size = N * N;

  // Generate a synthetic single-component image (deterministic seed). Pixel
  // bytes are centered to signed samples (byte - 128), matching Rodinia's
  // component preprocessing and exercising negative-value lifting divisions.
  std::vector<int> h_img(size);
  srand(9);
  for (int i = 0; i < size; ++i)
    h_img[i] = (rand() % 256) - 128;

  std::vector<int> h_ref(h_img);  // copy for the CPU reference

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_data = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * size, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // Pass the work-group size to the kernel to match the host launch config.
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-DBLOCK_SIZE=%d", block_size);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));

  kernel_h = CL_CHECK2(clCreateKernel(program, "fdwt53_horizontal", &_err));
  kernel_v = CL_CHECK2(clCreateKernel(program, "fdwt53_vertical", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  size_t global_work_size = (size_t)N;
  size_t local_work_size = (size_t)block_size;

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int lvl = 0; lvl < levels; ++lvl) {
    // Re-upload the current image for each 2-D level (single-level default).
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_data, CL_TRUE, 0,
                                  sizeof(int) * size, h_img.data(), 0, NULL, NULL));

    // Horizontal (row) pass.
    CL_CHECK(clSetKernelArg(kernel_h, 0, sizeof(cl_mem), &d_data));
    CL_CHECK(clSetKernelArg(kernel_h, 1, sizeof(int), &N));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_h, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));

    // Vertical (column) pass.
    CL_CHECK(clSetKernelArg(kernel_v, 0, sizeof(cl_mem), &d_data));
    CL_CHECK(clSetKernelArg(kernel_v, 1, sizeof(int), &N));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_v, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));

    // Read back the transformed coefficients (fed into the next level, if any).
    CL_CHECK(clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
                                 sizeof(int) * size, h_img.data(), 0, NULL, NULL));

    // Matching CPU reference for this level.
    fdwt53_cpu(h_ref, N);
  }
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Exact integer comparison (5/3 transform is reversible / bit-exact).
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    if (h_img[i] != h_ref[i]) {
      if (errors < 20)
        printf("*** error: [%d] expected=%d, actual=%d\n", i, h_ref[i], h_img[i]);
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
