// Hotspot3D (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// 3-D transient thermal stencil: each cell's next temperature is a weighted
// combination of its six neighbours (clamped at the chip boundaries) plus a
// per-cell power dissipation term and a fixed ambient contribution. The GPU
// kernel `hotspotOpt1` advances one time-step per launch, sweeping the z
// layers internally; the host loops it `iterations` times. Correctness is
// checked against a serial CPU reference running the identical stencil over the
// same deterministically-generated power/temperature grids (no external files).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

// Physical constants (from Rodinia hotspot3D).
#define MAX_PD       (3.0e6)
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100
#define FACTOR_CHIP  0.5
#define T_CHIP       0.0005f
#define CHIP_HEIGHT  0.016f
#define CHIP_WIDTH   0.016f
#define AMB_TEMP     80.0f

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

// Workload parameters (tiny by default so RTL simulation stays under budget).
// The 2-D work-group is block x block; it must not exceed the device max
// work-group size (NUM_WARPS*NUM_THREADS = 16 in the default CI config), hence
// block=4 (16 work-items) by default.
static int nx = 16;   // rows/cols (x and y are square)
static int ny = 16;
static int nz = 4;    // layers
static int iterations = 2;
static int block = 4; // 2-D local size = block*block

static void show_usage() {
  printf("Usage: [-n rows/cols] [-l layers] [-i iterations] [-b block] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:l:i:b:h")) != -1) {
    switch (c) {
    case 'n': nx = ny = atoi(optarg); break;
    case 'l': nz = atoi(optarg); break;
    case 'i': iterations = atoi(optarg); break;
    case 'b': block = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (nx < 2 || ny < 2 || nz < 2 || iterations < 1 || block < 1) {
    printf("Error: invalid parameters (need nx,ny,nz>=2, iterations>=1, block>=1)\n");
    exit(-1);
  }
  if (block * block > 16) {
    printf("Error: block*block=%d exceeds device max work-group size (16)\n", block * block);
    exit(-1);
  }
  if ((nx % block) != 0 || (ny % block) != 0) {
    printf("Error: nx(%d) and ny(%d) must be multiples of block(%d)\n", nx, ny, block);
    exit(-1);
  }
}

// Serial CPU reference: identical six-neighbour stencil, boundary-clamped,
// advanced `iterations` time-steps with ping-pong buffers.
static void hotspot_cpu(const std::vector<float>& power,
                        std::vector<float> tIn, std::vector<float>& tOut,
                        float sdc, float ce, float cw, float cn, float cs,
                        float ct, float cb, float cc) {
  int xy = nx * ny;
  std::vector<float> a = tIn;
  std::vector<float> b(a.size());
  for (int it = 0; it < iterations; ++it) {
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          int c = x + y * nx + z * xy;
          int w = (x == 0)      ? c : c - 1;
          int e = (x == nx - 1) ? c : c + 1;
          int n = (y == 0)      ? c : c - nx;
          int s = (y == ny - 1) ? c : c + nx;
          int bo = (z == 0)      ? c : c - xy;
          int to = (z == nz - 1) ? c : c + xy;
          b[c] = cc * a[c] + cw * a[w] + ce * a[e] + cs * a[s]
               + cn * a[n] + cb * a[bo] + ct * a[to] + sdc * power[c] + ct * AMB_TEMP;
        }
    a.swap(b);
  }
  tOut.swap(a);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("Hotspot3D: nx=%d ny=%d nz=%d iterations=%d block=%dx%d (wg=%d)\n",
         nx, ny, nz, iterations, block, block, block * block);

  int size = nx * ny * nz;

  // Derive the stencil coefficients exactly as the Rodinia host does.
  float dx = CHIP_HEIGHT / nx;
  float dy = CHIP_WIDTH  / ny;
  float dz = T_CHIP / nz;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * dx * dy;
  float Rx  = dy / (2.0f * K_SI * T_CHIP * dx);
  float Ry  = dx / (2.0f * K_SI * T_CHIP * dy);
  float Rz  = dz / (K_SI * dx * dy);

  float max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);
  float dt = PRECISION / max_slope;

  float stepDivCap = dt / Cap;
  float ce = stepDivCap / Rx, cw = ce;
  float cn = stepDivCap / Ry, cs = cn;
  float ct = stepDivCap / Rz, cb = ct;
  float cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);

  // Deterministic input grids: temperature ~300-324 K, power small [0,1).
  std::vector<float> h_power(size);
  std::vector<float> h_temp(size);
  srand(7);
  for (int i = 0; i < size; ++i) {
    h_temp[i]  = 300.0f + (rand() % 25000) / 1000.0f;   // 300.000 .. 324.999
    h_power[i] = (rand() % 1000) / 1000.0f;             // 0.000 .. 0.999
  }

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_power  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * size, NULL, &_err));
  d_temp[0] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &_err));
  d_temp[1] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, "hotspotOpt1", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Upload power and the initial temperature (into buffer 0).
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_power, CL_TRUE, 0,
                                sizeof(float) * size, h_power.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_temp[0], CL_TRUE, 0,
                                sizeof(float) * size, h_temp.data(), 0, NULL, NULL));

  size_t global[2] = {(size_t)nx, (size_t)ny};
  size_t local[2]  = {(size_t)block, (size_t)block};

  auto time_start = std::chrono::high_resolution_clock::now();
  int in = 0;  // ping-pong: kernel reads d_temp[in], writes d_temp[out]
  for (int it = 0; it < iterations; ++it) {
    int out = 1 - in;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_power));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_temp[in]));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_temp[out]));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float), &stepDivCap));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &nx));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ny));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &nz));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(float), &ce));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float), &cw));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(float), &cn));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(float), &cs));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float), &ct));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float), &cb));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float), &cc));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                    global, local, 0, NULL, NULL));
    in = out;  // result of this step becomes next step's input
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Final result is in d_temp[in].
  std::vector<float> h_gpu(size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_temp[in], CL_TRUE, 0,
                               sizeof(float) * size, h_gpu.data(), 0, NULL, NULL));

  // CPU reference over the identical grids.
  std::vector<float> h_ref(size);
  hotspot_cpu(h_power, h_temp, h_ref, stepDivCap, ce, cw, cn, cs, ct, cb, cc);

  // Compare with a float tolerance (the kernel and reference sum the same terms
  // in a different order, so results match only to within fp rounding).
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float a = h_ref[i], b = h_gpu[i];
    float diff = fabsf(a - b);
    float tol = 1e-3f * fabsf(a) + 1e-3f;
    if (diff > tol) {
      if (errors < 20)
        printf("*** error: [%d] expected=%f, actual=%f (diff=%g)\n", i, a, b, diff);
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
