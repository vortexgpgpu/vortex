// Heartwall (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Upstream heartwall tracks features across ultrasound video frames read from an
// AVI file. This port BYPASSES the AVI reader and avilib entirely: it synthesizes
// one small grayscale frame in-host (fixed seed) with bright feature blobs at
// known positions, then runs heartwall's CORE template-matching computation.
//
// Scope: the full upstream kernel is a single monolithic kernel with a fixed
// large work-group, tightly coupled to the AVI/frame pipeline — it cannot run at
// the device max work-group size of 16. This test therefore ports the core
// computational kernel of heartwall: template matching by rotation + full 2D
// convolution (= cross-correlation of the feature template with a search window).
// The rotation and convolution arithmetic are IDENTICAL to upstream's ROTATION
// and ACTUAL CONVOLUTION blocks (kernel_gpu_opencl.cl). The later normalization /
// argmax / position-update stages of heartwall are not included.
//
// For each feature the device produces the convolution (correlation) map; the
// result is checked element-by-element against a serial CPU reference that runs
// the identical rotation + convolution on the identical synthetic input.
//
// No atomics and no local memory are required (each work-item writes a distinct
// output element). Work-group size defaults to 16 (<= device max).

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
cl_kernel kernel_rotate = NULL;
cl_kernel kernel_convolve = NULL;
cl_mem d_in_all = NULL;
cl_mem d_in_mod_all = NULL;
cl_mem d_in2_all = NULL;
cl_mem d_conv_all = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel_rotate) clReleaseKernel(kernel_rotate);
  if (kernel_convolve) clReleaseKernel(kernel_convolve);
  if (program) clReleaseProgram(program);
  if (d_in_all) clReleaseMemObject(d_in_all);
  if (d_in_mod_all) clReleaseMemObject(d_in_mod_all);
  if (d_in2_all) clReleaseMemObject(d_in2_all);
  if (d_conv_all) clReleaseMemObject(d_conv_all);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (small by default so RTL simulation stays under budget).
// tSize/sSize are the template/search half-widths, exactly as upstream:
//   in_rows  = 2*tSize+1 (template),  in2_rows = 2*sSize+1 (search window).
// block_size is the OpenCL local work-group size; it must not exceed the device
// max work-group size (NUM_WARPS*NUM_THREADS = 16 in the default CI config).
static int tSize = 2;       // template half-width  -> 5x5 template
static int sSize = 4;       // search half-width    -> 9x9 window
static int num_features = 4;
static int block_size = 16;
static unsigned seed = 9;

static void show_usage() {
  printf("Usage: [-t tSize] [-s sSize] [-n num_features] [-b block_size] [-e seed] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:s:n:b:e:h")) != -1) {
    switch (c) {
    case 't': tSize = atoi(optarg); break;
    case 's': sSize = atoi(optarg); break;
    case 'n': num_features = atoi(optarg); break;
    case 'b': block_size = atoi(optarg); break;
    case 'e': seed = (unsigned)atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (tSize < 1 || sSize < tSize || num_features < 1 || block_size < 1 || block_size > 16) {
    printf("Error: invalid parameters (need tSize>=1, sSize>=tSize, features>=1, 1<=block_size<=16)\n");
    exit(-1);
  }
}

// 180-degree rotation of one feature template (column-major), matching upstream.
static void rotate_cpu(const float* d_in, float* d_in_mod, int in_rows, int in_elem) {
  for (int ei_new = 0; ei_new < in_elem; ++ei_new) {
    int row = (ei_new + 1) % in_rows - 1;
    int col = (ei_new + 1) / in_rows + 1 - 1;
    if ((ei_new + 1) % in_rows == 0) {
      row = in_rows - 1;
      col = col - 1;
    }
    int rot_row = (in_rows - 1) - row;
    int rot_col = (in_rows - 1) - col;
    d_in_mod[ei_new] = d_in[rot_col * in_rows + rot_row];
  }
}

// Full 2D convolution of the rotated template against the search window, matching
// upstream's ACTUAL CONVOLUTION block exactly (ioffset = joffset = 0).
static void convolve_cpu(const float* d_in_mod, const float* d_in2, float* d_conv,
                         int in_rows, int in_cols, int in2_rows, int in2_cols,
                         int conv_rows, int conv_elem) {
  const int ioffset = 0, joffset = 0;
  for (int ei_new = 0; ei_new < conv_elem; ++ei_new) {
    int ic = (ei_new + 1) % conv_rows;
    int jc = (ei_new + 1) / conv_rows + 1;
    if ((ei_new + 1) % conv_rows == 0) {
      ic = conv_rows;
      jc = jc - 1;
    }
    int j = jc + joffset;
    int jp1 = j + 1;
    int ja1 = (in2_cols < jp1) ? (jp1 - in2_cols) : 1;
    int ja2 = (in_cols < j) ? in_cols : j;

    int i = ic + ioffset;
    int ip1 = i + 1;
    int ia1 = (in2_rows < ip1) ? (ip1 - in2_rows) : 1;
    int ia2 = (in_rows < i) ? in_rows : i;

    float s = 0;
    for (int ja = ja1; ja <= ja2; ja++) {
      int jb = jp1 - ja;
      for (int ia = ia1; ia <= ia2; ia++) {
        int ib = ip1 - ia;
        s += d_in_mod[in_rows * (ja - 1) + ia - 1] * d_in2[in2_rows * (jb - 1) + ib - 1];
      }
    }
    d_conv[ei_new] = s;
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // Derived sizes (identical relationships to upstream kernel_gpu_opencl_wrapper.c).
  int in_rows = 2 * tSize + 1, in_cols = in_rows;
  int in_elem = in_rows * in_cols;
  int in2_rows = 2 * sSize + 1, in2_cols = in2_rows;
  int in2_elem = in2_rows * in2_cols;
  int conv_rows = in_rows + in2_rows - 1;
  int conv_cols = in_cols + in2_cols - 1;
  int conv_elem = conv_rows * conv_cols;
  int allPoints = num_features;

  printf("Heartwall: tSize=%d sSize=%d features=%d block_size=%d\n",
         tSize, sSize, allPoints, block_size);
  printf("  template=%dx%d search=%dx%d conv=%dx%d\n",
         in_rows, in_cols, in2_rows, in2_cols, conv_rows, conv_cols);

  // ----- Synthesize a small grayscale frame (deterministic, no AVI) -----------
  // Lay features on a near-square grid with >= sSize margin so each search
  // window is fully in-bounds. Background is low-level pseudo-random noise; a
  // bright Gaussian-like blob marks each feature center (a trackable feature).
  int gw = (int)ceil(sqrt((double)allPoints));
  int gh = (allPoints + gw - 1) / gw;
  int spacing = 2 * sSize + 2;
  int frame_rows = spacing * gh + 2 * sSize;
  int frame_cols = spacing * gw + 2 * sSize;
  std::vector<float> frame(frame_rows * frame_cols);

  srand(seed);
  for (int r = 0; r < frame_rows; ++r)
    for (int c = 0; c < frame_cols; ++c)
      frame[r * frame_cols + c] = (float)(rand() % 64) / 255.0f;  // background in [0, ~0.25]

  std::vector<int> feat_r(allPoints), feat_c(allPoints);
  for (int f = 0; f < allPoints; ++f) {
    int gr = f / gw, gc = f % gw;
    int cr = sSize + spacing / 2 + gr * spacing;
    int cc = sSize + spacing / 2 + gc * spacing;
    feat_r[f] = cr;
    feat_c[f] = cc;
    // stamp a bright blob (peak + falloff) so the feature is distinctive
    for (int dr = -sSize; dr <= sSize; ++dr) {
      for (int dc = -sSize; dc <= sSize; ++dc) {
        int rr = cr + dr, ccl = cc + dc;
        float d2 = (float)(dr * dr + dc * dc);
        float val = 0.9f * expf(-d2 / (2.0f * (float)tSize * (float)tSize + 1.0f));
        frame[rr * frame_cols + ccl] += val;
      }
    }
  }

  // ----- Extract per-feature template (in) and search window (in2) ------------
  // Both are ROIs centered on the feature point, stored column-major per feature
  // (element (row,col) of an R-row matrix at col*R+row), matching kernel layout.
  std::vector<float> h_in(allPoints * in_elem);
  std::vector<float> h_in2(allPoints * in2_elem);
  for (int f = 0; f < allPoints; ++f) {
    int cr = feat_r[f], cc = feat_c[f];
    for (int col = 0; col < in_cols; ++col)
      for (int row = 0; row < in_rows; ++row) {
        int fr = cr - tSize + row, fc = cc - tSize + col;
        h_in[f * in_elem + col * in_rows + row] = frame[fr * frame_cols + fc];
      }
    for (int col = 0; col < in2_cols; ++col)
      for (int row = 0; row < in2_rows; ++row) {
        int fr = cr - sSize + row, fc = cc - sSize + col;
        h_in2[f * in2_elem + col * in2_rows + row] = frame[fr * frame_cols + fc];
      }
  }

  // ----- OpenCL setup ---------------------------------------------------------
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_in_all = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(float) * allPoints * in_elem, NULL, &_err));
  d_in_mod_all = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(float) * allPoints * in_elem, NULL, &_err));
  d_in2_all = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       sizeof(float) * allPoints * in2_elem, NULL, &_err));
  d_conv_all = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        sizeof(float) * allPoints * conv_elem, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel_rotate = CL_CHECK2(clCreateKernel(program, "rotate_template", &_err));
  kernel_convolve = CL_CHECK2(clCreateKernel(program, "convolve", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_in_all, CL_TRUE, 0,
                                sizeof(float) * allPoints * in_elem, h_in.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_in2_all, CL_TRUE, 0,
                                sizeof(float) * allPoints * in2_elem, h_in2.data(), 0, NULL, NULL));

  size_t local_work_size = (size_t)block_size;
  auto round_up = [](size_t total, size_t local) {
    return ((total + local - 1) / local) * local;
  };

  auto time_start = std::chrono::high_resolution_clock::now();

  // Pass 1: rotate every feature template 180 degrees.
  {
    int total = allPoints * in_elem;
    size_t global_work_size = round_up(total, local_work_size);
    CL_CHECK(clSetKernelArg(kernel_rotate, 0, sizeof(cl_mem), &d_in_all));
    CL_CHECK(clSetKernelArg(kernel_rotate, 1, sizeof(cl_mem), &d_in_mod_all));
    CL_CHECK(clSetKernelArg(kernel_rotate, 2, sizeof(int), &in_rows));
    CL_CHECK(clSetKernelArg(kernel_rotate, 3, sizeof(int), &in_elem));
    CL_CHECK(clSetKernelArg(kernel_rotate, 4, sizeof(int), &allPoints));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_rotate, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }

  // Pass 2: full 2D convolution (cross-correlation map) per feature.
  {
    int total = allPoints * conv_elem;
    size_t global_work_size = round_up(total, local_work_size);
    int ioffset = 0, joffset = 0;
    CL_CHECK(clSetKernelArg(kernel_convolve, 0, sizeof(cl_mem), &d_in_mod_all));
    CL_CHECK(clSetKernelArg(kernel_convolve, 1, sizeof(cl_mem), &d_in2_all));
    CL_CHECK(clSetKernelArg(kernel_convolve, 2, sizeof(cl_mem), &d_conv_all));
    CL_CHECK(clSetKernelArg(kernel_convolve, 3, sizeof(int), &in_rows));
    CL_CHECK(clSetKernelArg(kernel_convolve, 4, sizeof(int), &in_cols));
    CL_CHECK(clSetKernelArg(kernel_convolve, 5, sizeof(int), &in2_rows));
    CL_CHECK(clSetKernelArg(kernel_convolve, 6, sizeof(int), &in2_cols));
    CL_CHECK(clSetKernelArg(kernel_convolve, 7, sizeof(int), &conv_rows));
    CL_CHECK(clSetKernelArg(kernel_convolve, 8, sizeof(int), &conv_elem));
    CL_CHECK(clSetKernelArg(kernel_convolve, 9, sizeof(int), &in_elem));
    CL_CHECK(clSetKernelArg(kernel_convolve, 10, sizeof(int), &in2_elem));
    CL_CHECK(clSetKernelArg(kernel_convolve, 11, sizeof(int), &ioffset));
    CL_CHECK(clSetKernelArg(kernel_convolve, 12, sizeof(int), &joffset));
    CL_CHECK(clSetKernelArg(kernel_convolve, 13, sizeof(int), &allPoints));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_convolve, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<float> h_conv(allPoints * conv_elem);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_conv_all, CL_TRUE, 0,
                               sizeof(float) * allPoints * conv_elem, h_conv.data(), 0, NULL, NULL));

  // ----- CPU golden reference (identical math, serial) ------------------------
  std::vector<float> ref_in_mod(in_elem);
  std::vector<float> ref_conv(allPoints * conv_elem);
  for (int f = 0; f < allPoints; ++f) {
    rotate_cpu(&h_in[f * in_elem], ref_in_mod.data(), in_rows, in_elem);
    convolve_cpu(ref_in_mod.data(), &h_in2[f * in2_elem], &ref_conv[f * conv_elem],
                 in_rows, in_cols, in2_rows, in2_cols, conv_rows, conv_elem);
  }

  int errors = 0;
  for (int idx = 0; idx < allPoints * conv_elem; ++idx) {
    float g = h_conv[idx], r = ref_conv[idx];
    if (fabsf(g - r) > 1e-3f + 1e-4f * fabsf(r)) {
      if (errors < 20)
        printf("*** error: feature=%d elem=%d expected=%f actual=%f\n",
               idx / conv_elem, idx % conv_elem, r, g);
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
