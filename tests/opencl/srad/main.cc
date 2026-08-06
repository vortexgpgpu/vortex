// SRAD (Speckle Reducing Anisotropic Diffusion, Rodinia) — standalone
// self-checking OpenCL port for Vortex.
//
// SRAD is an iterative edge-preserving denoiser. Each iteration:
//   1. compute image statistics (mean/variance of the ROI) -> speckle scale q0sqr,
//   2. srad  : per-pixel N/S/W/E derivatives + diffusion coefficient c,
//   3. srad2 : divergence of the flux and the diffusion image update.
// An extract (log-uncompress) precedes the loop and a compress (log-recompress)
// follows it. This port runs the non-tiled srad/srad2 kernels on the device and
// computes the statistics reduction on the host (no shared memory / atomics).
// The result image is checked against a serial CPU reference running the exact
// same single-precision math over the identical seeded input.

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
cl_kernel extract_kernel = NULL;
cl_kernel srad_kernel = NULL;
cl_kernel srad2_kernel = NULL;
cl_kernel compress_kernel = NULL;
cl_mem d_I = NULL;
cl_mem d_iN = NULL, d_iS = NULL, d_jE = NULL, d_jW = NULL;
cl_mem d_dN = NULL, d_dS = NULL, d_dE = NULL, d_dW = NULL;
cl_mem d_c = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (extract_kernel) clReleaseKernel(extract_kernel);
  if (srad_kernel) clReleaseKernel(srad_kernel);
  if (srad2_kernel) clReleaseKernel(srad2_kernel);
  if (compress_kernel) clReleaseKernel(compress_kernel);
  if (program) clReleaseProgram(program);
  if (d_I) clReleaseMemObject(d_I);
  if (d_iN) clReleaseMemObject(d_iN);
  if (d_iS) clReleaseMemObject(d_iS);
  if (d_jE) clReleaseMemObject(d_jE);
  if (d_jW) clReleaseMemObject(d_jW);
  if (d_dN) clReleaseMemObject(d_dN);
  if (d_dS) clReleaseMemObject(d_dS);
  if (d_dE) clReleaseMemObject(d_dE);
  if (d_dW) clReleaseMemObject(d_dW);
  if (d_c) clReleaseMemObject(d_c);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (tiny by default so RTL simulation stays under budget).
// local_size is the OpenCL work-group size and must not exceed the device's max
// work-group size (NUM_WARPS*NUM_THREADS = 16 in the default CI config).
static int Nr = 32;          // image rows
static int Nc = 32;          // image cols
static int niter = 2;        // SRAD iterations
static float lambda = 0.5f;  // update step size
static int local_size = 16;  // work-group size (<= 16)

static void show_usage() {
  printf("Usage: [-r rows] [-c cols] [-n niter] [-l lambda] [-b local_size] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:n:l:b:h")) != -1) {
    switch (c) {
    case 'r': Nr = atoi(optarg); break;
    case 'c': Nc = atoi(optarg); break;
    case 'n': niter = atoi(optarg); break;
    case 'l': lambda = atof(optarg); break;
    case 'b': local_size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (Nr < 2 || Nc < 2 || niter < 1 || local_size < 1 || local_size > 16) {
    printf("Error: invalid parameters (need Nr,Nc>=2, niter>=1, 1<=local_size<=16)\n");
    exit(-1);
  }
}

// Serial CPU reference: identical single-precision SRAD math over the same
// (extracted) image. Mirrors extract -> [stats, srad, srad2]*niter -> compress.
static void srad_cpu(std::vector<float>& I, int Nr, int Nc, long Ne, int niter,
                     float lambda, long NeROI,
                     const std::vector<int>& iN, const std::vector<int>& iS,
                     const std::vector<int>& jE, const std::vector<int>& jW) {
  std::vector<float> dN(Ne), dS(Ne), dW(Ne), dE(Ne), cc(Ne);

  // extract: log-uncompress
  for (long ei = 0; ei < Ne; ++ei)
    I[ei] = expf(I[ei] / 255);

  for (int iter = 0; iter < niter; ++iter) {
    // statistics reduction (host)
    float total = 0.f, total2 = 0.f;
    for (long ei = 0; ei < Ne; ++ei) {
      total += I[ei];
      total2 += I[ei] * I[ei];
    }
    float meanROI = total / (float)NeROI;
    float meanROI2 = meanROI * meanROI;
    float varROI = (total2 / (float)NeROI) - meanROI2;
    float q0sqr = varROI / meanROI2;

    // srad: derivatives + diffusion coefficient
    for (long ei = 0; ei < Ne; ++ei) {
      int row = ei % Nr;
      int col = ei / Nr;
      float Jc = I[ei];
      float dN_loc = I[iN[row] + Nr * col] - Jc;
      float dS_loc = I[iS[row] + Nr * col] - Jc;
      float dW_loc = I[row + Nr * jW[col]] - Jc;
      float dE_loc = I[row + Nr * jE[col]] - Jc;

      float G2 = (dN_loc * dN_loc + dS_loc * dS_loc +
                  dW_loc * dW_loc + dE_loc * dE_loc) / (Jc * Jc);
      float L = (dN_loc + dS_loc + dW_loc + dE_loc) / Jc;

      float num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
      float den = 1 + (0.25f * L);
      float qsqr = num / (den * den);

      den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
      float c_loc = 1.0f / (1.0f + den);
      if (c_loc < 0) c_loc = 0;
      else if (c_loc > 1) c_loc = 1;

      dN[ei] = dN_loc; dS[ei] = dS_loc; dW[ei] = dW_loc; dE[ei] = dE_loc;
      cc[ei] = c_loc;
    }

    // srad2: divergence + update
    for (long ei = 0; ei < Ne; ++ei) {
      int row = ei % Nr;
      int col = ei / Nr;
      float cN = cc[ei];
      float cS = cc[iS[row] + Nr * col];
      float cW = cc[ei];
      float cE = cc[row + Nr * jE[col]];
      float D = cN * dN[ei] + cS * dS[ei] + cW * dW[ei] + cE * dE[ei];
      I[ei] = I[ei] + 0.25f * lambda * D;
    }
  }

  // compress: log-recompress
  for (long ei = 0; ei < Ne; ++ei)
    I[ei] = logf(I[ei]) * 255;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  long Ne = (long)Nr * Nc;
  long NeROI = Ne;  // full-image ROI (r1=0,r2=Nr-1,c1=0,c2=Nc-1)
  printf("SRAD: Nr=%d Nc=%d Ne=%ld niter=%d lambda=%g local_size=%d\n",
         Nr, Nc, Ne, niter, lambda, local_size);

  // N/S/W/E neighbour indices with boundary clamping (same as Rodinia host).
  std::vector<int> iN(Nr), iS(Nr), jW(Nc), jE(Nc);
  for (int i = 0; i < Nr; ++i) { iN[i] = i - 1; iS[i] = i + 1; }
  for (int j = 0; j < Nc; ++j) { jW[j] = j - 1; jE[j] = j + 1; }
  iN[0] = 0; iS[Nr - 1] = Nr - 1; jW[0] = 0; jE[Nc - 1] = Nc - 1;

  // Deterministic input image: raw intensities in [0,255] (positive after the
  // extract exp()). Column-major, same layout host and device see.
  srand(7);
  std::vector<float> h_image(Ne);
  for (long i = 0; i < Ne; ++i)
    h_image[i] = (float)(rand() % 256);

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  size_t mem_size = sizeof(float) * Ne;
  d_I  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_iN = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * Nr, NULL, &_err));
  d_iS = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * Nr, NULL, &_err));
  d_jE = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * Nc, NULL, &_err));
  d_jW = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * Nc, NULL, &_err));
  d_dN = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_dS = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_dE = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_dW = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_c  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // Pass the work-group size as NUMBER_THREADS so kernel indexing matches the
  // launch geometry.
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-DNUMBER_THREADS=%d", local_size);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));

  extract_kernel  = CL_CHECK2(clCreateKernel(program, "extract_kernel", &_err));
  srad_kernel     = CL_CHECK2(clCreateKernel(program, "srad_kernel", &_err));
  srad2_kernel    = CL_CHECK2(clCreateKernel(program, "srad2_kernel", &_err));
  compress_kernel = CL_CHECK2(clCreateKernel(program, "compress_kernel", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Upload image + neighbour indices.
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_I, CL_TRUE, 0, mem_size, h_image.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_iN, CL_TRUE, 0, sizeof(int) * Nr, iN.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_iS, CL_TRUE, 0, sizeof(int) * Nr, iS.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_jE, CL_TRUE, 0, sizeof(int) * Nc, jE.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_jW, CL_TRUE, 0, sizeof(int) * Nc, jW.data(), 0, NULL, NULL));

  // Launch geometry: 1-D, round global up to a multiple of local (kernels guard
  // with ei < Ne).
  size_t local_work_size = local_size;
  size_t num_groups = (Ne + local_size - 1) / local_size;
  size_t global_work_size = num_groups * local_size;

  // Static srad/srad2 kernel arguments (q0sqr is refreshed each iteration).
  CL_CHECK(clSetKernelArg(srad_kernel, 0, sizeof(float), &lambda));
  CL_CHECK(clSetKernelArg(srad_kernel, 1, sizeof(int), &Nr));
  CL_CHECK(clSetKernelArg(srad_kernel, 2, sizeof(int), &Nc));
  CL_CHECK(clSetKernelArg(srad_kernel, 3, sizeof(long), &Ne));
  CL_CHECK(clSetKernelArg(srad_kernel, 4, sizeof(cl_mem), &d_iN));
  CL_CHECK(clSetKernelArg(srad_kernel, 5, sizeof(cl_mem), &d_iS));
  CL_CHECK(clSetKernelArg(srad_kernel, 6, sizeof(cl_mem), &d_jE));
  CL_CHECK(clSetKernelArg(srad_kernel, 7, sizeof(cl_mem), &d_jW));
  CL_CHECK(clSetKernelArg(srad_kernel, 8, sizeof(cl_mem), &d_dN));
  CL_CHECK(clSetKernelArg(srad_kernel, 9, sizeof(cl_mem), &d_dS));
  CL_CHECK(clSetKernelArg(srad_kernel, 10, sizeof(cl_mem), &d_dE));
  CL_CHECK(clSetKernelArg(srad_kernel, 11, sizeof(cl_mem), &d_dW));
  CL_CHECK(clSetKernelArg(srad_kernel, 13, sizeof(cl_mem), &d_c));
  CL_CHECK(clSetKernelArg(srad_kernel, 14, sizeof(cl_mem), &d_I));

  CL_CHECK(clSetKernelArg(srad2_kernel, 0, sizeof(float), &lambda));
  CL_CHECK(clSetKernelArg(srad2_kernel, 1, sizeof(int), &Nr));
  CL_CHECK(clSetKernelArg(srad2_kernel, 2, sizeof(int), &Nc));
  CL_CHECK(clSetKernelArg(srad2_kernel, 3, sizeof(long), &Ne));
  CL_CHECK(clSetKernelArg(srad2_kernel, 4, sizeof(cl_mem), &d_iN));
  CL_CHECK(clSetKernelArg(srad2_kernel, 5, sizeof(cl_mem), &d_iS));
  CL_CHECK(clSetKernelArg(srad2_kernel, 6, sizeof(cl_mem), &d_jE));
  CL_CHECK(clSetKernelArg(srad2_kernel, 7, sizeof(cl_mem), &d_jW));
  CL_CHECK(clSetKernelArg(srad2_kernel, 8, sizeof(cl_mem), &d_dN));
  CL_CHECK(clSetKernelArg(srad2_kernel, 9, sizeof(cl_mem), &d_dS));
  CL_CHECK(clSetKernelArg(srad2_kernel, 10, sizeof(cl_mem), &d_dE));
  CL_CHECK(clSetKernelArg(srad2_kernel, 11, sizeof(cl_mem), &d_dW));
  CL_CHECK(clSetKernelArg(srad2_kernel, 12, sizeof(cl_mem), &d_c));
  CL_CHECK(clSetKernelArg(srad2_kernel, 13, sizeof(cl_mem), &d_I));

  auto time_start = std::chrono::high_resolution_clock::now();

  // Extract (log-uncompress).
  CL_CHECK(clSetKernelArg(extract_kernel, 0, sizeof(long), &Ne));
  CL_CHECK(clSetKernelArg(extract_kernel, 1, sizeof(cl_mem), &d_I));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, extract_kernel, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL));

  std::vector<float> host_I(Ne);
  for (int iter = 0; iter < niter; ++iter) {
    // Statistics reduction on the host: read back current image, sum I and I^2.
    CL_CHECK(clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, mem_size, host_I.data(), 0, NULL, NULL));
    float total = 0.f, total2 = 0.f;
    for (long ei = 0; ei < Ne; ++ei) {
      total += host_I[ei];
      total2 += host_I[ei] * host_I[ei];
    }
    float meanROI = total / (float)NeROI;
    float meanROI2 = meanROI * meanROI;
    float varROI = (total2 / (float)NeROI) - meanROI2;
    float q0sqr = varROI / meanROI2;

    CL_CHECK(clSetKernelArg(srad_kernel, 12, sizeof(float), &q0sqr));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, srad_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, srad2_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }

  // Compress (log-recompress).
  CL_CHECK(clSetKernelArg(compress_kernel, 0, sizeof(long), &Ne));
  CL_CHECK(clSetKernelArg(compress_kernel, 1, sizeof(cl_mem), &d_I));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, compress_kernel, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL));

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<float> h_gpu(Ne);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_I, CL_TRUE, 0, mem_size, h_gpu.data(), 0, NULL, NULL));

  // CPU golden reference over the same seeded input.
  std::vector<float> h_ref(h_image);
  srad_cpu(h_ref, Nr, Nc, Ne, niter, lambda, NeROI, iN, iS, jE, jW);

  // Compare with a mixed absolute/relative float tolerance (exp/log + iterated
  // accumulation across host and device libm differ by a few ULP).
  const float atol = 1e-2f;
  const float rtol = 1e-3f;
  int errors = 0;
  for (long i = 0; i < Ne; ++i) {
    float a = h_ref[i], b = h_gpu[i];
    if (fabsf(a - b) > atol + rtol * fabsf(a)) {
      if (errors < 20)
        printf("*** error: [%ld] expected=%f, actual=%f\n", i, a, b);
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
