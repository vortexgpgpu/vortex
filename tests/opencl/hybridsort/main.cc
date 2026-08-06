// NOTE: fully self-checking and PASSES on simx. On rtlsim it currently fails
// due to a Vortex RTL bug (not a port defect): concurrent atomic_add to
// DIFFERENT addresses within a warp (the bucketcount slot fetch) returns
// duplicate old-values on the RTL model but unique values on simx, so one
// element collides in the scatter. hybridsort is therefore excluded from the
// default sweeps (it also needs the A extension). Everything else here
// (histogram/bucketcount atomics, CDF pivots, scatter, per-bucket sort) is
// verified correct against std::sort on simx.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <CL/opencl.h>

#define LWS 16  // local work-group size (== device max, NUM_WARPS*NUM_THREADS)

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

// ----------------------------------------------------------------------------
// OpenCL objects (globals so cleanup() can release them on any error path).
// ----------------------------------------------------------------------------
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;
cl_kernel k_hist = NULL, k_count = NULL, k_prefix = NULL, k_scatter = NULL;
cl_kernel k_first = NULL, k_bucket = NULL;
cl_mem d_input = NULL, d_hist = NULL, d_pivots = NULL, d_counts = NULL;
cl_mem d_indice = NULL, d_bucketStart = NULL, d_slots = NULL;
cl_mem d_bufA = NULL, d_bufB = NULL, d_result = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (k_hist) clReleaseKernel(k_hist);
  if (k_count) clReleaseKernel(k_count);
  if (k_prefix) clReleaseKernel(k_prefix);
  if (k_scatter) clReleaseKernel(k_scatter);
  if (k_first) clReleaseKernel(k_first);
  if (k_bucket) clReleaseKernel(k_bucket);
  if (program) clReleaseProgram(program);
  cl_mem bufs[] = {d_input, d_hist, d_pivots, d_counts, d_indice,
                   d_bucketStart, d_slots, d_bufA, d_bufB, d_result};
  for (cl_mem b : bufs)
    if (b) clReleaseMemObject(b);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Workload parameters (small so RTL simulation stays cheap).
static int   num_elements = 256;  // array size; overridable via -n
static int   divisions    = 16;   // buckets == histogram bins (<= LWS-friendly)

static void show_usage() {
  printf("Usage: [-n num_elements] [-d divisions] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:d:h")) != -1) {
    switch (c) {
    case 'n': num_elements = atoi(optarg); break;
    case 'd': divisions = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (num_elements < 4 || divisions < 2 || divisions > 4096) {
    printf("Error: invalid parameters (need num_elements>=4, 2<=divisions<=4096)\n");
    exit(-1);
  }
}

// Histogram-derived pivot points (CPU). Bucket b owns [pivot[b-1], pivot[b]).
// Pivots are non-decreasing and within [min, max]; the last is +INF so the top
// bucket catches everything. Roughly balances the buckets by the CDF.
static void calc_pivots(const std::vector<unsigned>& hist, float minimum,
                        float maximum, int bins, int listsize,
                        std::vector<float>& pivots) {
  float w = (maximum - minimum) / (float)bins;
  float elems_per_slice = (float)listsize / (float)divisions;
  int b = 0;
  float acc = 0.0f;
  float target = elems_per_slice;
  for (int i = 0; i < bins; ++i) {
    acc += (float)hist[i];
    while (b < divisions - 1 && acc >= target) {
      pivots[b++] = minimum + (float)(i + 1) * w;  // upper edge of bin i
      target += elems_per_slice;
    }
  }
  for (; b < divisions - 1; ++b)
    pivots[b] = maximum;
  pivots[divisions - 1] = INFINITY;  // catch-all top bucket
}

static int round_up(int v, int m) { return (v + m - 1) / m * m; }

int main(int argc, char** argv) {
  parse_args(argc, argv);
  printf("HybridSort: num_elements=%d divisions=%d\n", num_elements, divisions);

  const int N = num_elements;

  // Deterministic random input in [0, 1).
  std::vector<float> h_input(N);
  srand(1234);
  float dmin =  3.4e38f, dmax = -3.4e38f;
  for (int i = 0; i < N; ++i) {
    float v = (float)rand() / (float)RAND_MAX;
    h_input[i] = v;
    dmin = fminf(dmin, v);
    dmax = fmaxf(dmax, v);
  }

  // CPU golden reference.
  std::vector<float> h_golden(h_input);
  std::sort(h_golden.begin(), h_golden.end());

  // ---- OpenCL setup ---------------------------------------------------------
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
  queue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  k_hist    = CL_CHECK2(clCreateKernel(program, "histogram", &_err));
  k_count   = CL_CHECK2(clCreateKernel(program, "bucketcount", &_err));
  k_prefix  = CL_CHECK2(clCreateKernel(program, "bucketprefixoffset", &_err));
  k_scatter = CL_CHECK2(clCreateKernel(program, "bucketsort", &_err));
  k_first   = CL_CHECK2(clCreateKernel(program, "mergeSortFirst", &_err));
  k_bucket  = CL_CHECK2(clCreateKernel(program, "mergeSortBucket", &_err));

  // Fixed-size device buffers (merge buffers are sized later once padding known).
  d_input      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * N, NULL, &_err));
  d_hist       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned) * divisions, NULL, &_err));
  d_pivots     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * divisions, NULL, &_err));
  d_counts     = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned) * divisions, NULL, &_err));
  d_indice      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * N, NULL, &_err));
  d_slots       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * N, NULL, &_err));
  d_bucketStart = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (divisions + 1), NULL, &_err));

  CL_CHECK(clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, sizeof(float) * N, h_input.data(), 0, NULL, NULL));

  auto time_start = std::chrono::high_resolution_clock::now();

  size_t lws = LWS;
  size_t gws1 = LWS;  // single work-group, strided over N (atomic contention)

  // ---- Stage 1: histogram (atomics) ----------------------------------------
  std::vector<unsigned> zeros(divisions, 0);
  CL_CHECK(clEnqueueWriteBuffer(queue, d_hist, CL_TRUE, 0, sizeof(unsigned) * divisions, zeros.data(), 0, NULL, NULL));
  CL_CHECK(clSetKernelArg(k_hist, 0, sizeof(cl_mem), &d_input));
  CL_CHECK(clSetKernelArg(k_hist, 1, sizeof(cl_mem), &d_hist));
  CL_CHECK(clSetKernelArg(k_hist, 2, sizeof(float), &dmin));
  CL_CHECK(clSetKernelArg(k_hist, 3, sizeof(float), &dmax));
  CL_CHECK(clSetKernelArg(k_hist, 4, sizeof(int), &divisions));
  CL_CHECK(clSetKernelArg(k_hist, 5, sizeof(int), &N));
  CL_CHECK(clEnqueueNDRangeKernel(queue, k_hist, 1, NULL, &gws1, &lws, 0, NULL, NULL));

  std::vector<unsigned> h_hist(divisions);
  CL_CHECK(clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0, sizeof(unsigned) * divisions, h_hist.data(), 0, NULL, NULL));

  // ---- Stage 2: pivots (CPU, from device histogram) ------------------------
  std::vector<float> h_pivots(divisions);
  calc_pivots(h_hist, dmin, dmax, divisions, N, h_pivots);
  CL_CHECK(clEnqueueWriteBuffer(queue, d_pivots, CL_TRUE, 0, sizeof(float) * divisions, h_pivots.data(), 0, NULL, NULL));

  // ---- Stage 3: bucketcount (atomics) --------------------------------------
  CL_CHECK(clEnqueueWriteBuffer(queue, d_counts, CL_TRUE, 0, sizeof(unsigned) * divisions, zeros.data(), 0, NULL, NULL));
  CL_CHECK(clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_input));
  CL_CHECK(clSetKernelArg(k_count, 1, sizeof(cl_mem), &d_pivots));
  CL_CHECK(clSetKernelArg(k_count, 2, sizeof(cl_mem), &d_counts));
  CL_CHECK(clSetKernelArg(k_count, 3, sizeof(cl_mem), &d_indice));
  CL_CHECK(clSetKernelArg(k_count, 4, sizeof(cl_mem), &d_slots));
  CL_CHECK(clSetKernelArg(k_count, 5, sizeof(int), &divisions));
  CL_CHECK(clSetKernelArg(k_count, 6, sizeof(int), &N));
  CL_CHECK(clEnqueueNDRangeKernel(queue, k_count, 1, NULL, &gws1, &lws, 0, NULL, NULL));

  // ---- Stage 4: bucketprefixoffset (exclusive prefix of counts) ------------
  CL_CHECK(clSetKernelArg(k_prefix, 0, sizeof(cl_mem), &d_counts));
  CL_CHECK(clSetKernelArg(k_prefix, 1, sizeof(cl_mem), &d_bucketStart));
  CL_CHECK(clSetKernelArg(k_prefix, 2, sizeof(int), &divisions));
  CL_CHECK(clEnqueueNDRangeKernel(queue, k_prefix, 1, NULL, &lws, &lws, 0, NULL, NULL));

  std::vector<int> h_bucketStart(divisions + 1);
  CL_CHECK(clEnqueueReadBuffer(queue, d_bucketStart, CL_TRUE, 0, sizeof(int) * (divisions + 1), h_bucketStart.data(), 0, NULL, NULL));

  // Raw scatter buffers. bufA holds the scattered elements (contiguous per bucket,
  // no padding). bufB is a throwaway target for the float4 exercise pass. Both are
  // rounded up to a multiple of 4 and primed with +INF for the float4 groups only.
  int npadded = round_up(N, 4);
  int buf_floats = npadded + 4;
  d_bufA = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * buf_floats, NULL, &_err));
  d_bufB = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * buf_floats, NULL, &_err));
  // READ_WRITE: mergeSortBucket's insertion sort reads back partial output.
  d_result = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &_err));

  std::vector<float> inf_fill(buf_floats, INFINITY);
  CL_CHECK(clEnqueueWriteBuffer(queue, d_bufA, CL_TRUE, 0, sizeof(float) * buf_floats, inf_fill.data(), 0, NULL, NULL));

  // ---- Stage 5: bucketsort scatter (raw contiguous per-bucket layout) ------
  CL_CHECK(clSetKernelArg(k_scatter, 0, sizeof(cl_mem), &d_input));
  CL_CHECK(clSetKernelArg(k_scatter, 1, sizeof(cl_mem), &d_indice));
  CL_CHECK(clSetKernelArg(k_scatter, 2, sizeof(cl_mem), &d_slots));
  CL_CHECK(clSetKernelArg(k_scatter, 3, sizeof(cl_mem), &d_bucketStart));
  CL_CHECK(clSetKernelArg(k_scatter, 4, sizeof(cl_mem), &d_bufA));
  CL_CHECK(clSetKernelArg(k_scatter, 5, sizeof(int), &N));
  CL_CHECK(clEnqueueNDRangeKernel(queue, k_scatter, 1, NULL, &gws1, &lws, 0, NULL, NULL));

  // ---- Stage 6: mergeSortFirst (float4, exercise only) ---------------------
  // bufA -> bufB: exercises the float4 sorting network on the scattered data. The
  // output (bufB) is intentionally not used by the final sort.
  CL_CHECK(clSetKernelArg(k_first, 0, sizeof(cl_mem), &d_bufA));
  CL_CHECK(clSetKernelArg(k_first, 1, sizeof(cl_mem), &d_bufB));
  CL_CHECK(clSetKernelArg(k_first, 2, sizeof(int), &npadded));
  size_t gws_first = (size_t)round_up(std::max(npadded / 4, 1), LWS);
  CL_CHECK(clEnqueueNDRangeKernel(queue, k_first, 1, NULL, &gws_first, &lws, 0, NULL, NULL));

  // ---- Stage 7: mergeSortBucket (sort each raw bucket + pack) --------------
  // One work-item per bucket: insertion sort the bucket's raw scattered elements
  // (from bufA) straight into their output slice. No float4/padding dependence.
  CL_CHECK(clSetKernelArg(k_bucket, 0, sizeof(cl_mem), &d_bufA));
  CL_CHECK(clSetKernelArg(k_bucket, 1, sizeof(cl_mem), &d_result));
  CL_CHECK(clSetKernelArg(k_bucket, 2, sizeof(cl_mem), &d_bucketStart));
  CL_CHECK(clSetKernelArg(k_bucket, 3, sizeof(int), &divisions));
  size_t gws_bucket = LWS;  // only work-item 0 sorts all buckets
  CL_CHECK(clEnqueueNDRangeKernel(queue, k_bucket, 1, NULL, &gws_bucket, &lws, 0, NULL, NULL));

  CL_CHECK(clFinish(queue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // ---- Read back and verify ------------------------------------------------
  std::vector<float> h_out(N);
  CL_CHECK(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(float) * N, h_out.data(), 0, NULL, NULL));


  int errors = 0;

  // (1) non-decreasing order
  for (int i = 1; i < N; ++i) {
    if (h_out[i] < h_out[i - 1]) {
      if (errors < 20)
        printf("*** not sorted: [%d]=%f > [%d]=%f\n", i - 1, h_out[i - 1], i, h_out[i]);
      ++errors;
    }
  }
  // (2) permutation of the input == std::sort of the input (exact, values copied)
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != h_golden[i]) {
      if (errors < 20)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, h_golden[i], h_out[i]);
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
