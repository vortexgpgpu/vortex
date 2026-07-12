// Backprop (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// One training iteration of a fully-connected neural net: a forward pass over
// the input->hidden layer (bpnn_layerforward_ocl) followed by a momentum weight
// update (bpnn_adjust_weights_ocl). Both device kernels are checked against the
// serial CPU routines from Rodinia's backprop.c on the identical (seeded) net.
//
// The kernel tile is 4x4 (WIDTH=HEIGHT=4), so the 2-D local work-group is
// 4*4 = 16 work-items, matching the device max work-group size
// (NUM_WARPS*NUM_THREADS = 16). The hidden layer therefore has HEIGHT=4 units
// and the input layer size must be a multiple of BLOCK_SIZE=4.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <CL/opencl.h>

// Must match kernel.cl.
#define WIDTH      4
#define HEIGHT     4
#define BLOCK_SIZE 4
#define ETA        0.3f
#define MOMENTUM   0.3f

// Float comparison tolerance (relative + absolute floor). The GPU accumulates
// the forward sum as per-block partial sums reduced on the host, so the term
// ordering differs from the sequential CPU reference; a strict ULP match would
// be too tight for the summation.
#define EPSILON 1e-3f

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
cl_mem input_ocl = NULL;
cl_mem output_hidden_ocl = NULL;
cl_mem input_hidden_ocl = NULL;
cl_mem hidden_partial_sum_ocl = NULL;
cl_mem hidden_delta_ocl = NULL;
cl_mem input_prev_weights_ocl = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (program) clReleaseProgram(program);
  if (input_ocl) clReleaseMemObject(input_ocl);
  if (output_hidden_ocl) clReleaseMemObject(output_hidden_ocl);
  if (input_hidden_ocl) clReleaseMemObject(input_hidden_ocl);
  if (hidden_partial_sum_ocl) clReleaseMemObject(hidden_partial_sum_ocl);
  if (hidden_delta_ocl) clReleaseMemObject(hidden_delta_ocl);
  if (input_prev_weights_ocl) clReleaseMemObject(input_prev_weights_ocl);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Squashing (sigmoid) activation, as in backprop.c.
static inline float squash(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Serial reference: bpnn_layerforward from backprop.c.
// conn is row-major [k][j] with row stride (n2+1); l1[0] is the threshold unit.
static void bpnn_layerforward_cpu(const float* l1, float* l2,
                                  const float* conn, int n1, int n2) {
  for (int j = 1; j <= n2; ++j) {
    float sum = 0.0f;
    for (int k = 0; k <= n1; ++k)
      sum += conn[k * (n2 + 1) + j] * l1[k];
    l2[j] = squash(sum);
  }
}

// Serial reference: bpnn_adjust_weights from backprop.c.
// w/oldw are row-major [k][j] with row stride (ndelta+1); ly[0] is the bias.
static void bpnn_adjust_weights_cpu(const float* delta, int ndelta,
                                    float* ly, int nly,
                                    float* w, float* oldw) {
  ly[0] = 1.0f;
  int stride = ndelta + 1;
  for (int j = 1; j <= ndelta; ++j) {
    for (int k = 0; k <= nly; ++k) {
      float new_dw = (ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * stride + j]);
      w[k * stride + j] += new_dw;
      oldw[k * stride + j] = new_dw;
    }
  }
}

static inline bool almost_equal(float ref, float got) {
  return fabsf(ref - got) <= EPSILON * (1.0f + fabsf(ref));
}

// Input layer size (number of input units). Must be a multiple of BLOCK_SIZE.
// Default is a small multiple of 16 so RTL simulation stays under budget.
static int layer_size = 16;

static void show_usage() {
  printf("Usage: [-n input_layer_size (multiple of %d)] [-h]\n", HEIGHT);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n': layer_size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (layer_size < HEIGHT || (layer_size % HEIGHT) != 0) {
    printf("Error: input layer size must be a positive multiple of %d\n", HEIGHT);
    exit(-1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  const int in  = layer_size;   // input units
  const int hid = WIDTH;        // hidden units (tile width == hidden layer size)
  const int num_blocks = in / BLOCK_SIZE;

  printf("Backprop: input=%d hidden=%d tile=%dx%d (work-group=%d) blocks=%d\n",
         in, hid, WIDTH, HEIGHT, WIDTH * HEIGHT, num_blocks);

  const int wsz = (in + 1) * (hid + 1);  // input->hidden weight matrix size

  // Deterministic net setup (mirrors backprop.c bpnn_create + load).
  srand(7);
  std::vector<float> input_units(in + 1);
  std::vector<float> input_weights(wsz);
  std::vector<float> input_prev_weights(wsz, 0.0f);  // momentum, zero-initialized
  std::vector<float> hidden_delta(hid + 1);

  input_units[0] = 1.0f;  // threshold unit
  for (int i = 1; i <= in; ++i)
    input_units[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < wsz; ++i)
    input_weights[i] = (float)rand() / RAND_MAX;
  // Deterministic hidden-layer error term (would come from the backward pass).
  hidden_delta[0] = 0.0f;
  for (int j = 1; j <= hid; ++j)
    hidden_delta[j] = (float)rand() / RAND_MAX - 0.5f;

  // OpenCL setup.
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  input_ocl              = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * sizeof(float), NULL, &_err));
  output_hidden_ocl      = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &_err));
  input_hidden_ocl       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, wsz * sizeof(float), NULL, &_err));
  hidden_partial_sum_ocl = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, num_blocks * WIDTH * sizeof(float), NULL, &_err));
  hidden_delta_ocl       = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &_err));
  input_prev_weights_ocl = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, wsz * sizeof(float), NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  kernel1 = CL_CHECK2(clCreateKernel(program, "bpnn_layerforward_ocl", &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, "bpnn_adjust_weights_ocl", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // 2-D NDRange: local work-group is BLOCK_SIZE x BLOCK_SIZE = 16 work-items.
  size_t global_work[2] = { (size_t)BLOCK_SIZE, (size_t)BLOCK_SIZE * num_blocks };
  size_t local_work[2]  = { (size_t)BLOCK_SIZE, (size_t)BLOCK_SIZE };

  int errors = 0;

  // ---- Kernel 1: forward pass (input -> hidden) ----
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_ocl, CL_TRUE, 0, (in + 1) * sizeof(float), input_units.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_hidden_ocl, CL_TRUE, 0, wsz * sizeof(float), input_weights.data(), 0, NULL, NULL));

  CL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_mem), &input_ocl));
  CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), &output_hidden_ocl));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_mem), &input_hidden_ocl));
  CL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), &hidden_partial_sum_ocl));
  CL_CHECK(clSetKernelArg(kernel1, 4, sizeof(float) * HEIGHT, NULL));
  CL_CHECK(clSetKernelArg(kernel1, 5, sizeof(float) * HEIGHT * WIDTH, NULL));
  CL_CHECK(clSetKernelArg(kernel1, 6, sizeof(cl_int), &in));
  CL_CHECK(clSetKernelArg(kernel1, 7, sizeof(cl_int), &hid));

  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 2, NULL, global_work, local_work, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  std::vector<float> partial_sum(num_blocks * WIDTH);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, hidden_partial_sum_ocl, CL_TRUE, 0,
                               num_blocks * WIDTH * sizeof(float), partial_sum.data(), 0, NULL, NULL));

  // Host reduction of the per-block partial sums into hidden activations.
  // The kernel's tree reduction (fixed to start at power_two=2) returns the
  // true block sums, so no scaling is needed here.
  std::vector<float> hidden_gpu(hid + 1);
  for (int j = 1; j <= hid; ++j) {
    float sum = 0.0f;
    for (int k = 0; k < num_blocks; ++k)
      sum += partial_sum[k * hid + (j - 1)];
    sum += input_weights[0 * (hid + 1) + j];  // threshold weight (row k=0)
    hidden_gpu[j] = squash(sum);
  }

  // CPU golden: serial forward pass.
  std::vector<float> hidden_ref(hid + 1);
  bpnn_layerforward_cpu(input_units.data(), hidden_ref.data(), input_weights.data(), in, hid);

  for (int j = 1; j <= hid; ++j) {
    if (!almost_equal(hidden_ref[j], hidden_gpu[j])) {
      if (errors < 20)
        printf("*** error: hidden[%d] expected=%f, actual=%f\n", j, hidden_ref[j], hidden_gpu[j]);
      ++errors;
    }
  }

  // ---- Kernel 2: momentum weight update (input -> hidden) ----
  // Kernel 1 scribbled intermediate values into input_hidden_ocl; re-upload the
  // pristine weights (as the Rodinia host does) so the update starts fresh.
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_hidden_ocl, CL_TRUE, 0, wsz * sizeof(float), input_weights.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_prev_weights_ocl, CL_TRUE, 0, wsz * sizeof(float), input_prev_weights.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, hidden_delta_ocl, CL_TRUE, 0, (hid + 1) * sizeof(float), hidden_delta.data(), 0, NULL, NULL));

  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), &hidden_delta_ocl));
  CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_int), &hid));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_mem), &input_ocl));
  CL_CHECK(clSetKernelArg(kernel2, 3, sizeof(cl_int), &in));
  CL_CHECK(clSetKernelArg(kernel2, 4, sizeof(cl_mem), &input_hidden_ocl));
  CL_CHECK(clSetKernelArg(kernel2, 5, sizeof(cl_mem), &input_prev_weights_ocl));

  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 2, NULL, global_work, local_work, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  std::vector<float> weights_gpu(wsz);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, input_hidden_ocl, CL_TRUE, 0,
                               wsz * sizeof(float), weights_gpu.data(), 0, NULL, NULL));

  // CPU golden: serial weight update on a copy of the pristine net.
  std::vector<float> weights_ref(input_weights);
  std::vector<float> prev_ref(input_prev_weights);
  bpnn_adjust_weights_cpu(hidden_delta.data(), hid, input_units.data(), in,
                          weights_ref.data(), prev_ref.data());

  for (int i = 0; i < wsz; ++i) {
    if (!almost_equal(weights_ref[i], weights_gpu[i])) {
      if (errors < 20)
        printf("*** error: weight[%d] expected=%f, actual=%f\n", i, weights_ref[i], weights_gpu[i]);
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
