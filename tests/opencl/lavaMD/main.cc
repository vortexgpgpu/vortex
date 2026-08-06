// lavaMD (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Computes inter-particle forces (Lennard-Jones-like) over a 3-D grid of boxes.
// Each box holds NUMBER_PAR_PER_BOX particles and the OpenCL local work-group
// size equals NUMBER_PAR_PER_BOX. Particle positions/charges are generated
// deterministically (fixed srand seed); the GPU result is checked against a
// serial CPU reference running the identical force computation.
//
// Device max work-group size = NUM_WARPS*NUM_THREADS = 16 in the default CI
// config, so NUMBER_PAR_PER_BOX is fixed at 16 (fits the local group exactly).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

// Particles per box == OpenCL local work-group size. Must not exceed the device
// max work-group size (16). Passed to the kernel build as a -D define so host
// and device agree.
#define NUMBER_PAR_PER_BOX 16

// Force tolerance: exp() and the long accumulation chain differ slightly between
// the host libm and the device, so compare with a combined abs/rel tolerance.
#define ABS_TOL 1e-3f
#define REL_TOL 1e-3f

#define fp float

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

// Host-side mirrors of the kernel structs. `int` offsets (not `long`) so the
// struct ABI matches the 32-bit device byte-for-byte.
typedef struct {
  fp x, y, z;
} THREE_VECTOR;

typedef struct {
  fp v, x, y, z;
} FOUR_VECTOR;

typedef struct nei_str {
  int x, y, z;
  int number;
  int offset;
} nei_str;

typedef struct box_str {
  int x, y, z;
  int number;
  int offset;
  int nn;
  nei_str nei[26];
} box_str;

typedef struct par_str {
  fp alpha;
} par_str;

typedef struct dim_str {
  int cur_arg;
  int arch_arg;
  int cores_arg;
  int boxes1d_arg;
  int number_boxes;
  int box_mem;
  int space_elem;
  int space_mem;
  int space_mem2;
} dim_str;

#define DOT(A, B) ((A.x) * (B.x) + (A.y) * (B.y) + (A.z) * (B.z))

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;
  FILE* fh = fopen(filename, "r");
  if (NULL == fh) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }
  fseek(fh, 0, SEEK_END);
  long fsize = ftell(fh);
  rewind(fh);
  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fh);
  fclose(fh);
  return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem d_box_gpu = NULL;
cl_mem d_rv_gpu = NULL;
cl_mem d_qv_gpu = NULL;
cl_mem d_fv_gpu = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (d_box_gpu) clReleaseMemObject(d_box_gpu);
  if (d_rv_gpu) clReleaseMemObject(d_rv_gpu);
  if (d_qv_gpu) clReleaseMemObject(d_qv_gpu);
  if (d_fv_gpu) clReleaseMemObject(d_fv_gpu);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Small by default so RTL simulation stays under budget: boxes1d=2 -> 8 boxes,
// each with 16 particles = 128 particles total.
static int boxes1d = 2;

static void show_usage() {
  printf("Usage: [-b boxes1d] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "b:h")) != -1) {
    switch (c) {
    case 'b': boxes1d = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (boxes1d < 1) {
    printf("Error: boxes1d must be >= 1\n");
    exit(-1);
  }
}

// Build the box grid and its neighbour lists (identical to the benchmark).
static void build_boxes(std::vector<box_str>& box, int boxes1d) {
  int nh = 0;
  for (int i = 0; i < boxes1d; i++) {
    for (int j = 0; j < boxes1d; j++) {
      for (int k = 0; k < boxes1d; k++) {
        box[nh].x = k;
        box[nh].y = j;
        box[nh].z = i;
        box[nh].number = nh;
        box[nh].offset = nh * NUMBER_PAR_PER_BOX;
        box[nh].nn = 0;
        for (int l = -1; l < 2; l++) {
          for (int m = -1; m < 2; m++) {
            for (int n = -1; n < 2; n++) {
              if ((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0 &&
                  (i + l) < boxes1d && (j + m) < boxes1d && (k + n) < boxes1d &&
                  !(l == 0 && m == 0 && n == 0)) {
                int nn = box[nh].nn;
                box[nh].nei[nn].x = k + n;
                box[nh].nei[nn].y = j + m;
                box[nh].nei[nn].z = i + l;
                box[nh].nei[nn].number = (box[nh].nei[nn].z * boxes1d * boxes1d) +
                                         (box[nh].nei[nn].y * boxes1d) +
                                         box[nh].nei[nn].x;
                box[nh].nei[nn].offset = box[nh].nei[nn].number * NUMBER_PAR_PER_BOX;
                box[nh].nn++;
              }
            }
          }
        }
        nh++;
      }
    }
  }
}

// Serial CPU reference — same force computation as the kernel.
static void lavaMD_cpu(const par_str& par, int number_boxes,
                       const std::vector<box_str>& box,
                       const std::vector<FOUR_VECTOR>& rv,
                       const std::vector<fp>& qv,
                       std::vector<FOUR_VECTOR>& fv) {
  fp a2 = 2 * par.alpha * par.alpha;
  for (int bx = 0; bx < number_boxes; bx++) {
    int first_i = box[bx].offset;
    for (int k = 0; k < (1 + box[bx].nn); k++) {
      int pointer = (k == 0) ? bx : box[bx].nei[k - 1].number;
      int first_j = box[pointer].offset;
      for (int wtx = 0; wtx < NUMBER_PAR_PER_BOX; wtx++) {
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
          fp r2 = rv[first_i + wtx].v + rv[first_j + j].v -
                  DOT(rv[first_i + wtx], rv[first_j + j]);
          fp u2 = a2 * r2;
          fp vij = expf(-u2);
          fp fs = 2 * vij;
          THREE_VECTOR d;
          d.x = rv[first_i + wtx].x - rv[first_j + j].x;
          fp fxij = fs * d.x;
          d.y = rv[first_i + wtx].y - rv[first_j + j].y;
          fp fyij = fs * d.y;
          d.z = rv[first_i + wtx].z - rv[first_j + j].z;
          fp fzij = fs * d.z;
          fv[first_i + wtx].v += qv[first_j + j] * vij;
          fv[first_i + wtx].x += qv[first_j + j] * fxij;
          fv[first_i + wtx].y += qv[first_j + j] * fyij;
          fv[first_i + wtx].z += qv[first_j + j] * fzij;
        }
      }
    }
  }
}

static bool close_enough(fp ref, fp act) {
  return fabsf(ref - act) <= (ABS_TOL + REL_TOL * fabsf(ref));
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  par_str par;
  par.alpha = 0.5f;

  dim_str dim;
  dim.boxes1d_arg = boxes1d;
  dim.number_boxes = boxes1d * boxes1d * boxes1d;
  dim.space_elem = dim.number_boxes * NUMBER_PAR_PER_BOX;
  dim.box_mem = dim.number_boxes * (int)sizeof(box_str);
  dim.space_mem = dim.space_elem * (int)sizeof(FOUR_VECTOR);
  dim.space_mem2 = dim.space_elem * (int)sizeof(fp);

  printf("lavaMD: boxes1d=%d number_boxes=%d particles/box=%d total_particles=%d\n",
         boxes1d, dim.number_boxes, NUMBER_PAR_PER_BOX, dim.space_elem);

  // Box grid and neighbour lists.
  std::vector<box_str> box(dim.number_boxes);
  build_boxes(box, boxes1d);

  // Deterministic particle positions and charges (values in 0.1 .. 1.0).
  srand(2);
  std::vector<FOUR_VECTOR> rv(dim.space_elem);
  std::vector<fp> qv(dim.space_elem);
  for (int i = 0; i < dim.space_elem; i++) {
    rv[i].v = (rand() % 10 + 1) / 10.0f;
    rv[i].x = (rand() % 10 + 1) / 10.0f;
    rv[i].y = (rand() % 10 + 1) / 10.0f;
    rv[i].z = (rand() % 10 + 1) / 10.0f;
  }
  for (int i = 0; i < dim.space_elem; i++) {
    qv[i] = (rand() % 10 + 1) / 10.0f;
  }

  // Output forces start at zero (kernel accumulates).
  std::vector<FOUR_VECTOR> fv(dim.space_elem);
  for (int i = 0; i < dim.space_elem; i++) {
    fv[i].v = fv[i].x = fv[i].y = fv[i].z = 0;
  }

  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  d_box_gpu = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, dim.box_mem, NULL, &_err));
  d_rv_gpu = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, dim.space_mem, NULL, &_err));
  d_qv_gpu = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, dim.space_mem2, NULL, &_err));
  d_fv_gpu = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, dim.space_mem, NULL, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  // Pass NUMBER_PAR_PER_BOX so the device particle/thread counts match the host.
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-D NUMBER_PAR_PER_BOX=%d", NUMBER_PAR_PER_BOX);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));
  kernel = CL_CHECK2(clCreateKernel(program, "kernel_gpu_opencl", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_box_gpu, CL_TRUE, 0, dim.box_mem, box.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_rv_gpu, CL_TRUE, 0, dim.space_mem, rv.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_qv_gpu, CL_TRUE, 0, dim.space_mem2, qv.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_fv_gpu, CL_TRUE, 0, dim.space_mem, fv.data(), 0, NULL, NULL));

  // Pass alpha and number_boxes as scalars (not the by-value par_str/dim_str
  // structs, which pocl-vortex does not marshal reliably).
  cl_float k_alpha = par.alpha;
  cl_int k_number_boxes = dim.number_boxes;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_float), &k_alpha));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_int), &k_number_boxes));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_box_gpu));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_rv_gpu));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_qv_gpu));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_fv_gpu));

  // One work-group per box; local size == NUMBER_PAR_PER_BOX.
  size_t local_work_size = NUMBER_PAR_PER_BOX;
  size_t global_work_size = (size_t)dim.number_boxes * local_work_size;

  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<FOUR_VECTOR> h_gpu(dim.space_elem);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_fv_gpu, CL_TRUE, 0, dim.space_mem, h_gpu.data(), 0, NULL, NULL));

  // CPU golden reference over the same data.
  std::vector<FOUR_VECTOR> h_ref(dim.space_elem);
  for (int i = 0; i < dim.space_elem; i++)
    h_ref[i].v = h_ref[i].x = h_ref[i].y = h_ref[i].z = 0;
  lavaMD_cpu(par, dim.number_boxes, box, rv, qv, h_ref);

  int errors = 0;
  for (int i = 0; i < dim.space_elem; i++) {
    const fp ref[4] = {h_ref[i].v, h_ref[i].x, h_ref[i].y, h_ref[i].z};
    const fp act[4] = {h_gpu[i].v, h_gpu[i].x, h_gpu[i].y, h_gpu[i].z};
    for (int c = 0; c < 4; c++) {
      if (!close_enough(ref[c], act[c])) {
        if (errors < 20)
          printf("*** error: [%d].%c expected=%f, actual=%f\n",
                 i, "vxyz"[c], ref[c], act[c]);
        ++errors;
      }
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
