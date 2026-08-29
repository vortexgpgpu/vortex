#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex2.h>
#include <cmath>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE* out,
                       const TYPE* A, const TYPE* B,
                       uint32_t M, uint32_t N, uint32_t K,
                       TYPE alpha, TYPE beta) {
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      TYPE sum(0);
      for (uint32_t e = 0; e < K; ++e) {
        sum += A[row * K + e] * B[e * N + col];
      }
      out[row * N + col] = alpha * sum + beta * out[row * N + col];
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t size  = 64;
float    alpha = 1.0f;
float    beta  = 0.0f;

vx_device_h device      = nullptr;
vx_buffer_h A_buffer    = nullptr;
vx_buffer_h B_buffer    = nullptr;
vx_buffer_h C_buffer    = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex sgemmx Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (A_buffer) vx_buffer_release(A_buffer);
    if (B_buffer) vx_buffer_release(B_buffer);
    if (C_buffer) vx_buffer_release(C_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint32_t M = size, N = size, K = size;
  uint32_t buf_size_A = M * K * sizeof(TYPE);
  uint32_t buf_size_B = K * N * sizeof(TYPE);
  uint32_t buf_size_C = M * N * sizeof(TYPE);

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix: M=" << M << " N=" << N << " K=" << K << std::endl;
  std::cout << "alpha=" << alpha << " beta=" << beta << std::endl;

  // Derived tile sizes — must match kernel's #defines
  const uint32_t BLOCK_SIZE_M = BLOCK_DIM_Y * THREAD_SIZE_Y;
  const uint32_t BLOCK_SIZE_N = BLOCK_DIM_X * THREAD_SIZE_X;

  // Tile alignment checks (kernel has no boundary guards)
  if (M % BLOCK_SIZE_M != 0 || N % BLOCK_SIZE_N != 0 || K % BLOCK_SIZE_K != 0) {
    printf("Error: M(%d) must be multiple of %d, N(%d) of %d, K(%d) of %d\n",
           M, BLOCK_SIZE_M, N, BLOCK_SIZE_N, K, BLOCK_SIZE_K);
    return -1;
  }
  if (BLOCK_DIM_X < BLOCK_SIZE_K || BLOCK_DIM_Y < BLOCK_SIZE_K) {
    printf("Error: BLOCK_DIM_X(%d) and BLOCK_DIM_Y(%d) must be >= BLOCK_SIZE_K(%d)\n",
           BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_SIZE_K);
    return -1;
  }

  // block_dim[0] = BLOCK_DIM_X = warp size  (threadIdx.x range)
  // block_dim[1] = BLOCK_DIM_Y = num warps  (threadIdx.y range)
  uint32_t block_dim[2] = {BLOCK_DIM_X, BLOCK_DIM_Y};
  uint32_t grid_dim[2]  = {N / BLOCK_SIZE_N, M / BLOCK_SIZE_M};

  kernel_arg.M     = M;
  kernel_arg.N     = N;
  kernel_arg.K     = K;
  kernel_arg.alpha = alpha;
  kernel_arg.beta  = beta;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size_A, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, buf_size_B, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_buffer_create(device, buf_size_C, VX_MEM_READ_WRITE, &C_buffer));
  RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

  // generate source data
  std::vector<TYPE> h_A(M * K);
  std::vector<TYPE> h_B(K * N);
  std::vector<TYPE> h_C(M * N, TYPE(0));
  for (uint32_t i = 0; i < M * K; ++i) h_A[i] = Comparator<TYPE>::generate();
  for (uint32_t i = 0; i < K * N; ++i) h_B[i] = Comparator<TYPE>::generate();

  std::cout << "upload matrix A buffer" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A.data(), buf_size_A, 0, nullptr, nullptr));

  // B is row-major (no pre-transposition)
  std::cout << "upload matrix B buffer" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, h_B.data(), buf_size_B, 0, nullptr, nullptr));

  // C initialized to zero; beta=0 so the initial value doesn't matter for default run
  std::cout << "upload matrix C buffer" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, C_buffer, 0, h_C.data(), buf_size_C, 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "launch kernel" << std::endl;
  // LMEM per CTA: As[BLOCK_SIZE_M][BLOCK_SIZE_K] + Bs[BLOCK_SIZE_K][BLOCK_SIZE_N]
  uint32_t lmem_size = BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N) * sizeof(TYPE);
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = lmem_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::cout << "download destination buffer" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_C.data(), C_buffer, 0, buf_size_C, 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(M * N, TYPE(0));
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K, alpha, beta);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
