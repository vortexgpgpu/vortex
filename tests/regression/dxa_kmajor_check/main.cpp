#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t M = 64;
uint32_t N = 32;
uint32_t K = 16;
uint32_t tileM = 64;
uint32_t tileN = 16;
uint32_t tileK = 16;

vx_device_h device = nullptr;
vx_buffer_h srcA_buffer = nullptr;
vx_buffer_h srcB_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void parse_args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) { M = atoi(argv[++i]); continue; }
    if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) { N = atoi(argv[++i]); continue; }
    if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) { K = atoi(argv[++i]); continue; }
    if (strcmp(argv[i], "-tM") == 0 && i + 1 < argc) { tileM = atoi(argv[++i]); continue; }
    if (strcmp(argv[i], "-tN") == 0 && i + 1 < argc) { tileN = atoi(argv[++i]); continue; }
    if (strcmp(argv[i], "-tK") == 0 && i + 1 < argc) { tileK = atoi(argv[++i]); continue; }
  }
}

void cleanup() {
  if (device) {
    if (srcA_buffer) vx_buffer_release(srcA_buffer);
    if (srcB_buffer) vx_buffer_release(srcB_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if (M % tileM != 0 || N % tileN != 0 || K != tileK) {
    std::cerr << "Error: M/N must be multiples of tileM/tileN, K==tileK\n";
    return -1;
  }

  const uint32_t grid_m = M / tileM;
  const uint32_t grid_n = N / tileN;
  const uint32_t num_cta = grid_m * grid_n;
  const uint32_t a_bytes = tileM * tileK * sizeof(elem_t);
  const uint32_t b_bytes = tileN * tileK * sizeof(elem_t);
  const uint32_t cta_bytes = a_bytes + b_bytes;
  const uint32_t srcA_bytes = M * K * sizeof(elem_t);
  const uint32_t srcB_bytes = N * K * sizeof(elem_t);
  const uint32_t dst_bytes = num_cta * cta_bytes;

  std::cout << "M=" << M << " N=" << N << " K=" << K
            << " tileM=" << tileM << " tileN=" << tileN << " tileK=" << tileK
            << " num_cta=" << num_cta << "\n";

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_buffer_create(device, srcA_bytes, VX_MEM_READ,  &srcA_buffer));
  RT_CHECK(vx_buffer_create(device, srcB_bytes, VX_MEM_READ,  &srcB_buffer));
  RT_CHECK(vx_buffer_create(device, dst_bytes,  VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(srcA_buffer, &kernel_arg.srcA_addr));
  RT_CHECK(vx_buffer_address(srcB_buffer, &kernel_arg.srcB_addr));
  RT_CHECK(vx_buffer_address(dst_buffer,  &kernel_arg.dst_addr));

  kernel_arg.M = M; kernel_arg.N = N; kernel_arg.K = K;
  kernel_arg.tileM = tileM; kernel_arg.tileN = tileN; kernel_arg.tileK = tileK;
  kernel_arg.a_bytes = a_bytes;
  kernel_arg.b_bytes = b_bytes;
  kernel_arg.cta_bytes = cta_bytes;

  // Fill sources with unique-per-position values so any mismatch is obvious.
  std::vector<elem_t> h_srcA(M * K), h_srcB(N * K);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < K; ++k)
      h_srcA[m * K + k] = static_cast<elem_t>(0xA000 | (m * K + k));
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      h_srcB[k * N + n] = static_cast<elem_t>(0xB000 | (k * N + n));
  RT_CHECK(vx_enqueue_write(queue, srcA_buffer, 0, h_srcA.data(), srcA_bytes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, srcB_buffer, 0, h_srcB.data(), srcB_bytes, 0, nullptr, nullptr));

  std::vector<elem_t> h_dst(dst_bytes / sizeof(elem_t), 0xCCCC);
  RT_CHECK(vx_enqueue_write(queue, dst_buffer, 0, h_dst.data(), dst_bytes, 0, nullptr, nullptr));

  constexpr uint32_t kDescA = 0, kDescB = 1;

  // Program A: row-major (default). Iterates dim0=K inner, dim1=M outer.
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescA, kernel_arg.srcA_addr,
    /*size0=*/K, /*size1=*/M,
    /*stride0_bytes=*/K * sizeof(elem_t),
    /*tile0=*/tileK, /*tile1=*/tileM,
    /*elem_bytes=*/sizeof(elem_t)));
  // (no set_layout call → row-major)

  // Program B: K-major (LAYOUT=K_MAJOR). dim0=N inner, dim1=K outer.
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescB, kernel_arg.srcB_addr,
    /*size0=*/N, /*size1=*/K,
    /*stride0_bytes=*/N * sizeof(elem_t),
    /*tile0=*/tileN, /*tile1=*/tileK,
    /*elem_bytes=*/sizeof(elem_t)));
  RT_CHECK(vx_dxa_program_desc_set_layout(device, kDescB,
    VX_DXA_LAYOUT_K_MAJOR, /*rank=*/2, /*elem_bytes=*/sizeof(elem_t)));

  constexpr uint32_t kDescA_host = 0, kDescB_host = 1;
  (void)kDescA_host; (void)kDescB_host;

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 1;
  li.grid_dim[0]  = num_cta;
  // Match sgemm_tcu_wg_dxa per-CTA shape: warps_per_cta = ISSUE_WIDTH.
  li.block_dim[0] = 4 * VX_CFG_NUM_THREADS;
  li.lmem_size    = cta_bytes;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(launch_ev);

  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_bytes, 0, nullptr, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);

  // Verify each CTA's dump:
  //   A region: row-major. cta dump[0 .. a_bytes/2) → A_smem.
  //   B region: K-major.   cta dump[a_bytes/2 .. (a_bytes+b_bytes)/2).
  uint32_t errors = 0;
  std::string first_err;
  for (uint32_t cta_id = 0; cta_id < num_cta; ++cta_id) {
    uint32_t cta_row = cta_id / grid_n;
    uint32_t cta_col = cta_id % grid_n;
    const elem_t* dump = h_dst.data() + cta_id * (cta_bytes / sizeof(elem_t));
    const elem_t* A_dump = dump;
    const elem_t* B_dump = dump + a_bytes / sizeof(elem_t);

    // A: smem[m * tileK + k] = src[(cta_row*tileM + m) * K + k]
    for (uint32_t m = 0; m < tileM; ++m) {
      for (uint32_t k = 0; k < tileK; ++k) {
        elem_t exp = h_srcA[(cta_row * tileM + m) * K + k];
        elem_t act = A_dump[m * tileK + k];
        if (exp != act) {
          if (errors == 0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
              "A: cta=%u m=%u k=%u expected=0x%04x actual=0x%04x",
              cta_id, m, k, exp, act);
            first_err = buf;
          }
          ++errors;
        }
      }
    }
    // B: smem[n * tileK + k] = src[k * N + (cta_col * tileN + n)]
    for (uint32_t n = 0; n < tileN; ++n) {
      for (uint32_t k = 0; k < tileK; ++k) {
        elem_t exp = h_srcB[k * N + (cta_col * tileN + n)];
        elem_t act = B_dump[n * tileK + k];
        if (exp != act) {
          if (errors == 0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
              "B: cta=%u n=%u k=%u expected=0x%04x actual=0x%04x",
              cta_id, n, k, exp, act);
            first_err = buf;
          }
          ++errors;
        }
      }
    }
  }

  if (errors) {
    std::cout << "FAILED — " << errors << " mismatched elements (of "
              << (num_cta * (tileM*tileK + tileN*tileK)) << ")\n";
    std::cout << "first: " << first_err << "\n";
    cleanup();
    return 1;
  }
  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
