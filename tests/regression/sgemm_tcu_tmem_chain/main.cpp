#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex2.h>
#include <dxa.h>

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;
using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

static_assert(std::is_same<vt::ITYPE, vt::fp16>::value && std::is_same<vt::OTYPE, vt::fp32>::value,
              "the chain converts fp32 S to fp16 P");

uint32_t xl = 64;
uint32_t xn = 32;
uint32_t xk = 64;

vx_device_h device = nullptr;
vx_buffer_h buffers[5] = {};
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    for (auto b : buffers) {
      if (b) vx_buffer_release(b);
    }
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static uint16_t gen_fp16() {
  auto f = float(rand()) / RAND_MAX;
  return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr);
}

static float htof(uint16_t h) {
  return bit_cast<float>(rv_htof_s(h, 0, nullptr));
}

static int check(const char* name, const std::vector<float>& actual, const std::vector<float>& ref, uint32_t ulp) {
  int errors = 0;
  uint32_t max_ulp = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    uint32_t d = std::abs(bit_cast<int32_t>(actual[i]) - bit_cast<int32_t>(ref[i]));
    max_ulp = std::max(max_ulp, d);
    if (d > ulp && errors++ < 20) {
      printf("*** %s[%zu]: expected=%f, actual=%f\n", name, i, ref[i], actual[i]);
    }
  }
  std::cout << name << ": max drift " << max_ulp << " ULP, bound " << ulp << std::endl;
  return errors;
}

int main(int argc, char* argv[]) {
  int c;
  while ((c = getopt(argc, argv, "l:n:k:")) != -1) {
    switch (c) {
    case 'l': xl = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'k': xk = atoi(optarg); break;
    default: return -1;
    }
  }

  srand(50);
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t NT, issue_width;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &NT));
  RT_CHECK(vx_device_query(device, VX_CAPS_ISSUE_WIDTH, &issue_width));

  const uint32_t warps = issue_width;
  const uint32_t M = warps * cfg::xtileM;
  const uint32_t L = xl, N = xn, K1 = xk;
  const uint32_t nr = WGMMA_NRC;
  const uint32_t stage_k = KB * cfg::tileK;
  if ((L % (2 * nr)) || (N % nr) || (K1 % stage_k) || (L % stage_k)) {
    std::cout << "Error: L must be a multiple of " << 2 * nr << ", N of " << nr
              << ", K and L of KB*tileK=" << stage_k << std::endl;
    return -1;
  }

  // Random operands; K_T and V_T are the K-major forms.
  std::vector<uint16_t> h_Q(size_t(M) * K1), h_KT(size_t(L) * K1), h_VT(size_t(N) * L);
  for (auto& x : h_Q) x = gen_fp16();
  for (auto& x : h_KT) x = gen_fp16();
  for (auto& x : h_VT) x = gen_fp16();

  uint32_t smem_size = 2 * (M * stage_k + stage_k * std::max(L, N)) * sizeof(uint16_t);
  smem_size += TMEM_SLOT_BYTES;

  kernel_arg.M = M; kernel_arg.L = L; kernel_arg.N = N; kernel_arg.K1 = K1;
  RT_CHECK(vx_buffer_create(device, h_Q.size() * 2, VX_MEM_READ, &buffers[0]));
  RT_CHECK(vx_buffer_create(device, h_KT.size() * 2, VX_MEM_READ, &buffers[1]));
  RT_CHECK(vx_buffer_create(device, h_VT.size() * 2, VX_MEM_READ, &buffers[2]));
  RT_CHECK(vx_buffer_create(device, size_t(M) * L * 4, VX_MEM_WRITE, &buffers[3]));
  RT_CHECK(vx_buffer_create(device, size_t(M) * N * 4, VX_MEM_WRITE, &buffers[4]));
  RT_CHECK(vx_buffer_address(buffers[0], &kernel_arg.Q_addr));
  RT_CHECK(vx_buffer_address(buffers[1], &kernel_arg.KT_addr));
  RT_CHECK(vx_buffer_address(buffers[2], &kernel_arg.VT_addr));
  RT_CHECK(vx_buffer_address(buffers[3], &kernel_arg.S_addr));
  RT_CHECK(vx_buffer_address(buffers[4], &kernel_arg.O_addr));
  RT_CHECK(vx_enqueue_write(queue, buffers[0], 0, h_Q.data(), h_Q.size() * 2, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, buffers[1], 0, h_KT.data(), h_KT.size() * 2, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, buffers[2], 0, h_VT.data(), h_VT.size() * 2, 0, nullptr, nullptr));

  auto program = [&](uint32_t slot, uint64_t addr, uint32_t k, uint32_t rows, uint32_t tile_rows) {
    RT_CHECK(vortex::dxa::program_2d(queue, slot, addr, k, rows, k * 2, stage_k, tile_rows, 2));
    RT_CHECK(vortex::dxa::set_layout(queue, slot, vortex::dxa::Layout::RowMajor, 2, 2));
  };
  program(0, kernel_arg.Q_addr, K1, M, M);
  program(1, kernel_arg.KT_addr, K1, L, L);
  program(2, kernel_arg.VT_addr, L, N, N);

  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<float> h_S(size_t(M) * L), h_O(size_t(M) * N);

  vx_event_h launch_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 2;
  li.grid_dim[0]  = 1;
  li.grid_dim[1]  = 1;
  li.block_dim[0] = warps * uint32_t(NT);
  li.block_dim[1] = 1;
  li.lmem_size    = smem_size;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  vx_event_h read_s = nullptr, read_o = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_S.data(), buffers[3], 0, h_S.size() * 4, 1, &launch_ev, &read_s));
  RT_CHECK(vx_enqueue_read(queue, h_O.data(), buffers[4], 0, h_O.size() * 4, 1, &read_s, &read_o));
  RT_CHECK(vx_event_wait_value(read_o, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_o);
  vx_event_release(read_s);
  vx_event_release(launch_ev);

  std::vector<float> ref_S(h_S.size()), ref_O(h_O.size());
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t l = 0; l < L; ++l) {
      double acc = 0;
      for (uint32_t k = 0; k < K1; ++k) {
        acc += double(htof(h_Q[size_t(m) * K1 + k])) * htof(h_KT[size_t(l) * K1 + k]);
      }
      ref_S[size_t(m) * L + l] = float(acc);
    }
    // P derives from the device's S so O's check isolates the second GEMM.
    for (uint32_t n = 0; n < N; ++n) {
      double acc = 0;
      for (uint32_t l = 0; l < L; ++l) {
        uint16_t p = rv_ftoh_s(bit_cast<uint32_t>(h_S[size_t(m) * L + l]), 0, nullptr);
        acc += double(htof(p)) * htof(h_VT[size_t(n) * L + l]);
      }
      ref_O[size_t(m) * N + n] = float(acc);
    }
  }

  int errors = check("S", h_S, ref_S, std::max(8u, K1 / 5));
  errors += check("O", h_O, ref_O, std::max(8u, L / 5));

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!" << std::endl << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
