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
              "P is fp16 and S, O are fp32");

uint32_t xl = 64;
uint32_t xn = 32;
uint32_t xd = 64;
uint32_t xb = 2;
uint32_t xc = 1;

// fp16 rounding of P and MUFU's approximations.
constexpr double kMaxRelError = 0.01;

vx_device_h device = nullptr;
vx_buffer_h buffers[4] = {};
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

static double htod(uint16_t h) {
  return bit_cast<float>(rv_htof_s(h, 0, nullptr));
}

int main(int argc, char* argv[]) {
  int c;
  while ((c = getopt(argc, argv, "l:n:d:b:c:")) != -1) {
    switch (c) {
    case 'l': xl = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'd': xd = atoi(optarg); break;
    case 'b': xb = atoi(optarg); break;
    case 'c': xc = atoi(optarg); break;
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
  const uint32_t MT = M * xc;
  const uint32_t L = xl, N = xn, D = xd, NB = xb, LT = L * NB;
  const uint32_t nr = WGMMA_NRC;
  const uint32_t stage_k = KB * cfg::tileK;
  if ((L % (2 * nr)) || (N % nr) || (D % stage_k) || (L % stage_k)) {
    std::cout << "Error: L must be a multiple of " << 2 * nr << ", N of " << nr
              << ", D and L of KB*tileK=" << stage_k << std::endl;
    return -1;
  }

  // K_T and V_T are the K-major forms.
  std::vector<uint16_t> h_Q(size_t(MT) * D), h_KT(size_t(LT) * D), h_VT(size_t(N) * LT);
  for (auto& x : h_Q) x = gen_fp16();
  for (auto& x : h_KT) x = gen_fp16();
  for (auto& x : h_VT) x = gen_fp16();

  uint32_t smem_size = 2 * (M * stage_k + stage_k * std::max(L, N)) * sizeof(uint16_t);
  smem_size += TMEM_SLOT_BYTES;

  kernel_arg.L = L; kernel_arg.N = N; kernel_arg.D = D; kernel_arg.NB = NB;
  kernel_arg.scale = 1.0f / std::sqrt(float(D));
  uint64_t q_addr, kt_addr, vt_addr;
  RT_CHECK(vx_buffer_create(device, h_Q.size() * 2, VX_MEM_READ, &buffers[0]));
  RT_CHECK(vx_buffer_create(device, h_KT.size() * 2, VX_MEM_READ, &buffers[1]));
  RT_CHECK(vx_buffer_create(device, h_VT.size() * 2, VX_MEM_READ, &buffers[2]));
  RT_CHECK(vx_buffer_create(device, size_t(MT) * N * 4, VX_MEM_WRITE, &buffers[3]));
  RT_CHECK(vx_buffer_address(buffers[0], &q_addr));
  RT_CHECK(vx_buffer_address(buffers[1], &kt_addr));
  RT_CHECK(vx_buffer_address(buffers[2], &vt_addr));
  RT_CHECK(vx_buffer_address(buffers[3], &kernel_arg.O_addr));
  RT_CHECK(vx_enqueue_write(queue, buffers[0], 0, h_Q.data(), h_Q.size() * 2, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, buffers[1], 0, h_KT.data(), h_KT.size() * 2, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, buffers[2], 0, h_VT.data(), h_VT.size() * 2, 0, nullptr, nullptr));

  auto program = [&](uint32_t slot, uint64_t addr, uint32_t k, uint32_t rows, uint32_t tile_rows) {
    RT_CHECK(vortex::dxa::program_2d(queue, slot, addr, k, rows, k * 2, stage_k, tile_rows, 2));
    RT_CHECK(vortex::dxa::set_layout(queue, slot, vortex::dxa::Layout::RowMajor, 2, 2));
  };
  program(0, q_addr, D, MT, M);
  program(1, kt_addr, D, LT, L);
  program(2, vt_addr, LT, N, N);

  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<float> h_O(size_t(MT) * N);

  vx_event_h launch_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 2;
  li.grid_dim[0]  = xc;
  li.grid_dim[1]  = 1;
  li.block_dim[0] = warps * uint32_t(NT);
  li.block_dim[1] = 1;
  li.lmem_size    = smem_size;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  vx_event_h read_o = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_O.data(), buffers[3], 0, h_O.size() * 4, 1, &launch_ev, &read_o));
  RT_CHECK(vx_event_wait_value(read_o, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_o);
  vx_event_release(launch_ev);

  // Exact softmax over all keys at once; the device computes it online per block.
  int errors = 0;
  double max_err = 0;
  const double scale = kernel_arg.scale;
  std::vector<double> w(LT);
  for (uint32_t m = 0; m < MT; ++m) {
    double s_max = -INFINITY;
    for (uint32_t l = 0; l < LT; ++l) {
      double acc = 0;
      for (uint32_t k = 0; k < D; ++k) {
        acc += htod(h_Q[size_t(m) * D + k]) * htod(h_KT[size_t(l) * D + k]);
      }
      w[l] = acc * scale;
      s_max = std::max(s_max, w[l]);
    }
    double sum = 0;
    for (auto& x : w) {
      x = std::exp(x - s_max);
      sum += x;
    }
    for (uint32_t n = 0; n < N; ++n) {
      double ref = 0;
      for (uint32_t l = 0; l < LT; ++l) {
        ref += (w[l] / sum) * htod(h_VT[size_t(n) * LT + l]);
      }
      double actual = h_O[size_t(m) * N + n];
      double err = std::abs(actual - ref) / std::abs(ref);
      max_err = std::max(max_err, err);
      if (!(err <= kMaxRelError) && errors++ < 20) {
        printf("*** O[%u][%u]: expected=%f, actual=%f\n", m, n, ref, actual);
      }
    }
  }
  std::cout << "O: max relative error " << max_err << ", bound " << kMaxRelError << std::endl;

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!" << std::endl << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
