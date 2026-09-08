#include "common.h"
#include <VX_types.h>
#include <dxa.h>
#include <tensor_cfg.h>
#include <vortex2.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace vt = vortex::tensor;
using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, WGMMA_NRC>;
constexpr uint32_t kM = TMA_PIPE_COMPUTE_WARPS * cfg::xtileM, kN = cfg::xtileN;
constexpr uint32_t kK = TMA_PIPE_K_TILES * cfg::tileK, kOutWords = kM * kN;
constexpr uint32_t kInputWords = kM * cfg::tileK + cfg::tileK * kN;
constexpr uint32_t kOutStride = kOutWords + kN + TMA_PIPE_CANARY_WORDS;
constexpr uint32_t kStateStride = 3 * kM + TMA_PIPE_META_WORDS + TMA_PIPE_CANARY_WORDS;
constexpr uint32_t kSmemBytes = TMA_PIPE_IN_STAGES * kInputWords * 2 + (TMA_PIPE_OUT_STAGES * kOutStride + TMA_PIPE_STATE_STAGES * kStateStride) * 4;
constexpr uint32_t kGuardWords = TMA_PIPE_CANARY_WORDS;

static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static uint16_t half_bits(uint32_t x) { const uint16_t v[] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500}; return v[x]; }
static float a_value(uint32_t r, uint32_t c) { return float(1 + ((r + c) & 3)); }
static float b_value(uint32_t r, uint32_t c) { return float(1 + (r + c) % 5); }
static float result(uint32_t iter, uint32_t r, uint32_t c) {
  float x = 0;
  for (uint32_t k = 0; k < kK; ++k) { x += a_value(r, k) * b_value(k, c); }
  return x * 0.5f + float(iter + 1);
}

struct Device {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module = nullptr;
  vx_kernel_h kernel = nullptr;
  std::vector<vx_buffer_h> buffers;
  ~Device() {
    for (auto b : buffers) { vx_buffer_release(b); }
    if (kernel) { vx_kernel_release(kernel); }
    if (module) { vx_module_release(module); }
    if (queue) { vx_queue_release(queue); }
    if (device) { vx_device_release(device); }
  }
  int allocate(uint32_t bytes, vx_buffer_h& buffer, uint64_t& addr) {
    int rc = vx_buffer_create(device, bytes, VX_MEM_READ_WRITE | VX_MEM_PHYS, &buffer);
    if (rc) { return rc; }
    buffers.push_back(buffer);
    return vx_buffer_address(buffer, &addr);
  }
  int write(vx_buffer_h b, const void* data, uint32_t bytes) { return vx_enqueue_write(queue, b, 0, data, bytes, 0, nullptr, nullptr); }
  int read(vx_buffer_h b, std::vector<uint32_t>& data) { return vx_enqueue_read(queue, data.data(), b, 0, data.size() * 4, 0, nullptr, nullptr); }
  int finish() { return vx_queue_finish(queue, VX_TIMEOUT_INFINITE); }
};

#define CHECK(call) do { const int rc = (call); if (rc) { std::cerr << #call << " failed: " << rc << '\n'; return 1; } } while (0)

int main(int argc, char** argv) {
  Device d;
  CHECK(vx_device_open(0, &d.device));
  vx_queue_info_t qi{sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  CHECK(vx_queue_create(d.device, &qi, &d.queue));
  uint64_t threads = 0, warps = 0;
  CHECK(vx_device_query(d.device, VX_CAPS_NUM_THREADS, &threads));
  CHECK(vx_device_query(d.device, VX_CAPS_NUM_WARPS, &warps));
  if (threads != VX_CFG_NUM_THREADS || warps < TMA_PIPE_WARPS) { std::cerr << "requires 8 warps of 8 lanes\n"; return 1; }
  CHECK(vx_check_occupancy(d.device, TMA_PIPE_WARPS * VX_CFG_NUM_THREADS, kSmemBytes));
  kernel_arg_t args{};
  vx_buffer_h ab, bb;
  CHECK(d.allocate(kM * kK * 2, ab, args.a_addr));
  CHECK(d.allocate(kK * kN * 2, bb, args.b_addr));
  std::vector<uint16_t> a(kM * kK), b(kK * kN);
  for (uint32_t r = 0; r < kM; ++r) { for (uint32_t c = 0; c < kK; ++c) { a[r * kK + c] = half_bits((r + c) & 3); } }
  for (uint32_t r = 0; r < kK; ++r) { for (uint32_t c = 0; c < kN; ++c) { b[r * kN + c] = half_bits((r + c) % 5); } }
  CHECK(d.write(ab, a.data(), a.size() * 2));
  CHECK(d.write(bb, b.data(), b.size() * 2));
  const uint32_t sizes[] = {TMA_PIPE_ITERS * kOutWords, TMA_PIPE_ITERS * kN, TMA_PIPE_ITERS * 2 * kM,
    TMA_PIPE_ITERS * kM, TMA_PIPE_ITERS * TMA_PIPE_META_WORDS, (TMA_PIPE_OUT_STAGES + TMA_PIPE_STATE_STAGES) * kGuardWords};
  const char* names[] = {"output", "bias", "row statistics", "mean checkpoint", "metadata", "SMEM guards"};
  uint64_t* addresses[] = {&args.out_addr, &args.bias_addr, &args.stats_addr, &args.state_addr, &args.meta_addr, &args.canary_addr};
  vx_buffer_h outputs[6];
  std::vector<uint32_t> expected[6], actual[6];
  for (uint32_t i = 0; i < 6; ++i) {
    CHECK(d.allocate((sizes[i] + 2 * kGuardWords) * 4, outputs[i], *addresses[i]));
    *addresses[i] += kGuardWords * 4;
    expected[i].assign(sizes[i] + 2 * kGuardWords, TMA_PIPE_CANARY);
    actual[i] = expected[i];
  }
  for (uint32_t it = 0; it < TMA_PIPE_ITERS; ++it) {
    for (uint32_t r = 0; r < kM; ++r) {
      float sum = 0, maximum = 0;
      for (uint32_t c = 0; c < kN; ++c) {
        const float x = result(it, r, c);
        expected[0][kGuardWords + it * kOutWords + r * kN + c] = bits(x);
        sum += x;
        maximum = std::max(maximum, x);
      }
      expected[2][kGuardWords + it * 2 * kM + r] = bits(sum);
      expected[2][kGuardWords + it * 2 * kM + kM + r] = bits(maximum);
      if ((it & 1u) == 0) { expected[3][kGuardWords + it * kM + r] = bits(sum / kN); }
    }
    for (uint32_t c = 0; c < kN; ++c) { expected[1][kGuardWords + it * kN + c] = bits(float(it + c) + 0.25f); }
    if ((it & 3u) == 0) { for (uint32_t c = 0; c < TMA_PIPE_META_WORDS; ++c) { expected[4][kGuardWords + it * TMA_PIPE_META_WORDS + c] = bits(float(it * TMA_PIPE_META_WORDS + c)); } }
  }
  CHECK(d.finish());
  CHECK(vortex::dxa::program_2d(d.device, 0, args.a_addr, kK, kM, kK * 2, cfg::tileK, kM, 2));
  CHECK(vortex::dxa::program_2d(d.device, 1, args.b_addr, kN, kK, kN * 2, kN, cfg::tileK, 2));
  CHECK(vortex::dxa::set_layout(d.device, 1, vortex::dxa::Layout::BlockMajor, 2, 2));
  CHECK(vortex::dxa::set_tile_geometry(d.device, 1, cfg::tcN));
  CHECK(vortex::dxa::program_2d(d.device, 2, args.out_addr, kN, TMA_PIPE_ITERS * kM, kN * 4, kN, kM, 4));
  CHECK(vortex::dxa::program_2d(d.device, 3, args.bias_addr, kN, TMA_PIPE_ITERS, kN * 4, kN, 1, 4));
  CHECK(vortex::dxa::program_3d(d.device, 4, args.stats_addr, kM, 1, TMA_PIPE_ITERS * 2, kM * 4, kM * 4, kM, 1, 2, 4));
  CHECK(vortex::dxa::program_2d(d.device, 5, args.state_addr, kM, TMA_PIPE_ITERS, kM * 4, kM, 1, 4));
  CHECK(vortex::dxa::program_2d(d.device, 6, args.meta_addr, TMA_PIPE_META_WORDS, TMA_PIPE_ITERS, TMA_PIPE_META_WORDS * 4, TMA_PIPE_META_WORDS, 1, 4));
  CHECK(vx_module_load_file(d.device, argc > 1 ? argv[1] : "kernel.vxbin", &d.module));
  CHECK(vx_module_get_kernel(d.module, "main", &d.kernel));
  const char* mode_env = std::getenv("S2G_ONLY_MODE");
  const int selected = mode_env ? std::atoi(mode_env) : -1;
  if (selected < -1 || selected > 3) { std::cerr << "S2G_ONLY_MODE must be 0, 1, 2, or 3\n"; return 1; }
  const char* modes[] = {"TMA + pipelined S2G", "TMA + staged global", "LSU + staged global", "TMA + serial S2G"};
  for (uint32_t mode = selected < 0 ? 0 : selected; mode < (selected < 0 ? 4u : uint32_t(selected + 1)); ++mode) {
    for (uint32_t i = 0; i < 6; ++i) {
      std::fill(actual[i].begin(), actual[i].end(), TMA_PIPE_CANARY);
      CHECK(d.write(outputs[i], actual[i].data(), actual[i].size() * 4));
    }
    CHECK(d.finish());
    args.mode = mode;
    vx_launch_info_t info{};
    info.struct_size = sizeof(info); info.kernel = d.kernel; info.args_host = &args; info.args_size = sizeof(args);
    info.ndim = 1; info.grid_dim[0] = 1; info.block_dim[0] = TMA_PIPE_WARPS * VX_CFG_NUM_THREADS; info.lmem_size = kSmemBytes;
    CHECK(vx_enqueue_launch(d.queue, &info, 0, nullptr, nullptr));
    CHECK(d.finish()); // Runtime/DXA drain, not wait.read, establishes host observation.
    if (std::getenv("S2G_DUMP_PERF")) { CHECK(vx_device_dump_perf(d.device, stdout)); }
    for (uint32_t i = 0; i < 6; ++i) { CHECK(d.read(outputs[i], actual[i])); }
    CHECK(d.finish());
    uint32_t errors = 0;
    for (uint32_t i = 0; i < 6; ++i) {
      for (uint32_t j = 0; j < expected[i].size(); ++j) {
        if (actual[i][j] != expected[i][j]) {
          if (errors++ < 8) { std::cerr << names[i] << '[' << j << "] got=0x" << std::hex << actual[i][j] << " expected=0x" << expected[i][j] << std::dec << '\n'; }
        }
      }
    }
    if (errors) { std::cerr << modes[mode] << " FAIL mismatches=" << errors << '\n'; return 1; }
    std::cout << "mode " << mode << " " << modes[mode] << " PASS; " << kM << 'x' << kN << 'x' << kK << ", iters=" << TMA_PIPE_ITERS
              << ", stages=" << TMA_PIPE_IN_STAGES << '/' << TMA_PIPE_OUT_STAGES << '/' << TMA_PIPE_STATE_STAGES << '\n';
  }
  std::cout << "PASS: independent output/state issuers; 2D/3D groups; optional 1/2/3-op groups; source/destination canaries; L2 enabled\n";
}
