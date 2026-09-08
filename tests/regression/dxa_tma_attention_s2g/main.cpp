#include "common.h"

#include <VX_types.h>
#include <dxa.h>
#include <tensor_cfg.h>
#include <vortex2.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <vortex.h>

namespace vt = vortex::tensor;
using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, WGMMA_NRC>;

static constexpr uint32_t kM = ATT_COMPUTE_WARPS * cfg::xtileM;
static constexpr uint32_t kN = cfg::xtileN;
static constexpr uint32_t kD = kN;
static constexpr uint32_t kK = ATT_K_TILES * cfg::tileK;
static constexpr uint32_t kRows = ATT_ITERS * kM;
static constexpr uint32_t kQWords = kRows * kK;
static constexpr uint32_t kKWords = kK * kN;
static constexpr uint32_t kVWords = kN * kD;
static constexpr uint32_t kScoreWords = kM * kN;
static constexpr uint32_t kStageOutputWords = kM * kD;
static constexpr uint32_t kOutputWords = kRows * kD;
static constexpr uint32_t kStatsWords = kRows;
static constexpr uint32_t kQKStageWords = kM * cfg::tileK + cfg::tileK * kN;
static constexpr uint32_t kSmemBytes = 2 * kQKStageWords * sizeof(uint16_t)
    + kVWords * sizeof(float) + kScoreWords * sizeof(float)
    + ATT_OUT_STAGES * (kStageOutputWords + kM) * sizeof(float);

static vx_device_h device = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module = nullptr;
static vx_kernel_h kernel = nullptr;
static vx_buffer_h q_buffer = nullptr;
static vx_buffer_h k_buffer = nullptr;
static vx_buffer_h v_buffer = nullptr;
static vx_buffer_h out_buffer = nullptr;
static vx_buffer_h stats_buffer = nullptr;

static uint32_t f32_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static bool close_f32(uint32_t bits, float expected) {
  float got;
  std::memcpy(&got, &bits, sizeof(got));
  return std::isfinite(got) && std::fabs(got - expected) <= 1.0e-5f * std::max(1.0f, std::fabs(expected));
}

static uint16_t fp16_bits(uint32_t value) {
  static constexpr uint16_t bits[] = {0x3c00, 0x4000, 0x4200, 0x4400};
  return bits[value & 3u];
}

static float q_value(uint32_t row, uint32_t col) { return static_cast<float>(1u + ((row + col) & 3u)); }
static float k_value(uint32_t row, uint32_t col) { return static_cast<float>(1u + ((row + 2 * col) % 4u)); }
static float v_value(uint32_t row, uint32_t col) { return static_cast<float>(1u + ((2 * row + col) & 3u)); }
static float attention_weight(float score) { return score * 0.0625f + 1.0f; }

static void cleanup() {
  if (q_buffer) vx_buffer_release(q_buffer);
  if (k_buffer) vx_buffer_release(k_buffer);
  if (v_buffer) vx_buffer_release(v_buffer);
  if (out_buffer) vx_buffer_release(out_buffer);
  if (stats_buffer) vx_buffer_release(stats_buffer);
  if (kernel) vx_kernel_release(kernel);
  if (module) vx_module_release(module);
  if (queue) vx_queue_release(queue);
  if (device) vx_device_release(device);
  q_buffer = k_buffer = v_buffer = out_buffer = stats_buffer = nullptr;
  kernel = nullptr; module = nullptr; queue = nullptr; device = nullptr;
}

#define RT_CHECK(expr) do { int ret = (expr); if (ret) { std::cerr << #expr << " failed: " << ret << "\n"; cleanup(); return -1; } } while (0)

static int alloc_buffer(vx_buffer_h* buffer, uint64_t bytes, uint64_t* addr, uint32_t fill = 0) {
  int ret = vx_buffer_create(device, bytes, VX_MEM_READ_WRITE | VX_MEM_PHYS, buffer);
  if (ret) return ret;
  if ((ret = vx_buffer_address(*buffer, addr))) return ret;
  std::vector<uint32_t> init((bytes + 3) / 4, fill);
  if ((ret = vx_enqueue_write(queue, *buffer, 0, init.data(), bytes, 0, nullptr, nullptr))) return ret;
  return vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
}

static int read_words(vx_buffer_h buffer, uint32_t words, std::vector<uint32_t>* data) {
  data->assign(words, 0);
  vx_event_h event = nullptr;
  int ret = vx_enqueue_read(queue, data->data(), buffer, 0, words * sizeof(uint32_t), 0, nullptr, &event);
  if (!ret) ret = vx_event_wait_value(event, 1, VX_TIMEOUT_INFINITE);
  if (event) vx_event_release(event);
  return ret;
}

static int launch(const kernel_arg_t& args, uint32_t threads) {
  vx_launch_info_t info{};
  info.struct_size = sizeof(info); info.kernel = kernel; info.args_host = &args;
  info.args_size = sizeof(args); info.ndim = 1; info.grid_dim[0] = 1;
  info.block_dim[0] = threads; info.lmem_size = kSmemBytes;
  vx_event_h event = nullptr;
  int ret = vx_enqueue_launch(queue, &info, 0, nullptr, &event);
  if (!ret) ret = vx_event_wait_value(event, 1, VX_TIMEOUT_INFINITE);
  if (event) vx_event_release(event);
  return ret;
}

int main(int argc, char** argv) {
  const char* kernel_file = argc > 1 ? argv[1] : "kernel.vxbin";
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t queue_info{sizeof(queue_info), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(device, &queue_info, &queue));
  uint64_t threads = 0, warps = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps));
  if (threads != VX_CFG_NUM_THREADS || warps < ATT_WARPS) {
    std::cout << "SKIPPED: requires " << VX_CFG_NUM_THREADS << " threads/warp and " << ATT_WARPS << " resident warps\n";
    cleanup();
    return 0;
  }

  const uint32_t block_threads = VX_CFG_NUM_THREADS * ATT_WARPS;
  kernel_arg_t args{};
  RT_CHECK(alloc_buffer(&q_buffer, kQWords * sizeof(uint16_t), &args.q_addr));
  RT_CHECK(alloc_buffer(&k_buffer, kKWords * sizeof(uint16_t), &args.k_addr));
  RT_CHECK(alloc_buffer(&v_buffer, kVWords * sizeof(float), &args.v_addr));
  RT_CHECK(alloc_buffer(&out_buffer, kOutputWords * sizeof(uint32_t), &args.out_addr, 0xbaadf00d));
  RT_CHECK(alloc_buffer(&stats_buffer, kStatsWords * sizeof(uint32_t), &args.stats_addr, 0xbaadf00d));
  if (std::getenv("ATT_DUMP")) std::cout << "addr out=0x" << std::hex << args.out_addr << " stats=0x" << args.stats_addr << std::dec << "\n";

  std::vector<uint16_t> q(kQWords), k(kKWords);
  std::vector<float> v(kVWords);
  for (uint32_t row = 0; row < kRows; ++row)
    for (uint32_t col = 0; col < kK; ++col) q[row * kK + col] = fp16_bits((row + col) & 3u);
  for (uint32_t row = 0; row < kN; ++row)
    for (uint32_t col = 0; col < kD; ++col) v[row * kD + col] = v_value(row, col);
  for (uint32_t row = 0; row < kK; ++row)
    for (uint32_t col = 0; col < kN; ++col) k[row * kN + col] = fp16_bits((row + 2 * col) % 4u);
  RT_CHECK(vx_enqueue_write(queue, q_buffer, 0, q.data(), q.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, k_buffer, 0, k.data(), k.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, v_buffer, 0, v.data(), v.size() * sizeof(float), 0, nullptr, nullptr));
  RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));

  RT_CHECK(vortex::dxa::program_2d(device, 0, args.q_addr, kK, kRows, kK * sizeof(uint16_t), cfg::tileK, kM, sizeof(uint16_t)));
  RT_CHECK(vortex::dxa::program_2d(device, 1, args.k_addr, kN, kK, kN * sizeof(uint16_t), kN, cfg::tileK, sizeof(uint16_t)));
  RT_CHECK(vortex::dxa::set_layout(device, 1, vortex::dxa::Layout::BlockMajor, 2, sizeof(uint16_t)));
  RT_CHECK(vortex::dxa::set_tile_geometry(device, 1, cfg::tcN));
  RT_CHECK(vortex::dxa::program_2d(device, 2, args.v_addr, kD, kN, kD * sizeof(float), kD, kN, sizeof(float)));
  RT_CHECK(vortex::dxa::program_2d(device, 3, args.out_addr, kN, kRows, kN * sizeof(float), kN, kM, sizeof(float)));
  RT_CHECK(vortex::dxa::program_2d(device, 4, args.stats_addr, 1, kRows, sizeof(float), 1, kM, sizeof(float)));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module));
  RT_CHECK(vx_module_get_kernel(module, "main", &kernel));
  RT_CHECK(vx_check_occupancy(device, block_threads, kSmemBytes));

  // Run one mode in isolation when diagnosing a stalled asynchronous path.
  // Keeping this opt-in avoids combining mode counters or poisoning state
  // from the preceding baseline run.
  const char* mode_env = std::getenv("S2G_ONLY_MODE");
  const int selected_mode = mode_env ? std::atoi(mode_env) : -1;
  if (selected_mode < -1 || selected_mode > 2) {
    std::cerr << "S2G_ONLY_MODE must be 0, 1, or 2\n";
    cleanup();
    return -1;
  }
  const uint32_t first_mode = selected_mode < 0 ? 0u : uint32_t(selected_mode);
  const uint32_t last_mode = selected_mode < 0 ? 3u : first_mode + 1u;
  for (uint32_t mode = first_mode; mode < last_mode; ++mode) {
    std::vector<uint32_t> poison_out(kOutputWords, 0xbaadf00d), poison_stats(kStatsWords, 0xbaadf00d);
    RT_CHECK(vx_enqueue_write(queue, out_buffer, 0, poison_out.data(), poison_out.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, stats_buffer, 0, poison_stats.data(), poison_stats.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
    args.mode = mode;
    RT_CHECK(launch(args, block_threads));
    RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
    if (std::getenv("ATT_DUMP_PERF")) RT_CHECK(vx_device_dump_perf(device, stdout));
    std::vector<uint32_t> out, stats;
    RT_CHECK(read_words(out_buffer, kOutputWords, &out));
    RT_CHECK(read_words(stats_buffer, kStatsWords, &stats));
    if (std::getenv("ATT_DUMP")) {
      for (uint32_t rr = 0; rr < kM; ++rr) {
        std::cout << "out[" << rr << "]";
        for (uint32_t c = 0; c < kN; ++c) { float x; std::memcpy(&x, &out[rr * kN + c], sizeof(x)); std::cout << " " << x; }
        std::cout << "\n";
      }
      std::cout << "stats";
      for (uint32_t rr = 0; rr < kM; ++rr) { float x; std::memcpy(&x, &stats[rr], sizeof(x)); std::cout << " " << x; }
      std::cout << "\n";
    }

    uint32_t errors = 0, printed = 0;
    for (uint32_t iter = 0; iter < ATT_ITERS; ++iter) {
      for (uint32_t r = 0; r < kM; ++r) {
        const uint32_t global_row = iter * kM + r;
        float denom = 0.0f;
        for (uint32_t c = 0; c < kN; ++c) {
          float dot = 0.0f;
          for (uint32_t kk = 0; kk < kK; ++kk) dot += q_value(global_row, kk) * k_value(kk, c);
          denom += attention_weight(dot);
        }
        for (uint32_t c = 0; c < kD; ++c) {
          float value = 0.0f;
          for (uint32_t n = 0; n < kN; ++n) {
            float dot = 0.0f;
            for (uint32_t kk = 0; kk < kK; ++kk) dot += q_value(global_row, kk) * k_value(kk, n);
            value += attention_weight(dot) * v_value(n, c);
          }
          const float expected = value / denom;
          const uint32_t got = out[iter * kM * kD + r * kD + c];
          if (!close_f32(got, expected)) {
            ++errors;
            if (printed++ < 8) std::cerr << "out mismatch i=" << iter << " r=" << r << " c=" << c << " got=0x" << std::hex << got << " exp=0x" << f32_bits(expected) << std::dec << "\n";
          }
        }
        float expected_denom = 0.0f;
        for (uint32_t n = 0; n < kN; ++n) {
          float dot = 0.0f;
          for (uint32_t kk = 0; kk < kK; ++kk) dot += q_value(global_row, kk) * k_value(kk, n);
          expected_denom += attention_weight(dot);
        }
        const uint32_t expected_stat = f32_bits(expected_denom);
        if (stats[global_row] != expected_stat) {
          ++errors;
          if (printed++ < 8) std::cerr << "stats mismatch row=" << global_row << " got=0x" << std::hex << stats[global_row] << " exp=0x" << expected_stat << std::dec << "\n";
        }
      }
    }
    if (errors) {
      std::cerr << "mode " << mode << " FAILED: " << errors << " mismatches\n";
      cleanup();
      return -1;
    }
    const char* mode_name = mode == 0 ? "TMA+S2G" : (mode == 1 ? "TMA+global" : "LSU+global");
    std::cout << "mode " << mode << " (" << mode_name << ") PASS (QK WGMMA + V gate, " << kM << "x" << kN << ")\n";
  }
  std::cout << "PASS: TMA Q/K/V -> WGMMA score -> stable positive epilogue -> grouped S2G tile+stats (L2 enabled)\n";
  cleanup();
  return 0;
}
