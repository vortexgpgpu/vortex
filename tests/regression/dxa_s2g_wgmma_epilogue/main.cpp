#include <VX_types.h>
#include <dxa.h>
#include <tensor_cfg.h>
#include <vortex2.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "common.h"

namespace vt = vortex::tensor;
using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, WGMMA_NRC>;

static constexpr uint32_t kCtaM = S2G_TE_COMPUTE_WARPS * cfg::xtileM;
static constexpr uint32_t kOutWords = kCtaM * cfg::xtileN;
static constexpr uint32_t kBiasWords = cfg::xtileN;
static constexpr uint32_t kStatsWords = S2G_TE_STATS_X * S2G_TE_STATS_Y * S2G_TE_STATS_Z;
static constexpr uint32_t kOutStageWords = kOutWords + kBiasWords;
static constexpr uint32_t kOutStages = 2;
static constexpr uint32_t kStatsStages = S2G_TE_STATS_STAGES;

static uint32_t f32_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint16_t a_bits(uint32_t row, uint32_t col) {
  static constexpr uint16_t values[] = {0x3c00, 0x4000, 0x4200, 0x4400};
  return values[(row + col) & 3u];
}

static float a_value(uint32_t row, uint32_t col) {
  return static_cast<float>(1u + ((row + col) & 3u));
}

static uint16_t b_bits(uint32_t row, uint32_t col) {
  static constexpr uint16_t values[] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500};
  return values[(row + col) % 5u];
}

static float b_value(uint32_t row, uint32_t col) {
  return static_cast<float>(1u + ((row + col) % 5u));
}

static float reference_dot(uint32_t row, uint32_t col, uint32_t k) {
  float result = 0.0f;
  for (uint32_t i = 0; i < k; ++i)
    result += a_value(row, i) * b_value(i, col);
  return result;
}

#define RT_CHECK(expr) do { \
  int ret = (expr); \
  if (ret) { std::cerr << #expr << " failed: " << ret << "\n"; cleanup(); return -1; } \
} while (0)

static vx_device_h device;
static vx_queue_h queue;
static vx_module_h module;
static vx_kernel_h kernel;
static vx_buffer_h a_buffer;
static vx_buffer_h b_buffer;
static vx_buffer_h out_buffer;
static vx_buffer_h bias_buffer;
static vx_buffer_h stats_buffer;

static void cleanup() {
  if (a_buffer) vx_buffer_release(a_buffer);
  if (b_buffer) vx_buffer_release(b_buffer);
  if (out_buffer) vx_buffer_release(out_buffer);
  if (bias_buffer) vx_buffer_release(bias_buffer);
  if (stats_buffer) vx_buffer_release(stats_buffer);
  if (kernel) vx_kernel_release(kernel);
  if (module) vx_module_release(module);
  if (queue) vx_queue_release(queue);
  if (device) vx_device_release(device);
  a_buffer = b_buffer = out_buffer = bias_buffer = stats_buffer = nullptr;
  kernel = nullptr; module = nullptr; queue = nullptr; device = nullptr;
}

static int alloc(vx_buffer_h* buffer, uint64_t bytes, uint64_t* address, uint32_t fill = 0) {
  int ret = vx_buffer_create(device, bytes, VX_MEM_READ_WRITE | VX_MEM_PHYS, buffer);
  if (ret) return ret;
  if ((ret = vx_buffer_address(*buffer, address))) return ret;
  std::vector<uint32_t> data((bytes + 3) / 4, fill);
  if ((ret = vx_enqueue_write(queue, *buffer, 0, data.data(), bytes, 0, nullptr, nullptr)))
    return ret;
  return vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
}

static int read_buffer(vx_buffer_h buffer, uint32_t words, std::vector<uint32_t>* data) {
  data->assign(words, 0);
  vx_event_h event = nullptr;
  int ret = vx_enqueue_read(queue, data->data(), buffer, 0, words * sizeof(uint32_t), 0, nullptr, &event);
  if (!ret) ret = vx_event_wait_value(event, 1, VX_TIMEOUT_INFINITE);
  if (event) vx_event_release(event);
  return ret;
}

static int launch(const kernel_arg_t& args, uint32_t block_threads, uint32_t smem_bytes) {
  vx_launch_info_t info{};
  info.struct_size = sizeof(info);
  info.kernel = kernel;
  info.args_host = &args;
  info.args_size = sizeof(args);
  info.ndim = 1; info.grid_dim[0] = 1; info.block_dim[0] = block_threads; info.lmem_size = smem_bytes;
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
  if (threads != VX_CFG_NUM_THREADS || warps < S2G_TE_WARPS) {
    std::cout << "SKIPPED: requires " << VX_CFG_NUM_THREADS << "-thread warps and "
              << S2G_TE_WARPS << " resident warps\n";
    cleanup();
    return 0;
  }

  const uint32_t block_threads = static_cast<uint32_t>(threads * S2G_TE_WARPS);
  const uint32_t m = kCtaM;
  const uint32_t k = S2G_TE_K_TILES * cfg::tileK;
  const uint32_t out_words = S2G_TE_ITERS * kOutWords;
  const uint32_t bias_words = S2G_TE_ITERS * kBiasWords;
  const uint32_t stats_words = S2G_TE_ITERS * kStatsWords;
  const uint32_t smem_words = kOutStages * kOutStageWords
                            + kStatsStages * kStatsWords
                            + m * cfg::tileK + cfg::tileK * cfg::xtileN;
  const uint32_t smem_bytes = smem_words * sizeof(uint32_t);

  kernel_arg_t args{};
  const uint32_t a_words = m * k;
  const uint32_t b_words = k * cfg::xtileN;
  RT_CHECK(alloc(&a_buffer, a_words * sizeof(uint16_t), &args.a_addr));
  RT_CHECK(alloc(&b_buffer, b_words * sizeof(uint16_t), &args.b_addr));
  uint64_t out_addr = 0, bias_addr = 0, stats_addr = 0;
  RT_CHECK(alloc(&out_buffer, out_words * sizeof(uint32_t), &out_addr, 0xbaadf00d));
  RT_CHECK(alloc(&bias_buffer, bias_words * sizeof(uint32_t), &bias_addr, 0xbaadf00d));
  RT_CHECK(alloc(&stats_buffer, stats_words * sizeof(uint32_t), &stats_addr, 0xbaadf00d));

  std::vector<uint16_t> a_data(a_words), b_data(b_words);
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t col = 0; col < k; ++col)
      a_data[row * k + col] = a_bits(row, col);
  }
  for (uint32_t row = 0; row < k; ++row) {
    for (uint32_t col = 0; col < cfg::xtileN; ++col)
      b_data[row * cfg::xtileN + col] = b_bits(row, col);
  }
  RT_CHECK(vx_enqueue_write(queue, a_buffer, 0, a_data.data(),
                            a_data.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, b_buffer, 0, b_data.data(),
                            b_data.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));

  RT_CHECK(vortex::dxa::program_2d(device, 0, out_addr, cfg::xtileN,
                                   S2G_TE_ITERS * m, cfg::xtileN * 4,
                                   cfg::xtileN, m, 4));
  RT_CHECK(vortex::dxa::program_2d(device, 1, bias_addr, cfg::xtileN,
                                   S2G_TE_ITERS, cfg::xtileN * 4,
                                   cfg::xtileN, 1, 4));
  RT_CHECK(vortex::dxa::program_3d(device, 2, stats_addr,
                                   S2G_TE_STATS_X, S2G_TE_STATS_Y,
                                   S2G_TE_ITERS * S2G_TE_STATS_Z,
                                   S2G_TE_STATS_X * 4,
                                   S2G_TE_STATS_X * S2G_TE_STATS_Y * 4,
                                   S2G_TE_STATS_X, S2G_TE_STATS_Y,
                                   S2G_TE_STATS_Z, 4));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module));
  RT_CHECK(vx_module_get_kernel(module, "main", &kernel));
  RT_CHECK(vx_check_occupancy(device, block_threads, smem_bytes));

  for (uint32_t mode = 0; mode < 2; ++mode) {
    std::vector<uint32_t> poison_out(out_words, 0xbaadf00d);
    std::vector<uint32_t> poison_bias(bias_words, 0xbaadf00d);
    std::vector<uint32_t> poison_stats(stats_words, 0xbaadf00d);
    RT_CHECK(vx_enqueue_write(queue, out_buffer, 0, poison_out.data(),
                              poison_out.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, bias_buffer, 0, poison_bias.data(),
                              poison_bias.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, stats_buffer, 0, poison_stats.data(),
                              poison_stats.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
    args.mode = mode;
    RT_CHECK(launch(args, block_threads, smem_bytes));
    RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
    std::vector<uint32_t> out, bias, stats;
    RT_CHECK(read_buffer(out_buffer, out_words, &out));
    RT_CHECK(read_buffer(bias_buffer, bias_words, &bias));
    RT_CHECK(read_buffer(stats_buffer, stats_words, &stats));
    uint32_t errors = 0, out_errors = 0, bias_errors = 0, stats_errors = 0;
    for (uint32_t iter = 0; iter < S2G_TE_ITERS; ++iter) {
      for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < cfg::xtileN; ++col) {
          const float dot = reference_dot(row, col, k);
          const uint32_t expected = f32_bits((dot < 0.0f ? 0.0f : dot) * 0.5f
                                             + static_cast<float>(iter + 1));
          if (out[iter * kOutWords + row * cfg::xtileN + col] != expected) {
            ++errors;
            ++out_errors;
          }
        }
      }
      for (uint32_t i = 0; i < kBiasWords; ++i) {
        const uint32_t expected = f32_bits(static_cast<float>(iter * cfg::xtileN + i));
        if (bias[iter * kBiasWords + i] != expected) { ++errors; ++bias_errors; }
      }
      for (uint32_t i = 0; i < kStatsWords; ++i) {
        const uint32_t expected = 0x70000000u + iter * 0x1000u + i;
        if (stats[iter * kStatsWords + i] != expected) { ++errors; ++stats_errors; }
      }
    }
    if (errors) {
      std::cerr << "mode " << mode << ": " << errors << " mismatches (out="
                << out_errors << ", bias=" << bias_errors << ", stats="
                << stats_errors << ")\n";
      cleanup();
      return -1;
    }
    std::cout << "mode " << mode << " PASS (" << m << "x" << cfg::xtileN << " WGMMA output, 3D stats)\n";
  }
  std::cout << "PASS: WGMMA register epilogue -> SMEM -> grouped S2G 2D/3D\n";
  cleanup();
  return 0;
}
