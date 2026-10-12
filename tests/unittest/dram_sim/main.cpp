// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dram_sim.h>
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace vortex;

static uint64_t g_responses = 0;

static bool on_response(void*) {
  ++g_responses;
  return true;
}

// The a100 profile: 32 HBM2 channels at 3,038 MT/s (the card's 40 at 2,430)
// behind a 1,410 MHz core, with the fixed latency (190 cycles) and request
// queue (257) of GPGPU-Sim's SM80 configuration.
static DramSim::Config a100_config() {
  DramSim::Config config;
  config.type         = DramSim::Type::HBM2;
  config.num_channels = 32;
  config.block_size   = 64;
  config.core_clock   = 1410;
  config.data_rate    = 3038;
  config.latency      = 190;
  config.queue_size   = 257;
  return config;
}

// Peak bytes per core cycle: a 32 B burst every two DRAM clocks per
// pseudo-channel (two per channel), or every eight clocks per GDDR6 channel.
static double peak_bandwidth(const DramSim::Config& config) {
  double bytes_per_clock = (config.type == DramSim::Type::GDDR6) ? 4.0 : 32.0;
  return config.num_channels * bytes_per_clock * (config.data_rate / 2.0) / config.core_clock;
}

// Median core cycles for one read on an idle channel; `addr_of(i)` picks the
// i-th address, and the first read (cold) is not counted.
template <typename F>
static double unloaded_latency(const DramSim::Config& config, F addr_of) {
  DramSim dram(config);
  dram.reset();
  const int n = 33;
  std::vector<uint64_t> samples;
  for (int i = 0; i < n; ++i) {
    g_responses = 0;
    uint64_t cycles = 0;
    dram.send_request(0, addr_of(i), false, on_response, nullptr);
    while (g_responses == 0) {
      dram.tick();
      ++cycles;
    }
    if (i != 0) {
      samples.push_back(cycles);
    }
    for (int k = 0; k < 256; ++k) {
      dram.tick();
    }
  }
  std::sort(samples.begin(), samples.end());
  return double(samples[samples.size() / 2]);
}

// Bytes per core cycle when every channel streams `blocks` blocks, taken in
// turn from `streams` sequential streams `stride` bytes apart;
// `peak_inflight` receives the most reads outstanding at once.
static double stream_bandwidth(const DramSim::Config& config, uint32_t blocks, uint64_t* peak_inflight = nullptr,
                               uint32_t streams = 1, uint64_t stride = 0) {
  DramSim dram(config);
  dram.reset();
  g_responses = 0;
  std::vector<uint32_t> sent(config.num_channels, 0);
  uint64_t total = uint64_t(blocks) * config.num_channels;
  uint64_t issued = 0;
  uint64_t cycles = 0;
  uint64_t peak = 0;
  while (g_responses < total) {
    for (uint32_t ch = 0; ch < config.num_channels; ++ch) {
      if (sent[ch] < blocks && dram.can_accept(ch)) {
        uint64_t addr = uint64_t(sent[ch] % streams) * stride + uint64_t(sent[ch] / streams) * config.block_size;
        dram.send_request(ch, addr, false, on_response, nullptr);
        ++sent[ch];
        ++issued;
      }
    }
    peak = std::max(peak, issued - g_responses);
    dram.tick();
    ++cycles;
  }
  if (peak_inflight) {
    *peak_inflight = peak;
  }
  return double(total) * config.block_size / cycles;
}

// Alternates between two rows of one bank: the rows from address bit 15 on
// fold to the same bank permutation (HBM 33 = 1|1 in 5-bit chunks, GDDR6
// 17 = 1|1 in 4-bit chunks).
static uint64_t hbm_conflict(int i) { return uint64_t(i % 2) * 33 << 15; }
static uint64_t gddr6_conflict(int i) { return uint64_t(i % 2) * 17 << 15; }

static int g_errors = 0;

static void check(bool cond, const char* what, double got) {
  printf("%s %s (%.1f)\n", cond ? "PASS" : "FAIL", what, got);
  if (!cond) {
    ++g_errors;
  }
}

int main() {
  auto a100 = a100_config();

  // Unloaded latency: open-row hits, then row conflicts (precharge + activate).
  double open = unloaded_latency(a100, [](int i) { return uint64_t(i % 8) * 128; });
  double conflict = unloaded_latency(a100, hbm_conflict);
  check(open > 205 && open < 225, "A100 open-row read latency ~215 core cycles", open);
  check(conflict > 245 && conflict < 265, "A100 row-conflict read latency ~255 core cycles", conflict);

  // The fixed latency is added exactly once per read (measured at a 1:1 clock
  // ratio, so the DRAM clock phase cannot shift the comparison).
  auto unit_clock = a100;
  unit_clock.core_clock = unit_clock.data_rate / 2;
  auto no_latency = unit_clock;
  no_latency.latency = 0;
  auto open_row = [](int i) { return uint64_t(i % 8) * 128; };
  double added = unloaded_latency(unit_clock, open_row) - unloaded_latency(no_latency, open_row);
  check(added == 190, "fixed latency adds 190 cycles", added);

  // Every channel is an independent controller: 32 channels stream ~32x one,
  // and the count need not be a power of two.
  auto one = a100;
  one.num_channels = 1;
  double bw1 = stream_bandwidth(one, 4096);
  double bw32 = stream_bandwidth(a100, 4096);
  check(bw32 / bw1 > 29 && bw32 / bw1 < 33, "bandwidth scales with the channel count", bw32 / bw1);
  auto five = a100;
  five.num_channels = 5;
  double bw5 = stream_bandwidth(five, 4096);
  check(bw5 / bw1 > 4.5 && bw5 / bw1 < 5.1, "5 channels stream ~5x one", bw5 / bw1);

  // Saturated stream efficiency against the computed peak (1,103 B/cycle, the
  // card's 1,555 GB/s).
  double a100_peak = peak_bandwidth(a100);
  check(bw32 > 0.75 * a100_peak && bw32 <= a100_peak, "A100 stream reaches 75-100% of peak", bw32);

  // The controller queue bounds the requests in flight.
  auto narrow = a100;
  narrow.num_channels = 1;
  narrow.queue_size = 8;
  uint64_t peak = 0;
  stream_bandwidth(narrow, 1024, &peak);
  check(peak <= 8, "queue of 8 bounds requests in flight", peak);

  // A block of two bursts returns one response.
  auto wide = a100;
  wide.num_channels = 1;
  wide.block_size = 128;
  {
    DramSim dram(wide);
    dram.reset();
    g_responses = 0;
    dram.send_request(0, 0, false, on_response, nullptr);
    for (int k = 0; k < 2000; ++k) {
      dram.tick();
    }
    check(g_responses == 1, "a 128 B block returns one response", double(g_responses));
  }

  // The platform default: 2 HBM2 channels clocked with a 400 MHz core.
  DramSim::Config base{DramSim::Type::HBM2, 2, 64, 400, 800, 0, 0};
  double base_open = unloaded_latency(base, [](int i) { return uint64_t(i % 8) * 64; });
  double base_conflict = unloaded_latency(base, hbm_conflict);
  check(base_open == 14, "default open-row read latency 14 cycles", base_open);
  check(base_conflict == 26, "default row-conflict read latency 26 cycles", base_conflict);

  // Streams a power of two apart open their rows in different banks: four
  // streams 1 MB apart on one channel keep over half a single stream's rate
  // (an unpermuted map keeps a third). They still share banks over time, so
  // each row visit pays an activation.
  auto base_one = base;
  base_one.num_channels = 1;
  double seq = stream_bandwidth(base_one, 4096);
  double strided = stream_bandwidth(base_one, 4096, nullptr, 4, 1 << 20);
  check(strided > 0.55 * seq, "four streams 1 MB apart keep over half one stream's rate", strided / seq);
  double bw2 = stream_bandwidth(base, 4096);
  double bw1h = stream_bandwidth(base_one, 4096);
  check(bw2 / bw1h > 1.8 && bw2 / bw1h < 2.05, "HBM2 2 channels stream ~2x one", bw2 / bw1h);

  // HBM3 at the h100 profile's shape: 32 channels at 6,545 MT/s behind a
  // 1,980 MHz core, the card's 3.35 TB/s.
  DramSim::Config h100{DramSim::Type::HBM3, 32, 64, 1980, 6545, 0, 0};
  double h100_peak = peak_bandwidth(h100);
  double hbw = stream_bandwidth(h100, 4096);
  check(hbw > 0.75 * h100_peak && hbw <= h100_peak, "H100 stream reaches 75-100% of peak", hbw);

  // GDDR6 at the rtx5080 profile's shape: 32 x16 channels at 15 GT/s behind a
  // 2,295 MHz core: 418 B/cycle peak, the card's 960 GB/s.
  DramSim::Config gddr6{DramSim::Type::GDDR6, 32, 64, 2295, 15000, 0, 0};
  double gpeak = peak_bandwidth(gddr6);
  double gbw32 = stream_bandwidth(gddr6, 4096);
  check(gbw32 > 0.7 * gpeak && gbw32 <= gpeak, "GDDR6 stream reaches 70-100% of peak", gbw32);
  auto gddr6_one = gddr6;
  gddr6_one.num_channels = 1;
  double gbw1 = stream_bandwidth(gddr6_one, 4096);
  check(gbw32 / gbw1 > 29 && gbw32 / gbw1 < 33, "GDDR6 bandwidth scales with the channel count", gbw32 / gbw1);
  double gopen = unloaded_latency(gddr6_one, [](int i) { return uint64_t(i % 8) * 32; });
  double gconflict = unloaded_latency(gddr6_one, gddr6_conflict);
  check(gopen < gconflict, "GDDR6 row conflicts cost more than row hits", gconflict - gopen);

  if (g_errors != 0) {
    printf("FAILED: %d check(s)\n", g_errors);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
