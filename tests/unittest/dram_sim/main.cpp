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

// A100 HBM as GPGPU-Sim's SM80 configuration models it: 40 channels at
// 1,512 MHz behind a 1,410 MHz core, 190 cycles of fixed latency, a 257-request
// window per channel.
static DramSim::Config a100_config() {
  DramSim::Config config;
  config.type         = DramSim::Type::HBM2_A100;
  config.num_channels = 40;
  config.block_size   = 64;
  config.core_clock   = 1410;
  config.dram_clock   = 1512;
  config.latency      = 190;
  config.window       = 257;
  return config;
}

// Median core cycles for one read on an idle channel (the median, so the few
// reads that wait behind a refresh do not count); `addr_of(i)` picks the i-th
// address, and the first read (cold) is not counted.
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

// Bytes per core cycle when every channel streams `blocks` sequential blocks;
// `peak_inflight` receives the most reads outstanding at once.
static double stream_bandwidth(const DramSim::Config& config, uint32_t blocks, uint64_t* peak_inflight = nullptr) {
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
        dram.send_request(ch, uint64_t(sent[ch]) * config.block_size, false, on_response, nullptr);
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
  double conflict = unloaded_latency(a100, [](int i) { return uint64_t(i) << 20; });
  check(open > 205 && open < 225, "A100 open-row read latency ~215 core cycles", open);
  check(conflict > 245 && conflict < 265, "A100 row-conflict read latency ~255 core cycles", conflict);

  // The fixed latency is added exactly once per read (measured at a 1:1 clock
  // ratio, so the DRAM clock phase cannot shift the comparison).
  auto unit_clock = a100;
  unit_clock.core_clock = unit_clock.dram_clock;
  auto no_latency = unit_clock;
  no_latency.latency = 0;
  auto open_row = [](int i) { return uint64_t(i % 8) * 128; };
  double added = unloaded_latency(unit_clock, open_row) - unloaded_latency(no_latency, open_row);
  check(added == 190, "fixed latency adds 190 cycles", added);

  // Every channel is an independent controller: 40 channels stream ~40x one.
  auto one = a100;
  one.num_channels = 1;
  double bw1 = stream_bandwidth(one, 4096);
  double bw40 = stream_bandwidth(a100, 4096);
  check(bw40 / bw1 > 36 && bw40 / bw1 < 41, "bandwidth scales with the channel count", bw40 / bw1);

  // Saturated stream efficiency against the A100's 1,373 B/cycle peak.
  check(bw40 > 0.6 * 1373 && bw40 <= 1373, "A100 stream reaches 60-100% of peak", bw40);

  // The window bounds the requests in flight.
  auto narrow = a100;
  narrow.num_channels = 1;
  narrow.window = 8;
  uint64_t peak = 0;
  stream_bandwidth(narrow, 1024, &peak);
  check(peak <= 8, "window of 8 bounds requests in flight", peak);

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

  // Plain HBM2 at the platform default: still independent channels.
  auto hbm2 = a100;
  hbm2.type = DramSim::Type::HBM2;
  hbm2.num_channels = 2;
  hbm2.latency = 0;
  hbm2.window = 0;
  double bw2 = stream_bandwidth(hbm2, 4096);
  hbm2.num_channels = 1;
  double bw1h = stream_bandwidth(hbm2, 4096);
  check(bw2 / bw1h > 1.8 && bw2 / bw1h < 2.05, "HBM2 2 channels stream ~2x one", bw2 / bw1h);

  // Every channel refreshes: over many refresh intervals some open-row reads
  // wait behind a refresh (nRFC = 160 cycles at a 1:1 clock).
  {
    auto refresh = unit_clock;
    refresh.type = DramSim::Type::HBM2;
    refresh.num_channels = 1;
    refresh.latency = 0;
    DramSim dram(refresh);
    dram.reset();
    uint64_t min_lat = ~0ull, max_lat = 0;
    for (int i = 0; i < 400; ++i) {
      g_responses = 0;
      uint64_t cycles = 0;
      dram.send_request(0, uint64_t(i % 8) * 128, false, on_response, nullptr);
      while (g_responses == 0) {
        dram.tick();
        ++cycles;
      }
      if (i != 0) {
        min_lat = std::min(min_lat, cycles);
        max_lat = std::max(max_lat, cycles);
      }
      for (int k = 0; k < 97; ++k) {
        dram.tick();
      }
    }
    check(max_lat - min_lat > 100, "refresh stalls reads by ~nRFC", double(max_lat - min_lat));
  }

  // GDDR6 at the rtx5080 profile's shape: 32 x16 channels at 15 GT/s (7,500
  // MHz data clock) behind a 2,295 MHz core, 32 B per 8-clock burst: 418 B/cycle
  // peak, the card's 960 GB/s.
  auto gddr6 = a100;
  gddr6.type = DramSim::Type::GDDR6;
  gddr6.num_channels = 32;
  gddr6.core_clock = 2295;
  gddr6.dram_clock = 7500;
  gddr6.latency = 0;
  gddr6.window = 0;
  double gpeak = 32.0 * 32 / 8 * 7500 / 2295;
  double gbw32 = stream_bandwidth(gddr6, 4096);
  check(gbw32 > 0.6 * gpeak && gbw32 <= gpeak, "GDDR6 stream reaches 60-100% of peak", gbw32);
  auto gddr6_one = gddr6;
  gddr6_one.num_channels = 1;
  double gbw1 = stream_bandwidth(gddr6_one, 4096);
  check(gbw32 / gbw1 > 29 && gbw32 / gbw1 < 33, "GDDR6 bandwidth scales with the channel count", gbw32 / gbw1);
  double gopen = unloaded_latency(gddr6_one, [](int i) { return uint64_t(i % 8) * 32; });
  double gconflict = unloaded_latency(gddr6_one, [](int i) { return uint64_t(i) << 22; });
  check(gopen < gconflict, "GDDR6 row conflicts cost more than row hits", gconflict - gopen);

  if (g_errors != 0) {
    printf("FAILED: %d check(s)\n", g_errors);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
