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

// The bank model (sim/common/dram_bank_model.h) against Ramulator (DramSim) on
// the same closed-loop traffic: per channel, `window` requests outstanding,
// reads until their response, writes until acknowledged. Both take the
// traffic through DramSim's interface, so the clock loop, ingress, fixed
// latency and response drains are the same and only the scheduling differs.
//
// The gate: an idle channel's reads (one outstanding) take the same latency in
// both, read for read, on every configuration -- row hits, misses and
// conflicts. Loaded traffic is reported: bandwidth, mean / p50 / p99 read
// latency, the Kolmogorov-Smirnov distance between the latency distributions,
// and the bank model's row-hit rate.
//
//   ./dram_bank_model            the gate, on short runs
//   ./dram_bank_model full [cfg] the full table (20,000 requests per cell)
//   ./dram_bank_model refresh    the bandwidth refresh costs the bank model,
//                                per bank and all banks, at 64 outstanding
//                                (dram_sim models no refresh). The gate: one
//                                refresh per interval, to the operation; and a
//                                stream, which moves data every clock it may,
//                                loses at least tRFC / tREFI to all-bank
//                                refresh, the time no bank is open

#include <dram_bank_model.h>
#include <dram_sim.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace vortex;

namespace {

struct Machine {
  const char* name;
  DramSim::Config cfg;
};

// The default machine and the profiles' channels (profiles/*.json), one
// channel each unless named otherwise.
const Machine kMachines[] = {
  {"default",  {DramSim::Type::HBM2, 1, 64, 400, 800, 0, 0}},
  {"default4", {DramSim::Type::HBM2, 4, 64, 400, 800, 0, 0}},
  {"v100",     {DramSim::Type::HBM2, 1, 64, 1530, 1752, 0, 0}},
  {"a100",     {DramSim::Type::HBM2, 1, 64, 1410, 3038, 190, 257}},
  {"h100",     {DramSim::Type::HBM3, 1, 64, 1980, 6545, 0, 0}},
  {"rtx1080",  {DramSim::Type::GDDR6, 1, 64, 1607, 10000, 0, 0}},
  {"rtx5080",  {DramSim::Type::GDDR6, 1, 64, 2295, 15000, 339, 0}},
};

struct Rec {
  uint64_t sent, done;
  bool w;
};

uint64_t g_cycle = 0;

bool on_rsp(void* arg) {
  static_cast<Rec*>(arg)->done = g_cycle;
  return true;
}

struct Pattern {
  const char* name;
  std::function<uint64_t(uint64_t i, std::mt19937_64& rng)> block;
  double write_frac;
};

struct Result {
  double bw = 0, mean = 0, p50 = 0, p99 = 0;
  std::vector<uint64_t> lat;
};

template <typename M>
Result run(M& m, const DramSim::Config& cfg, const Pattern& p, uint32_t window, uint32_t n) {
  std::mt19937_64 rng(12345), rng_w(777);
  std::vector<Rec> recs(uint64_t(n) * cfg.num_channels);
  std::vector<uint32_t> sent(cfg.num_channels, 0);
  std::vector<std::vector<Rec*>> live(cfg.num_channels);
  const uint64_t total = recs.size();
  uint64_t done = 0;
  g_cycle = 0;
  std::uniform_real_distribution<double> u(0, 1);
  while (done < total) {
    for (uint32_t ch = 0; ch < cfg.num_channels; ++ch) {
      auto& l = live[ch];
      for (size_t i = 0; i < l.size();) {
        if (l[i]->done != ~0ull) {
          l[i] = l.back();
          l.pop_back();
          ++done;
        } else {
          ++i;
        }
      }
      if (sent[ch] < n && l.size() < window && m.can_accept(ch)) {
        Rec* r = &recs[uint64_t(ch) * n + sent[ch]];
        r->sent = g_cycle;
        r->done = ~0ull;
        r->w = u(rng_w) < p.write_frac;
        m.send_request(ch, p.block(sent[ch], rng) * cfg.block_size, r->w, on_rsp, r);
        l.push_back(r);
        ++sent[ch];
      }
    }
    m.tick();
    ++g_cycle;
  }
  Result res;
  res.bw = double(total) * cfg.block_size / double(g_cycle);
  for (auto& r : recs) {
    if (!r.w) res.lat.push_back(r.done - r.sent);
  }
  std::sort(res.lat.begin(), res.lat.end());
  double s = 0;
  for (auto v : res.lat) s += double(v);
  if (!res.lat.empty()) {
    res.mean = s / double(res.lat.size());
    res.p50 = double(res.lat[res.lat.size() / 2]);
    res.p99 = double(res.lat[std::min(res.lat.size() - 1, size_t(0.99 * double(res.lat.size())))]);
  }
  return res;
}

double ks_distance(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  if (a.empty() || b.empty()) return 0;
  double ks = 0;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    uint64_t v = std::min(a[i], b[j]);
    while (i < a.size() && a[i] <= v) ++i;
    while (j < b.size() && b[j] <= v) ++j;
    ks = std::max(ks, std::fabs(double(i) / double(a.size()) - double(j) / double(b.size())));
  }
  return ks;
}

DramBankSim::Config bank_config(const DramSim::Config& c,
                                DramBankModel::Refresh refresh = DramBankModel::Refresh::Off) {
  return DramBankSim::Config{c.type, c.num_channels, c.block_size, c.core_clock, c.data_rate,
                             c.latency, c.queue_size, refresh};
}

std::vector<Pattern> patterns() {
  const uint64_t blocks = (256ull << 20) / 64;
  return {
    {"seq", [](uint64_t i, std::mt19937_64&) { return i; }, 0.0},
    {"rand", [blocks](uint64_t, std::mt19937_64& r) { return r() % blocks; }, 0.0},
    // Four sequential streams 1 MB apart, interleaved.
    {"strided4", [](uint64_t i, std::mt19937_64&) { return (i % 4) * (1ull << 14) + i / 4; }, 0.0},
    // Blocks in random rows among 64 that all map to one bank.
    {"conflict", [](uint64_t i, std::mt19937_64& r) {
       uint64_t row = r() % 64;
       return (row * 33 << 15) / 64 + (i % 2);
     }, 0.0},
    {"rand_w30", [blocks](uint64_t, std::mt19937_64& r) { return r() % blocks; }, 0.3},
    {"seq_w50", [](uint64_t i, std::mt19937_64&) { return i; }, 0.5},
  };
}

}  // namespace

// Bandwidth with refresh off, per bank and all banks.
int refresh_table() {
  using R = DramBankModel::Refresh;
  int failures = 0;
  const uint32_t n = 60000, w = 64;
  printf("%-9s %-9s | %7s | %7s %6s %6s | %7s %6s %6s %6s\n", "machine", "pattern", "off B/c",
         "bank", "loss%", "refs", "all", "loss%", "refs", "tRFC/tREFI%");
  for (const auto& mach : kMachines) {
    if (mach.cfg.type == DramSim::Type::GDDR6) continue;
    const auto std = (mach.cfg.type == DramSim::Type::HBM3) ? dram_timing::Standard::HBM3
                                                             : dram_timing::Standard::HBM2;
    const auto r = dram_timing::refresh(std, mach.cfg.data_rate);
    const double analytic = 100.0 * double(r.RFC) / double(r.REFI);
    for (const auto& p : patterns()) {
      if (p.write_frac != 0 || !std::strcmp(p.name, "conflict")) continue;
      double bw[3];
      uint64_t refs[3];
      const R modes[3] = {R::Off, R::PerBank, R::AllBank};
      for (int k = 0; k < 3; ++k) {
        DramBankSim mod(bank_config(mach.cfg, modes[k]));
        bw[k] = run(mod, mach.cfg, p, w, n).bw;
        refs[k] = 0;
        for (uint32_t ch = 0; ch < mach.cfg.num_channels; ++ch) {
          const auto& m = mod.channel(ch);
          refs[k] += m.stats().refreshes;
          // The refreshes due by the last clock, but one in progress.
          const uint32_t refi = m.config().refi;
          const uint64_t due = refi ? m.clock() / refi : 0;
          if (m.stats().refreshes + 1 < due || m.stats().refreshes > due) {
            printf("FAIL %s %s: %lu refreshes in %lu clocks of %u\n", mach.name, p.name,
                   (unsigned long)m.stats().refreshes, (unsigned long)m.clock(), refi);
            ++failures;
          }
        }
      }
      const double lb = 100.0 * (1 - bw[1] / bw[0]), la = 100.0 * (1 - bw[2] / bw[0]);
      printf("%-9s %-9s | %7.2f | %7.2f %6.2f %6lu | %7.2f %6.2f %6lu %6.2f\n", mach.name, p.name,
             bw[0], bw[1], lb, (unsigned long)refs[1], bw[2], la, (unsigned long)refs[2], analytic);
      if (!std::strcmp(p.name, "seq") && la < analytic) {
        printf("FAIL %s %s: all-bank refresh costs %.2f%%, less than tRFC/tREFI (%.2f%%)\n", mach.name,
               p.name, la, analytic);
        ++failures;
      }
    }
  }
  if (failures) {
    printf("FAILED: %d\n", failures);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && !std::strcmp(argv[1], "refresh"))
    return refresh_table();
  const bool full = argc > 1 && !std::strcmp(argv[1], "full");
  const char* only = (full && argc > 2) ? argv[2] : nullptr;
  const uint32_t n = full ? 20000 : 2000;
  const std::vector<uint32_t> windows = full ? std::vector<uint32_t>{1, 4, 16, 64}
                                             : std::vector<uint32_t>{1, 16};
  int failures = 0;
  printf("%-9s %-9s %4s | %7s %7s %6s %6s | %7s %7s %6s %6s | %6s %6s %5s %5s\n", "machine",
         "pattern", "win", "ref B/c", "mean", "p50", "p99", "mod B/c", "mean", "p50", "p99", "dBW%",
         "dMean%", "KS", "hit%");
  for (const auto& mach : kMachines) {
    if (only && std::strcmp(only, mach.name)) continue;
    for (const auto& p : patterns()) {
      for (uint32_t w : windows) {
        DramSim ref(mach.cfg);
        ref.reset();
        Result a = run(ref, mach.cfg, p, w, n);
        DramBankSim mod(bank_config(mach.cfg));
        Result b = run(mod, mach.cfg, p, w, n);
        uint64_t hits = 0, ops = 0;
        for (uint32_t ch = 0; ch < mach.cfg.num_channels; ++ch) {
          const auto& st = mod.channel(ch).stats();
          hits += st.hits;
          ops += st.hits + st.misses + st.conflicts;
        }
        const double ks = ks_distance(a.lat, b.lat);
        printf("%-9s %-9s %4u | %7.2f %7.1f %6.0f %6.0f | %7.2f %7.1f %6.0f %6.0f | %6.1f %6.1f %5.3f %5.1f\n",
               mach.name, p.name, w, a.bw, a.mean, a.p50, a.p99, b.bw, b.mean, b.p50, b.p99,
               100.0 * (b.bw - a.bw) / a.bw, a.mean ? 100.0 * (b.mean - a.mean) / a.mean : 0.0, ks,
               ops ? 100.0 * double(hits) / double(ops) : 0.0);
        // One read outstanding on an idle channel: every latency the same.
        if (w == 1 && p.write_frac == 0 && a.lat != b.lat) {
          printf("FAIL %s %s: idle read latencies differ\n", mach.name, p.name);
          ++failures;
        }
      }
    }
  }
  if (failures) {
    printf("FAILED: %d\n", failures);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
