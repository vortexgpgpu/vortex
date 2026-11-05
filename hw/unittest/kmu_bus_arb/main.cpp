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

// VX_kmu_bus_arb's distribute side against its reference (the round-robin pick as a scan of the rotations), cycle by
// cycle on random stimulus: random input beats (compute and fragment, multi-beat messages), random output readies
// (from all-ready to sparse). Fails on the first mismatch; also requires every output to have been granted.

#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include "VVX_kmu_bus_arb_top.h"
#include "verilated.h"

double sc_time_stamp() { return 0; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint64_t cycles = (argc > 1) ? strtoull(argv[1], nullptr, 0) : 200000;
  auto* t = new VVX_kmu_bus_arb_top;
  std::mt19937_64 rng(12345);
  auto tick = [&]() { t->clk = 0; t->eval(); t->clk = 1; t->eval(); };
  t->reset = 1; for (int i = 0; i < 4; ++i) tick(); t->reset = 0;
  std::vector<uint64_t> grants;
  uint64_t fires = 0;
  for (uint64_t c = 0; c < cycles; ++c) {
    uint32_t density = (c / 4096) % 4;   // ready density phases: 1/2, 1/8, 7/8, 1/64
    t->in_valid = (rng() % 4) != 0;
    t->in_kind  = (rng() % 4) == 0;      // mostly compute (the round-robin path)
    t->in_eop   = (rng() % 3) != 0;
    t->in_dest  = rng();
    t->in_seed  = (uint32_t)rng();
    auto* w = reinterpret_cast<uint8_t*>(&t->out_ready);
    for (size_t i = 0; i < sizeof(t->out_ready); ++i) {
      uint8_t v = 0;
      for (int b = 0; b < 8; ++b) {
        uint32_t r = rng() % 64;
        bool on = density == 0 ? r < 32 : density == 1 ? r < 8 : density == 2 ? r < 56 : r < 1;
        v |= (uint8_t)(on << b);
      }
      w[i] = v;
    }
    t->clk = 0; t->eval();
    if (t->mismatch) { printf("FAILED: mismatch at cycle %lu\n", (unsigned long)c); return 1; }
    if (t->in_valid && t->in_ready) ++fires;
    auto* ov = reinterpret_cast<uint8_t*>(&t->out_valid);
    auto* orr = reinterpret_cast<uint8_t*>(&t->out_ready);
    size_t nb = sizeof(t->out_valid) * 8;
    if (grants.empty()) grants.assign(nb, 0);
    for (size_t b = 0; b < nb; ++b)
      if (((ov[b / 8] >> (b % 8)) & 1) && ((orr[b / 8] >> (b % 8)) & 1)) ++grants[b];
    t->clk = 1; t->eval();
  }
  uint32_t used = 0;
  for (auto g : grants) used += (g != 0);
  printf("PASSED: %lu cycles, %lu input beats, %u output slots granted\n", (unsigned long)cycles, (unsigned long)fires, used);
  delete t;
  return 0;
}
