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

#include "cout_drainer.h"
#include "mem.h"
#include <cstdint>
#include <iostream>
#include <vector>

using namespace vortex;

namespace {
constexpr uint32_t SLOTS     = VX_MEM_IO_COUT_SLOTS;
constexpr uint32_t RING      = VX_MEM_IO_COUT_RING;
constexpr uint64_t WR_BASE   = VX_MEM_IO_COUT_ADDR;
constexpr uint64_t RD_BASE   = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 4;
constexpr uint64_t DATA_BASE = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 8;
constexpr uint64_t LOST_BASE = DATA_BASE + uint64_t(SLOTS) * RING;
} // namespace

CoutDrainer::CoutDrainer(RAM& ram) : ram_(ram) {
  // Mirror Device::vx_init: zero wr[], rd[], and lost[]. Skip data[] —
  // vx_putchar overwrites it before the drainer reads.
  std::vector<uint8_t> zeros(SLOTS * 4, 0);
  ram_.write(zeros.data(), WR_BASE,   SLOTS * 4);
  ram_.write(zeros.data(), RD_BASE,   SLOTS * 4);
  ram_.write(zeros.data(), LOST_BASE, SLOTS * 4);
}

void CoutDrainer::tick() {
  uint32_t wr[SLOTS];
  uint32_t lost[SLOTS];
  ram_.read(wr,   WR_BASE,   sizeof(wr));
  ram_.read(lost, LOST_BASE, sizeof(lost));

  bool advanced = false;
  for (uint32_t s = 0; s < SLOTS; ++s) {
    const uint32_t rd = rd_[s];
    // Surface lost-byte deltas — kernel atomically bumps lost[slot] on a
    // full-ring drop; host reports the delta and remembers the latest.
    if (lost[s] != lost_seen_[s]) {
      uint32_t delta = lost[s] - lost_seen_[s];
      std::cout << "[#" << s << ": lost " << delta << " bytes]"
                << std::endl;
      lost_seen_[s] = lost[s];
    }
    if (wr[s] == rd) continue;
    uint32_t n = wr[s] - rd;
    if (n > RING) n = RING; // defensive — should not exceed RING in the lossy ring
    char data[RING];
    ram_.read(data, DATA_BASE + uint64_t(s) * RING, RING);
    for (uint32_t i = 0; i < n; ++i) {
      const char c = data[(rd + i) & (RING - 1)];
      line_[s].push_back(c);
      if (c == '\n') {
        std::cout << "#" << s << ": " << line_[s] << std::flush;
        line_[s].clear();
      }
    }
    rd_[s] = wr[s];
    advanced = true;
  }
  if (advanced)
    ram_.write(rd_.data(), RD_BASE, sizeof(uint32_t) * SLOTS);
}
