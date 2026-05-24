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

using namespace vortex;

void CoutDrainer::tick(RAM& ram) {
  constexpr uint32_t SLOTS = VX_MEM_IO_COUT_SLOTS;
  constexpr uint32_t RING  = VX_MEM_IO_COUT_RING;
  const uint64_t WR_BASE   = VX_MEM_IO_COUT_ADDR;
  const uint64_t RD_BASE   = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 4;
  const uint64_t DATA_BASE = VX_MEM_IO_COUT_ADDR + uint64_t(SLOTS) * 8;

  uint32_t wr[SLOTS];
  ram.read(wr, WR_BASE, sizeof(wr));

  bool advanced = false;
  for (uint32_t s = 0; s < SLOTS; ++s) {
    const uint32_t rd = rd_[s];
    if (wr[s] == rd) continue;
    uint32_t n = wr[s] - rd;
    if (n > RING) n = RING; // defensive — a lossless ring never overruns
    char data[RING];
    ram.read(data, DATA_BASE + uint64_t(s) * RING, RING);
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
    ram.write(rd_.data(), RD_BASE, sizeof(uint32_t) * SLOTS);
}
