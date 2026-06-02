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

#pragma once

#include <VX_types.h>
#include <array>
#include <cstdint>
#include <string>

namespace vortex {

class RAM;

// Drainer for the per-hart COUT stream ring used by standalone simulators.
// Matches the layout and semantics of sw/runtime/common/device.cpp::Device::drain_cout.
// Standalone simulators need their own drainer because they have no CP
// launch-wait poll loop driving the host runtime's drainer.
//
// Layout: wr[SLOTS], rd[SLOTS], data[SLOTS][RING], lost[SLOTS].
// Kernel-side vx_putchar drops bytes on a full ring and atomically bumps
// lost[slot]; this drainer surfaces "[#N: lost K bytes]" alongside streamed text.
//
// Construction resets the COUT ring (wr[]/rd[]/lost[]) in `ram` so the first
// drain sees an empty ring and a zero overflow baseline. Without this reset,
// uninitialized RAM values appear as live write pointers or phantom lost-byte
// counts, causing spurious "[#N: lost N bytes]" output.
//
// `tick()` reads each slot's wr[] and lost[] from the bound ram, copies out
// [rd, wr) into stdout, and advances rd[]. Safe to call every cycle; call
// once more after the simulator finishes to flush any unterminated lines.
class CoutDrainer {
public:
  explicit CoutDrainer(RAM& ram);

  void tick();

private:
  RAM& ram_;
  std::array<uint32_t, VX_MEM_IO_COUT_SLOTS> rd_{};
  std::array<uint32_t, VX_MEM_IO_COUT_SLOTS> lost_seen_{};
  std::array<std::string, VX_MEM_IO_COUT_SLOTS> line_{};
};

} // namespace vortex
