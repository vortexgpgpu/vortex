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

// Standalone-sim drainer for the per-hart COUT stream ring (proposal §10,
// post-O-1: lossy). Mirrors sw/runtime/common/device.cpp::Device::drain_cout
// — same layout, same semantics. Standalone simx/rtlsim need their own
// drainer because they have no CP launch-wait poll loop driving the host
// runtime's drainer.
//
// Layout (matches VX_types.toml): wr[SLOTS], rd[SLOTS], data[SLOTS][RING],
// lost[SLOTS]. Kernel-side vx_putchar drops bytes on a full ring and
// atomically bumps lost[slot]; this drainer surfaces "[#N: lost K bytes]"
// alongside the streamed text.
//
// Construction resets the COUT ring (wr[]/rd[]/lost[]) in `ram` so the
// first drain sees an empty ring and a zero overflow baseline. Without
// this, the drainer would read RAM's `0xbaadf00d` sentinel as a live
// write pointer or as phantom lost-byte counts, surfacing as a wall of
// `[#N: lost 3131961357 bytes]` lines and tripping any test that
// asserts on stdout cleanliness (e.g. dhrystone). The host runtime
// (Device::vx_init) performs the equivalent init via dev_write on the
// runtime-driven path; the drainer here is only instantiated by the
// standalone simx/rtlsim binaries that have no host runtime.
//
// `tick()` reads each slot's wr[] and lost[] from the bound ram,
// copies out [rd, wr) into stdout, and publishes the advanced rd[].
// Safe to call on every cycle (cheap when the slot is empty); main
// loops should still call it once more after the simulator finishes
// so the kernel's final unterminated lines are flushed.
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
