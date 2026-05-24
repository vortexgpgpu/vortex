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

// Standalone-sim drainer for the lossless per-hart COUT stream ring
// (proposal §10), used by vx_putchar / vx_print.S. The host runtime
// (sw/runtime/common/device.cpp Device::drain_cout) drains this ring in
// its CP launch-wait poll; standalone simx/rtlsim have no such loop, so
// without an in-process drainer any kernel that overruns RING bytes per
// hart back-pressures forever in the producer's spin.
//
// `tick(ram)` reads each slot's wr[], copies out [rd, wr) into stdout,
// and publishes the advanced rd[]. Safe to call on every cycle (cheap
// when the slot is empty); main loops should still call it once more
// after the simulator finishes so the kernel's final unterminated lines
// are flushed.
class CoutDrainer {
public:
  void tick(RAM& ram);

private:
  std::array<uint32_t, VX_MEM_IO_COUT_SLOTS> rd_{};
  std::array<std::string, VX_MEM_IO_COUT_SLOTS> line_{};
};

} // namespace vortex
