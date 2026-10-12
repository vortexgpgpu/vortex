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

#include <cstdint>

namespace vortex {

// The maxsim kernel (hw/rtl/afu/maxsim) simulated in-process, as the
// runtime's avedsim target reaches it: its control registers, and the device
// memory behind its AXI4 port. The kernel clock runs only while the host
// touches a register, each access costing the kernel cycles a register access
// over PCIe would (MAXSIM_EMUL_MMIO_CYCLES, 250 by default), so a host that
// polls a register is what lets the target run.
class maxsim_emul {
public:
  maxsim_emul();
  ~maxsim_emul();

  int init();

  int reg_write(uint32_t offset, uint32_t value);
  int reg_read(uint32_t offset, uint32_t* value);

  // Device memory, by device address.
  int mem_write(uint64_t addr, const void* data, uint64_t size);
  int mem_read(uint64_t addr, void* data, uint64_t size);

private:
  class Impl;
  Impl* impl_;
};

} // namespace vortex
