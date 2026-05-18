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

// Device-memory accessor seam for the gem5 backend.
//
// Per gem5_v2_cp_migration_proposal §2.5: every device-memory access —
// CP ring fetches, completion writebacks, CMD_MEM_* DMA payload, Vortex
// MemSim loads/stores — funnels through this interface. In v1 the only
// implementation is InProcessDevMem (wraps simx::RAM). In v2 a
// DmaPortDevMem will replace it; CP hooks and Vortex memory code are
// untouched.
//
// Layered on top of simx::RAM rather than replacing it because Vortex's
// existing MemSim already knows how to talk to RAM; we only need the
// accessor seam for the CP side.

#pragma once

#include <cstddef>
#include <cstdint>

namespace vortex {
class RAM;
} // namespace vortex

namespace vortex_gem5 {

class DevMemAccessor {
public:
    virtual ~DevMemAccessor() = default;

    virtual void read (uint64_t addr, void* dst,       std::size_t bytes) = 0;
    virtual void write(uint64_t addr, const void* src, std::size_t bytes) = 0;
};

// v1 backing: the simx::RAM the Processor already uses. ACL bypass is
// the same pattern the simx/rtlsim CP hooks apply (sw/runtime/simx/
// vortex.cpp:271-280) — the CP/DMA is a peer of the host runtime, not
// a userspace caller subject to per-region page protections.
class InProcessDevMem final : public DevMemAccessor {
public:
    explicit InProcessDevMem(vortex::RAM& ram) : ram_(ram) {}

    void read (uint64_t addr, void* dst,       std::size_t bytes) override;
    void write(uint64_t addr, const void* src, std::size_t bytes) override;

private:
    vortex::RAM& ram_;
};

} // namespace vortex_gem5
