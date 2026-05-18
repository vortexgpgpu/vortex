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

// VortexGPGPU — gem5 SimObject wrapper for libvortex-gem5.so.
//
// Lives at $GEM5_HOME/src/dev/vortex/vortex_gpgpu_dev.{cc,hh} after
// sim/simx/gem5/install.sh runs. The host-side source of truth is
// the Vortex tree (sim/simx/gem5/) so API drift between gem5 and the
// Vortex C ABI shows up as a build error in Vortex CI, not as a gem5
// integration mystery.
//
// Design points (see docs/proposals/gem5_simx_v3_proposal.md §3.1):
//   - dlopen the Vortex library at construction time; resolve all
//     vortex_gem5_* symbols up-front. This keeps gem5 decoupled from
//     the Vortex C++ ABI, so we can iterate on SimX internals without
//     rebuilding gem5.
//   - Drive Vortex's clock from a self-rescheduling EventFunctionWrapper
//     (sim/simx/gem5/gem5_api_notes.md §"EventFunctionWrapper"). One
//     vortex_gem5_tick() per gem5 cycle.
//   - Inherits DmaDevice (not just PioDevice) so Phase 4's host runtime
//     gets DMA "for free" via gem5's DmaPort; the Phase 3 entry just
//     declares the inheritance and leaves DMA paths unexercised.

#ifndef __DEV_VORTEX_VORTEX_GPGPU_DEV_HH__
#define __DEV_VORTEX_VORTEX_GPGPU_DEV_HH__

#include "dev/dma_device.hh"
#include "dev/io_device.hh"
#include "params/VortexGPGPU.hh"
#include "sim/eventq.hh"

#include <cstdint>
#include <string>

namespace gem5
{

class VortexGPGPU : public DmaDevice
{
public:
    using Params = VortexGPGPUParams;

    VortexGPGPU(const Params &p);
    ~VortexGPGPU() override;

    // PioDevice interface
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;
    AddrRangeList getAddrRanges() const override;

    // SimObject lifecycle
    void init() override;
    void startup() override;

private:
    // Self-rescheduling clock tick — calls vortex_gem5_tick() once per
    // device cycle. Returns false (program done) ⇒ exitSimLoop.
    void tick();

    // Decode an MMIO command type write (MMIO_CMD_TYPE) and route
    // CMD_MEM_{READ,WRITE} to the DMA path. Phase 3 routes other
    // command types via vortex_gem5_mmio_write64; Phase 4 promotes
    // CMD_MEM_* to the real DmaPort flow.
    void handleCmdType(uint64_t value);

    // Library binding ------------------------------------------------
    // Opaque dlopen handle; closed in dtor.
    void* libHandle_;
    // Vortex device handle returned by vortex_gem5_create.
    void* deviceHandle_;

    // Cached function pointers — resolved once at construction so the
    // hot path (tick, read, write) is straight indirect calls with no
    // string lookups.
    struct AbiV1 {
        const char* (*build_info)(void);
        void*       (*create)(void);
        void        (*destroy)(void* h);
        int         (*load_kernel)(void* h, const char* path);
        bool        (*tick)(void* h);
        uint64_t    (*mmio_read64)(void* h, uint64_t off);
        void        (*mmio_write64)(void* h, uint64_t off, uint64_t value);
        void        (*vram_write)(void* h, uint64_t addr, const uint8_t* src, uint32_t size);
        void        (*vram_read)(void* h, uint64_t addr, uint8_t* dst, uint32_t size);
        int         (*dcr_write)(void* h, uint32_t addr, uint32_t value);
        int         (*dcr_read)(void* h, uint32_t addr, uint32_t tag, uint32_t* value);
        uint64_t    (*pop_pending_cmd)(void* h);
        uint64_t    (*get_cmd_arg)(void* h, int which);
        void        (*set_busy)(void* h, bool busy);
    } abi_;

    // Configuration --------------------------------------------------
    const std::string libraryPath_;
    const std::string kernelPath_;
    const Addr        pioAddr_;
    const Addr        pioSize_;
    const Tick        pioLatency_;

    // Tick scheduling
    EventFunctionWrapper tickEvent_;

    // Standalone vs. hosted mode (selected at startup based on
    // whether the `kernel=` Python param was set). In standalone
    // mode the device drives a single preloaded kernel to
    // completion and exits the sim loop; in hosted mode it sits
    // idle until the host CPU issues CMD_RUN via MMIO.
    bool standalone_;
};

} // namespace gem5

#endif // __DEV_VORTEX_VORTEX_GPGPU_DEV_HH__
