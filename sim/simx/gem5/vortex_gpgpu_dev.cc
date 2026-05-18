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

#include "dev/vortex/vortex_gpgpu_dev.hh"

#include "base/logging.hh"
#include "base/trace.hh"
#include "mem/packet_access.hh"
#include "sim/sim_exit.hh"

#include <dlfcn.h>

// OPAE MMIO command-set constants. Hardcoded to match the layout
// the host runtime uses (sw/runtime/gem5/vortex.cpp:50-66, also
// hw/syn/altera/opae/vortex_afu.json). Hardcoded — not pulled from
// vortex_opae.h — because gem5 is compiled out-of-tree and we
// don't want a build-time dep on the Vortex source.
static constexpr uint64_t MMIO_CMD_TYPE = 10 * 4;  // byte offset
static constexpr uint64_t CMD_MEM_READ  = 1;
static constexpr uint64_t CMD_MEM_WRITE = 2;
static constexpr uint64_t CMD_RUN       = 3;

// Cache line size — args are stored shifted by log2(CACHE_BLOCK_SIZE)
// in the OPAE protocol; both directions agree at log2(64) = 6.
static constexpr unsigned CACHE_BLOCK_LOG2 = 6;

namespace gem5
{

namespace {

// Helper for dlsym + null-check in one line. Returns the resolved
// pointer cast to T, or fatals out with a stable error message.
template <typename T>
T dlsym_or_fatal(void* handle, const char* symbol, const char* libpath)
{
    void* p = dlsym(handle, symbol);
    if (p == nullptr) {
        fatal("VortexGPGPU: dlsym(%s) failed in %s: %s",
              symbol, libpath, dlerror());
    }
    return reinterpret_cast<T>(p);
}

} // namespace

VortexGPGPU::VortexGPGPU(const Params &p)
  : DmaDevice(p),
    libHandle_(nullptr),
    deviceHandle_(nullptr),
    abi_{},
    libraryPath_(p.library),
    kernelPath_(p.kernel),
    pioAddr_(p.pio_addr),
    pioSize_(p.pio_size),
    pioLatency_(p.pio_latency),
    tickEvent_([this]{ this->tick(); }, name() + ".tickEvent")
{
    if (libraryPath_.empty()) {
        fatal("VortexGPGPU: 'library' parameter is required "
              "(path to libvortex-gem5.so)");
    }

    // dlopen with RTLD_LAZY|RTLD_LOCAL — local so multiple SimObject
    // instances don't share symbol scope, lazy because we resolve
    // explicitly with dlsym below anyway.
    libHandle_ = dlopen(libraryPath_.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (libHandle_ == nullptr) {
        fatal("VortexGPGPU: dlopen('%s') failed: %s",
              libraryPath_, dlerror());
    }

    // Resolve the full v1 C ABI surface. Any missing symbol is a hard
    // build mismatch between gem5 and the Vortex library — fatal so
    // we fail fast at construction rather than mid-simulation.
    abi_.build_info   = dlsym_or_fatal<const char*(*)(void)>
                          (libHandle_, "vortex_gem5_build_info",   libraryPath_.c_str());
    abi_.create       = dlsym_or_fatal<void*(*)(void)>
                          (libHandle_, "vortex_gem5_create",       libraryPath_.c_str());
    abi_.destroy      = dlsym_or_fatal<void(*)(void*)>
                          (libHandle_, "vortex_gem5_destroy",      libraryPath_.c_str());
    abi_.load_kernel  = dlsym_or_fatal<int(*)(void*, const char*)>
                          (libHandle_, "vortex_gem5_load_kernel",  libraryPath_.c_str());
    abi_.tick         = dlsym_or_fatal<bool(*)(void*)>
                          (libHandle_, "vortex_gem5_tick",         libraryPath_.c_str());
    abi_.mmio_read64  = dlsym_or_fatal<uint64_t(*)(void*, uint64_t)>
                          (libHandle_, "vortex_gem5_mmio_read64",  libraryPath_.c_str());
    abi_.mmio_write64 = dlsym_or_fatal<void(*)(void*, uint64_t, uint64_t)>
                          (libHandle_, "vortex_gem5_mmio_write64", libraryPath_.c_str());
    abi_.vram_write   = dlsym_or_fatal<void(*)(void*, uint64_t, const uint8_t*, uint32_t)>
                          (libHandle_, "vortex_gem5_vram_write",   libraryPath_.c_str());
    abi_.vram_read    = dlsym_or_fatal<void(*)(void*, uint64_t, uint8_t*, uint32_t)>
                          (libHandle_, "vortex_gem5_vram_read",    libraryPath_.c_str());
    abi_.dcr_write    = dlsym_or_fatal<int(*)(void*, uint32_t, uint32_t)>
                          (libHandle_, "vortex_gem5_dcr_write",    libraryPath_.c_str());
    abi_.dcr_read     = dlsym_or_fatal<int(*)(void*, uint32_t, uint32_t, uint32_t*)>
                          (libHandle_, "vortex_gem5_dcr_read",     libraryPath_.c_str());
    abi_.pop_pending_cmd = dlsym_or_fatal<uint64_t(*)(void*)>
                          (libHandle_, "vortex_gem5_pop_pending_cmd", libraryPath_.c_str());
    abi_.get_cmd_arg  = dlsym_or_fatal<uint64_t(*)(void*, int)>
                          (libHandle_, "vortex_gem5_get_cmd_arg",  libraryPath_.c_str());
    abi_.set_busy     = dlsym_or_fatal<void(*)(void*, bool)>
                          (libHandle_, "vortex_gem5_set_busy",     libraryPath_.c_str());

    inform("VortexGPGPU: %s", abi_.build_info());
    inform("VortexGPGPU: library=%s pio=[0x%llx,+0x%llx)",
           libraryPath_,
           static_cast<unsigned long long>(pioAddr_),
           static_cast<unsigned long long>(pioSize_));

    deviceHandle_ = abi_.create();
    if (deviceHandle_ == nullptr) {
        fatal("VortexGPGPU: vortex_gem5_create returned NULL");
    }
}

VortexGPGPU::~VortexGPGPU()
{
    if (deviceHandle_ != nullptr && abi_.destroy != nullptr) {
        abi_.destroy(deviceHandle_);
    }
    if (libHandle_ != nullptr) {
        dlclose(libHandle_);
    }
}

void
VortexGPGPU::init()
{
    DmaDevice::init();
}

void
VortexGPGPU::startup()
{
    DmaDevice::startup();

    if (!kernelPath_.empty()) {
        // Standalone mode (Phase 3): preload a kernel and self-drive
        // to completion. Used by ci/gem5_test_vortex_hello.py — no
        // host CPU needed.
        inform("VortexGPGPU: standalone mode (preload + auto-tick)");
        inform("VortexGPGPU: preloading kernel=%s", kernelPath_);
        if (abi_.load_kernel(deviceHandle_, kernelPath_.c_str()) != 0) {
            fatal("VortexGPGPU: vortex_gem5_load_kernel('%s') failed",
                  kernelPath_);
        }
        standalone_ = true;
        schedule(tickEvent_, clockEdge(Cycles(1)));
    } else {
        // Hosted mode (Phase 5+): the host CPU uploads kernels via
        // MMIO/DMA and triggers execution with CMD_RUN. We sit idle
        // until then; CMD_RUN's write handler schedules tickEvent_.
        inform("VortexGPGPU: hosted mode (waiting for host CMD_RUN)");
        standalone_ = false;
    }
}

void
VortexGPGPU::tick()
{
    bool running = abi_.tick(deviceHandle_);
    if (running) {
        schedule(tickEvent_, clockEdge(Cycles(1)));
        return;
    }
    // Kernel finished.
    if (standalone_) {
        inform("VortexGPGPU: standalone kernel complete — exiting sim loop");
        exitSimLoop("VortexGPGPU: kernel complete");
    } else {
        // Host CPU is polling MMIO_STATUS waiting for busy bit to
        // clear; do that now so vx_ready_wait returns.
        abi_.set_busy(deviceHandle_, false);
    }
}

Tick
VortexGPGPU::read(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr_;
    const uint64_t value = abi_.mmio_read64(deviceHandle_, offset);

    // 64-bit aligned access is the only shape the OPAE protocol uses.
    // Stuff the result into the packet regardless of size (gem5 will
    // truncate based on getSize); narrow reads are unsupported by the
    // protocol but harmless here.
    pkt->setUintX(value, ByteOrder::little);
    pkt->makeAtomicResponse();
    return pioLatency_;
}

Tick
VortexGPGPU::write(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr_;
    const uint64_t value = pkt->getUintX(ByteOrder::little);

    // Always forward the write to the Vortex library first so the
    // device sees the args/CMD_TYPE in order.
    abi_.mmio_write64(deviceHandle_, offset, value);

    // Then react to commands that need gem5-side action (kicking the
    // tick scheduler for CMD_RUN; Phase 5+ will add CMD_MEM_*
    // dispatch through dmaPort).
    if (offset == MMIO_CMD_TYPE) {
        handleCmdType(value);
    }

    pkt->makeAtomicResponse();
    return pioLatency_;
}

void
VortexGPGPU::handleCmdType(uint64_t /*value*/)
{
    // Read which async command the library wants us to handle.
    // Sync commands (DCR_*) already completed inside mmio_write64
    // and don't surface here (pop returns 0).
    const uint64_t cmd = abi_.pop_pending_cmd(deviceHandle_);
    if (cmd == 0) return;

    if (cmd == CMD_RUN) {
        // Schedule the tick loop. tick() clears busy_ when the
        // kernel finishes (via abi_.set_busy(false)).
        if (!tickEvent_.scheduled()) {
            schedule(tickEvent_, clockEdge(Cycles(1)));
        }
        return;
    }

    if (cmd == CMD_MEM_WRITE || cmd == CMD_MEM_READ) {
        // Args are CACHE-LINE shifted in the OPAE protocol.
        const Addr host_addr = abi_.get_cmd_arg(deviceHandle_, 0)
                                 << CACHE_BLOCK_LOG2;
        const Addr dev_addr  = abi_.get_cmd_arg(deviceHandle_, 1)
                                 << CACHE_BLOCK_LOG2;
        const uint64_t size  = abi_.get_cmd_arg(deviceHandle_, 2)
                                 << CACHE_BLOCK_LOG2;

        // Scratch buffer for the transfer; freed inside the
        // completion callback. EventFunctionWrapper's `true` tail
        // arg flags auto-delete after firing.
        auto* scratch = new uint8_t[size];
        void* deviceHandle = deviceHandle_;
        auto& abi = abi_;

        if (cmd == CMD_MEM_WRITE) {
            // Host pinned buffer → device VRAM.
            auto* done = new EventFunctionWrapper(
                [&abi, deviceHandle, dev_addr, scratch, size]() {
                    abi.vram_write(deviceHandle, dev_addr, scratch,
                                   static_cast<uint32_t>(size));
                    delete[] scratch;
                    abi.set_busy(deviceHandle, false);
                },
                name() + ".dmaReadDone",
                /*deletePostEvent=*/true);
            dmaRead(host_addr, size, done, scratch);
        } else {
            // Device VRAM → host pinned buffer.
            abi.vram_read(deviceHandle, dev_addr, scratch,
                          static_cast<uint32_t>(size));
            auto* done = new EventFunctionWrapper(
                [&abi, deviceHandle, scratch]() {
                    delete[] scratch;
                    abi.set_busy(deviceHandle, false);
                },
                name() + ".dmaWriteDone",
                /*deletePostEvent=*/true);
            dmaWrite(host_addr, size, done, scratch);
        }
        return;
    }
}

AddrRangeList
VortexGPGPU::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(RangeSize(pioAddr_, pioSize_));
    return ranges;
}

} // namespace gem5
