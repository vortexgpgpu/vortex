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

namespace gem5
{

namespace {

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
    pinAddr_(p.pin_addr),
    pinSize_(p.pin_size),
    pioLatency_(p.pio_latency),
    cpTickEvent_([this]{ this->cpTick(); }, name() + ".cpTickEvent"),
    vortexTickEvent_([this]{ this->vortexTick(); }, name() + ".vortexTickEvent"),
    standalone_(false)
{
    if (libraryPath_.empty()) {
        fatal("VortexGPGPU: 'library' parameter is required "
              "(path to libvortex-gem5.so)");
    }

    libHandle_ = dlopen(libraryPath_.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (libHandle_ == nullptr) {
        fatal("VortexGPGPU: dlopen('%s') failed: %s",
              libraryPath_, dlerror());
    }

    // Resolve the ABI surface. Any missing symbol is a hard build
    // mismatch — fatal at construction rather than mid-simulation.
    abi_.build_info        = dlsym_or_fatal<const char*(*)(void)>
                              (libHandle_, "vortex_gem5_build_info",        libraryPath_.c_str());
    abi_.create            = dlsym_or_fatal<void*(*)(void)>
                              (libHandle_, "vortex_gem5_create",            libraryPath_.c_str());
    abi_.destroy           = dlsym_or_fatal<void(*)(void*)>
                              (libHandle_, "vortex_gem5_destroy",           libraryPath_.c_str());
    abi_.set_start_handler = dlsym_or_fatal<void(*)(void*, void(*)(void*), void*)>
                              (libHandle_, "vortex_gem5_set_start_handler", libraryPath_.c_str());
    abi_.load_kernel       = dlsym_or_fatal<int(*)(void*, const char*)>
                              (libHandle_, "vortex_gem5_load_kernel",       libraryPath_.c_str());
    abi_.cp_mmio_write     = dlsym_or_fatal<void(*)(void*, uint32_t, uint32_t)>
                              (libHandle_, "vortex_gem5_cp_mmio_write",     libraryPath_.c_str());
    abi_.cp_mmio_read      = dlsym_or_fatal<uint32_t(*)(void*, uint32_t)>
                              (libHandle_, "vortex_gem5_cp_mmio_read",      libraryPath_.c_str());
    abi_.cp_tick           = dlsym_or_fatal<bool(*)(void*)>
                              (libHandle_, "vortex_gem5_cp_tick",           libraryPath_.c_str());
    abi_.cp_has_work       = dlsym_or_fatal<bool(*)(void*)>
                              (libHandle_, "vortex_gem5_cp_has_work",       libraryPath_.c_str());
    abi_.vortex_tick       = dlsym_or_fatal<bool(*)(void*)>
                              (libHandle_, "vortex_gem5_vortex_tick",       libraryPath_.c_str());
    abi_.vortex_busy       = dlsym_or_fatal<bool(*)(void*)>
                              (libHandle_, "vortex_gem5_vortex_busy",       libraryPath_.c_str());
    abi_.vram_write        = dlsym_or_fatal<void(*)(void*, uint64_t, const uint8_t*, uint32_t)>
                              (libHandle_, "vortex_gem5_vram_write",        libraryPath_.c_str());
    abi_.vram_read         = dlsym_or_fatal<void(*)(void*, uint64_t, uint8_t*, uint32_t)>
                              (libHandle_, "vortex_gem5_vram_read",         libraryPath_.c_str());

    inform("VortexGPGPU: %s", abi_.build_info());
    inform("VortexGPGPU: library=%s", libraryPath_);
    inform("VortexGPGPU: pio[CP regfile]=[0x%llx,+0x%llx)",
           static_cast<unsigned long long>(pioAddr_),
           static_cast<unsigned long long>(pioSize_));
    if (pinSize_ != 0) {
        inform("VortexGPGPU: pin[BAR-mapped VRAM]=[0x%llx,+0x%llx)",
               static_cast<unsigned long long>(pinAddr_),
               static_cast<unsigned long long>(pinSize_));
    }

    deviceHandle_ = abi_.create();
    if (deviceHandle_ == nullptr) {
        fatal("VortexGPGPU: vortex_gem5_create returned NULL");
    }

    // Register the vortex_start trampoline so the CP can schedule
    // Vortex ticks from inside cp_tick.
    abi_.set_start_handler(deviceHandle_, &VortexGPGPU::onVortexStartTrampoline, this);
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
        // Standalone mode: preload a kernel and self-drive to completion.
        // No host CPU involvement; used as a smoke test for the device library.
        inform("VortexGPGPU: standalone mode (preload + auto-tick)");
        inform("VortexGPGPU: preloading kernel=%s", kernelPath_);
        if (abi_.load_kernel(deviceHandle_, kernelPath_.c_str()) != 0) {
            fatal("VortexGPGPU: vortex_gem5_load_kernel('%s') failed",
                  kernelPath_);
        }
        standalone_ = true;
        schedule(vortexTickEvent_, clockEdge(Cycles(1)));
    } else {
        // Hosted mode: the host runtime issues CP MMIO writes to configure
        // queues and commit commands; the CP schedules ticks via maybeWakeCp()
        // and the vortex tick via the start handler. Idle at boot.
        inform("VortexGPGPU: hosted mode (waiting for CP enable)");
        standalone_ = false;
    }
}

void
VortexGPGPU::cpTick()
{
    const bool still_busy = abi_.cp_tick(deviceHandle_);
    if (still_busy) {
        schedule(cpTickEvent_, clockEdge(Cycles(1)));
    }
    // Idle drop-out: no reschedule. PIO writes that arm new work will
    // call maybeWakeCp() and reschedule us.
}

void
VortexGPGPU::vortexTick()
{
    const bool still_running = abi_.vortex_tick(deviceHandle_);
    if (still_running) {
        schedule(vortexTickEvent_, clockEdge(Cycles(1)));
        return;
    }
    if (standalone_) {
        inform("VortexGPGPU: standalone kernel complete — exiting sim loop");
        exitSimLoop("VortexGPGPU: kernel complete");
        return;
    }
    // Hosted mode: Vortex finished. The CP's launch FSM observes
    // vortex_busy() == false on its next tick and retires the
    // CMD_LAUNCH. If the CP is already idle (no scheduled tick) we
    // need to wake it so the retirement actually happens.
    maybeWakeCp();
}

void
VortexGPGPU::maybeWakeCp()
{
    if (abi_.cp_has_work(deviceHandle_) && !cpTickEvent_.scheduled()) {
        schedule(cpTickEvent_, clockEdge(Cycles(1)));
    }
}

void
VortexGPGPU::onVortexStartTrampoline(void* ctx)
{
    static_cast<VortexGPGPU*>(ctx)->onVortexStart();
}

void
VortexGPGPU::onVortexStart()
{
    if (!vortexTickEvent_.scheduled()) {
        schedule(vortexTickEvent_, clockEdge(Cycles(1)));
    }
}

Tick
VortexGPGPU::read(PacketPtr pkt)
{
    const Addr a = pkt->getAddr();
    if (a >= pioAddr_ && a < pioAddr_ + pioSize_) {
        // CP regfile access — 32-bit only.
        const uint32_t off = uint32_t(a - pioAddr_);
        const uint32_t value = abi_.cp_mmio_read(deviceHandle_, off);
        pkt->setUintX(static_cast<uint64_t>(value), ByteOrder::little);
        pkt->makeAtomicResponse();
        return pioLatency_;
    }
    // BAR-mapped VRAM access (CPU is reading device memory directly).
    // Variable-width packet (host load / cache-line fill).
    const uint64_t dev_addr = a - pinAddr_;
    abi_.vram_read(deviceHandle_,
                   dev_addr,
                   pkt->getPtr<uint8_t>(),
                   uint32_t(pkt->getSize()));
    pkt->makeAtomicResponse();
    return pioLatency_;
}

Tick
VortexGPGPU::write(PacketPtr pkt)
{
    const Addr a = pkt->getAddr();
    if (a >= pioAddr_ && a < pioAddr_ + pioSize_) {
        // CP regfile write — 32-bit only.
        const uint32_t off = uint32_t(a - pioAddr_);
        const uint64_t raw = pkt->getUintX(ByteOrder::little);
        abi_.cp_mmio_write(deviceHandle_, off, uint32_t(raw));
        maybeWakeCp();
        pkt->makeAtomicResponse();
        return pioLatency_;
    }
    // BAR-mapped VRAM write — variable-width packet (host store /
    // cache writeback). Subsequent device reads at the same address
    // see the bytes written here.
    const uint64_t dev_addr = a - pinAddr_;
    abi_.vram_write(deviceHandle_,
                    dev_addr,
                    pkt->getConstPtr<uint8_t>(),
                    uint32_t(pkt->getSize()));
    // Writes to device VRAM may seed CP ring entries; if the CP is
    // dormant, leave it dormant (the CP only wakes on a doorbell PIO
    // write, not on a ring-fill).
    pkt->makeAtomicResponse();
    return pioLatency_;
}

AddrRangeList
VortexGPGPU::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(RangeSize(pioAddr_, pioSize_));
    if (pinSize_ != 0) {
        ranges.push_back(RangeSize(pinAddr_, pinSize_));
    }
    return ranges;
}

} // namespace gem5
