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

#include "vortex_gpgpu.h"

#include "constants.h"
#include "dev_mem.h"
#include "processor.h"
#include <cmd_processor.h>
#include <mem.h>
#include <util.h>
#include <VX_config.h>
#include <VX_types.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace vortex;

// Mirrors sw/runtime/common/common.h's GLOBAL_MEM_SIZE so the bounds
// check in vram_{read,write} matches what the host runtime enforces.
// Inlined rather than including common.h because that header drags in
// the full runtime ABI which a device library has no business touching.
#if (XLEN == 64)
static constexpr uint64_t GEM5_GLOBAL_MEM_SIZE = 0x200000000ull;  // 8 GB
#else
static constexpr uint64_t GEM5_GLOBAL_MEM_SIZE = 0x100000000ull;  // 4 GB
#endif

namespace {

// Gem5Device — owns the Vortex Processor + RAM + CommandProcessor
// triplet. The CP's hooks call back into proc_/dev_mem_, and the
// SimObject drives cp_tick / vortex_tick on independent gem5 events.
class Gem5Device {
public:
    Gem5Device()
        : ram_(0, MEM_PAGE_SIZE),
          proc_(std::make_unique<Processor>()),
          dev_mem_(std::make_unique<vortex_gem5::InProcessDevMem>(ram_)),
          cp_(make_cp_hooks()) {
        proc_->attach_ram(&ram_);
    }

    ~Gem5Device() = default;

    // ---------------- Standalone (Phase 3) kernel preload ---------------
    // Primes the KMU DCRs for a 1×1×1 CTA at STARTUP_ADDR and loads the
    // ELF/bin/hex into VRAM. After this, calling vortex_tick repeatedly
    // dispatches the kernel to completion (ProcessorImpl::cycle's lazy
    // init resets SimPlatform and calls kmu_->start() on first tick).
    // The hosted (CP-driven) path never calls this — kernel ELFs land
    // in VRAM via mem_upload, and KMU programming goes through CMD_DCR_*.
    bool load_kernel(const std::string& path) {
        const uint64_t startup_addr(STARTUP_ADDR);
        proc_->dcr_write(VX_DCR_KMU_STARTUP_ADDR0, startup_addr & 0xffffffff);
    #if (XLEN == 64)
        proc_->dcr_write(VX_DCR_KMU_STARTUP_ADDR1, startup_addr >> 32);
    #endif
        proc_->dcr_write(VX_DCR_KMU_STARTUP_ARG0, 0);
        proc_->dcr_write(VX_DCR_KMU_STARTUP_ARG1, 0);
        proc_->dcr_write(VX_DCR_KMU_GRID_DIM_X,   1);
        proc_->dcr_write(VX_DCR_KMU_GRID_DIM_Y,   1);
        proc_->dcr_write(VX_DCR_KMU_GRID_DIM_Z,   1);
        proc_->dcr_write(VX_DCR_KMU_BLOCK_DIM_X,  1);
        proc_->dcr_write(VX_DCR_KMU_BLOCK_DIM_Y,  1);
        proc_->dcr_write(VX_DCR_KMU_BLOCK_DIM_Z,  1);
        proc_->dcr_write(VX_DCR_KMU_LMEM_SIZE,    0);
        proc_->dcr_write(VX_DCR_KMU_BLOCK_SIZE,   1);
        proc_->dcr_write(VX_DCR_KMU_WARP_STEP_X,  NUM_THREADS);
        proc_->dcr_write(VX_DCR_KMU_WARP_STEP_Y,  0);
        proc_->dcr_write(VX_DCR_KMU_WARP_STEP_Z,  0);

        std::string ext(fileExtension(path.c_str()));
        if (ext == "vxbin") {
            ram_.loadVxImage(path.c_str());
        } else if (ext == "bin") {
            ram_.loadBinImage(path.c_str(), startup_addr);
        } else if (ext == "hex") {
            ram_.loadHexImage(path.c_str());
        } else {
            std::cerr << "vortex_gem5: unsupported kernel extension '" << ext
                      << "' (need .vxbin, .bin, or .hex)" << std::endl;
            return false;
        }
        // Mark the device as "running" so the SimObject's standalone
        // path advances vortexTickEvent_ until ProcessorImpl::cycle()
        // reports done. Hosted launches set this via vortex_start.
        vortex_running_ = true;
        return true;
    }

    // ---------------- VRAM direct access --------------------------------
    void vram_write(uint64_t addr, const uint8_t* src, uint32_t size) {
        if (addr + size > GEM5_GLOBAL_MEM_SIZE) {
        #ifndef NDEBUG
            std::cerr << "vortex_gem5: vram_write overflow addr=0x"
                      << std::hex << addr << " size=" << std::dec << size
                      << std::endl;
        #endif
            return;
        }
        dev_mem_->write(addr, src, size);
    }
    void vram_read(uint64_t addr, uint8_t* dst, uint32_t size) {
        if (addr + size > GEM5_GLOBAL_MEM_SIZE) {
        #ifndef NDEBUG
            std::cerr << "vortex_gem5: vram_read overflow addr=0x"
                      << std::hex << addr << " size=" << std::dec << size
                      << std::endl;
        #endif
            return;
        }
        dev_mem_->read(addr, dst, size);
    }

    // ---------------- CP regfile MMIO -----------------------------------
    // The SimObject's PIO handlers translate `cp_mmio_write(off,v)` to
    // a single call here. The CommandProcessor's regfile is 32-bit and
    // its address map is documented in sim/common/cmd_processor.h.
    void cp_mmio_write(uint32_t off, uint32_t value) { cp_.mmio_write(off, value); }
    uint32_t cp_mmio_read (uint32_t off) const       { return cp_.mmio_read(off); }

    // ---------------- CP tick / introspection ---------------------------
    // tick() advances the CP one functional cycle and returns true iff
    // the CP still has work to do. The SimObject reschedules
    // cpTickEvent_ while true and sleeps otherwise — proposal §2.3.
    bool cp_tick() {
        cp_.tick();
        return cp_.busy();
    }
    bool cp_has_work() const { return cp_.enabled() && cp_.busy(); }

    // ---------------- Vortex tick / introspection -----------------------
    // vortex_tick advances ProcessorImpl::cycle() one step. cycle() does
    // lazy init (resets SimPlatform + calls kmu_->start()) on first call.
    // For back-to-back launches the CP's vortex_start hook calls
    // processor_.start_kmu() explicitly to re-arm the KMU for the next
    // kernel (kmu_->start is idempotent — first launch redundantly
    // re-starts inside the lazy init, no harm).
    bool vortex_tick() {
        bool still_running = proc_->cycle();
        if (!still_running) {
            vortex_running_ = false;
        }
        return vortex_running_;
    }
    bool vortex_busy() const { return vortex_running_; }

    // ---------------- vortex_start handler registration -----------------
    // The SimObject registers a callback the CP fires when retiring a
    // CMD_LAUNCH. The callback schedules vortexTickEvent_ at the next
    // clock edge, decoupling CP and Vortex tick chains (proposal §2.4).
    void set_start_handler(vortex_gem5_start_handler_t fn, void* ctx) {
        start_fn_  = fn;
        start_ctx_ = ctx;
    }

private:
    vortex::CommandProcessor::Hooks make_cp_hooks() {
        vortex::CommandProcessor::Hooks h;
        h.dram_read = [this](uint64_t addr, void* dst, std::size_t bytes) {
            dev_mem_->read(addr, dst, bytes);
        };
        h.dram_write = [this](uint64_t addr, const void* src, std::size_t bytes) {
            dev_mem_->write(addr, src, bytes);
        };
        h.vortex_dcr_write = [this](uint32_t addr, uint32_t value) {
            proc_->dcr_write(addr, value);
        };
        h.vortex_dcr_read = [this](uint32_t addr, uint32_t tag) -> uint32_t {
            uint32_t v = 0;
            proc_->dcr_read(addr, tag, &v);
            return v;
        };
        h.vortex_start = [this]() {
            // Mark Vortex as in-flight so vortex_busy returns true on
            // the very next CP poll (before the first cycle() runs).
            // Then re-arm the KMU for the (possibly back-to-back)
            // kernel and ask the SimObject to begin ticking Vortex.
            vortex_running_ = true;
            proc_->start_kmu();
            if (start_fn_) start_fn_(start_ctx_);
        };
        h.vortex_busy = [this]() -> bool { return vortex_running_; };
        return h;
    }

    RAM ram_;
    std::unique_ptr<Processor> proc_;
    std::unique_ptr<vortex_gem5::DevMemAccessor> dev_mem_;
    vortex::CommandProcessor cp_;
    bool vortex_running_ = false;
    vortex_gem5_start_handler_t start_fn_  = nullptr;
    void* start_ctx_ = nullptr;
};

} // namespace

// ----- C ABI -----------------------------------------------------------------

extern "C" {

const char* vortex_gem5_build_info(void) {
    static char info[256];
    std::snprintf(info, sizeof(info),
                  "vortex-gem5 (XLEN=%d, threads=%d, warps=%d, cores=%d, clusters=%d)",
                  XLEN, NUM_THREADS, NUM_WARPS, NUM_CORES, NUM_CLUSTERS);
    return info;
}

vortex_gem5_handle_t vortex_gem5_create(void) {
    try {
        return reinterpret_cast<vortex_gem5_handle_t>(new Gem5Device());
    } catch (const std::exception& e) {
        std::cerr << "vortex_gem5_create: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "vortex_gem5_create: unknown exception" << std::endl;
        return nullptr;
    }
}

void vortex_gem5_destroy(vortex_gem5_handle_t h) {
    if (h == nullptr) return;
    delete reinterpret_cast<Gem5Device*>(h);
}

void vortex_gem5_set_start_handler(vortex_gem5_handle_t h,
                                   vortex_gem5_start_handler_t fn,
                                   void* ctx) {
    if (h == nullptr) return;
    reinterpret_cast<Gem5Device*>(h)->set_start_handler(fn, ctx);
}

int vortex_gem5_load_kernel(vortex_gem5_handle_t h, const char* path) {
    if (h == nullptr || path == nullptr) return -1;
    return reinterpret_cast<Gem5Device*>(h)->load_kernel(path) ? 0 : -1;
}

void vortex_gem5_cp_mmio_write(vortex_gem5_handle_t h,
                               uint32_t off, uint32_t value) {
    if (h == nullptr) return;
    reinterpret_cast<Gem5Device*>(h)->cp_mmio_write(off, value);
}

uint32_t vortex_gem5_cp_mmio_read(vortex_gem5_handle_t h, uint32_t off) {
    if (h == nullptr) return 0;
    return reinterpret_cast<Gem5Device*>(h)->cp_mmio_read(off);
}

bool vortex_gem5_cp_tick(vortex_gem5_handle_t h) {
    if (h == nullptr) return false;
    return reinterpret_cast<Gem5Device*>(h)->cp_tick();
}

bool vortex_gem5_cp_has_work(vortex_gem5_handle_t h) {
    if (h == nullptr) return false;
    return reinterpret_cast<Gem5Device*>(h)->cp_has_work();
}

bool vortex_gem5_vortex_tick(vortex_gem5_handle_t h) {
    if (h == nullptr) return false;
    return reinterpret_cast<Gem5Device*>(h)->vortex_tick();
}

bool vortex_gem5_vortex_busy(vortex_gem5_handle_t h) {
    if (h == nullptr) return false;
    return reinterpret_cast<Gem5Device*>(h)->vortex_busy();
}

void vortex_gem5_vram_write(vortex_gem5_handle_t h,
                            uint64_t dev_addr, const uint8_t* src,
                            uint32_t size) {
    if (h == nullptr || src == nullptr) return;
    reinterpret_cast<Gem5Device*>(h)->vram_write(dev_addr, src, size);
}

void vortex_gem5_vram_read(vortex_gem5_handle_t h,
                           uint64_t dev_addr, uint8_t* dst,
                           uint32_t size) {
    if (h == nullptr || dst == nullptr) return;
    reinterpret_cast<Gem5Device*>(h)->vram_read(dev_addr, dst, size);
}

} // extern "C"
