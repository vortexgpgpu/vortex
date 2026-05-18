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
#include "processor.h"
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
// check in vram_{read,write} matches what the host runtime enforces
// on its side. Inlined rather than including common.h because that
// header drags in the full runtime ABI (vortex.h + callbacks.h +
// mem_alloc.h) which a device library has no business touching.
#if (XLEN == 64)
static constexpr uint64_t GEM5_GLOBAL_MEM_SIZE = 0x200000000ull;  // 8 GB
#else
static constexpr uint64_t GEM5_GLOBAL_MEM_SIZE = 0x100000000ull;  // 4 GB
#endif

// OPAE MMIO command-set constants (same as
// hw/syn/altera/opae/vortex_afu.json + sw/runtime/gem5/vortex.cpp).
// Hardcoded — no #include of vortex_opae.h — to keep the device
// library independent of the OPAE header generator.
namespace cmd {
constexpr uint64_t MEM_READ  = 1;
constexpr uint64_t MEM_WRITE = 2;
constexpr uint64_t RUN       = 3;
constexpr uint64_t DCR_WRITE = 4;
constexpr uint64_t DCR_READ  = 5;
} // namespace cmd
namespace mmio {
constexpr uint64_t CMD_TYPE  = 10 * 4;  // byte offsets, matching the
constexpr uint64_t CMD_ARG0  = 12 * 4;  // sw/runtime side
constexpr uint64_t CMD_ARG1  = 14 * 4;
constexpr uint64_t CMD_ARG2  = 16 * 4;
constexpr uint64_t STATUS    = 18 * 4;
constexpr uint64_t DCR_RSP   = 28 * 4;
} // namespace mmio

// Internal C++ class. Mirrors the shape of vortex::VortexSimulator in
// sim/simx/sst/ — same Processor + RAM ownership, same KMU DCR priming,
// same load_kernel paths — but with no SST types in the interface.
namespace {

class Gem5Device {
public:
  Gem5Device()
    : ram_(0, MEM_PAGE_SIZE)
    , proc_(std::make_unique<Processor>()) {
    proc_->attach_ram(&ram_);
  }

  ~Gem5Device() = default;

  // Load a kernel image and prime the KMU for a 1×1×1 CTA at
  // STARTUP_ADDR. After this, cycle() will dispatch the kernel.
  // Returns true on success.
  bool load_kernel(const std::string& path) {
    // KMU DCRs — same sequence as sim/simx/main.cpp:101–116 and
    // sim/simx/sst/vortex_simulator.cpp:22–39.
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
    return true;
  }

  bool tick()  { return proc_->cycle(); }

  // Memory access uses the same ACL-bypass pattern as
  // sw/runtime/simx/vortex.cpp upload()/download(); the gem5 DMA path
  // is a peer of the host runtime, not a userspace caller subject to
  // page protections.
  void vram_write(uint64_t addr, const uint8_t* src, uint32_t size) {
    if (addr + size > GEM5_GLOBAL_MEM_SIZE) {
    #ifndef NDEBUG
      std::cerr << "vortex_gem5: vram_write overflow addr=0x"
                << std::hex << addr << " size=" << std::dec << size << std::endl;
    #endif
      return;
    }
    ram_.enable_acl(false);
    ram_.write(src, addr, size);
    ram_.enable_acl(true);
  }

  void vram_read(uint64_t addr, uint8_t* dst, uint32_t size) {
    if (addr + size > GEM5_GLOBAL_MEM_SIZE) {
    #ifndef NDEBUG
      std::cerr << "vortex_gem5: vram_read overflow addr=0x"
                << std::hex << addr << " size=" << std::dec << size << std::endl;
    #endif
      return;
    }
    ram_.enable_acl(false);
    ram_.read(dst, addr, size);
    ram_.enable_acl(true);
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    return proc_->dcr_write(addr, value);
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
    return proc_->dcr_read(addr, tag, value);
  }

  // OPAE MMIO command-set state machine. The host runtime
  // (sw/runtime/gem5/vortex.cpp) drives it in exactly the same
  // shape as sw/runtime/opae/vortex.cpp:
  //   1. Write CMD_ARG0/1/2 with command-specific args
  //   2. Write CMD_TYPE — triggers the command
  //   3. Poll MMIO_STATUS until busy bit clears
  //   4. (For DCR_READ) read MMIO_DCR_RSP for the response
  //
  // Synchronous commands (DCR_*) complete inside this function and
  // clear the busy bit immediately. Async commands (RUN, MEM_*)
  // surface to the gem5 SimObject via pop_pending_cmd; the SimObject
  // performs the gem5-side work (clock ticks, DMA) and clears busy
  // when done.
  uint64_t mmio_read64(uint64_t offset) {
    if (offset == mmio::STATUS)  return busy_ ? 1u : 0u;
    if (offset == mmio::DCR_RSP) return dcr_rsp_;
    return 0;
  }

  void mmio_write64(uint64_t offset, uint64_t value) {
    if (offset == mmio::CMD_ARG0) { cmd_args_[0] = value; return; }
    if (offset == mmio::CMD_ARG1) { cmd_args_[1] = value; return; }
    if (offset == mmio::CMD_ARG2) { cmd_args_[2] = value; return; }
    if (offset != mmio::CMD_TYPE) return;  // unknown reg — ignore

    busy_ = true;
    switch (value) {
    case cmd::DCR_WRITE: {
      proc_->dcr_write(uint32_t(cmd_args_[0]), uint32_t(cmd_args_[1]));
      busy_ = false;
      break;
    }
    case cmd::DCR_READ: {
      uint32_t v = 0;
      proc_->dcr_read(uint32_t(cmd_args_[0]),
                      uint32_t(cmd_args_[1]),
                      &v);
      dcr_rsp_ = v;
      busy_ = false;
      break;
    }
    case cmd::RUN:
    case cmd::MEM_READ:
    case cmd::MEM_WRITE:
      // Async — gem5 SimObject reads pending_cmd_ on the same MMIO
      // dispatch tick and routes the work (clock cycles for RUN,
      // dmaAction for MEM_*). It clears busy when done.
      pending_cmd_ = value;
      break;
    default:
      // Unknown command: drop the busy bit so the host doesn't hang.
      busy_ = false;
      break;
    }
  }

  uint64_t pop_pending_cmd() {
    uint64_t c = pending_cmd_;
    pending_cmd_ = 0;
    return c;
  }
  uint64_t get_cmd_arg(int which) const {
    return (which >= 0 && which < 3) ? cmd_args_[which] : 0;
  }
  void set_busy(bool busy) { busy_ = busy; }

private:
  RAM ram_;
  std::unique_ptr<Processor> proc_;

  // OPAE protocol state.
  uint64_t cmd_args_[3] = {0, 0, 0};
  uint64_t pending_cmd_ = 0;
  uint64_t dcr_rsp_     = 0;
  bool     busy_        = false;
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

int vortex_gem5_load_kernel(vortex_gem5_handle_t h, const char* path) {
  if (h == nullptr || path == nullptr) return -1;
  return reinterpret_cast<Gem5Device*>(h)->load_kernel(path) ? 0 : -1;
}

bool vortex_gem5_tick(vortex_gem5_handle_t h) {
  if (h == nullptr) return false;
  return reinterpret_cast<Gem5Device*>(h)->tick();
}

uint64_t vortex_gem5_mmio_read64(vortex_gem5_handle_t h, uint64_t offset) {
  if (h == nullptr) return 0;
  return reinterpret_cast<Gem5Device*>(h)->mmio_read64(offset);
}

void vortex_gem5_mmio_write64(vortex_gem5_handle_t h, uint64_t offset, uint64_t value) {
  if (h == nullptr) return;
  reinterpret_cast<Gem5Device*>(h)->mmio_write64(offset, value);
}

void vortex_gem5_vram_write(vortex_gem5_handle_t h, uint64_t dev_addr, const uint8_t* src, uint32_t size) {
  if (h == nullptr || src == nullptr) return;
  reinterpret_cast<Gem5Device*>(h)->vram_write(dev_addr, src, size);
}

void vortex_gem5_vram_read(vortex_gem5_handle_t h, uint64_t dev_addr, uint8_t* dst, uint32_t size) {
  if (h == nullptr || dst == nullptr) return;
  reinterpret_cast<Gem5Device*>(h)->vram_read(dev_addr, dst, size);
}

int vortex_gem5_dcr_write(vortex_gem5_handle_t h, uint32_t addr, uint32_t value) {
  if (h == nullptr) return -1;
  return reinterpret_cast<Gem5Device*>(h)->dcr_write(addr, value);
}

int vortex_gem5_dcr_read(vortex_gem5_handle_t h, uint32_t addr, uint32_t tag, uint32_t* value) {
  if (h == nullptr || value == nullptr) return -1;
  return reinterpret_cast<Gem5Device*>(h)->dcr_read(addr, tag, value);
}

uint64_t vortex_gem5_pop_pending_cmd(vortex_gem5_handle_t h) {
  if (h == nullptr) return 0;
  return reinterpret_cast<Gem5Device*>(h)->pop_pending_cmd();
}

uint64_t vortex_gem5_get_cmd_arg(vortex_gem5_handle_t h, int which) {
  if (h == nullptr) return 0;
  return reinterpret_cast<Gem5Device*>(h)->get_cmd_arg(which);
}

void vortex_gem5_set_busy(vortex_gem5_handle_t h, bool busy) {
  if (h == nullptr) return;
  reinterpret_cast<Gem5Device*>(h)->set_busy(busy);
}

} // extern "C"
