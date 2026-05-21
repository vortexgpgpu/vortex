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

#include <VX_types.h>
#include <common.h>

#include <constants.h>
#include <mem.h>
#include <processor.h>
#include <util.h>
#include <cmd_processor.h>

#ifdef VX_CFG_VM_ENABLE
#include <vm.h>
#include <memory>
#endif

#include <assert.h>
#include <chrono>
#include <future>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


using namespace vortex;

#ifdef VX_CFG_VM_ENABLE
// DeviceMemIO adapter over the simx RAM backing store. The ACL bypass
// is encapsulated here so VMManager itself stays driver-agnostic.
class RamMemIO : public DeviceMemIO {
public:
  explicit RamMemIO(RAM* ram) : ram_(ram) {}
  void read(void* dst, uint64_t addr, size_t size) override {
    ram_->enable_acl(false);
    ram_->read((uint8_t*)dst, addr, size);
    ram_->enable_acl(true);
  }
  void write(const void* src, uint64_t addr, size_t size) override {
    ram_->enable_acl(false);
    ram_->write((const uint8_t*)src, addr, size);
    ram_->enable_acl(true);
  }
private:
  RAM* ram_;
};
#endif

class vx_device {
public:
  vx_device()
      : ram_(0, VX_VM_PAGE_SIZE),
        processor_(),
        global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                    VX_VM_PAGE_SIZE, CACHE_BLOCK_SIZE),
        cp_(make_cp_hooks()) {
    // attach memory module
    processor_.attach_ram(&ram_);
  }

  ~vx_device() {
    if (future_.valid()) {
      future_.wait();
    }
  }

  int init() {
#ifdef VX_CFG_VM_ENABLE
    // Boot-time VM init: allocate the page table inside RAM, push SATP
    // into the simulator. Must run after attach_ram (constructor) and
    // before the first vx_mem_alloc (so phy_to_virt_map can mint VAs).
    dev_io_ = std::make_unique<RamMemIO>(&ram_);
    vm_mgr_ = std::make_unique<VMManager>(dev_io_.get());
    CHECK_ERR(vm_mgr_->init(), { return err; });
#endif
    return 0;
  }

  int get_caps(uint32_t caps_id, uint64_t *value) {
    uint64_t _value;
    switch (caps_id) {
    case VX_CAPS_VERSION:
      _value = VX_ISA_IMPL_ID;
      break;
    case VX_CAPS_NUM_THREADS:
      _value = VX_CFG_NUM_THREADS;
      break;
    case VX_CAPS_NUM_WARPS:
      _value = VX_CFG_NUM_WARPS;
      break;
    case VX_CAPS_NUM_CORES:
      _value = VX_CFG_NUM_CORES * VX_CFG_NUM_CLUSTERS;
      break;
    case VX_CAPS_NUM_CLUSTERS:
      _value = VX_CFG_NUM_CLUSTERS;
      break;
    case VX_CAPS_SOCKET_SIZE:
      _value = VX_CFG_SOCKET_SIZE;
      break;
    case VX_CAPS_ISSUE_WIDTH:
      _value = VX_CFG_ISSUE_WIDTH;
      break;
    case VX_CAPS_CACHE_LINE_SIZE:
      _value = CACHE_BLOCK_SIZE;
      break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
      _value = GLOBAL_MEM_SIZE;
      break;
    case VX_CAPS_LOCAL_MEM_SIZE:
      _value = (1 << VX_CFG_LMEM_LOG_SIZE);
      break;
    case VX_CAPS_ISA_FLAGS:
      _value = ((uint64_t(VX_CFG_MISA_EXT)) << 32) | ((log2floor(VX_CFG_XLEN) - 4) << 30) | VX_CFG_MISA_STD;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = VX_CFG_PLATFORM_MEMORY_NUM_BANKS;
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (VX_CFG_MEM_ADDR_WIDTH / VX_CFG_PLATFORM_MEMORY_NUM_BANKS);
      break;
    case VX_CAPS_CLOCK_RATE:
      _value = 0;
      break;
    case VX_CAPS_PEAK_MEM_BW:
      _value = VX_CFG_PLATFORM_MEMORY_PEAK_BW;
      break;
    default:
      std::cout << "invalid caps id: " << caps_id << std::endl;
      std::abort();
      return -1;
    }
    *value = _value;
    return 0;
  }

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr) {
#ifdef VX_CFG_VM_ENABLE
    uint64_t asize = aligned_size(size, VX_VM_PAGE_SIZE);
#else
    uint64_t asize = size;
#endif
    uint64_t addr = 0;

    DBGPRINT("[RT:mem_alloc] size: 0x%lx, asize, 0x%lx,flag : 0x%d\n", size, asize, flags);
    CHECK_ERR(global_mem_.allocate(asize, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, asize, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
#ifdef VX_CFG_VM_ENABLE
    if (flags & VX_MEM_PHYS) {
      // PHYS request: keep *dev_addr as the PA and identity-map it so
      // kernel loads (via the MMU) and fixed-function units (raster/
      // tex/om — bypass the MMU) see the same address.
      CHECK_ERR(vm_mgr_->install_identity_map(addr, asize), {
        global_mem_.release(addr);
        return err;
      });
    } else {
      // Replace the PA in *dev_addr with a freshly-minted VA. After
      // this call, the user-facing API uses VAs end-to-end.
      vm_mgr_->phy_to_virt_map(asize, dev_addr, flags);
    }
#endif
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
#ifdef VX_CFG_VM_ENABLE
    uint64_t asize = aligned_size(size, VX_VM_PAGE_SIZE);
#else
    uint64_t asize = size;
#endif
    CHECK_ERR(global_mem_.reserve(dev_addr, asize), {
      return err;
    });
    DBGPRINT("[RT:mem_reserve] addr: 0x%lx, asize:0x%lx, size: 0x%lx\n", dev_addr, asize, size);
    CHECK_ERR(this->mem_access(dev_addr, asize, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
#ifdef VX_CFG_VM_ENABLE
    // mem_reserve places content at the caller-chosen PA (vs mem_alloc,
    // which mints a fresh VA). The kernel will later access this region
    // via that same PA through the MMU, so install identity PTEs.
    CHECK_ERR(vm_mgr_->install_identity_map(dev_addr, asize), {
      global_mem_.release(dev_addr);
      return err;
    });
#endif
    return 0;
  }

  int mem_free(uint64_t dev_addr) {
#ifdef VX_CFG_VM_ENABLE
    // dev_addr is a VA; resolve to PA before releasing from the
    // physical-address-keyed global allocator.
    uint64_t paddr = vm_mgr_->page_table_walk(dev_addr);
    return global_mem_.release(paddr);
#else
    return global_mem_.release(dev_addr);
#endif
  }

  int mem_access(uint64_t dev_addr, uint64_t size, int flags) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dev_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    ram_.set_acl(dev_addr, size, flags);
    return 0;
  }

  int mem_info(uint64_t *mem_free, uint64_t *mem_used) const {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int copy(uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE || dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
    ram_.enable_acl(false);
    ram_.copy(dest_addr, src_addr, size);
    ram_.enable_acl(true);
    return 0;
  }

  int upload(uint64_t dest_addr, const void *src, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
#ifdef VX_CFG_VM_ENABLE
    // dest_addr is a VA; translate before touching backing RAM.
    dest_addr = vm_mgr_->page_table_walk(dest_addr);
#endif
    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, dest_addr, size);
    ram_.enable_acl(true);

    /*
    DBGPRINT("upload %ld bytes to 0x%lx\n", size, dest_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + i));
    }*/

    return 0;
  }

  int download(void *dest, uint64_t src_addr, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    // flush GPU caches before reading back results
    {
      uint32_t dummy;
      for (uint32_t cid = 0; cid < VX_CFG_NUM_CORES * VX_CFG_NUM_CLUSTERS; ++cid) {
        this->dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
      }
    }
#ifdef VX_CFG_VM_ENABLE
    // src_addr is a VA; translate before reading from backing RAM.
    src_addr = vm_mgr_->page_table_walk(src_addr);
#endif
    ram_.enable_acl(false);
    ram_.read((uint8_t *)dest, src_addr, size);
    ram_.enable_acl(true);

    /*DBGPRINT("download %ld bytes from 0x%lx\n", size, src_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + i));
    }*/

    return 0;
  }

  int start() {
    // DCRs already written by stub; just trigger execution
    future_ = std::async(std::launch::async, [&] { processor_.run(); });
    return 0;
  }

  int ready_wait(uint64_t timeout) {
    if (!future_.valid())
      return 0;
    uint64_t timeout_sec = timeout / 1000;
    std::chrono::seconds wait_time(1);
    for (;;) {
      // wait for 1 sec and check status
      auto status = future_.wait_for(wait_time);
      if (status == std::future_status::ready)
        break;
      if (0 == timeout_sec--)
        return -1;
    }
    return 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_write(addr, value);
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t *value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_read(addr, tag, value);
  }

  // ----- CP MMIO surface -----
  // simx has no hardware CP; the regfile surface is provided by a
  // functional CommandProcessor C++ model. A bounded tick burst around
  // each MMIO transaction keeps the CP responsive without a dedicated
  // simulation thread.
  int cp_mmio_write(uint32_t off, uint32_t value) {
    cp_.mmio_write(off, value);
    for (int i = 0; i < 256 && cp_.busy(); ++i) cp_.tick();
    return 0;
  }
  int cp_mmio_read(uint32_t off, uint32_t* value) {
    for (int i = 0; i < 256 && cp_.busy(); ++i) cp_.tick();
    *value = cp_.mmio_read(off);
    return 0;
  }

private:
  vortex::CommandProcessor::Hooks make_cp_hooks() {
    vortex::CommandProcessor::Hooks h;
    h.dram_read = [this](uint64_t addr, void* dst, std::size_t bytes) {
      ram_.enable_acl(false);
      ram_.read(static_cast<uint8_t*>(dst), addr, bytes);
      ram_.enable_acl(true);
    };
    h.dram_write = [this](uint64_t addr, const void* src, std::size_t bytes) {
      ram_.enable_acl(false);
      ram_.write(static_cast<const uint8_t*>(src), addr, bytes);
      ram_.enable_acl(true);
    };
    h.vortex_dcr_write = [this](uint32_t addr, uint32_t value) {
      processor_.dcr_write(addr, value);
    };
    h.vortex_dcr_read = [this](uint32_t addr, uint32_t tag) -> uint32_t {
      // Wait for any background processor_.run() to finish so dcr_read
      // does not race the Verilator state.
      if (future_.valid()) future_.wait();
      uint32_t v = 0;
      processor_.dcr_read(addr, tag, &v);
      return v;
    };
    h.vortex_start = [this]() {
      future_ = std::async(std::launch::async, [&] { processor_.run(); });
    };
    h.vortex_busy = [this]() -> bool {
      if (!future_.valid()) return false;
      return future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    };
    return h;
  }

  RAM ram_;
  Processor processor_;
  MemoryAllocator global_mem_;
  std::future<void> future_;
  vortex::CommandProcessor cp_;
#ifdef VX_CFG_VM_ENABLE
  std::unique_ptr<RamMemIO> dev_io_;
  std::unique_ptr<VMManager> vm_mgr_;
#endif
};

#include <callbacks.inc>
