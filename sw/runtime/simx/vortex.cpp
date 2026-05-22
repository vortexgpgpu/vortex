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

// ============================================================================
// simx backend — a pure transport HAL (see callbacks.h). It exposes only:
//   * device lifecycle      — init() / ~vx_device()
//   * CP register channel   — cp_reg_read / cp_reg_write
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// Device-memory allocation and caps decoding live in the common core; the
// Command Processor is the sole memory engine. simx has unified memory, so
// host memory is a plain process allocation: the software CommandProcessor's
// dram hooks dereference it directly (the sim runs in-process).
// ============================================================================

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

#include <chrono>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <stdint.h>
#include <stdlib.h>

using namespace vortex;

#ifdef VX_CFG_VM_ENABLE
// DeviceMemIO adapter over the simx RAM backing store.
class RamMemIO : public DeviceMemIO {
public:
  explicit RamMemIO(RAM* ram) : ram_(ram) {}
  void read(void* dst, uint64_t addr, size_t size) override {
    ram_->read((uint8_t*)dst, addr, size);
  }
  void write(const void* src, uint64_t addr, size_t size) override {
    ram_->write((const uint8_t*)src, addr, size);
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
        cp_(make_cp_hooks()) {
    ram_.enable_acl(false);   // the common-core allocator owns the address map
    processor_.attach_ram(&ram_);
  }

  ~vx_device() {
    if (future_.valid()) {
      future_.wait();
    }
    for (auto& kv : host_regions_)
      free(reinterpret_cast<void*>(kv.first));
    host_regions_.clear();
  }

  int init() {
#ifdef VX_CFG_VM_ENABLE
    // Boot-time VM init: allocate the page table inside RAM, push SATP
    // into the simulator.
    dev_io_ = std::make_unique<RamMemIO>(&ram_);
    vm_mgr_ = std::make_unique<VMManager>(dev_io_.get());
    CHECK_ERR(vm_mgr_->init(), { return err; });
#endif
    return 0;
  }

  // ----- CP register channel -----
  // simx has no hardware CP; the regfile surface is provided by a
  // functional CommandProcessor C++ model. A bounded tick burst around
  // each MMIO transaction keeps the CP responsive without a dedicated
  // simulation thread.
  int cp_reg_write(uint32_t off, uint32_t value) {
    cp_.mmio_write(off, value);
    for (int i = 0; i < 256 && cp_.busy(); ++i) cp_.tick();
    return 0;
  }
  int cp_reg_read(uint32_t off, uint32_t* value) {
    for (int i = 0; i < 256 && cp_.busy(); ++i) cp_.tick();
    *value = cp_.mmio_read(off);
    return 0;
  }

  // ----- CP-visible host memory (command ring + DMA staging) -----
  // Unified memory: a plain process allocation. cp_addr is the pointer
  // value itself; the CP model's dram hooks route by registered region.
  int host_mem_alloc(uint64_t size, void** host_ptr, uint64_t* cp_addr) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    void* ptr = aligned_alloc(CACHE_BLOCK_SIZE, asize);
    if (ptr == nullptr)
      return -1;
    std::lock_guard<std::mutex> g(host_mu_);
    host_regions_[reinterpret_cast<uint64_t>(ptr)] = asize;
    *host_ptr = ptr;
    *cp_addr  = reinterpret_cast<uint64_t>(ptr);
    return 0;
  }

  int host_mem_free(uint64_t cp_addr) {
    {
      std::lock_guard<std::mutex> g(host_mu_);
      auto it = host_regions_.find(cp_addr);
      if (it == host_regions_.end())
        return -1;
      host_regions_.erase(it);
    }
    free(reinterpret_cast<void*>(cp_addr));
    return 0;
  }

private:
  // If `addr` falls in a registered host region, return it as a host
  // pointer (cp_addr == the pointer); otherwise nullptr → device memory.
  void* host_region_ptr(uint64_t addr) {
    std::lock_guard<std::mutex> g(host_mu_);
    if (host_regions_.empty())
      return nullptr;
    auto it = host_regions_.upper_bound(addr);
    if (it == host_regions_.begin())
      return nullptr;
    --it;
    if (addr >= it->first && addr < it->first + it->second)
      return reinterpret_cast<void*>(addr);
    return nullptr;
  }

  vortex::CommandProcessor::Hooks make_cp_hooks() {
    vortex::CommandProcessor::Hooks h;
    h.dram_read = [this](uint64_t addr, void* dst, std::size_t bytes) {
      if (void* hp = host_region_ptr(addr)) {
        std::memcpy(dst, hp, bytes);
        return;
      }
      ram_.read(static_cast<uint8_t*>(dst), addr, bytes);
    };
    h.dram_write = [this](uint64_t addr, const void* src, std::size_t bytes) {
      if (void* hp = host_region_ptr(addr)) {
        std::memcpy(hp, src, bytes);
        return;
      }
      ram_.write(static_cast<const uint8_t*>(src), addr, bytes);
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
  std::future<void> future_;
  vortex::CommandProcessor cp_;
  std::mutex host_mu_;
  std::map<uint64_t, uint64_t> host_regions_;   // base -> size
#ifdef VX_CFG_VM_ENABLE
  std::unique_ptr<RamMemIO> dev_io_;
  std::unique_ptr<VMManager> vm_mgr_;
#endif
};

#include <callbacks.inc>
