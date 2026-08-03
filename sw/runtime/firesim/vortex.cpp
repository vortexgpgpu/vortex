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
// firesim backend — a pure transport HAL (see callbacks.h). It exposes only:
//   * device lifecycle      — init() / ~vx_device()
//   * CP register channel   — cp_reg_read / cp_reg_write
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// The target carries no Command Processor, so the regfile surface comes from
// the same functional CommandProcessor model rtlsim uses. The difference is
// where its dram hooks land: device memory is behind the simulator's LoadMem
// widget rather than in this process, so those accesses go through the
// transport instead of being dereferenced.
// ============================================================================

#include <VX_types.h>
#include <common.h>

#include <cmd_processor.h>
#include <firesim_sim.h>
#include <util.h>

#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <map>
#include <mutex>

using namespace vortex;

class vx_device {
public:
  vx_device()
    : cp_(make_cp_hooks())
  {}

  ~vx_device() {
    for (auto& kv : host_regions_)
      free(reinterpret_cast<void*>(kv.first));
    host_regions_.clear();
  }

  int init() {
    // The bitstream is selected by the environment so that the same runtime
    // serves a metasimulation and an FPGA without a rebuild.
    const char* bitstream = getenv("VORTEX_FIRESIM_XCLBIN");
    return sim_.init(bitstream != nullptr ? bitstream : "");
  }

  // ----- CP register channel -----
  // No hardware CP on this target; the regfile surface is provided by the
  // functional model. A bounded tick burst around each MMIO transaction keeps
  // it responsive without a dedicated thread.
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
  // Host memory stays a plain process allocation: it is the host's, and the CP
  // model reads it directly. Only device memory crosses the transport.
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
      sim_.mem_read(addr, bytes, dst);
    };
    h.dram_write = [this](uint64_t addr, const void* src, std::size_t bytes) {
      if (void* hp = host_region_ptr(addr)) {
        std::memcpy(hp, src, bytes);
        return;
      }
      sim_.mem_write(addr, bytes, src);
    };
    h.vortex_dcr_write = [this](uint32_t addr, uint32_t value) {
      sim_.dcr_write(addr, value);
    };
    h.vortex_dcr_read = [this](uint32_t addr, uint32_t tag) -> uint32_t {
      uint32_t v = 0;
      sim_.dcr_read(addr, tag, &v);
      return v;
    };
    h.vortex_start = [this]() {
      sim_.start();
    };
    h.vortex_busy = [this]() -> bool {
      bool busy = false;
      if (sim_.is_busy(&busy) != 0)
        return false;
      return busy;
    };
#ifdef VX_CFG_VM_ENABLE
    // The device MMU control surface answers the fault-report DCRs.
    h.mmu_fault_report = true;
#endif
    return h;
  }

  vortex::firesim_sim sim_;
  vortex::CommandProcessor cp_;
  std::mutex host_mu_;
  std::map<uint64_t, uint64_t> host_regions_;   // base -> size
};

#include <callbacks.inc>
