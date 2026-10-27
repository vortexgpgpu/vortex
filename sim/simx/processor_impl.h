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

#include "memory.h"
#include "cache.h"
#include "constants.h"
#include "cluster.h"
#include "kmu.h"
#include "gbar_unit.h"
#include "cp_llc_port.h"

namespace vortex {

class ProcessorImpl {
public:
  struct PerfStats {
    Cache::PerfStats l3cache;
    Memory::PerfStats memsim;
#ifdef VX_CFG_VM_ENABLE
    Ptw::PerfStats ptw;
#endif
    uint64_t mem_reads = 0;
    uint64_t mem_writes = 0;
    uint64_t mem_latency = 0;
  };

  ProcessorImpl();
  ~ProcessorImpl();

  void attach_ram(RAM* mem);

  void reset();

  int run();

  // Single-cycle step; see Processor::cycle() doc. Lazily initializes
  // (resets + starts KMU) on the first call after construction or
  // after reset() has been invoked.
  bool cycle();

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  PerfStats perf_stats() const;

  Kmu& kmu()       { return *kmu_; }

  void set_mem_telemetry_hook(Memory::PreSendHook hook) {
    memsim_->set_pre_send_hook(std::move(hook));
  }

  // Functional backing store (device physical memory). Exposed so the raster
  // early-Z stage can read the committed depth buffer synchronously during its
  // walk (a peek, not a substitute for the cycle-modeled ocache path).
  RAM* ram() const { return ram_; }

  bool any_running() const;

  class Core* get_first_core() const;

  // Drain dirty data from caches (write-back path) all the way to DRAM.
  // Walks L1 dcaches → L2 → L3, ticking the simulator between phases so
  // each level's evictions reach the next before that level itself flushes.
  void flush_caches(bool l1_only);

  // Device-memory access through the command processor's port into the
  // last-level cache (CP_ON_L2 / CP_ON_L3). A write returns once every
  // request entered the port, a read once its data returned.
  void cp_mem_write(uint64_t addr, const void* src, uint64_t size, bool noalloc);
  void cp_mem_read(uint64_t addr, void* dst, uint64_t size);

private:

  // A grid-less launch is a delegated draw launch: the KMU walks no CTAs and
  // the frame kick is forwarded to every cluster's raster engine instead.
  void forward_delegated_launch();

  // Queue a transfer on the CP port and tick until it has entered, or for a
  // read returned; the last-level cache keeps what it took into the next launch.
  void cp_mem_transfer(uint64_t addr, const uint8_t* src, uint8_t* dst, uint64_t size, bool noalloc);

  // Reset the model for a launch. After DMA writes it waits for the port's
  // fence and keeps the memory side, whose line fills may still be in flight.
  void launch_reset();

  void llc_keep_lines();

  // True once the device walker has latched a page fault.
  bool mmu_fault_pending() const;

  Kmu::Ptr    kmu_;
  std::vector<Cluster::Ptr> clusters_;
  GbarUnit::Ptr gbar_unit_;     // device-level global barrier (multi-cluster)
  Memory::Ptr memsim_;
#ifdef VX_CFG_VM_ENABLE
  uint64_t    mmu_satp_ = 0;    // assembled from the two DCR halves
  // Device-level walker: one Ptw serves every cluster's L2 TLB, its PTE
  // fetches riding the LLC.
  Ptw::Ptr    dev_ptw_;
  PtwMux::Ptr dev_ptw_mux_;
#endif
  RAM*        ram_ = nullptr;   // functional backing store (set by attach_ram)
  Cache::Ptr l3cache_;
  CpLlcPort::Ptr cp_llc_port_;  // CP DMA client of the last-level cache
  bool platform_reset_ = false;
  uint64_t perf_mem_reads_ = 0;
  uint64_t perf_mem_writes_ = 0;
  uint64_t perf_mem_latency_ = 0;
  uint64_t perf_mem_pending_reads_ = 0;
  // Tracks whether cycle() has done its first-call init (reset +
  // kmu_->start()). reset() clears it so a back-to-back kernel launch
  // via cycle() re-dispatches the KMU.
  bool is_cycle_initialized_;
  bool flush_l1_only_ = false;  // kind of the flush the last start write opened
};

}
