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

#include "processor.h"
#include "processor_impl.h"
#include <VX_types.h>

#include <cstdlib>
#include <execinfo.h>

using namespace vortex;

static void simx_print_backtrace() {
  void* addrs[64];
  int count = ::backtrace(addrs, int(std::size(addrs)));
  char** symbols = ::backtrace_symbols(addrs, count);
  if (symbols == nullptr)
    return;
  std::cerr << "Backtrace (" << count << " frames):" << std::endl;
  for (int i = 0; i < count; ++i) {
    std::cerr << "  " << symbols[i] << std::endl;
  }
  std::free(symbols);
}

ProcessorImpl::ProcessorImpl()
  : clusters_(VX_CFG_NUM_CLUSTERS)
{
  SimPlatform::instance().initialize();

	assert(VX_CFG_PLATFORM_MEMORY_DATA_SIZE == VX_CFG_MEM_BLOCK_SIZE);

  // create kernel management unit (SimObject)
  kmu_ = Kmu::Create("kmu");

  // create memory simulator
  memsim_ = Memory::Create("dram", Memory::Config{
    VX_CFG_PLATFORM_MEMORY_NUM_BANKS,
    VX_CFG_L3_MEM_PORTS,
    VX_CFG_MEM_BLOCK_SIZE,
    MEM_CLOCK_RATIO
  });

  char sname[100];

  // create clusters
  for (uint32_t i = 0; i < VX_CFG_NUM_CLUSTERS; ++i) {
    snprintf(sname, 100, "cluster%d", i);
    clusters_.at(i) = Cluster::Create(sname, i, this);
  }

  // create L3 cache
  // §3.1.3: when L3 is enabled it is the LLC; otherwise the L3 instance
  // is a transparent bypass arbiter and the L2 (or L1) is the LLC.
  l3cache_ = Cache::Create("l3cache", Cache::Config{
    !VX_CFG_L3_ENABLED,
    log2ceil(VX_CFG_L3_CACHE_SIZE),  // C
    log2ceil(VX_CFG_MEM_BLOCK_SIZE), // L
    log2ceil(VX_CFG_L2_LINE_SIZE),   // W
    log2ceil(VX_CFG_L3_NUM_WAYS),    // A
    log2ceil(VX_CFG_L3_NUM_BANKS),   // B
    VX_CFG_XLEN,                     // address bits
    VX_CFG_L3_NUM_REQS,              // request size
    VX_CFG_L3_MEM_PORTS,             // memory ports
    VX_CFG_L3_WRITEBACK,             // write-back
    false,                    // write response
    VX_CFG_L3_MSHR_SIZE,             // mshr size
    2,                        // pipeline latency
    VX_CFG_L3_REPL_POLICY,           // replacement policy
    VX_CFG_L3_ENABLED != 0,          // is_llc when L3 is the LLC
    }
  );

#if VX_CFG_EXT_A_ENABLED
  // §3.1.5 build-time invariant: every cache strictly above the LLC
  // must be write-through. A write-back intermediate could absorb a
  // store from hart B without the LLC seeing it; a later SC from hart
  // A on the same line would spuriously succeed (RVA spec violation:
  // RVA permits spurious failure, not spurious success).
#if VX_CFG_L3_ENABLED
  static_assert(!VX_CFG_DCACHE_WRITEBACK, "AMO requires write-through L1 (VX_CFG_DCACHE_WRITEBACK=0) when L3 is the LLC");
  static_assert(!VX_CFG_L2_WRITEBACK,     "AMO requires write-through L2 (VX_CFG_L2_WRITEBACK=0) when L3 is the LLC");
#elif VX_CFG_L2_ENABLED
  static_assert(!VX_CFG_DCACHE_WRITEBACK, "AMO requires write-through L1 (VX_CFG_DCACHE_WRITEBACK=0) when L2 is the LLC");
  // L1 is unconstrained when L1 itself is the LLC.
#endif

  // Non-LLC AMO passthrough (proposal §3.8) is implemented in the
  // bank pipeline: AmoProbe entries probe-and-invalidate the local
  // line then forward via mem_req_out tagged with
  // AMO_PASSTHRU_TAG_FLAG so the response routes back to core_rsp_out
  // without installing a fill. Multi-level (L1+L2 / L1+L2+L3) builds
  // exercise this path; L1-only builds keep the dcache as the LLC and
  // never enter it.
#endif

  // connect L3 core interfaces
  for (uint32_t i = 0; i < VX_CFG_NUM_CLUSTERS; ++i) {
    for (uint32_t j = 0; j < VX_CFG_L2_MEM_PORTS; ++j) {
      clusters_.at(i)->mem_req_out.at(j).bind(&l3cache_->core_req_in.at(i * VX_CFG_L2_MEM_PORTS + j));
      l3cache_->core_rsp_out.at(i * VX_CFG_L2_MEM_PORTS + j).bind(&clusters_.at(i)->mem_rsp_in.at(j));
    }
  }

  // connect L3 memory interfaces
  for (uint32_t i = 0; i < VX_CFG_L3_MEM_PORTS; ++i) {
    l3cache_->mem_req_out.at(i).bind(&memsim_->mem_req_in.at(i));
    memsim_->mem_rsp_out.at(i).bind(&l3cache_->mem_rsp_in.at(i));
  }

  // set up memory profiling
  for (uint32_t i = 0; i < VX_CFG_L3_MEM_PORTS; ++i) {
    memsim_->mem_req_in.at(i).tx_callback([&](const MemReq& req, uint64_t cycle){
      __unused (cycle);
      perf_mem_reads_  += !req.is_write();
      perf_mem_writes_ += req.is_write();
      perf_mem_pending_reads_ += !req.is_write();
    });
    memsim_->mem_rsp_out.at(i).tx_callback([&](const MemRsp&, uint64_t cycle){
      __unused (cycle);
      --perf_mem_pending_reads_;
    });
  }

#ifndef NDEBUG
  // dump device configuration
  std::cout << "CONFIGS:"
            << " num_threads=" << VX_CFG_NUM_THREADS
            << ", num_warps=" << VX_CFG_NUM_WARPS
            << ", num_cores=" << VX_CFG_NUM_CORES
            << ", num_clusters=" << VX_CFG_NUM_CLUSTERS
            << ", socket_size=" << VX_CFG_SOCKET_SIZE
            << ", local_mem_base=0x" << std::hex << VX_MEM_LMEM_BASE_ADDR << std::dec
            << ", num_barriers=" << VX_CFG_NUM_BARRIERS
            << std::endl;
#endif
  // reset the device
  this->reset();
}

ProcessorImpl::~ProcessorImpl() {
  SimPlatform::instance().finalize();
}

void ProcessorImpl::attach_ram(RAM* ram) {
  memsim_->attach_ram(ram);
}

void ProcessorImpl::flush_caches() {
  // Cache hierarchy is drained inside-out so each level only walks
  // after the level above has emitted all its dirty writebacks and the
  // channels carrying them have settled.
  //
  // Per-level fanout mirrors the RTL VX_dcr_flush wiring in
  // [VX_core.sv:170](hw/rtl/core/VX_core.sv#L170) and
  // [VX_graphics.sv:182](hw/rtl/VX_graphics.sv#L182): one shared `req`
  // fires icache + dcache + {tcache, rcache, ocache} simultaneously,
  // and `done` AND-reduces across every instance before the host
  // releases the request. We model the same parallelism by issuing
  // every L1 `flush_begin()` up-front and ticking until *all* surfaces
  // report `flush_done()` together.

  // L1 surfaces: dcache + icache + graphics caches. Write-through
  // surfaces early-exit in Cache::flush_begin() (the no-op is harmless
  // and keeps the per-surface code path warm for future write-back
  // configs).
  for (auto& cluster : clusters_) {
    cluster->dcache_flush_begin();
    cluster->icache_flush_begin();
#ifdef VX_CFG_EXT_TEX_ENABLE
    cluster->tcache_flush_begin();
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
    cluster->rcache_flush_begin();
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
    cluster->ocache_flush_begin();
#endif
  }
  while (true) {
    bool all_done = true;
    for (auto& cluster : clusters_) {
      if (!cluster->dcache_flush_done()) { all_done = false; break; }
      if (!cluster->icache_flush_done()) { all_done = false; break; }
#ifdef VX_CFG_EXT_TEX_ENABLE
      if (!cluster->tcache_flush_done()) { all_done = false; break; }
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
      if (!cluster->rcache_flush_done()) { all_done = false; break; }
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
      if (!cluster->ocache_flush_done()) { all_done = false; break; }
#endif
    }
    if (all_done && SimChannelBase::inflight_count() == 0)
      break;
    SimPlatform::instance().tick();
  }

  // L2 caches.
  for (auto& cluster : clusters_) {
    cluster->l2_flush_begin();
  }
  while (true) {
    bool all_done = true;
    for (auto& cluster : clusters_) {
      if (!cluster->l2_flush_done()) { all_done = false; break; }
    }
    if (all_done && SimChannelBase::inflight_count() == 0)
      break;
    SimPlatform::instance().tick();
  }

  // L3 cache (single instance at processor level).
  l3cache_->flush_begin();
  while (!l3cache_->flush_done() || SimChannelBase::inflight_count() != 0) {
    SimPlatform::instance().tick();
  }
}

int ProcessorImpl::run() {
  this->reset();
  kmu_->start();

  bool done;
  int exitcode = 0;
  do {
    SimPlatform::instance().tick();
    bool any_running = false;
    for (auto cluster : clusters_) {
      if (cluster->running()) {
        any_running = true;
      } else {
        exitcode |= cluster->get_exitcode();
      }
    }
    // Stop only when cores are idle AND no channel still carries an
    // undelivered packet. Cache pipelines wrap a SimChannel inside TFifo,
    // so cache-pipe state (and any in-flight cache→memory writethrough)
    // shows up in the same counter — no per-module busy reporting needed.
    done = !any_running && (SimChannelBase::inflight_count() == 0);
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);

  return exitcode;
}

void ProcessorImpl::reset() {
  SimPlatform::instance().reset();
  perf_mem_reads_ = 0;
  perf_mem_writes_ = 0;
  perf_mem_latency_ = 0;
  perf_mem_pending_reads_ = 0;
  is_cycle_initialized_ = false;
}

bool ProcessorImpl::cycle() {
  // Lazy first-call init mirrors run()'s top-of-loop sequence so the
  // external driver doesn't need to choreograph reset + kmu start
  // separately. reset() clears is_cycle_initialized_ so a back-to-back
  // kernel launch re-dispatches.
  if (!is_cycle_initialized_) {
    this->reset();
    kmu_->start();
    is_cycle_initialized_ = true;
  }
  SimPlatform::instance().tick();
  perf_mem_latency_ += perf_mem_pending_reads_;
  return this->any_running();
}

int ProcessorImpl::dcr_write(uint32_t addr, uint32_t value) {
  // KMU DCRs are stored in the processor-level KMU and not broadcast to cores.
  // The cluster_dim trio lives at 0x003-0x005 (below the main KMU block
  // at 0x010-0x01F) so the texture/raster/om DCR layouts didn't shift; they
  // still belong to the KMU and must be routed there.
  bool is_kmu_dcr = (addr >= VX_DCR_KMU_STATE_BEGIN && addr < VX_DCR_KMU_STATE_END)
                 || (addr == VX_DCR_KMU_CLUSTER_DIM_X)
                 || (addr == VX_DCR_KMU_CLUSTER_DIM_Y)
                 || (addr == VX_DCR_KMU_CLUSTER_DIM_Z);
  if (is_kmu_dcr) {
    kmu_->dcr_write(addr, value);
    return 0;
  }
  for (auto& cluster : clusters_) {
    int ret = cluster->dcr_write(addr, value);
    if (ret != 0)
      return ret;
  }
  return 0;
}

int ProcessorImpl::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  if (addr == VX_DCR_BASE_CACHE_FLUSH) {
    // Drain dirty cache lines to DRAM before the host reads back results.
    // After flush_caches() returns every dirty line has reached memsim_'s
    // backing RAM.
    this->flush_caches();
    *value = 0;
    return 0;
  }
  for (auto& cluster : clusters_) {
    int ret = cluster->dcr_read(addr, tag, value);
    if (ret != 0)
      return ret;
  }
  return 0;
}

Core* ProcessorImpl::get_first_core() const {
  if (clusters_.empty()) return nullptr;
  return clusters_.at(0)->get_core(0);
}

bool ProcessorImpl::any_running() const {
  for (auto& cluster : clusters_) {
    if (cluster->running()) return true;
  }
  return (SimChannelBase::inflight_count() != 0);
}

ProcessorImpl::PerfStats ProcessorImpl::perf_stats() const {
  ProcessorImpl::PerfStats perf;
  perf.mem_reads   = perf_mem_reads_;
  perf.mem_writes  = perf_mem_writes_;
  perf.mem_latency = perf_mem_latency_;
  perf.l3cache     = l3cache_->perf_stats();
  perf.memsim      = memsim_->perf_stats();
  return perf;
}

///////////////////////////////////////////////////////////////////////////////

Processor::Processor()
  : impl_(new ProcessorImpl())
{}

Processor::~Processor() {
  delete impl_;
}

void Processor::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

void Processor::reset() {
  impl_->reset();
}

void Processor::start_kmu() {
  impl_->kmu().start();
}

bool Processor::any_running() const {
  return impl_->any_running();
}

Core* Processor::get_first_core() const {
  return impl_->get_first_core();
}

int Processor::run() {
  try {
    return impl_->run();
  } catch (const std::exception& e) {
    std::cerr << "Error: exception: " << e.what() << std::endl;
    if (std::getenv("SIMX_BACKTRACE") != nullptr) {
      simx_print_backtrace();
    }
  } catch (...) {
    std::cerr << "Error: unknown exception." << std::endl;
    if (std::getenv("SIMX_BACKTRACE") != nullptr) {
      simx_print_backtrace();
    }
  }
  return -1;
}

bool Processor::cycle() {
  return impl_->cycle();
}

Memory* Processor::memsim() {
  return impl_->memsim();
}

int Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Processor::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}
