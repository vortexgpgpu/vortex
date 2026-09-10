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
#include "core.h"
#include "scheduler.h"
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
  SimPlatform::instance().set_num_workers(SIMX_NUM_WORKERS);

	assert(VX_CFG_PLATFORM_MEMORY_DATA_SIZE == VX_CFG_MEM_BLOCK_SIZE);

  // create kernel management unit (SimObject)
  constexpr uint32_t total_cores = VX_CFG_NUM_CLUSTERS * NUM_SOCKETS * VX_CFG_SOCKET_SIZE;
  kmu_ = Kmu::Create("kmu", total_cores);

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

  // Launch bus: a registered, credit-gated lane from the KMU to each core's
  // dispatcher; CTAs cross into the core's domain through the slice, and
  // admission credits return to the KMU through a slice of their own.
  constexpr uint32_t cores_per_cluster = NUM_SOCKETS * VX_CFG_SOCKET_SIZE;
  for (uint32_t i = 0; i < total_cores; ++i) {
    snprintf(sname, 100, "kmu-lane%d", i);
    auto slice = RegSlice<kmu_req_t>::Create(sname, 1);
    auto* core = clusters_.at(i / cores_per_cluster)->get_core(i % cores_per_cluster);
    kmu_->bus_out.at(i).bind(&slice->In);
    slice->Out.bind(&core->scheduler().cta_dispatcher()->bus_in);
    snprintf(sname, 100, "kmu-credit%d", i);
    RegSlice<uint8_t>::Ptr cslice;
    {
      // The return slice is owned by the sending core's partition so its
      // output is the registered crossing back into the uncore domain.
      SimPlatform::DomainScope core_scope(core);
      cslice = RegSlice<uint8_t>::Create(sname, 1);
    }
    core->scheduler().cta_dispatcher()->credit_out.bind(&cslice->In);
    cslice->Out.bind(&kmu_->credit_in.at(i));
  }

  // create L3 cache; when L3 is enabled it is the LLC, otherwise it is a
  // transparent bypass arbiter and the L2 (or L1) is the LLC.
  l3cache_ = Cache::Create("l3cache", Cache::Config{
    !VX_CFG_L3_ENABLED,
    log2ceil(VX_CFG_L3_SIZE),  // C
    log2ceil(VX_CFG_L3_LINE_SIZE),   // L
    log2ceil(VX_CFG_L3_SECTOR_SIZE), // S
    log2ceil(VX_CFG_L2_SECTOR_SIZE), // W
    log2ceil(VX_CFG_L3_NUM_WAYS),    // A
    log2ceil(VX_CFG_L3_NUM_BANKS),   // B
    VX_CFG_XLEN,                     // address bits
    VX_CFG_L3_NUM_REQS,              // request size
    VX_CFG_L3_MEM_PORTS,             // memory ports
    VX_CFG_L3_WRITEBACK,             // write-back
    false,                    // write response
    VX_CFG_L3_MSHR_SIZE,             // mshr size
    VX_CFG_L3_LATENCY,               // pipeline latency
    VX_CFG_L3_REPL_POLICY,           // replacement policy
    VX_CFG_L3_ENABLED != 0,          // is_llc when L3 is the LLC
    }
  );

#if VX_CFG_EXT_A_ENABLED
  // Build-time invariant: every cache above the LLC must be write-through.
  // A write-back intermediate could absorb a store without the LLC seeing it;
  // a later SC on the same line would spuriously succeed (RVA permits
  // spurious failure, not spurious success).
#if VX_CFG_L3_ENABLED
  static_assert(!VX_CFG_DCACHE_WRITEBACK, "AMO requires write-through L1 (VX_CFG_DCACHE_WRITEBACK=0) when L3 is the LLC");
  static_assert(!VX_CFG_L2_WRITEBACK,     "AMO requires write-through L2 (VX_CFG_L2_WRITEBACK=0) when L3 is the LLC");
#elif VX_CFG_L2_ENABLED
  static_assert(!VX_CFG_DCACHE_WRITEBACK, "AMO requires write-through L1 (VX_CFG_DCACHE_WRITEBACK=0) when L2 is the LLC");
  // L1 is unconstrained when L1 itself is the LLC.
#endif

  // Non-LLC AMO passthrough: AmoProbe entries probe-and-invalidate the local
  // line then forward via mem_req_out tagged with AMO_PASSTHRU_TAG_FLAG so
  // the response routes back to core_rsp_out without installing a fill.
  // L1-only builds keep the dcache as the LLC and never enter this path.
#endif

  // connect L3 core interfaces
  for (uint32_t i = 0; i < VX_CFG_NUM_CLUSTERS; ++i) {
    for (uint32_t j = 0; j < VX_CFG_L2_MEM_PORTS; ++j) {
      clusters_.at(i)->mem_req_out.at(j).bind(&l3cache_->core_req_in.at(i * VX_CFG_L2_MEM_PORTS + j));
      l3cache_->core_rsp_out.at(i * VX_CFG_L2_MEM_PORTS + j).bind(&clusters_.at(i)->mem_rsp_in.at(j));
    }
  }

#ifdef VX_CFG_VM_ENABLE
  // Device-level walker: every cluster's L2-TLB miss link folds through the
  // mux into one shared Ptw whose PTE fetches ride the LLC's last input slot.
  dev_ptw_ = Ptw::Create("dev-ptw");
  dev_ptw_mux_ = PtwMux::Create("dev-ptwmux", VX_CFG_NUM_CLUSTERS);
  for (uint32_t i = 0; i < VX_CFG_NUM_CLUSTERS; ++i) {
    clusters_.at(i)->ptw_req_out().bind(&dev_ptw_mux_->ReqIn.at(i));
    dev_ptw_mux_->RspOut.at(i).bind(&clusters_.at(i)->ptw_rsp_in());
  }
  dev_ptw_mux_->ReqOut.bind(&dev_ptw_->ReqIn);
  dev_ptw_->RspOut.bind(&dev_ptw_mux_->RspIn);
  dev_ptw_->MemReqOut.bind(&l3cache_->core_req_in.at(VX_CFG_L3_PTW_IDX));
  l3cache_->core_rsp_out.at(VX_CFG_L3_PTW_IDX).bind(&dev_ptw_->MemRspIn);
#endif

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
  ram_ = ram;
  memsim_->attach_ram(ram);
}

void ProcessorImpl::flush_caches() {
  // Cache hierarchy is drained inside-out: issue all L1 flush_begin() calls
  // up-front so icache, dcache, and graphics caches flush in parallel, then
  // tick until all surfaces report flush_done().

  // L1 surfaces: dcache + icache + graphics caches.
  // Write-through surfaces early-exit in Cache::flush_begin().
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
#ifdef VX_CFG_EXT_RTU_ENABLE
    cluster->rtcache_flush_begin();
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
#ifdef VX_CFG_EXT_RTU_ENABLE
      if (!cluster->rtcache_flush_done()) { all_done = false; break; }
#endif
    }
    if (all_done && SimPlatform::instance().idle())
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
    if (all_done && SimPlatform::instance().idle())
      break;
    SimPlatform::instance().tick();
  }

  // L3 cache (single instance at processor level).
  l3cache_->flush_begin();
  while (!l3cache_->flush_done() || !SimPlatform::instance().idle()) {
    SimPlatform::instance().tick();
  }
}

int ProcessorImpl::run() {
  this->reset();
  kmu_->start();
  this->forward_delegated_launch();

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
#ifdef VX_CFG_VM_ENABLE
    // A walk in flight holds no channel packet while it waits on memory,
    // so quiescence must ask the device walker directly.
    if (dev_ptw_->busy()) {
      any_running = true;
    }
#endif
    // A page fault kills its accesses. Most warps drain on the kill
    // responses, but one on a path that owes no response (an instruction
    // fetch) would stall forever: end the launch as soon as a fault is
    // latched and the fabric is quiet, and let the host read the report.
    bool faulted = this->mmu_fault_pending();
    // Stop only when cores are idle AND the platform holds no undelivered
    // work: cache pipelines wrap a SimChannel inside TFifo, so cache-pipe
    // state (and any in-flight cache→memory writethrough) shows up in
    // idle(), as do launch-lane CTAs and barrier hops between domains.
    done = (!any_running || faulted) && SimPlatform::instance().idle();
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);

  return exitcode;
}

bool ProcessorImpl::mmu_fault_pending() const {
#ifdef VX_CFG_VM_ENABLE
  return dev_ptw_->fault_info().valid;
#else
  return false;
#endif
}

void ProcessorImpl::forward_delegated_launch() {
#ifdef VX_CFG_EXT_RASTER_ENABLE
  if (kmu_->launch_delegated()) {
    for (auto& cluster : clusters_) {
      cluster->raster_core()->frame_kick();
    }
  }
#endif
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
    this->forward_delegated_launch();
    is_cycle_initialized_ = true;
  }
  SimPlatform::instance().tick();
  perf_mem_latency_ += perf_mem_pending_reads_;
  return this->any_running();
}

int ProcessorImpl::dcr_write(uint32_t addr, uint32_t value) {
  // KMU DCRs route to the KMU and are not broadcast to cores.
  bool is_kmu_dcr = (addr >= VX_DCR_KMU_STATE_BEGIN && addr < VX_DCR_KMU_STATE_END);
  if (is_kmu_dcr) {
    kmu_->dcr_write(addr, value);
    return 0;
  }
#ifdef VX_CFG_VM_ENABLE
  // Device SATP for the shared walker complex, assembled from two
  // 32-bit halves and fanned out to every cluster on the high write.
  if (addr == VX_DCR_MMU_SATP_LO) {
    mmu_satp_ = (mmu_satp_ & ~uint64_t(0xFFFFFFFF)) | value;
    return 0;
  }
  if (addr == VX_DCR_MMU_FAULT_INFO) {
    // Write-to-clear: the host drops the report once it has read it, so a
    // fault raised by one launch stays readable across the next one's reset.
    dev_ptw_->clear_fault();
    return 0;
  }
  if (addr == VX_DCR_MMU_SATP_HI) {
    mmu_satp_ = (mmu_satp_ & 0xFFFFFFFF) | ((uint64_t)value << 32);
    dev_ptw_->set_satp(mmu_satp_);
    for (auto& cluster : clusters_) {
      cluster->set_mmu_satp(mmu_satp_);
    }
    return 0;
  }
#endif
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
#ifdef VX_CFG_VM_ENABLE
  if (addr == VX_DCR_MMU_FAULT_VA
   || addr == VX_DCR_MMU_FAULT_VA_HI
   || addr == VX_DCR_MMU_FAULT_INFO) {
    // The device walker owns the (single) fault latch; it clears on
    // reset, so each launch starts with a clean report.
    *value = 0;
    const auto& f = dev_ptw_->fault_info();
    if (f.valid) {
      if (addr == VX_DCR_MMU_FAULT_INFO) {
        *value = VX_MMU_FAULT_VALID
               | (((uint32_t)f.access << VX_MMU_FAULT_ACCESS_SH) & VX_MMU_FAULT_ACCESS)
               | (f.amo ? VX_MMU_FAULT_AMO : 0u);
      } else if (addr == VX_DCR_MMU_FAULT_VA) {
        *value = (uint32_t)f.va;
      } else {
        *value = (uint32_t)(f.va >> 32);
      }
    }
    return 0;
  }
#endif
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
  return !SimPlatform::instance().idle();
}

ProcessorImpl::PerfStats ProcessorImpl::perf_stats() const {
  ProcessorImpl::PerfStats perf;
  perf.mem_reads   = perf_mem_reads_;
  perf.mem_writes  = perf_mem_writes_;
  perf.mem_latency = perf_mem_latency_;
  perf.l3cache     = l3cache_->perf_stats();
  perf.memsim      = memsim_->perf_stats();
#ifdef VX_CFG_VM_ENABLE
  perf.ptw         = dev_ptw_->perf_stats();
#endif
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

void Processor::set_mem_telemetry_hook(std::function<void(const MemReq&)> hook) {
  impl_->set_mem_telemetry_hook(std::move(hook));
}

int Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Processor::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}
