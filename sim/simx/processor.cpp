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
  : clusters_(NUM_CLUSTERS)
{
  SimPlatform::instance().initialize();

	assert(PLATFORM_MEMORY_DATA_SIZE == MEM_BLOCK_SIZE);

  // create kernel management unit (SimObject)
  kmu_ = Kmu::Create("kmu");

  // create memory simulator
  memsim_ = Memory::Create("dram", Memory::Config{
    PLATFORM_MEMORY_NUM_BANKS,
    L3_MEM_PORTS,
    MEM_BLOCK_SIZE,
    MEM_CLOCK_RATIO
  });

  char sname[100];

  // create clusters
  for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
    snprintf(sname, 100, "cluster%d", i);
    clusters_.at(i) = Cluster::Create(sname, i, this);
  }

  // create L3 cache
  l3cache_ = Cache::Create("l3cache", Cache::Config{
    !L3_ENABLED,
    log2ceil(L3_CACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // L
    log2ceil(L2_LINE_SIZE),   // W
    log2ceil(L3_NUM_WAYS),    // A
    log2ceil(L3_NUM_BANKS),   // B
    XLEN,                     // address bits
    L3_NUM_REQS,              // request size
    L3_MEM_PORTS,             // memory ports
    L3_WRITEBACK,             // write-back
    false,                    // write response
    L3_MSHR_SIZE,             // mshr size
    2,                        // pipeline latency
    L3_REPL_POLICY,           // replacement policy
    }
  );

  // connect L3 core interfaces
  for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
    for (uint32_t j = 0; j < L2_MEM_PORTS; ++j) {
      clusters_.at(i)->mem_req_out.at(j).bind(&l3cache_->core_req_in.at(i * L2_MEM_PORTS + j));
      l3cache_->core_rsp_out.at(i * L2_MEM_PORTS + j).bind(&clusters_.at(i)->mem_rsp_in.at(j));
    }
  }

  // connect L3 memory interfaces
  for (uint32_t i = 0; i < L3_MEM_PORTS; ++i) {
    l3cache_->mem_req_out.at(i).bind(&memsim_->mem_req_in.at(i));
    memsim_->mem_rsp_out.at(i).bind(&l3cache_->mem_rsp_in.at(i));
  }

  // set up memory profiling
  for (uint32_t i = 0; i < L3_MEM_PORTS; ++i) {
    memsim_->mem_req_in.at(i).tx_callback([&](const MemReq& req, uint64_t cycle){
      __unused (cycle);
      perf_mem_reads_  += !req.write;
      perf_mem_writes_ += req.write;
      perf_mem_pending_reads_ += !req.write;
    });
    memsim_->mem_rsp_out.at(i).tx_callback([&](const MemRsp&, uint64_t cycle){
      __unused (cycle);
      --perf_mem_pending_reads_;
    });
  }

#ifndef NDEBUG
  // dump device configuration
  std::cout << "CONFIGS:"
            << " num_threads=" << NUM_THREADS
            << ", num_warps=" << NUM_WARPS
            << ", num_cores=" << NUM_CORES
            << ", num_clusters=" << NUM_CLUSTERS
            << ", socket_size=" << SOCKET_SIZE
            << ", local_mem_base=0x" << std::hex << LMEM_BASE_ADDR << std::dec
            << ", num_barriers=" << NUM_BARRIERS
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

  // L1 dcaches: walk dirty lines and emit writebacks to L2 (or directly
  // to memsim when L2 is bypassed). Tick until all dcaches report done
  // *and* the inflight channels have drained.
  for (auto& cluster : clusters_) {
    cluster->dcache_flush_begin();
  }
  while (true) {
    bool all_done = true;
    for (auto& cluster : clusters_) {
      if (!cluster->dcache_flush_done()) { all_done = false; break; }
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
}

int ProcessorImpl::dcr_write(uint32_t addr, uint32_t value) {
  // KMU DCRs are stored in the processor-level KMU and not broadcast to cores
  if (addr >= VX_DCR_KMU_STATE_BEGIN && addr < VX_DCR_KMU_STATE_END) {
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

int Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Processor::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}
