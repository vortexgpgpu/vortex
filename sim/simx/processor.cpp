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

using namespace vortex;

ProcessorImpl::ProcessorImpl(const Arch& arch)
  : arch_(arch)
  , clusters_(arch.num_clusters())
{
  SimPlatform::instance().initialize();

  // create memory simulator
  memsim_ = MemSim::Create("dram", MemSim::Config{
    PLATFORM_MEMORY_BANKS,
    L3_MEM_PORTS
  });

  // create clusters
  for (uint32_t i = 0; i < arch.num_clusters(); ++i) {
    clusters_.at(i) = Cluster::Create(i, this, arch, dcrs_);
  }

  // create L3 cache
  l3cache_ = CacheSim::Create("l3cache", CacheSim::Config{
    !L3_ENABLED,
    log2ceil(L3_CACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // L
    log2ceil(L2_LINE_SIZE),   // W
    log2ceil(L3_NUM_WAYS),    // A
    log2ceil(L3_NUM_BANKS),   // B
    XLEN,                     // address bits
    1,                        // number of ports
    L3_NUM_REQS,              // request size
    L3_MEM_PORTS,             // memory ports
    L3_WRITEBACK,             // write-back
    false,                    // write response
    L3_MSHR_SIZE,             // mshr size
    2,                        // pipeline latency
    }
  );

  // connect L3 core interfaces
  for (uint32_t i = 0; i < arch.num_clusters(); ++i) {
    for (uint32_t j = 0; j < L2_MEM_PORTS; ++j) {
      clusters_.at(i)->mem_req_ports.at(j).bind(&l3cache_->CoreReqPorts.at(i * L2_MEM_PORTS + j));
      l3cache_->CoreRspPorts.at(i * L2_MEM_PORTS + j).bind(&clusters_.at(i)->mem_rsp_ports.at(j));
    }
  }

  // connect L3 memory interfaces
  for (uint32_t i = 0; i < L3_MEM_PORTS; ++i) {
    l3cache_->MemReqPorts.at(i).bind(&memsim_->MemReqPorts.at(i));
    memsim_->MemRspPorts.at(i).bind(&l3cache_->MemRspPorts.at(i));
  }

  // set up memory profiling
  for (uint32_t i = 0; i < L3_MEM_PORTS; ++i) {
    memsim_->MemReqPorts.at(i).tx_callback([&](const MemReq& req, uint64_t cycle){
      __unused (cycle);
      perf_mem_reads_  += !req.write;
      perf_mem_writes_ += req.write;
      perf_mem_pending_reads_ += !req.write;
    });
    memsim_->MemRspPorts.at(i).tx_callback([&](const MemRsp&, uint64_t cycle){
      __unused (cycle);
      --perf_mem_pending_reads_;
    });
  }

#ifndef NDEBUG
  // dump device configuration
  std::cout << "CONFIGS:"
            << " num_threads=" << arch.num_threads()
            << ", num_warps=" << arch.num_warps()
            << ", num_cores=" << arch.num_cores()
            << ", num_clusters=" << arch.num_clusters()
            << ", socket_size=" << arch.socket_size()
            << ", local_mem_base=0x" << std::hex << arch.local_mem_base() << std::dec
            << ", num_barriers=" << arch.num_barriers()
            << std::endl;
#endif
  // reset the device
  this->reset();
}

ProcessorImpl::~ProcessorImpl() {
  SimPlatform::instance().finalize();
}

void ProcessorImpl::attach_ram(RAM* ram) {
  for (auto cluster : clusters_) {
    cluster->attach_ram(ram);
  }
}
#ifdef VM_ENABLE
void ProcessorImpl::set_satp(uint64_t satp) {
  for (auto cluster : clusters_) {
    cluster->set_satp(satp);
  }
}
#endif

int ProcessorImpl::run() {
  SimPlatform::instance().reset();
  this->reset();

  bool done;
  int exitcode = 0;
  do {
    SimPlatform::instance().tick();
    done = true;
    for (auto cluster : clusters_) {
      if (cluster->running()) {
        done = false;
        continue;
      }
      exitcode |= cluster->get_exitcode();
    }
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);

  return exitcode;
}

void ProcessorImpl::reset() {
  perf_mem_reads_ = 0;
  perf_mem_writes_ = 0;
  perf_mem_latency_ = 0;
  perf_mem_pending_reads_ = 0;
}

void ProcessorImpl::dcr_write(uint32_t addr, uint32_t value) {
  dcrs_.write(addr, value);
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

Processor::Processor(const Arch& arch)
  : impl_(new ProcessorImpl(arch))
{
#ifdef VM_ENABLE
  satp_ = NULL;
#endif
}

Processor::~Processor() {
  delete impl_;
#ifdef VM_ENABLE
  if (satp_ != NULL)
    delete satp_;
#endif
}

void Processor::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

int Processor::run() {
  return impl_->run();
}

void Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

#ifdef VM_ENABLE
int16_t Processor::set_satp_by_addr(uint64_t base_addr) {
  uint16_t asid = 0;
  satp_ = new SATP_t (base_addr,asid);
  if (satp_ == NULL)
    return 1;
  uint64_t satp = satp_->get_satp();
  impl_->set_satp(satp);
  return 0;
}
bool Processor::is_satp_unset() {
  return (satp_== NULL);
}
uint8_t Processor::get_satp_mode() {
  assert (satp_!=NULL);
  return satp_->get_mode();
}
uint64_t Processor::get_base_ppn() {
  assert (satp_!=NULL);
  return satp_->get_base_ppn();
}
#endif
