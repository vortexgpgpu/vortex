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
    MEMORY_BANKS,
    uint32_t(arch.num_cores()) * arch.num_clusters()
  });

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
    uint8_t(arch.num_clusters()), // request size
    true,                     // write-through
    false,                    // write response
    L3_MSHR_SIZE,             // mshr size
    2,                        // pipeline latency
    }
  );

  // connect L3 memory ports
  l3cache_->MemReqPort.bind(&memsim_->MemReqPort);
  memsim_->MemRspPort.bind(&l3cache_->MemRspPort);

  // create clusters
  for (uint32_t i = 0; i < arch.num_clusters(); ++i) {
    clusters_.at(i) = Cluster::Create(i, this, arch, dcrs_);
    // connect L3 core ports
    clusters_.at(i)->mem_req_port.bind(&l3cache_->CoreReqPorts.at(i));
    l3cache_->CoreRspPorts.at(i).bind(&clusters_.at(i)->mem_rsp_port);
  }

  // set up memory profiling
  memsim_->MemReqPort.tx_callback([&](const MemReq& req, uint64_t cycle){
    __unused (cycle);
    perf_mem_reads_   += !req.write;
    perf_mem_writes_  += req.write;
    perf_mem_pending_reads_ += !req.write;
  });
  memsim_->MemRspPort.tx_callback([&](const MemRsp&, uint64_t cycle){
    __unused (cycle);
    --perf_mem_pending_reads_;
  });

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

void ProcessorImpl::run() {
  SimPlatform::instance().reset();
  this->reset();

  bool done;
  do {
    SimPlatform::instance().tick();
    done = true;
    for (auto cluster : clusters_) {
      if (cluster->running()) {
        done = false;
        continue;
      }
    }
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);
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
  return perf;
}

///////////////////////////////////////////////////////////////////////////////

Processor::Processor(const Arch& arch)
  : impl_(new ProcessorImpl(arch))
{}

Processor::~Processor() {
  delete impl_;
}

void Processor::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

void Processor::run() {
  impl_->run();
}

void Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}