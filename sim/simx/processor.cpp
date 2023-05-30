#include "processor.h"
#include "processor_impl.h"

using namespace vortex;

ProcessorImpl::ProcessorImpl(const Arch& arch) 
  : arch_(arch)
  , clusters_(NUM_CLUSTERS)
{
  SimPlatform::instance().initialize();

  uint32_t cores_per_cluster = arch.num_cores() / NUM_CLUSTERS;

  // create memory simulator
  memsim_ = MemSim::Create("dram", MemSim::Config{
    MEMORY_BANKS,
    arch.num_cores()
  });

  // create L3 cache
  l3cache_ = CacheSim::Create("l3cache", CacheSim::Config{
    !L3_ENABLED,
    log2ceil(L3_CACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // B
    log2ceil(L3_NUM_WAYS),  // W
    0,                      // A
    XLEN,                   // address bits  
    L3_NUM_BANKS,           // number of banks
    L3_NUM_PORTS,           // number of ports
    NUM_CLUSTERS,           // request size 
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    L3_MSHR_SIZE,           // mshr
    2,                      // pipeline latency
    }
  );        
  
  // connect L3 memory ports
  l3cache_->MemReqPort.bind(&memsim_->MemReqPort);
  memsim_->MemRspPort.bind(&l3cache_->MemRspPort);

  // create clusters
  for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
    clusters_.at(i) = Cluster::Create(i, cores_per_cluster, this, arch, dcrs_);
    // connect L3 core ports
    clusters_.at(i)->mem_req_port.bind(&l3cache_->CoreReqPorts.at(i));
    l3cache_->CoreRspPorts.at(i).bind(&clusters_.at(i)->mem_rsp_port);
  }

  // set up memory perf recording
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

int ProcessorImpl::run() {
  SimPlatform::instance().reset();
  this->reset();
  
  bool done;
  Word exitcode = 0;
  do {
    SimPlatform::instance().tick();
    done = true;
    for (auto cluster : clusters_) {
      if (cluster->running()) {
        Word ec;   
        if (cluster->check_exit(&ec, 3)) {
          exitcode |= ec;
        } else {
          done = false;
        }
      }
    }
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);

  return exitcode;
}
 
void ProcessorImpl::reset() {
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
  perf_mem_reads_ = 0;
  perf_mem_writes_ = 0;
  perf_mem_latency_ = 0;
  perf_mem_pending_reads_ = 0;
}

void ProcessorImpl::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto& barrier = barriers_.at(bar_id);
  barrier.set(core_id);
  DP(3, "*** Suspend core #" << core_id << " at barrier #" << bar_id);

  if (barrier.count() == (size_t)count) {
      // resume suspended cores
      uint32_t cores_per_cluster = arch_.num_cores() / NUM_CLUSTERS;
      for (uint32_t i = 0; i < arch_.num_cores(); ++i) {
        if (barrier.test(i)) {
          DP(3, "*** Resume core #" << i << " at barrier #" << bar_id);
          uint32_t core_idx = i % cores_per_cluster;
          uint32_t cluster_idx = i / cores_per_cluster;
          clusters_.at(cluster_idx)->core(core_idx)->resume();
        }
      }
      barrier.reset();
    }
}

void ProcessorImpl::write_dcr(uint32_t addr, uint32_t value) {
  dcrs_.write(addr, value);
}

ProcessorImpl::PerfStats ProcessorImpl::perf_stats() const {
  ProcessorImpl::PerfStats perf;
  perf.mem_reads   = perf_mem_reads_;
  perf.mem_writes  = perf_mem_writes_;
  perf.mem_latency = perf_mem_latency_;
  perf.l3cache     = l3cache_->perf_stats();
  for (auto cluster : clusters_) {
    perf.clusters += cluster->perf_stats();
  }   
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

int Processor::run() {
  return impl_->run();
}

void Processor::write_dcr(uint32_t addr, uint32_t value) {
  return impl_->write_dcr(addr, value);
}