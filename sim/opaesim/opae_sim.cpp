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

#include "opae_sim.h"

#include <verilated.h>
#include "Vvortex_afu_shim.h"
#include "Vvortex_afu_shim__Syms.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <mem.h>

#include <dram_sim.h>

#include <VX_config.h>
#include <vortex_afu.h>

#include <future>
#include <list>
#include <queue>
#include <unordered_map>
#include <util.h>

#ifndef MEMORY_BANKS
  #ifdef PLATFORM_PARAM_LOCAL_MEMORY_BANKS
    #define MEMORY_BANKS PLATFORM_PARAM_LOCAL_MEMORY_BANKS
  #else
    #define MEMORY_BANKS 2
  #endif
#endif

#ifndef MEM_CLOCK_RATIO
#define MEM_CLOCK_RATIO 1
#endif

#undef MEM_BLOCK_SIZE
#define MEM_BLOCK_SIZE (PLATFORM_PARAM_LOCAL_MEMORY_DATA_WIDTH / 8)

#define CACHE_BLOCK_SIZE  64

#define CCI_LATENCY  8
#define CCI_RAND_MOD 8
#define CCI_RQ_SIZE 16
#define CCI_WQ_SIZE 16

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

#ifndef VERILATOR_RESET_VALUE
#define VERILATOR_RESET_VALUE 2
#endif

#define RAM_PAGE_SIZE 4096

#define CPU_GPU_LATENCY 200

using namespace vortex;

static uint64_t timestamp = 0;

double sc_time_stamp() {
  return timestamp;
}

static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time = TRACE_STOP_TIME;

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

///////////////////////////////////////////////////////////////////////////////

class opae_sim::Impl {
public:
  Impl()
  : device_(nullptr)
  , ram_(nullptr)
  , dram_sim_(MEM_CLOCK_RATIO)
  , stop_(false)
  , host_buffer_ids_(0)
#ifdef VCD_OUTPUT
  , trace_(nullptr)
#endif
  {}

  ~Impl() {
    stop_ = true;
    if (future_.valid()) {
      future_.wait();
    }
    for (auto& buffer : host_buffers_) {
      aligned_free(buffer.second.data);
    }
  #ifdef VCD_OUTPUT
    if (trace_) {
      trace_->close();
      delete trace_;
    }
  #endif
    if (device_) {
      delete device_;
    }
    if (ram_) {
      delete ram_;
    }
  }

  int init() {
    // create RTL module instance
    device_ = new Vvortex_afu_shim();
    
  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    trace_ = new VerilatedVcdC();
    device_->trace(trace_, 99);
    trace_->open("trace.vcd");
  #endif

    // force random values for unitialized signals
    Verilated::randReset(VERILATOR_RESET_VALUE);
    Verilated::randSeed(50);

    // turn off assertion before reset
    Verilated::assertOn(false);

    ram_ = new RAM(0, RAM_PAGE_SIZE);

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

    // launch execution thread
    future_ = std::async(std::launch::async, [&]{
        while (!stop_) {
            std::lock_guard<std::mutex> guard(mutex_);
            this->tick();
        }
    });

    return 0;
  }

  int prepare_buffer(uint64_t len, void **buf_addr, uint64_t *wsid, int flags) {
    auto alloc = aligned_malloc(len, CACHE_BLOCK_SIZE);
    if (alloc == NULL)
      return -1;
    // set uninitialized data to "baadf00d"
    for (uint32_t i = 0; i < len; ++i) {
        ((uint8_t*)alloc)[i] = (0xbaadf00d >> ((i & 0x3) * 8)) & 0xff;
    }
    host_buffer_t buffer;
    buffer.data   = (uint64_t*)alloc;
    buffer.size   = len;
    buffer.ioaddr = uintptr_t(alloc);
    auto buffer_id = host_buffer_ids_++;
    host_buffers_.emplace(buffer_id, buffer);
    *buf_addr = alloc;
    *wsid = buffer_id;
    return 0;
  }

  void release_buffer(uint64_t wsid) {
    auto it = host_buffers_.find(wsid);
    if (it != host_buffers_.end()) {
      aligned_free(it->second.data);
      host_buffers_.erase(it);
    }
  }

  void get_io_address(uint64_t wsid, uint64_t *ioaddr) {
    *ioaddr = host_buffers_[wsid].ioaddr;
  }

  void read_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t *value) {
    std::lock_guard<std::mutex> guard(mutex_);

    // simulate CPU-GPU latency
    for (uint32_t i = 0; i < CPU_GPU_LATENCY; ++i)
      this->tick();

    // simulate mmio request
    device_->vcp2af_sRxPort_c0_mmioRdValid = 1;
    device_->vcp2af_sRxPort_c0_ReqMmioHdr_address = offset / 4;
    device_->vcp2af_sRxPort_c0_ReqMmioHdr_length = 1;
    device_->vcp2af_sRxPort_c0_ReqMmioHdr_tid = 0;
    this->tick();
    device_->vcp2af_sRxPort_c0_mmioRdValid = 0;
    assert(device_->af2cp_sTxPort_c2_mmioRdValid);
    *value = device_->af2cp_sTxPort_c2_data;
  }

  void write_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t value) {
    std::lock_guard<std::mutex> guard(mutex_);

    // simulate CPU-GPU latency
    for (uint32_t i = 0; i < CPU_GPU_LATENCY; ++i)
      this->tick();

    // simulate mmio request
    device_->vcp2af_sRxPort_c0_mmioWrValid = 1;
    device_->vcp2af_sRxPort_c0_ReqMmioHdr_address = offset / 4;
    device_->vcp2af_sRxPort_c0_ReqMmioHdr_length = 1;
    device_->vcp2af_sRxPort_c0_ReqMmioHdr_tid = 0;
    memcpy(device_->vcp2af_sRxPort_c0_data, &value, 8);
    this->tick();
    device_->vcp2af_sRxPort_c0_mmioWrValid = 0;
  }

private:

  void reset() {
    this->cci_bus_reset();
    this->avs_bus_reset();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queue_, empty);
    }

    device_->reset = 1;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }

    device_->reset = 0;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }

    // Turn on assertion after reset
    Verilated::assertOn(true);
  }

  void tick() {
    this->cci_bus_eval();
    this->avs_bus_eval();

    if (!dram_queue_.empty()) {
      auto mem_req = dram_queue_.front();
      if (dram_sim_.send_request(mem_req->write, mem_req->addr, mem_req->bank_id, [](void* arg) {
        auto orig_req = reinterpret_cast<mem_req_t*>(arg);
        if (orig_req->ready) {
          delete orig_req;
        } else {
          orig_req->ready = true;
        }
      }, mem_req)) {
        dram_queue_.pop();
      }
    }

    device_->clk = 0;
    this->eval();
    device_->clk = 1;
    this->eval();

    dram_sim_.tick();

  #ifndef NDEBUG
    fflush(stdout);
  #endif
  }

  void eval() {
    device_->eval();
  #ifdef VCD_OUTPUT
    if (sim_trace_enabled()) {
      trace_->dump(timestamp);
    }
  #endif
    ++timestamp;
  }

  void cci_bus_reset() {
    cci_reads_.clear();
    cci_writes_.clear();
    device_->vcp2af_sRxPort_c0_mmioRdValid = 0;
    device_->vcp2af_sRxPort_c0_mmioWrValid = 0;
    device_->vcp2af_sRxPort_c0_rspValid = 0;
    device_->vcp2af_sRxPort_c1_rspValid = 0;
    device_->vcp2af_sRxPort_c0_TxAlmFull = 0;
    device_->vcp2af_sRxPort_c1_TxAlmFull = 0;
  }

  void cci_bus_eval() {
    this->sRxPort_bus_eval();
    this->sTxPort_bus_eval();
  }

  void sRxPort_bus_eval() {
    // check mmio request
    bool mmio_req_enabled = device_->vcp2af_sRxPort_c0_mmioRdValid
                        || device_->vcp2af_sRxPort_c0_mmioWrValid;

    // schedule CCI read responses
    std::list<cci_rd_req_t>::iterator cci_rd_it(cci_reads_.end());
    for (auto it = cci_reads_.begin(), ie = cci_reads_.end(); it != ie; ++it) {
      if (it->cycles_left > 0)
        it->cycles_left -= 1;
      if ((cci_rd_it == ie) && (it->cycles_left == 0)) {
        cci_rd_it = it;
      }
    }

    // schedule CCI write responses
    std::list<cci_wr_req_t>::iterator cci_wr_it(cci_writes_.end());
    for (auto it = cci_writes_.begin(), ie = cci_writes_.end(); it != ie; ++it) {
      if (it->cycles_left > 0)
        it->cycles_left -= 1;
      if ((cci_wr_it == ie) && (it->cycles_left == 0)) {
        cci_wr_it = it;
      }
    }

    // send CCI write response
    device_->vcp2af_sRxPort_c1_rspValid = 0;
    if (cci_wr_it != cci_writes_.end()) {
      device_->vcp2af_sRxPort_c1_rspValid = 1;
      device_->vcp2af_sRxPort_c1_hdr_resp_type = 0;
      device_->vcp2af_sRxPort_c1_hdr_mdata = cci_wr_it->mdata;
      cci_writes_.erase(cci_wr_it);
    }

    // send CCI read response (ensure mmio disabled)
    device_->vcp2af_sRxPort_c0_rspValid = 0;
    if (!mmio_req_enabled
    && (cci_rd_it != cci_reads_.end())) {
      device_->vcp2af_sRxPort_c0_rspValid = 1;
      device_->vcp2af_sRxPort_c0_hdr_resp_type = 0;
      memcpy(device_->vcp2af_sRxPort_c0_data, cci_rd_it->data.data(), CACHE_BLOCK_SIZE);
      device_->vcp2af_sRxPort_c0_hdr_mdata = cci_rd_it->mdata;
      /*printf("%0ld: [sim] CCI Rd Rsp: addr=%ld, mdata=%d, data=", timestamp, cci_rd_it->addr, cci_rd_it->mdata);
      for (int i = 0; i < CACHE_BLOCK_SIZE; ++i)
        printf("%02x", cci_rd_it->data[CACHE_BLOCK_SIZE-1-i]);
      printf("\n");*/
      cci_reads_.erase(cci_rd_it);
    }
  }

  void sTxPort_bus_eval() {
    // process read requests
    if (device_->af2cp_sTxPort_c0_valid) {
      assert(!device_->vcp2af_sRxPort_c0_TxAlmFull);
      cci_rd_req_t cci_req;
      cci_req.cycles_left = CCI_LATENCY + (timestamp % CCI_RAND_MOD);
      cci_req.addr = device_->af2cp_sTxPort_c0_hdr_address;
      cci_req.mdata = device_->af2cp_sTxPort_c0_hdr_mdata;
      auto host_ptr = (uint64_t*)(device_->af2cp_sTxPort_c0_hdr_address * CACHE_BLOCK_SIZE);
      memcpy(cci_req.data.data(), host_ptr, CACHE_BLOCK_SIZE);
      //printf("%0ld: [sim] CCI Rd Req: addr=%ld, mdata=%d\n", timestamp, device_->af2cp_sTxPort_c0_hdr_address, cci_req.mdata);
      cci_reads_.emplace_back(cci_req);
    }

    // process write requests
    if (device_->af2cp_sTxPort_c1_valid) {
      assert(!device_->vcp2af_sRxPort_c1_TxAlmFull);
      cci_wr_req_t cci_req;
      cci_req.cycles_left = CCI_LATENCY + (timestamp % CCI_RAND_MOD);
      cci_req.mdata = device_->af2cp_sTxPort_c1_hdr_mdata;
      auto host_ptr = (uint64_t*)(device_->af2cp_sTxPort_c1_hdr_address * CACHE_BLOCK_SIZE);
      memcpy(host_ptr, device_->af2cp_sTxPort_c1_data, CACHE_BLOCK_SIZE);
      cci_writes_.emplace_back(cci_req);
    }

    // check queues overflow
    device_->vcp2af_sRxPort_c0_TxAlmFull = (cci_reads_.size() >= (CCI_RQ_SIZE-1));
    device_->vcp2af_sRxPort_c1_TxAlmFull = (cci_writes_.size() >= (CCI_WQ_SIZE-1));
  }

  void avs_bus_reset() {
    for (int b = 0; b < MEMORY_BANKS; ++b) {
      pending_mem_reqs_[b].clear();
      device_->avs_readdatavalid[b] = 0;
      device_->avs_waitrequest[b] = 0;
    }
  }

  void avs_bus_eval() {
    for (int b = 0; b < MEMORY_BANKS; ++b) {
      // process memory responses
      device_->avs_readdatavalid[b] = 0;
      if (!pending_mem_reqs_[b].empty()
       && (*pending_mem_reqs_[b].begin())->ready) {
        auto mem_rd_it = pending_mem_reqs_[b].begin();
        auto mem_req = *mem_rd_it;
        device_->avs_readdatavalid[b] = 1;
        memcpy(device_->avs_readdata[b], mem_req->data.data(), MEM_BLOCK_SIZE);
        uint32_t addr = mem_req->addr;
        pending_mem_reqs_[b].erase(mem_rd_it);
        delete mem_req;
      }

      // process memory requests
      assert(!device_->avs_read[b] || !device_->avs_write[b]);
      unsigned byte_addr = (device_->avs_address[b] * MEMORY_BANKS + b) * MEM_BLOCK_SIZE;
      if (device_->avs_write[b]) {
        uint64_t byteen = device_->avs_byteenable[b];
        uint8_t* data = (uint8_t*)(device_->avs_writedata[b].data());
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          if ((byteen >> i) & 0x1) {
            (*ram_)[byte_addr + i] = data[i];
          }
        }

        /*printf("%0ld: [sim] MEM Wr Req: bank=%d, addr=%x, data=", timestamp, b, byte_addr);
        for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
          printf("%02x", data[(MEM_BLOCK_SIZE-1)-i]);
        }
        printf("\n");*/

        // send dram request
        auto mem_req = new mem_req_t();
        mem_req->addr  = device_->avs_address[b];
        mem_req->bank_id = b;
        mem_req->write = true;
        mem_req->ready = true;

        dram_queue_.push(mem_req);
      } else
      if (device_->avs_read[b]) {
        auto mem_req = new mem_req_t();
        mem_req->addr = device_->avs_address[b];
        mem_req->bank_id = b;
        ram_->read(mem_req->data.data(), byte_addr, MEM_BLOCK_SIZE);
        mem_req->write = false;
        mem_req->ready = false;
        pending_mem_reqs_[b].emplace_back(mem_req);

        /*printf("%0ld: [sim] MEM Rd Req: bank=%d, addr=%x, pending={", timestamp, b, mem_req.addr * MEM_BLOCK_SIZE);
        for (auto& req : pending_mem_reqs_[b]) {
          if (req.cycles_left != 0)
            printf(" !%0x", req.addr * MEM_BLOCK_SIZE);
          else
            printf(" %0x", req.addr * MEM_BLOCK_SIZE);
        }
        printf("}\n");*/

        // send dram request
        dram_queue_.push(mem_req);
      }

      device_->avs_waitrequest[b] = false;
    }
  }

  typedef struct {
    std::array<uint8_t, MEM_BLOCK_SIZE> data;
    uint32_t addr;
    uint32_t bank_id;
    bool write;
    bool ready;
  } mem_req_t;

  typedef struct {
    int cycles_left;
    std::array<uint8_t, CACHE_BLOCK_SIZE> data;
    uint64_t addr;
    uint32_t mdata;
  } cci_rd_req_t;

  typedef struct {
    int cycles_left;
    uint32_t mdata;
  } cci_wr_req_t;

  typedef struct {
    uint64_t* data;
    size_t    size;
    uint64_t  ioaddr;
  } host_buffer_t;

  Vvortex_afu_shim *device_;
  RAM* ram_;
  DramSim dram_sim_;

  std::future<void> future_;
  bool stop_;

  std::unordered_map<int64_t, host_buffer_t> host_buffers_;
  int64_t host_buffer_ids_;

  std::list<mem_req_t*> pending_mem_reqs_[MEMORY_BANKS];

  std::list<cci_rd_req_t> cci_reads_;
  std::list<cci_wr_req_t> cci_writes_;

  std::mutex mutex_;

  std::queue<mem_req_t*> dram_queue_;

#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

opae_sim::opae_sim()
  : impl_(new Impl())
{}

opae_sim::~opae_sim() {
  delete impl_;
}

int opae_sim::init() {
  return impl_->init();
}

int opae_sim::prepare_buffer(uint64_t len, void **buf_addr, uint64_t *wsid, int flags) {
  return impl_->prepare_buffer(len, buf_addr, wsid, flags);
}

void opae_sim::release_buffer(uint64_t wsid) {
  impl_->release_buffer(wsid);
}

void opae_sim::get_io_address(uint64_t wsid, uint64_t *ioaddr) {
  impl_->get_io_address(wsid, ioaddr);
}

void opae_sim::write_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t value) {
  impl_->write_mmio64(mmio_num, offset, value);
}

void opae_sim::read_mmio64(uint32_t mmio_num, uint64_t offset, uint64_t *value) {
  impl_->read_mmio64(mmio_num, offset, value);
}
