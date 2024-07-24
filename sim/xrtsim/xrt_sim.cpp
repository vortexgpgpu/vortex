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

#include "xrt_sim.h"

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

class xrt_sim::Impl {
public:
  Impl()
  : device_(nullptr)
  , ram_(nullptr)
  , dram_sim_(MEM_CLOCK_RATIO)
  , stop_(false)
#ifdef VCD_OUTPUT
  , trace_(nullptr)
#endif
  {}

  ~Impl() {
    stop_ = true;
    if (future_.valid()) {
      future_.wait();
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

private:

  void reset() {
    this->axi_ctrl_bus_reset();
    this->axi_mem_bus_reset();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queue_, empty);
    }

    device_->ap_rst_n = 0;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->ap_clk = 0;
      this->eval();
      device_->ap_clk = 1;
      this->eval();
    }

    device_->ap_rst_n = 1;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->ap_clk = 0;
      this->eval();
      device_->ap_clk = 1;
      this->eval();
    }

    // Turn on assertion after reset
    Verilated::assertOn(true);
  }

  void tick() {
    this->axi_ctrl_bus_eval();
    this->axi_mem_bus_eval();

    if (!dram_queue_.empty()) {
      auto mem_req = dram_queue_.front();
      if (dram_sim_.send_request(mem_req->write, mem_req->addr, 0, [](void* arg) {
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

    device_->ap_clk = 0;
    this->eval();
    device_->ap_clk = 1;
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

  void axi_ctrl_bus_reset() {
    // address write request
    device_->s_axi_ctrl_awvalid = 0;
    //device_->s_axi_ctrl_awaddr = 0;

    // data write request
    device_->s_axi_ctrl_wvalid = 0;
    //device_->s_axi_ctrl_wdata = 0;
    //device_->s_axi_ctrl_wstrb = 0;

    // address read request
    device_->s_axi_ctrl_arvalid = 0;
    //device_->s_axi_ctrl_araddr = 0;

    // data read response
    device_->s_axi_ctrl_rready = 0;

    // data write response
    device_->s_axi_ctrl_bready = 0;
  }

  void axi_ctrl_bus_eval() {
    //--
  }

  void axi_mem_bus_reset() {
    // address write request
    device_->m_axi_mem_0_awready = 0;

    // data write request
    device_->m_axi_mem_0_wready = 0;

    // address read request
    device_->m_axi_mem_0_arready = 0;

    // data read response
    device_->m_axi_mem_0_rvalid = 0;
    //device_->m_axi_mem_0_rdata = 0;
    //device_->m_axi_mem_0_rlast = 0;
    //device_->m_axi_mem_0_rid = 0;
    //device_->m_axi_mem_0_rresp = 0;

    // data write response
    device_->m_axi_mem_0_bvalid = 0;
    //device_->m_axi_mem_0_bresp = 0;
    //device_->m_axi_mem_0_bid = 0;
  }

  void axi_mem_bus_eval() {
    //--
  }

  typedef struct {
    std::array<uint8_t, MEM_BLOCK_SIZE> data;
    uint32_t addr;
    bool write;
    bool ready;
  } mem_req_t;

  Vvortex_afu_shim *device_;
  RAM* ram_;
  DramSim dram_sim_;

  std::future<void> future_;
  bool stop_;

  std::mutex mutex_;

  std::list<mem_req_t*> pending_mem_reqs_[MEMORY_BANKS];

  std::queue<mem_req_t*> dram_queue_;

#ifdef VCD_OUTPUT
  VerilatedVcdC *trace_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

xrt_sim::xrt_sim()
  : impl_(new Impl())
{}

xrt_sim::~xrt_sim() {
  delete impl_;
}

int xrt_sim::init() {
  return impl_->init();
}