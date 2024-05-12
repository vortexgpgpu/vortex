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

#define RAMULATOR
#include <ramulator/src/Gem5Wrapper.h>
#include <ramulator/src/Request.h>
#include <ramulator/src/Statistics.h>

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

#ifndef MEM_CYCLE_RATIO
#define MEM_CYCLE_RATIO -1
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
  , ramulator_(nullptr)
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
    if (ramulator_) {
      ramulator_->finish();
      Stats::statlist.printall();
      delete ramulator_;
    }
  }

  int init() {
    // force random values for unitialized signals
    Verilated::randReset(VERILATOR_RESET_VALUE);
    Verilated::randSeed(50);

    // turn off assertion before reset
    Verilated::assertOn(false);

    // create RTL module instance
    device_ = new Vvortex_afu_shim();

  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    trace_ = new VerilatedVcdC();
    device_->trace(trace_, 99);
    trace_->open("trace.vcd");
  #endif

    ram_ = new RAM(0, RAM_PAGE_SIZE);

    // initialize dram simulator
    ramulator::Config ram_config;
    ram_config.add("standard", "DDR4");
    ram_config.add("channels", std::to_string(MEMORY_BANKS));
    ram_config.add("ranks", "1");
    ram_config.add("speed", "DDR4_2400R");
    ram_config.add("org", "DDR4_4Gb_x8");
    ram_config.add("mapping", "defaultmapping");
    ram_config.set_core_num(1);
    ramulator_ = new ramulator::Gem5Wrapper(ram_config, MEM_BLOCK_SIZE);
    Stats::statlist.output("ramulator.ddr4.log");

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
    //--

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
    //--

    if (!dram_queue_.empty()) {
      if (ramulator_->send(dram_queue_.front()))
        dram_queue_.pop();
    }

    device_->ap_clk = 0;
    this->eval();
    device_->ap_clk = 1;
    this->eval();

    if (MEM_CYCLE_RATIO > 0) {
      auto cycle = timestamp / 2;
      if ((cycle % MEM_CYCLE_RATIO) == 0)
        ramulator_->tick();
    } else {
      for (int i = MEM_CYCLE_RATIO; i <= 0; ++i)
        ramulator_->tick();
    }

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

  Vvortex_afu_shim *device_;
  RAM* ram_;
  ramulator::Gem5Wrapper* ramulator_;

  std::future<void> future_;
  bool stop_;

  std::mutex mutex_;

  std::queue<ramulator::Request> dram_queue_;

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