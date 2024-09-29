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

#include "VVortex.h"

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <mem.h>

#include <VX_config.h>
#include <ostream>
#include <list>
#include <queue>
#include <vector>
#include <sstream>
#include <unordered_map>

#include <dram_sim.h>
#include <util.h>

#ifndef MEM_CLOCK_RATIO
#define MEM_CLOCK_RATIO 1
#endif

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

#ifndef VERILATOR_RESET_VALUE
#define VERILATOR_RESET_VALUE 2
#endif

#if (XLEN == 32)
typedef uint32_t Word;
#elif (XLEN == 64)
typedef uint64_t Word;
#else
#error unsupported XLEN
#endif

#define VL_WDATA_GETW(lwp, i, n, w) \
  VL_SEL_IWII(0, n * w, 0, 0, lwp, i * w, w)

using namespace vortex;

static uint64_t timestamp = 0;

double sc_time_stamp() {
  return timestamp;
}

///////////////////////////////////////////////////////////////////////////////

static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

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

class Processor::Impl {
public:
  Impl() : dram_sim_(MEM_CLOCK_RATIO) {
    // force random values for uninitialized signals
    Verilated::randReset(VERILATOR_RESET_VALUE);
    Verilated::randSeed(50);

    // turn off assertion before reset
    Verilated::assertOn(false);

    // create RTL module instance
    device_ = new VVortex();

  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    tfp_ = new VerilatedVcdC();
    device_->trace(tfp_, 99);
    tfp_->open("trace.vcd");
  #endif

    ram_ = nullptr;

    // reset the device
    this->reset();

    // Turn on assertion after reset
    Verilated::assertOn(true);
  }

  ~Impl() {
    this->cout_flush();

  #ifdef VCD_OUTPUT
    tfp_->close();
    delete tfp_;
  #endif

    delete device_;
  }

  void cout_flush() {
    for (auto& buf : print_bufs_) {
      auto str = buf.second.str();
      if (!str.empty()) {
        std::cout << "#" << buf.first << ": " << str << std::endl;
      }
    }
  }

  void attach_ram(RAM* ram) {
    ram_ = ram;
  }

  void run() {
  #ifndef NDEBUG
    std::cout << std::dec << timestamp << ": [sim] run()" << std::endl;
  #endif

    // reset device
    this->reset();

    // start
    device_->reset = 0;

    // wait on device to go busy
    while (!device_->busy) {
      this->tick();
    }

    // wait on device to go idle
    while (device_->busy) {
      this->tick();
    }

    // stop
    device_->reset = 1;

    this->cout_flush();
  }

  void dcr_write(uint32_t addr, uint32_t value) {
    device_->dcr_wr_valid = 1;
    device_->dcr_wr_addr  = addr;
    device_->dcr_wr_data  = value;
    this->tick();
    device_->dcr_wr_valid = 0;
  }

private:

  void reset() {
    this->mem_bus_reset();
    this->dcr_bus_reset();

    print_bufs_.clear();

    pending_mem_reqs_.clear();

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

    device_->mem_req_ready = 1;
  }

  void tick() {
    this->mem_bus_eval();

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

    dram_sim_.tick();

    device_->clk = 0;
    this->eval();
    device_->clk = 1;
    this->eval();

  #ifndef NDEBUG
    fflush(stdout);
  #endif
  }

  void eval() {
    device_->eval();
  #ifdef VCD_OUTPUT
    if (sim_trace_enabled()) {
      tfp_->dump(timestamp);
    } else {
      exit(-1);
    }
  #endif
    ++timestamp;
  }

  void mem_bus_reset() {
    device_->mem_req_ready = 0;
    device_->mem_rsp_valid = 0;
  }

  void mem_bus_eval() {
    // process memory read responses
    if (device_->mem_rsp_valid && device_->mem_rsp_ready) {
      device_->mem_rsp_valid = 0;
    }
    if (!device_->mem_rsp_valid) {
      if (!pending_mem_reqs_.empty()
       && (*pending_mem_reqs_.begin())->ready) {
        auto mem_rsp_it = pending_mem_reqs_.begin();
        auto mem_rsp = *mem_rsp_it;
        /*printf("%0ld: [sim] MEM Rd Rsp: tag=0x%0lx, addr=0x%0lx, data=0x", timestamp, mem_rsp->tag, mem_rsp->addr);
        for (int i = MEM_BLOCK_SIZE-1; i >= 0; --i) {
          printf("%02x", mem_rsp->data[i]);
        }
        printf("\n");
        */
        device_->mem_rsp_valid = 1;
        memcpy(VDataCast<void*, MEM_BLOCK_SIZE>::get(device_->mem_rsp_data), mem_rsp->data.data(), MEM_BLOCK_SIZE);
        device_->mem_rsp_tag = mem_rsp->tag;
        pending_mem_reqs_.erase(mem_rsp_it);
        delete mem_rsp;
      }
    }

    // process memory requests
    if (device_->mem_req_valid && device_->mem_req_ready) {
      uint64_t byte_addr = (device_->mem_req_addr * MEM_BLOCK_SIZE);
      if (device_->mem_req_rw) {
        auto byteen = device_->mem_req_byteen;
        auto data = VDataCast<uint8_t*, MEM_BLOCK_SIZE>::get(device_->mem_req_data);
        if (byte_addr >= uint64_t(IO_COUT_ADDR)
         && byte_addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {
          // process console output
          for (int i = 0; i < IO_COUT_SIZE; i++) {
            if ((byteen >> i) & 0x1) {
              auto& ss_buf = print_bufs_[i];
              char c = data[i];
              ss_buf << c;
              if (c == '\n') {
                std::cout << std::dec << "#" << i << ": " << ss_buf.str() << std::flush;
                ss_buf.str("");
              }
            }
          }
        } else {
          // process writes
          /*
          printf("%0ld: [sim] MEM Wr Req: tag=0x%0lx, addr=0x%0lx, byteen=0x", timestamp, device_->mem_req_tag, byte_addr);
          for (int i = (MEM_BLOCK_SIZE/4)-1; i >= 0; --i) {
            printf("%x", (int)((byteen >> (4 * i)) & 0xf));
          }
          printf(", data=0x");
          for (int i = MEM_BLOCK_SIZE-1; i >= 0; --i) {
            printf("%d=%02x,", i, data[i]);
          }
          printf("\n");
          */
          for (int i = 0; i < MEM_BLOCK_SIZE; i++) {
            if ((byteen >> i) & 0x1) {
              (*ram_)[byte_addr + i] = data[i];
            }
          }

          auto mem_req = new mem_req_t();
          mem_req->tag   = device_->mem_req_tag;
          mem_req->addr  = byte_addr;
          mem_req->write = true;
          mem_req->ready = true;

          // send dram request
          dram_queue_.push(mem_req);
        }
      } else {
        // process reads
        auto mem_req = new mem_req_t();
        mem_req->tag   = device_->mem_req_tag;
        mem_req->addr  = byte_addr;
        mem_req->write = false;
        mem_req->ready = false;
        ram_->read(mem_req->data.data(), byte_addr, MEM_BLOCK_SIZE);
        pending_mem_reqs_.emplace_back(mem_req);

        //printf("%0ld: [sim] MEM Rd Req: addr=0x%0lx, tag=0x%0lx\n", timestamp, byte_addr, device_->mem_req_tag);

        // send dram request
        dram_queue_.push(mem_req);
      }
    }
  }

  void dcr_bus_reset() {
    device_->dcr_wr_valid = 0;
  }

  void wait(uint32_t cycles) {
    for (int i = 0; i < cycles; ++i) {
      this->tick();
    }
  }

private:

  typedef struct {
    VVortex* device;
    std::array<uint8_t, MEM_BLOCK_SIZE> data;
    uint64_t addr;
    uint64_t tag;
    bool write;
    bool ready;
  } mem_req_t;

  std::unordered_map<int, std::stringstream> print_bufs_;

  std::list<mem_req_t*> pending_mem_reqs_;

  std::queue<mem_req_t*> dram_queue_;

  DramSim dram_sim_;

  VVortex* device_;

#ifdef VCD_OUTPUT
  VerilatedVcdC *tfp_;
#endif

  RAM* ram_;
};

///////////////////////////////////////////////////////////////////////////////

Processor::Processor()
  : impl_(new Impl())
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