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

#include "Vrtlsim_shim.h"

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

#define PLATFORM_MEMORY_DATA_SIZE (PLATFORM_MEMORY_DATA_WIDTH/8)

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
    device_ = new Vrtlsim_shim();

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
    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      device_->mem_req_ready[b] = 1;
    }

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
    this->tick();
  }

private:

  void reset() {
    this->mem_bus_reset();
    this->dcr_bus_reset();

    print_bufs_.clear();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queue_[b], empty);
    }

    device_->reset = 1;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }
  }

  void tick() {

    device_->clk = 0;
    this->eval();

    this->mem_bus_eval(0);

    device_->clk = 1;
    this->eval();

    this->mem_bus_eval(1);

    dram_sim_.tick();

    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      if (!dram_queue_[b].empty()) {
        auto mem_req = dram_queue_[b].front();
        if (dram_sim_.send_request(mem_req->write, mem_req->addr, b, [](void* arg) {
          // mark completed request as ready
          auto orig_req = reinterpret_cast<mem_req_t*>(arg);
          orig_req->ready = true;
        }, mem_req)) {
          // was successfully sent to dram, remove from queue
          dram_queue_[b].pop();
        }
      }
    }

  #ifndef NDEBUG
    fflush(stdout);
  #endif
  }

  void eval() {
    device_->eval();
  #ifdef VCD_OUTPUT
    if (sim_trace_enabled()) {
      tfp_->dump(timestamp);
    }
  #endif
    ++timestamp;
  }

  void mem_bus_reset() {
    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      device_->mem_req_ready[b] = 0;
      device_->mem_rsp_valid[b] = 0;
    }
  }

  void mem_bus_eval(bool clk) {
    if (!clk) {
      for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
        mem_rd_rsp_ready_[b] = device_->mem_rsp_ready[b];
      }
      return;
    }

    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      // process memory responses
      if (device_->mem_rsp_valid[b] && mem_rd_rsp_ready_[b]) {
        device_->mem_rsp_valid[b] = 0;
      }
      if (device_->mem_rsp_valid[b] == 0) {
        if (!pending_mem_reqs_[b].empty()) {
          auto mem_rsp_it = pending_mem_reqs_[b].begin();
          auto mem_rsp = *mem_rsp_it;
          if (mem_rsp->ready) {
            if (!mem_rsp->write) {
              // return read responses
              device_->mem_rsp_valid[b] = 1;
              memcpy(VDataCast<void*, PLATFORM_MEMORY_DATA_SIZE>::get(device_->mem_rsp_data[b]), mem_rsp->data.data(), PLATFORM_MEMORY_DATA_SIZE);
              device_->mem_rsp_tag[b] = mem_rsp->tag;
            }
            // delete the request
            pending_mem_reqs_[b].erase(mem_rsp_it);
            delete mem_rsp;
          }
        }
      }

      // process memory requests
      if (device_->mem_req_valid[b] && device_->mem_req_ready[b]) {
        uint64_t byte_addr = (device_->mem_req_addr[b] * PLATFORM_MEMORY_DATA_SIZE);
        if (device_->mem_req_rw[b]) {
          auto byteen = device_->mem_req_byteen[b];
          auto data = VDataCast<uint8_t*, PLATFORM_MEMORY_DATA_SIZE>::get(device_->mem_req_data[b]);
          // check address range
          if (byte_addr >= uint64_t(IO_COUT_ADDR)
           && byte_addr < (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE)) {
            // process console output
            for (int i = 0; i < PLATFORM_MEMORY_DATA_SIZE; i++) {
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
            /*printf("%0ld: [sim] MEM Wr Req[%d]: addr=0x%0lx, tag=0x%0lx, byteen=0x", timestamp, b, byte_addr, device_->mem_req_tag[b]);
            for (int i = (PLATFORM_MEMORY_DATA_SIZE/4)-1; i >= 0; --i) {
              printf("%x", (int)((byteen >> (4 * i)) & 0xf));
            }
            printf(", data=0x");
            for (int i = PLATFORM_MEMORY_DATA_SIZE-1; i >= 0; --i) {
              printf("%d=%02x,", i, data[i]);
            }
            printf("\n");*/
            for (int i = 0; i < PLATFORM_MEMORY_DATA_SIZE; i++) {
              if ((byteen >> i) & 0x1) {
                (*ram_)[byte_addr + i] = data[i];
              }
            }
            auto mem_req = new mem_req_t();
            mem_req->tag   = device_->mem_req_tag[b];
            mem_req->addr  = byte_addr;
            mem_req->write = true;
            mem_req->ready = false;

            // enqueue dram request
            dram_queue_[b].push(mem_req);

            // add to pending list
            pending_mem_reqs_[b].emplace_back(mem_req);
          }
        } else {
          // process reads
          auto mem_req = new mem_req_t();
          mem_req->tag   = device_->mem_req_tag[b];
          mem_req->addr  = byte_addr;
          mem_req->write = false;
          mem_req->ready = false;
          ram_->read(mem_req->data.data(), byte_addr, PLATFORM_MEMORY_DATA_SIZE);

          /*printf("%0ld: [sim] MEM Rd Req[%d]: addr=0x%0lx, tag=0x%0lx, data=0x", timestamp, b, byte_addr, device_->mem_req_tag[b]);
          for (int i = PLATFORM_MEMORY_DATA_SIZE-1; i >= 0; --i) {
            printf("%02x", mem_req->data[i]);
          }
          printf("\n");*/

          // enqueue dram request
          dram_queue_[b].push(mem_req);

          // add to pending list
          pending_mem_reqs_[b].emplace_back(mem_req);
        }
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
    Vrtlsim_shim* device;
    std::array<uint8_t, PLATFORM_MEMORY_DATA_SIZE> data;
    uint64_t addr;
    uint64_t tag;
    bool write;
    bool ready;
  } mem_req_t;

  std::unordered_map<int, std::stringstream> print_bufs_;

  std::list<mem_req_t*> pending_mem_reqs_[PLATFORM_MEMORY_BANKS];

  std::queue<mem_req_t*> dram_queue_[PLATFORM_MEMORY_BANKS];

  std::array<bool, PLATFORM_MEMORY_BANKS> mem_rd_rsp_ready_;

  DramSim dram_sim_;

  Vrtlsim_shim* device_;

  RAM* ram_;

#ifdef VCD_OUTPUT
  VerilatedVcdC *tfp_;
#endif
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