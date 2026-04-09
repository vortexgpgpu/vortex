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

#ifdef SAIF_OUTPUT
#include <verilated_saif_c.h>
#endif

#if defined(VCD_OUTPUT) && defined(SAIF_OUTPUT)
#error "VCD_OUTPUT and SAIF_OUTPUT cannot both be defined"
#endif


#include <cstdlib>
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

static uint32_t g_mem_bank_addr_width = (PLATFORM_MEMORY_ADDR_WIDTH - log2ceil(PLATFORM_MEMORY_NUM_BANKS));

static uint64_t timestamp = 0;

double sc_time_stamp() {
  return timestamp;
}

///////////////////////////////////////////////////////////////////////////////

static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return false;
}

///////////////////////////////////////////////////////////////////////////////

class Processor::Impl {
public:
  Impl() : dram_sim_(PLATFORM_MEMORY_NUM_BANKS, PLATFORM_MEMORY_DATA_SIZE, MEM_CLOCK_RATIO) {
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
    const char* vcd_file = std::getenv("VCD_FILE");
    tfp_->open(vcd_file ? vcd_file : "trace.vcd");
  #endif

  #ifdef SAIF_OUTPUT
    Verilated::traceEverOn(true);
    sfp_ = new VerilatedSaifC();
    device_->trace(sfp_, 99);
    const char* saif_file = std::getenv("SAIF_FILE");
    sfp_->open(saif_file ? saif_file : "trace.saif");
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
  #ifdef SAIF_OUTPUT
    sfp_->close();
    delete sfp_;
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
    // pulse start for one cycle
    device_->start = 1;
    this->tick();
    device_->start = 0;

    // wait for device to go busy
    while (!device_->busy) {
      this->tick();
    }

    // wait for device to go idle
    while (device_->busy) {
      this->tick();
    }

    this->cout_flush();
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    device_->dcr_req_valid = 1;
    device_->dcr_req_rw    = 1;
    device_->dcr_req_addr  = addr;
    device_->dcr_req_data  = value;
    this->tick();
    device_->dcr_req_valid = 0;
    this->tick();
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
    device_->dcr_req_valid = 1;
    device_->dcr_req_rw    = 0;
    device_->dcr_req_addr  = addr;
    device_->dcr_req_data  = tag;
    this->tick();
    device_->dcr_req_valid = 0;
    this->tick();
    // READ response is returned when dcr_rsp_valid is high
    while (!device_->dcr_rsp_valid) {
      this->tick();
    }
    *value = device_->dcr_rsp_data;
    return 0;
  }

private:

  void reset() {
    this->mem_bus_reset();
    this->dcr_bus_reset();

    print_bufs_.clear();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    for (int b = 0; b < PLATFORM_MEMORY_NUM_BANKS; ++b) {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queue_[b], empty);
    }

    device_->start = 0;
    device_->reset = 1;

    // Hold reset high until all internal pipeline state are initialized.
    // This mimics the behavior of the reset fanout buffering.
    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }

    device_->reset = 0;

    // Pump clocks after reset drops to allow internal pipeline states to settle.
    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }

    for (int b = 0; b < PLATFORM_MEMORY_NUM_BANKS; ++b) {
      device_->mem_req_ready[b] = 1;
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

    for (int b = 0; b < PLATFORM_MEMORY_NUM_BANKS; ++b) {
      if (!dram_queue_[b].empty()) {
        auto mem_req = dram_queue_[b].front();
        dram_sim_.send_request(mem_req->addr, mem_req->write, [](void* arg)->bool {
          // mark completed request as ready
          auto orig_req = reinterpret_cast<mem_req_t*>(arg);
          orig_req->ready = true;
          return true;
        }, mem_req);
        dram_queue_[b].pop();
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
  #ifdef SAIF_OUTPUT
    if (sim_trace_enabled()) {
      sfp_->dump(timestamp);
    }
  #endif
    ++timestamp;
  }

  void mem_bus_reset() {
    for (int b = 0; b < PLATFORM_MEMORY_NUM_BANKS; ++b) {
      device_->mem_req_ready[b] = 0;
      device_->mem_rsp_valid[b] = 0;
    }
  }

  void mem_bus_eval(bool clk) {
    if (!clk) {
      for (int b = 0; b < PLATFORM_MEMORY_NUM_BANKS; ++b) {
        mem_rd_rsp_ready_[b] = device_->mem_rsp_ready[b];
      }
      return;
    }

    for (int b = 0; b < PLATFORM_MEMORY_NUM_BANKS; ++b) {
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
      #if PLATFORM_MEMORY_INTERLEAVE == 1
        uint64_t byte_addr = (uint64_t(device_->mem_req_addr[b]) * PLATFORM_MEMORY_NUM_BANKS + b) * PLATFORM_MEMORY_DATA_SIZE;
      #else
        uint64_t byte_addr = (uint64_t(device_->mem_req_addr[b]) + (b << g_mem_bank_addr_width)) * PLATFORM_MEMORY_DATA_SIZE;
      #endif
        // check read/write
        if (device_->mem_req_rw[b]) {
          auto byteen = device_->mem_req_byteen[b];
          auto data = VDataCast<uint8_t*, PLATFORM_MEMORY_DATA_SIZE>::get(device_->mem_req_data[b]);
          // check if console output address
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
            // process memory writes
            /*printf("%0ld: [sim] MEM Wr Req[%d]: addr=0x%0lx, tag=0x%0lx, byteen=0x", timestamp, b, byte_addr, device_->mem_req_tag[b]);
            for (int i = (PLATFORM_MEMORY_DATA_SIZE/4)-1; i >= 0; --i) {
              printf("%x", (int)((byteen >> (4 * i)) & 0xf));
            }
            printf(", data=0x");
            for (int i = PLATFORM_MEMORY_DATA_SIZE-1; i >= 0; --i) {
              printf("%02x", data[i]);
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
          // process memory reads
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
    device_->dcr_req_valid = 0;
    device_->dcr_req_rw    = 0;
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

  std::list<mem_req_t*> pending_mem_reqs_[PLATFORM_MEMORY_NUM_BANKS];

  std::queue<mem_req_t*> dram_queue_[PLATFORM_MEMORY_NUM_BANKS];

  std::array<bool, PLATFORM_MEMORY_NUM_BANKS> mem_rd_rsp_ready_;

  DramSim dram_sim_;

  Vrtlsim_shim* device_;

  RAM* ram_;

#ifdef VCD_OUTPUT
  VerilatedVcdC *tfp_;
#endif
#ifdef SAIF_OUTPUT
  VerilatedSaifC *sfp_;
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

int Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Processor::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}