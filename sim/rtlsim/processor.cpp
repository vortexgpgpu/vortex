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

#include <VX_types.h>
#include "processor.h"

#ifdef AXI_BUS
#include "Vrtlsim_axi_shim.h"
#include <axi_latency_pipe.h>
typedef Vrtlsim_axi_shim Device;
#else
#include "Vrtlsim_shim.h"
typedef Vrtlsim_shim Device;
#endif

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
#include <host_monitor.h>
#include <cout_drainer.h>

#include <VX_config.h>
#include <ostream>
#include <list>
#include <queue>
#include <deque>
#include <cassert>
#include <vector>
#include <sstream>
#include <unordered_map>

#include <dram_sim.h>
#include <util.h>


// AXI_BUS memory timing; the defaults are FASED's.
#ifndef AXI_MEM_READ_LATENCY
#define AXI_MEM_READ_LATENCY 30
#endif

#ifndef AXI_MEM_WRITE_LATENCY
#define AXI_MEM_WRITE_LATENCY 30
#endif

#ifndef AXI_MEM_MAX_READS
#define AXI_MEM_MAX_READS 16
#endif

#ifndef AXI_MEM_MAX_WRITES
#define AXI_MEM_MAX_WRITES 16
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

#if (VX_CFG_XLEN == 32)
typedef uint32_t Word;
#elif (VX_CFG_XLEN == 64)
typedef uint64_t Word;
#else
#error unsupported VX_CFG_XLEN
#endif

#define VL_WDATA_GETW(lwp, i, n, w) \
  VL_SEL_IWII(0, n * w, 0, 0, lwp, i * w, w)

using namespace vortex;

#ifndef AXI_BUS
static uint32_t g_mem_bank_addr_width = (VX_CFG_PLATFORM_MEMORY_ADDR_WIDTH - log2ceil(VX_CFG_PLATFORM_MEMORY_NUM_BANKS));
#endif

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
#ifdef AXI_BUS
  Impl() {
#else
  Impl() : dram_sim_(DramSim::platform_config()) {
#endif
    // restart sim time so a re-created model registers at time zero
    timestamp = 0;

    // A context of our own rather than the process-wide default. A model
    // registers a trace callback on its context at construction, and
    // VerilatedContext::addModel has no counterpart -- the registration lives as
    // long as the context does. On the shared default context those entries
    // accumulate across device opens and keep naming models that have since been
    // deleted, so tracing a second open walks the list and dereferences freed
    // memory. Per-context, the list dies with the models it names. Reset values,
    // the seed and the assertion window are per-context settings and move with it.
    context_ = new VerilatedContext();

    // force random values for uninitialized signals
    context_->randReset(VERILATOR_RESET_VALUE);
    context_->randSeed(50);

    // turn off assertion before reset
    context_->assertOn(false);

    // create device instance
    device_ = new Device(context_);

  #ifdef VCD_OUTPUT
    context_->traceEverOn(true);
    tfp_ = new VerilatedVcdC();
    device_->trace(tfp_, 99);
    const char* vcd_file = std::getenv("VCD_FILE");
    tfp_->open(vcd_file ? vcd_file : "trace.vcd");
  #endif

  #ifdef SAIF_OUTPUT
    context_->traceEverOn(true);
    sfp_ = new VerilatedSaifC();
    device_->trace(sfp_, 99);
    const char* saif_file = std::getenv("SAIF_FILE");
    sfp_->open(saif_file ? saif_file : "trace.saif");
  #endif

    ram_ = nullptr;

    // reset the device
    this->reset();

    // Turn on assertion after reset
    context_->assertOn(true);
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

    // Model before context: the model holds a reference to the context it was
    // built with, so releasing the context first would leave that reference
    // dangling through the model's own destruction.
    delete device_;
    delete context_;
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

  void run(HostMonitor* monitor, CoutDrainer* cout_drainer) {
  #ifndef NDEBUG
    std::cout << std::dec << timestamp << ": [sim] run()" << std::endl;
  #endif
    // pulse start for one cycle
    device_->start = 1;
    this->tick();
    device_->start = 0;

    // Upper bound on the post-start wait for busy (see below); a no-work frame
    // never asserts busy, so the wait must not be unbounded.
    constexpr uint32_t NO_WORK_TIMEOUT = 100000;

    // wait for device to go busy. A frame may legitimately have no work — e.g. a
    // fully-culled graphics draw whose rasterizer emits zero fragment waves — and
    // then busy never re-asserts after the start pulse. Bound the wait so such a
    // frame drains immediately instead of spinning forever. Real work asserts
    // busy within a few cycles of start, far below this window.
    for (uint32_t i = 0; !device_->busy && i < NO_WORK_TIMEOUT; ++i) {
      this->tick();
    }

    // wait for device to go idle, or for the HTIF tohost write.
    //
    // Standalone rtlsim has no host runtime, so a kernel that uses
    // vx_printf would back-pressure on a full COUT ring forever unless
    // it's drained here (cf. Device::drain_cout). Drainer is only wired
    // by sim/rtlsim/main.cpp — the runtime path leaves it null so the
    // runtime's own drainer is the sole COUT consumer.
    while (device_->busy) {
      this->tick();
      if (cout_drainer) cout_drainer->tick();
      if (monitor && monitor->enabled() && monitor->tick(*ram_))
        break;
    }
    if (cout_drainer) cout_drainer->tick(); // final flush

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

#ifndef AXI_BUS
    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queue_[b], empty);
    }
#endif

    device_->start = 0;
    device_->reset = 1;

    // Hold reset high until all internal pipeline state are initialized.
    for (int i = 0; i < VX_CFG_RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }

    device_->reset = 0;

    // Pump clocks after reset drops to allow internal pipeline states to settle.
    for (int i = 0; i < VX_CFG_RESET_DELAY; ++i) {
      device_->clk = 0;
      this->eval();
      device_->clk = 1;
      this->eval();
    }

  #ifdef AXI_BUS
    this->axi_drive();
  #else
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      device_->mem_req_ready[b] = 1;
    }
  #endif
  }

#ifdef AXI_BUS
  void tick() {
    device_->clk = 0;
    this->eval();

    this->axi_sample();

    device_->clk = 1;
    this->eval();

    this->axi_step();

  #ifndef NDEBUG
    fflush(stdout);
  #endif
  }
#else
  void tick() {
    device_->clk = 0;
    this->eval();

    this->mem_bus_eval(0);

    device_->clk = 1;
    this->eval();

    this->mem_bus_eval(1);

    dram_sim_.tick();

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      if (!dram_queue_[b].empty() && dram_sim_.can_accept(b)) {
        auto mem_req = dram_queue_[b].front();
        dram_sim_.send_request(b, mem_req->bank_addr, mem_req->write, [](void* arg)->bool {
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
#endif

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

#ifdef AXI_BUS
  static constexpr uint32_t AXI_BEAT_SIZE = VX_CFG_PLATFORM_MEMORY_DATA_SIZE;

  void mem_bus_reset() {
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      AxiLatencyPipe::Config cfg;
      cfg.read_latency  = AXI_MEM_READ_LATENCY;
      cfg.write_latency = AXI_MEM_WRITE_LATENCY;
      cfg.max_reads     = AXI_MEM_MAX_READS;
      cfg.max_writes    = AXI_MEM_MAX_WRITES;
      axi_mem_[b] = AxiLatencyPipe(cfg);
      axi_rd_data_[b].clear();
      axi_wr_addr_[b].clear();
      axi_in_[b] = AxiLatencyPipe::Inputs();
    }
    this->axi_drive();
  }

  // Samples each port's request side before the clock edge and performs the
  // accepted transactions on memory: writes first, then reads.
  void axi_sample() {
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      auto& mem = axi_mem_[b];
      auto& in = axi_in_[b];
      in.ar_valid = device_->m_axi_arvalid[b];
      in.ar_id    = device_->m_axi_arid[b];
      in.ar_len   = device_->m_axi_arlen[b];
      in.aw_valid = device_->m_axi_awvalid[b];
      in.aw_id    = device_->m_axi_awid[b];
      in.w_valid  = device_->m_axi_wvalid[b];
      in.w_last   = device_->m_axi_wlast[b];
      in.r_ready  = device_->m_axi_rready[b];
      in.b_ready  = device_->m_axi_bready[b];

      if (in.aw_valid && mem.aw_ready()) {
        axi_wr_addr_[b].push_back(uint64_t(device_->m_axi_awaddr[b]));
      }
      if (in.w_valid && mem.w_ready()) {
        // a write beat never runs ahead of its address on this port
        assert(!axi_wr_addr_[b].empty());
        uint64_t addr = axi_wr_addr_[b].front();
        auto data = VDataCast<uint8_t*, AXI_BEAT_SIZE>::get(device_->m_axi_wdata[b]);
        uint64_t strb = device_->m_axi_wstrb[b];
        for (uint32_t i = 0; i < AXI_BEAT_SIZE; ++i) {
          if ((strb >> i) & 0x1) {
            (*ram_)[addr + i] = data[i];
          }
        }
        if (in.w_last) {
          axi_wr_addr_[b].pop_front();
        } else {
          axi_wr_addr_[b].front() += AXI_BEAT_SIZE;
        }
      }
      if (in.ar_valid && mem.ar_ready()) {
        uint64_t addr = uint64_t(device_->m_axi_araddr[b]);
        std::vector<uint8_t> data((in.ar_len + 1) * AXI_BEAT_SIZE);
        ram_->read(data.data(), addr, data.size());
        axi_rd_data_[b].push_back(std::move(data));
      }
      axi_rlast_fire_[b] = mem.r_valid() && in.r_ready && mem.r_last();
    }
  }

  void axi_step() {
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      if (axi_rlast_fire_[b]) {
        axi_rd_data_[b].pop_front();
      }
      axi_mem_[b].step(axi_in_[b]);
    }
    this->axi_drive();
  }

  void axi_drive() {
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      auto& mem = axi_mem_[b];
      device_->m_axi_arready[b] = mem.ar_ready();
      device_->m_axi_awready[b] = mem.aw_ready();
      device_->m_axi_wready[b]  = mem.w_ready();
      device_->m_axi_rvalid[b]  = mem.r_valid();
      device_->m_axi_rresp[b]   = 0;
      if (mem.r_valid()) {
        device_->m_axi_rid[b]   = mem.r_id();
        device_->m_axi_rlast[b] = mem.r_last();
        memcpy(VDataCast<void*, AXI_BEAT_SIZE>::get(device_->m_axi_rdata[b]),
               axi_rd_data_[b].front().data() + mem.r_beat() * AXI_BEAT_SIZE, AXI_BEAT_SIZE);
      }
      device_->m_axi_bvalid[b] = mem.b_valid();
      device_->m_axi_bresp[b]  = 0;
      if (mem.b_valid()) {
        device_->m_axi_bid[b] = mem.b_id();
      }
    }
  }
#else
  void mem_bus_reset() {
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      device_->mem_req_ready[b] = 0;
      device_->mem_rsp_valid[b] = 0;
    }
  }

  void mem_bus_eval(bool clk) {
    if (!clk) {
      for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
        mem_rd_rsp_ready_[b] = device_->mem_rsp_ready[b];
      }
      return;
    }

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
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
              memcpy(VDataCast<void*, VX_CFG_PLATFORM_MEMORY_DATA_SIZE>::get(device_->mem_rsp_data[b]), mem_rsp->data.data(), VX_CFG_PLATFORM_MEMORY_DATA_SIZE);
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
      #if VX_CFG_PLATFORM_MEMORY_INTERLEAVE == 1
        uint64_t byte_addr = (uint64_t(device_->mem_req_addr[b]) * VX_CFG_PLATFORM_MEMORY_NUM_BANKS + b) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
      #else
        uint64_t byte_addr = (uint64_t(device_->mem_req_addr[b]) + (b << g_mem_bank_addr_width)) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
      #endif
        // check read/write
        if (device_->mem_req_rw[b]) {
          auto byteen = device_->mem_req_byteen[b];
          auto data = VDataCast<uint8_t*, VX_CFG_PLATFORM_MEMORY_DATA_SIZE>::get(device_->mem_req_data[b]);
          // Console output (VX_MEM_IO_COUT range) is now ordinary device
          // memory — Option C: the host runtime drains it. No tap here.
          {
            // process memory writes
            for (int i = 0; i < VX_CFG_PLATFORM_MEMORY_DATA_SIZE; i++) {
              if ((byteen >> i) & 0x1) {
                (*ram_)[byte_addr + i] = data[i];
              }
            }

            auto mem_req = new mem_req_t();
            mem_req->tag   = device_->mem_req_tag[b];
            mem_req->addr  = byte_addr;
            mem_req->bank_addr = uint64_t(device_->mem_req_addr[b]) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
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
          mem_req->bank_addr = uint64_t(device_->mem_req_addr[b]) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
          mem_req->write = false;
          mem_req->ready = false;
          ram_->read(mem_req->data.data(), byte_addr, VX_CFG_PLATFORM_MEMORY_DATA_SIZE);

          // enqueue dram request
          dram_queue_[b].push(mem_req);

          // add to pending list
          pending_mem_reqs_[b].emplace_back(mem_req);
        }
      }
    }
  }
#endif

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

  std::unordered_map<int, std::stringstream> print_bufs_;

#ifdef AXI_BUS
  std::array<AxiLatencyPipe, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> axi_mem_;
  std::array<AxiLatencyPipe::Inputs, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> axi_in_;
  std::array<bool, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> axi_rlast_fire_;
  std::array<std::deque<std::vector<uint8_t>>, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> axi_rd_data_;
  std::array<std::deque<uint64_t>, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> axi_wr_addr_;
#else
  typedef struct {
    std::array<uint8_t, VX_CFG_PLATFORM_MEMORY_DATA_SIZE> data;
    uint64_t addr;
    uint64_t bank_addr;
    uint64_t tag;
    bool write;
    bool ready;
  } mem_req_t;

  std::list<mem_req_t*> pending_mem_reqs_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  std::queue<mem_req_t*> dram_queue_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  std::array<bool, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> mem_rd_rsp_ready_;

  DramSim dram_sim_;
#endif

  VerilatedContext* context_;
  Device* device_;

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

void Processor::run(HostMonitor* monitor, CoutDrainer* cout_drainer) {
  impl_->run(monitor, cout_drainer);
}

int Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Processor::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}