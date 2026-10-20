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


#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mem.h>
#include <host_monitor.h>
#include <cout_drainer.h>

#include <VX_config.h>
#include <ostream>
#include <deque>
#include <list>
#include <queue>
#include <vector>
#include <sstream>
#include <unordered_map>

#include <dram_sim.h>
#include <util.h>


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

// Command-processor DMA port into the last-level cache: one cache word per
// request, and as many reads in flight as one 4 KB DMA burst holds.
#if VX_CFG_L3_ENABLED
#define CP_MEM_DATA_SIZE VX_CFG_L2_SECTOR_SIZE
#else
#define CP_MEM_DATA_SIZE VX_CFG_L1_LINE_SIZE
#endif
#define CP_MEM_RD_SLOTS (4096 / CP_MEM_DATA_SIZE)

using namespace vortex;

static uint64_t g_mem_done_seq = 0;
static uint32_t g_mem_bank_addr_width = (VX_CFG_PLATFORM_MEMORY_ADDR_WIDTH - log2ceil(VX_CFG_PLATFORM_MEMORY_NUM_BANKS));

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
  Impl() : dram_sim_(DramSim::platform_config()) {
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
    device_ = new Vrtlsim_shim(context_);

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
    // A core whose flush already finished answers on the next cycle, and the
    // response is a one-cycle pulse, so start polling right away.
    while (!device_->dcr_rsp_valid) {
      this->tick();
    }
    *value = device_->dcr_rsp_data;
    return 0;
  }

  void drain() {
    while (device_->busy) {
      this->tick();
    }
  }

  void cp_mem_write(uint64_t addr, const uint8_t* src, uint64_t size, bool noalloc) {
    while (size != 0) {
      uint64_t offset = addr % CP_MEM_DATA_SIZE;
      uint64_t n = std::min<uint64_t>(size, CP_MEM_DATA_SIZE - offset);
      cp_req_t req;
      req.rw      = true;
      req.noalloc = noalloc;
      req.addr    = addr / CP_MEM_DATA_SIZE;
      req.byteen  = 0;
      req.data.fill(0);
      for (uint64_t i = 0; i < n; ++i) {
        req.data[offset + i] = src[i];
        req.byteen |= uint64_t(1) << (offset + i);
      }
      req.tag = 0;
      cp_reqs_.push_back(req);
      addr += n;
      src  += n;
      size -= n;
    }
    while (!cp_reqs_.empty()) {
      this->tick();
    }
  }

  void cp_mem_read(uint64_t addr, uint8_t* dst, uint64_t size) {
    uint32_t inflight = 0;
    while (size != 0 || inflight != 0) {
      // two queued requests keep one read entering the port every cycle
      while (size != 0 && cp_reqs_.size() < 2 && !cp_free_slots_.empty()) {
        uint64_t offset = addr % CP_MEM_DATA_SIZE;
        uint64_t n = std::min<uint64_t>(size, CP_MEM_DATA_SIZE - offset);
        uint32_t slot = cp_free_slots_.back();
        cp_free_slots_.pop_back();
        cp_rd_slots_[slot] = {dst, uint32_t(offset), uint32_t(n)};
        cp_req_t req;
        req.rw      = false;
        req.noalloc = false;
        req.addr    = addr / CP_MEM_DATA_SIZE;
        req.byteen  = ~uint64_t(0);
        req.data.fill(0);
        req.tag     = slot;
        cp_reqs_.push_back(req);
        ++inflight;
        addr += n;
        dst  += n;
        size -= n;
      }
      this->tick();
      inflight -= cp_rd_completed_;
      cp_rd_completed_ = 0;
    }
  }

private:

  void reset() {
    this->mem_bus_reset();
    this->dcr_bus_reset();
    this->cp_mem_reset();

    print_bufs_.clear();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queue_[b], empty);
    }

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

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      device_->mem_req_ready[b] = 1;
    }
  }

  void tick() {
    device_->clk = 0;
    this->eval();

    this->mem_bus_eval(0);
    this->cp_mem_eval(0);

    device_->clk = 1;
    this->eval();

    this->mem_bus_eval(1);
    this->cp_mem_eval(1);

    dram_sim_.tick();

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      if (!dram_queue_[b].empty() && dram_sim_.can_accept(b)) {
        auto mem_req = dram_queue_[b].front();
        dram_sim_.send_request(b, mem_req->bank_addr, mem_req->write, [](void* arg)->bool {
          // mark completed request as ready
          auto orig_req = reinterpret_cast<mem_req_t*>(arg);
          orig_req->ready = true;
          orig_req->done_seq = g_mem_done_seq++;
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
      // Reads carry distinct AXI IDs, so the memory returns them in the order
      // they complete; writes are posted and never hold back a read.
      auto& pending = pending_mem_reqs_[b];
      auto next_rd = pending.end();
      for (auto it = pending.begin(); it != pending.end();) {
        auto mem_req = *it;
        if (mem_req->ready && mem_req->write) {
          it = pending.erase(it);
          delete mem_req;
          continue;
        }
        if (mem_req->ready && (next_rd == pending.end() || mem_req->done_seq < (*next_rd)->done_seq)) {
          next_rd = it;
        }
        ++it;
      }
      if (next_rd != pending.end() && !device_->mem_rsp_valid[b]) {
        auto mem_rsp = *next_rd;
        device_->mem_rsp_valid[b] = 1;
        memcpy(VDataCast<void*, VX_CFG_PLATFORM_MEMORY_DATA_SIZE>::get(device_->mem_rsp_data[b]), mem_rsp->data.data(), VX_CFG_PLATFORM_MEMORY_DATA_SIZE);
        device_->mem_rsp_tag[b] = mem_rsp->tag;
        pending.erase(next_rd);
        delete mem_rsp;
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

  void dcr_bus_reset() {
    device_->dcr_req_valid = 0;
    device_->dcr_req_rw    = 0;
  }

  void cp_mem_reset() {
    cp_reqs_.clear();
    cp_free_slots_.clear();
    for (uint32_t i = 0; i < CP_MEM_RD_SLOTS; ++i) {
      cp_free_slots_.push_back(CP_MEM_RD_SLOTS - 1 - i);
    }
    cp_req_fire_ = false;
    cp_rd_completed_ = 0;
    device_->cp_mem_req_valid = 0;
    device_->cp_mem_rsp_ready = 1;
  }

  // Inputs change after the rising edge; the handshakes they take part in
  // are sampled after the falling-edge evaluation, ahead of the next edge.
  void cp_mem_eval(bool clk) {
    if (!clk) {
      cp_req_fire_ = device_->cp_mem_req_valid && device_->cp_mem_req_ready;
      if (device_->cp_mem_rsp_valid) {
        uint32_t slot = device_->cp_mem_rsp_tag;
        auto& rd = cp_rd_slots_.at(slot);
        auto data = VDataCast<const uint8_t*, CP_MEM_DATA_SIZE>::get(device_->cp_mem_rsp_data);
        std::memcpy(rd.dst, data + rd.offset, rd.size);
        cp_free_slots_.push_back(slot);
        ++cp_rd_completed_;
      }
      return;
    }
    if (cp_req_fire_) {
      cp_reqs_.pop_front();
      cp_req_fire_ = false;
    }
    if (cp_reqs_.empty()) {
      device_->cp_mem_req_valid = 0;
      return;
    }
    auto& req = cp_reqs_.front();
    device_->cp_mem_req_valid   = 1;
    device_->cp_mem_req_rw      = req.rw;
    device_->cp_mem_req_noalloc = req.noalloc;
    device_->cp_mem_req_addr    = req.addr;
    device_->cp_mem_req_byteen  = req.byteen;
    device_->cp_mem_req_tag     = req.tag;
    memcpy(VDataCast<void*, CP_MEM_DATA_SIZE>::get(device_->cp_mem_req_data), req.data.data(), CP_MEM_DATA_SIZE);
  }

  void wait(uint32_t cycles) {
    for (int i = 0; i < cycles; ++i) {
      this->tick();
    }
  }

private:

  typedef struct {
    Vrtlsim_shim* device;
    std::array<uint8_t, VX_CFG_PLATFORM_MEMORY_DATA_SIZE> data;
    uint64_t addr;
    uint64_t bank_addr;
    uint64_t tag;
    bool write;
    bool ready;
    uint64_t done_seq;
  } mem_req_t;

  std::unordered_map<int, std::stringstream> print_bufs_;

  typedef struct {
    std::array<uint8_t, CP_MEM_DATA_SIZE> data;
    uint64_t addr;
    uint64_t byteen;
    uint32_t tag;
    bool rw;
    bool noalloc;
  } cp_req_t;

  typedef struct {
    uint8_t* dst;
    uint32_t offset;
    uint32_t size;
  } cp_rd_t;

  std::deque<cp_req_t> cp_reqs_;
  std::array<cp_rd_t, CP_MEM_RD_SLOTS> cp_rd_slots_;
  std::vector<uint32_t> cp_free_slots_;
  uint32_t cp_rd_completed_;
  bool cp_req_fire_;

  std::list<mem_req_t*> pending_mem_reqs_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  std::queue<mem_req_t*> dram_queue_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  std::array<bool, VX_CFG_PLATFORM_MEMORY_NUM_BANKS> mem_rd_rsp_ready_;

  DramSim dram_sim_;

  VerilatedContext* context_;
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

void Processor::run(HostMonitor* monitor, CoutDrainer* cout_drainer) {
  impl_->run(monitor, cout_drainer);
}

int Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Processor::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}

void Processor::cp_mem_write(uint64_t addr, const void* src, uint64_t size, bool noalloc) {
  impl_->cp_mem_write(addr, static_cast<const uint8_t*>(src), size, noalloc);
}

void Processor::cp_mem_read(uint64_t addr, void* dst, uint64_t size) {
  impl_->cp_mem_read(addr, static_cast<uint8_t*>(dst), size);
}

bool Processor::cp_llc_client() const {
  return VX_CFG_CP_ON_LLC != 0;
}

uint64_t Processor::cp_llc_size() const {
  return VX_CFG_L3_ENABLED ? VX_CFG_L3_SIZE : VX_CFG_L2_SIZE;
}

void Processor::drain() {
  impl_->drain();
}