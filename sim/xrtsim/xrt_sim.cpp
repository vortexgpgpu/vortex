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

#include "Vvortex_afu_shim.h"

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
#include <mem_alloc.h>
#include <mp_macros.h>

#include <iostream>

#define PLATFORM_MEMORY_DATA_SIZE (PLATFORM_MEMORY_DATA_WIDTH/8)

#ifndef MEM_CLOCK_RATIO
#define MEM_CLOCK_RATIO 1
#endif

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

#if PLATFORM_MEMORY_DATA_WIDTH > 64
  typedef VlWide<(PLATFORM_MEMORY_DATA_WIDTH/32)> Vl_m_data_t;
#else
#if PLATFORM_MEMORY_DATA_WIDTH > 32
  typedef QData Vl_m_data_t;
#else
  typedef IData Vl_m_data_t;
#endif
#endif

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

#define MP_M_AXI_MEM_EACH(i) \
  m_axi_mem_[i].awvalid = &device_->m_axi_mem_##i##_awvalid; \
  m_axi_mem_[i].awready = &device_->m_axi_mem_##i##_awready; \
  m_axi_mem_[i].awaddr  = &device_->m_axi_mem_##i##_awaddr; \
  m_axi_mem_[i].awid    = &device_->m_axi_mem_##i##_awid; \
  m_axi_mem_[i].awlen   = &device_->m_axi_mem_##i##_awlen; \
  m_axi_mem_[i].wvalid  = &device_->m_axi_mem_##i##_wvalid; \
  m_axi_mem_[i].wready  = &device_->m_axi_mem_##i##_wready; \
  m_axi_mem_[i].wdata   = &device_->m_axi_mem_##i##_wdata; \
  m_axi_mem_[i].wstrb   = &device_->m_axi_mem_##i##_wstrb; \
  m_axi_mem_[i].wlast   = &device_->m_axi_mem_##i##_wlast; \
  m_axi_mem_[i].arvalid = &device_->m_axi_mem_##i##_arvalid; \
  m_axi_mem_[i].arready = &device_->m_axi_mem_##i##_arready; \
  m_axi_mem_[i].araddr  = &device_->m_axi_mem_##i##_araddr; \
  m_axi_mem_[i].arid    = &device_->m_axi_mem_##i##_arid; \
  m_axi_mem_[i].arlen   = &device_->m_axi_mem_##i##_arlen; \
  m_axi_mem_[i].rvalid  = &device_->m_axi_mem_##i##_rvalid; \
  m_axi_mem_[i].rready  = &device_->m_axi_mem_##i##_rready; \
  m_axi_mem_[i].rdata   = &device_->m_axi_mem_##i##_rdata; \
  m_axi_mem_[i].rlast   = &device_->m_axi_mem_##i##_rlast; \
  m_axi_mem_[i].rid     = &device_->m_axi_mem_##i##_rid; \
  m_axi_mem_[i].rresp   = &device_->m_axi_mem_##i##_rresp; \
  m_axi_mem_[i].bvalid  = &device_->m_axi_mem_##i##_bvalid; \
  m_axi_mem_[i].bready  = &device_->m_axi_mem_##i##_bready; \
  m_axi_mem_[i].bresp   = &device_->m_axi_mem_##i##_bresp; \
  m_axi_mem_[i].bid     = &device_->m_axi_mem_##i##_bid;

#define MP_M_AXI_MEM(n) MP_REPEAT(n, MP_M_AXI_MEM_EACH, ;)

class xrt_sim::Impl {
public:
  Impl()
  : device_(nullptr)
  , ram_(nullptr)
  , dram_sim_(MEM_CLOCK_RATIO)
  , stop_(false)
#ifdef VCD_OUTPUT
  , tfp_(nullptr)
#endif
  {}

  ~Impl() {
    stop_ = true;
    if (future_.valid()) {
      future_.wait();
    }
    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      delete mem_alloc_[b];
    }
    if (ram_) {
      delete ram_;
    }
  #ifdef VCD_OUTPUT
    if (tfp_) {
      tfp_->close();
      delete tfp_;
    }
  #endif
    if (device_) {
      delete device_;
    }
  }

  int init() {
    // force random values for uninitialized signals
    Verilated::randReset(VERILATOR_RESET_VALUE);
    Verilated::randSeed(50);

    // turn off assertion before reset
    Verilated::assertOn(false);

    // create RTL module instance
    device_ = new Vvortex_afu_shim();

  #ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    tfp_ = new VerilatedVcdC();
    device_->trace(tfp_, 99);
    tfp_->open("trace.vcd");
  #endif

    // calculate memory bank size
    mem_bank_size_ = 1ull << PLATFORM_MEMORY_ADDR_WIDTH;

    // allocate RAM
    ram_ = new RAM(0, RAM_PAGE_SIZE);

    // initialize AXI memory interfaces
    MP_M_AXI_MEM(PLATFORM_MEMORY_BANKS);

    // initialize memory allocator
    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      mem_alloc_[b] = new MemoryAllocator(0, mem_bank_size_, 4096, 64);
    }

    // reset the device
    this->reset();

    // Turn on assertion after reset
    Verilated::assertOn(true);

    // launch execution thread
    future_ = std::async(std::launch::async, [&]{
      while (!stop_) {
        std::lock_guard<std::mutex> guard(mutex_);
        this->tick();
      }
    });

    return 0;
  }

  int mem_alloc(uint64_t size, uint32_t bank_id, uint64_t* addr) {
    if (bank_id >= PLATFORM_MEMORY_BANKS)
      return -1;
    return mem_alloc_[bank_id]->allocate(size, addr);
  }

  int mem_free(uint32_t bank_id, uint64_t addr) {
    if (bank_id >= PLATFORM_MEMORY_BANKS)
      return -1;
    return mem_alloc_[bank_id]->release(addr);
  }

  int mem_write(uint32_t bank_id, uint64_t addr, uint64_t size, const void* data) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (bank_id >= PLATFORM_MEMORY_BANKS)
      return -1;
    uint64_t base_addr = bank_id * mem_bank_size_ + addr;
    ram_->write(data, base_addr, size);
    /*printf("%0ld: [sim] xrt-mem-write: bank_id=%0d, addr=0x%lx, size=%ld, data=0x", timestamp, bank_id, base_addr, size);
    for (int i = size-1; i >= 0; --i) {
      printf("%02x", ((const uint8_t*)data)[i]);
    }
    printf(")\n");*/
    return 0;
  }

  int mem_read(uint32_t bank_id, uint64_t addr, uint64_t size, void* data) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (bank_id >= PLATFORM_MEMORY_BANKS)
      return -1;
    uint64_t base_addr = bank_id * mem_bank_size_ + addr;
    ram_->read(data, base_addr, size);
    /*printf("%0ld: [sim] xrt-mem-read: bank_id=%0d, addr=0x%lx, size=%ld, data=0x", timestamp, bank_id, base_addr, size);
    for (int i = size-1; i >= 0; --i) {
      printf("%02x", ((uint8_t*)data)[i]);
    }
    printf(")\n");*/
    return 0;
  }

  int register_write(uint32_t offset, uint32_t value) {
    std::lock_guard<std::mutex> guard(mutex_);

    // write address
    //printf("%0ld: [sim] register_write: address=0x%x\n", timestamp, offset);
    device_->s_axi_ctrl_awvalid = 1;
    device_->s_axi_ctrl_awaddr = offset;
    while (!device_->s_axi_ctrl_awready) {
      this->tick();
    }
    this->tick();
    device_->s_axi_ctrl_awvalid = 0;

    // write data
    //printf("%0ld: [sim] register_write: data=0x%x\n", timestamp, value);
    device_->s_axi_ctrl_wvalid = 1;
    device_->s_axi_ctrl_wdata = value;
    device_->s_axi_ctrl_wstrb = 0xf;
    while (!device_->s_axi_ctrl_wready) {
      this->tick();
    }
    this->tick();
    device_->s_axi_ctrl_wvalid = 0;

    // write response
    //printf("%0ld: [sim] register_write: response\n", timestamp);
    do {
      this->tick();
    } while (!device_->s_axi_ctrl_bvalid);
    device_->s_axi_ctrl_bready = 1;
    this->tick();
    device_->s_axi_ctrl_bready = 0;
    //printf("%0ld: [sim] register_write: done\n", timestamp);
    return 0;
  }

  int register_read(uint32_t offset, uint32_t* value) {
    std::lock_guard<std::mutex> guard(mutex_);
    // read address
    //printf("%0ld: [sim] register_read: address=0x%x\n", timestamp, offset);
    device_->s_axi_ctrl_arvalid = 1;
    device_->s_axi_ctrl_araddr = offset;
    while (!device_->s_axi_ctrl_arready) {
      this->tick();
    }
    this->tick();
    device_->s_axi_ctrl_arvalid = 0;

    // read response
    //printf("%0ld: [sim] register_read: response\n", timestamp);
    do {
      this->tick();
    } while (!device_->s_axi_ctrl_rvalid);
    *value = device_->s_axi_ctrl_rdata;
    device_->s_axi_ctrl_rready = 1;
    this->tick();
    device_->s_axi_ctrl_rready = 0;
    //printf("%0ld: [sim] register_read: done (value=0x%x)\n", timestamp, *value);
    return 0;
  }

private:

  void reset() {
    this->axi_ctrl_bus_reset();
    this->axi_mem_bus_reset();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queues_[b], empty);
    }

    device_->ap_rst_n = 0;

    for (int i = 0; i < RESET_DELAY; ++i) {
      device_->ap_clk = 0;
      this->eval();
      device_->ap_clk = 1;
      this->eval();
    }

    device_->ap_rst_n = 1;

    // this AXI device is always ready to accept new requests
    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      *m_axi_mem_[b].arready = 1;
      *m_axi_mem_[b].awready = 1;
      *m_axi_mem_[b].wready  = 1;
    }
  }

  void tick() {
    device_->ap_clk = 0;
    this->eval();

    this->axi_mem_bus_eval(0);

    device_->ap_clk = 1;
    this->eval();

    this->axi_mem_bus_eval(1);

    dram_sim_.tick();

    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      if (!dram_queues_[b].empty()) {
        auto mem_req = dram_queues_[b].front();
        if (dram_sim_.send_request(mem_req->write, mem_req->addr, b, [](void* arg) {
          auto orig_req = reinterpret_cast<mem_req_t*>(arg);
          if (orig_req->ready) {
            delete orig_req;
          } else {
            orig_req->ready = true;
          }
        }, mem_req)) {
          dram_queues_[b].pop();
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

  void axi_ctrl_bus_reset() {
    // read request address
    device_->s_axi_ctrl_arvalid = 0;
    device_->s_axi_ctrl_araddr = 0;

    // read response
    device_->s_axi_ctrl_rready = 0;

    // write request address
    device_->s_axi_ctrl_awvalid = 0;
    device_->s_axi_ctrl_awaddr = 0;

    // write request data
    device_->s_axi_ctrl_wvalid = 0;
    device_->s_axi_ctrl_wdata = 0;
    device_->s_axi_ctrl_wstrb = 0;

    // write response
    device_->s_axi_ctrl_bready = 0;
  }

  void axi_mem_bus_reset() {
    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      // read request address
      *m_axi_mem_[b].arready = 0;

      // write request address
      *m_axi_mem_[b].awready = 0;

      // write request data
      *m_axi_mem_[b].wready = 0;

      // read response
      *m_axi_mem_[b].rvalid = 0;

      // write response
      *m_axi_mem_[b].bvalid = 0;

      // states
      m_axi_states_[b].write_req_addr_ack = false;
      m_axi_states_[b].write_req_data_ack = false;
    }
  }

  void axi_mem_bus_eval(bool clk) {
    if (!clk) {
      for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
        m_axi_states_[b].read_rsp_ready = *m_axi_mem_[b].rready;
        m_axi_states_[b].write_rsp_ready = *m_axi_mem_[b].bready;
      }
      return;
    }

    for (int b = 0; b < PLATFORM_MEMORY_BANKS; ++b) {
      // handle read responses
      if (*m_axi_mem_[b].rvalid && m_axi_states_[b].read_rsp_ready) {
        *m_axi_mem_[b].rvalid = 0;
      }
      if (!*m_axi_mem_[b].rvalid) {
        if (!pending_mem_reqs_[b].empty()
        && (*pending_mem_reqs_[b].begin())->ready
        && !(*pending_mem_reqs_[b].begin())->write) {
          auto mem_rsp_it = pending_mem_reqs_[b].begin();
          auto mem_rsp = *mem_rsp_it;
          *m_axi_mem_[b].rvalid = 1;
          *m_axi_mem_[b].rid    = mem_rsp->tag;
          *m_axi_mem_[b].rresp  = 0;
          *m_axi_mem_[b].rlast  = 1;
          memcpy(m_axi_mem_[b].rdata->data(), mem_rsp->data.data(), PLATFORM_MEMORY_DATA_SIZE);
          pending_mem_reqs_[b].erase(mem_rsp_it);
          delete mem_rsp;
        }
      }

      // handle write responses
      if (*m_axi_mem_[b].bvalid && m_axi_states_[b].write_rsp_ready) {
        *m_axi_mem_[b].bvalid = 0;
      }
      if (!*m_axi_mem_[b].bvalid) {
        if (!pending_mem_reqs_[b].empty()
        && (*pending_mem_reqs_[b].begin())->ready
        && (*pending_mem_reqs_[b].begin())->write) {
          auto mem_rsp_it = pending_mem_reqs_[b].begin();
          auto mem_rsp = *mem_rsp_it;
          *m_axi_mem_[b].bvalid = 1;
          *m_axi_mem_[b].bid    = mem_rsp->tag;
          *m_axi_mem_[b].bresp  = 0;
          pending_mem_reqs_[b].erase(mem_rsp_it);
          delete mem_rsp;
        }
      }

      // handle read requests
      if (*m_axi_mem_[b].arvalid && *m_axi_mem_[b].arready) {
        auto mem_req = new mem_req_t();
        mem_req->tag   = *m_axi_mem_[b].arid;
        mem_req->addr  = uint64_t(*m_axi_mem_[b].araddr);
        ram_->read(mem_req->data.data(), mem_req->addr, PLATFORM_MEMORY_DATA_SIZE);
        mem_req->write = false;
        mem_req->ready = false;
        pending_mem_reqs_[b].emplace_back(mem_req);

        /*printf("%0ld: [sim] axi-mem-read: bank=%d, addr=0x%lx, tag=0x%x, data=0x", timestamp, b, mem_req->addr, mem_req->tag);
        for (int i = PLATFORM_MEMORY_DATA_SIZE-1; i >= 0; --i) {
          printf("%02x", mem_req->data[b]);
        }
        printf("\n");*/

        // send dram request
        dram_queues_[b].push(mem_req);
      }

      // handle write address requests
      if (*m_axi_mem_[b].awvalid && *m_axi_mem_[b].awready && !m_axi_states_[b].write_req_addr_ack) {
        m_axi_states_[b].write_req_addr = *m_axi_mem_[b].awaddr;
        m_axi_states_[b].write_req_tag = *m_axi_mem_[b].awid;
        m_axi_states_[b].write_req_addr_ack = true;
      }

      // handle write data requests
      if (*m_axi_mem_[b].wvalid && *m_axi_mem_[b].wready && !m_axi_states_[b].write_req_data_ack) {
        m_axi_states_[b].write_req_byteen = *m_axi_mem_[b].wstrb;
        auto data = (const uint8_t*)m_axi_mem_[b].wdata->data();
        for (int i = 0; i < PLATFORM_MEMORY_DATA_SIZE; ++i) {
          m_axi_states_[b].write_req_data[i] = data[i];
        }
        m_axi_states_[b].write_req_data_ack = true;
      }

      // handle write requests
      if (m_axi_states_[b].write_req_addr_ack && m_axi_states_[b].write_req_data_ack) {
        auto byteen = m_axi_states_[b].write_req_byteen;
        auto byte_addr = m_axi_states_[b].write_req_addr;
        for (int i = 0; i < PLATFORM_MEMORY_DATA_SIZE; ++i) {
          if ((byteen >> i) & 0x1) {
            (*ram_)[byte_addr + i] = m_axi_states_[b].write_req_data[i];
          }
        }
        auto mem_req = new mem_req_t();
        mem_req->tag   = m_axi_states_[b].write_req_tag;
        mem_req->addr  = byte_addr;
        mem_req->write = true;
        mem_req->ready = false;
        pending_mem_reqs_[b].emplace_back(mem_req);

        /*printf("%0ld: [sim] axi-mem-write: bank=%d, addr=0x%lx, byteen=0x%lx, tag=0x%x, data=0x", timestamp, b, mem_req->addr, byteen, mem_req->tag);
        for (int i = PLATFORM_MEMORY_DATA_SIZE-1; i >= 0; --i) {
          printf("%02x", m_axi_states_[b].write_req_data[i]]);
        }
        printf("\n");*/

        // send dram request
        dram_queues_[b].push(mem_req);

        // clear acks
        m_axi_states_[b].write_req_addr_ack = false;
        m_axi_states_[b].write_req_data_ack = false;
      }
    }
  }

  typedef struct {
    std::array<uint8_t, PLATFORM_MEMORY_DATA_SIZE> write_req_data;
    uint64_t write_req_byteen;
    uint64_t write_req_addr;
    uint32_t write_req_tag;
    bool read_rsp_ready;
    bool write_rsp_ready;
    bool write_req_addr_ack;
    bool write_req_data_ack;
  } m_axi_state_t;

  typedef struct {
    std::array<uint8_t, PLATFORM_MEMORY_DATA_SIZE> data;
    uint32_t tag;
    uint64_t addr;
    bool write;
    bool ready;
  } mem_req_t;

  typedef struct {
    CData* awvalid;
    CData* awready;
    QData* awaddr;
    IData* awid;
    CData* awlen;
    CData* wvalid;
    CData* wready;
    Vl_m_data_t* wdata;
    QData* wstrb;
    CData* wlast;
    CData* arvalid;
    CData* arready;
    QData* araddr;
    IData* arid;
    CData* arlen;
    CData* rvalid;
    CData* rready;
    Vl_m_data_t* rdata;
    CData* rlast;
    IData* rid;
    CData* rresp;
    CData* bvalid;
    CData* bready;
    CData* bresp;
    IData* bid;
  } m_axi_mem_t;

  Vvortex_afu_shim* device_;
  RAM* ram_;
  DramSim dram_sim_;
  uint64_t mem_bank_size_;

  std::future<void> future_;
  bool stop_;

  std::mutex mutex_;

  std::list<mem_req_t*> pending_mem_reqs_[PLATFORM_MEMORY_BANKS];

  m_axi_mem_t m_axi_mem_[PLATFORM_MEMORY_BANKS];

  MemoryAllocator* mem_alloc_[PLATFORM_MEMORY_BANKS];

  m_axi_state_t m_axi_states_[PLATFORM_MEMORY_BANKS];

  std::queue<mem_req_t*> dram_queues_[PLATFORM_MEMORY_BANKS];

#ifdef VCD_OUTPUT
  VerilatedVcdC* tfp_;
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

int xrt_sim::mem_alloc(uint64_t size, uint32_t bank_id, uint64_t* addr) {
  return impl_->mem_alloc(size, bank_id, addr);
}

int xrt_sim::mem_free(uint32_t bank_id, uint64_t addr) {
  return impl_->mem_free(bank_id, addr);
}

int xrt_sim::mem_write(uint32_t bank_id, uint64_t addr, uint64_t size, const void* data) {
  return impl_->mem_write(bank_id, addr, size, data);
}

int xrt_sim::mem_read(uint32_t bank_id, uint64_t addr, uint64_t size, void* data) {
  return impl_->mem_read(bank_id, addr, size, data);
}

int xrt_sim::register_write(uint32_t offset, uint32_t value) {
  return impl_->register_write(offset, value);
}

int xrt_sim::register_read(uint32_t offset, uint32_t* value) {
  return impl_->register_read(offset, value);
}