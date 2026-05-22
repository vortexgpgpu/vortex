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

// Host-memory aperture — XRT host-only BOs live here; the kernel's
// m_axi_host master (the Command Processor's host-memory port) addresses
// into it. RAM is sparsely paged, so a large aperture costs nothing.
#define HOST_RAM_BASE  (1ull << 44)
#define HOST_RAM_SIZE  (1ull << 30)   // 1 GiB

#if VX_CFG_PLATFORM_MEMORY_DATA_SIZE > 8
  typedef VlWide<(VX_CFG_PLATFORM_MEMORY_DATA_SIZE/4)> Vl_m_data_t;
#else
#if VX_CFG_PLATFORM_MEMORY_DATA_SIZE > 4
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

static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time = TRACE_STOP_TIME;

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return false;
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
  , host_ram_(nullptr)
  , host_alloc_(nullptr)
  , dram_sim_(VX_CFG_PLATFORM_MEMORY_NUM_BANKS, VX_CFG_PLATFORM_MEMORY_DATA_SIZE, MEM_CLOCK_RATIO)
  , stop_(false)
#ifdef VCD_OUTPUT
  , tfp_(nullptr)
#endif
#ifdef SAIF_OUTPUT
  , sfp_(nullptr)
#endif
  {}

  ~Impl() {
    stop_ = true;
    if (future_.valid()) {
      future_.wait();
    }
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      delete mem_alloc_[b];
    }
    if (ram_) {
      delete ram_;
    }
    if (host_alloc_) {
      delete host_alloc_;
    }
    if (host_ram_) {
      delete host_ram_;
    }
  #ifdef VCD_OUTPUT
    if (tfp_) {
      tfp_->close();
      delete tfp_;
    }
  #endif
  #ifdef SAIF_OUTPUT
    if (sfp_) {
      sfp_->close();
      delete sfp_;
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

    // calculate memory bank size
    mem_bank_size_ = (1ull << VX_CFG_PLATFORM_MEMORY_ADDR_WIDTH) / VX_CFG_PLATFORM_MEMORY_NUM_BANKS;

    // allocate RAM
    ram_ = new RAM(0, RAM_PAGE_SIZE);

    // initialize AXI memory interfaces
    MP_M_AXI_MEM(VX_CFG_PLATFORM_MEMORY_NUM_BANKS);

    // initialize memory allocator
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      mem_alloc_[b] = new MemoryAllocator(0, mem_bank_size_, 4096, 64);
    }

    // allocate host RAM + allocator (backs XRT host-only BOs; reached by
    // the kernel's m_axi_host master)
    host_ram_   = new RAM(0, RAM_PAGE_SIZE);
    host_alloc_ = new MemoryAllocator(HOST_RAM_BASE, HOST_RAM_SIZE, 4096, 64);

    // bind the m_axi_host AXI slave port
    m_axi_host_.awvalid = &device_->m_axi_host_awvalid;
    m_axi_host_.awready = &device_->m_axi_host_awready;
    m_axi_host_.awaddr  = &device_->m_axi_host_awaddr;
    m_axi_host_.awid    = &device_->m_axi_host_awid;
    m_axi_host_.awlen   = &device_->m_axi_host_awlen;
    m_axi_host_.wvalid  = &device_->m_axi_host_wvalid;
    m_axi_host_.wready  = &device_->m_axi_host_wready;
    m_axi_host_.wdata   = &device_->m_axi_host_wdata;
    m_axi_host_.wstrb   = &device_->m_axi_host_wstrb;
    m_axi_host_.wlast   = &device_->m_axi_host_wlast;
    m_axi_host_.arvalid = &device_->m_axi_host_arvalid;
    m_axi_host_.arready = &device_->m_axi_host_arready;
    m_axi_host_.araddr  = &device_->m_axi_host_araddr;
    m_axi_host_.arid    = &device_->m_axi_host_arid;
    m_axi_host_.arlen   = &device_->m_axi_host_arlen;
    m_axi_host_.rvalid  = &device_->m_axi_host_rvalid;
    m_axi_host_.rready  = &device_->m_axi_host_rready;
    m_axi_host_.rdata   = &device_->m_axi_host_rdata;
    m_axi_host_.rlast   = &device_->m_axi_host_rlast;
    m_axi_host_.rid     = &device_->m_axi_host_rid;
    m_axi_host_.rresp   = &device_->m_axi_host_rresp;
    m_axi_host_.bvalid  = &device_->m_axi_host_bvalid;
    m_axi_host_.bready  = &device_->m_axi_host_bready;
    m_axi_host_.bresp   = &device_->m_axi_host_bresp;
    m_axi_host_.bid     = &device_->m_axi_host_bid;

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
    if (bank_id >= VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
      return -1;
    return mem_alloc_[bank_id]->allocate(size, addr);
  }

  int mem_free(uint32_t bank_id, uint64_t addr) {
    if (bank_id >= VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
      return -1;
    return mem_alloc_[bank_id]->release(addr);
  }

  int mem_write(uint32_t bank_id, uint64_t addr, uint64_t size, const void* data) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (bank_id >= VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
      return -1;
    uint64_t base_addr = bank_id * mem_bank_size_ + addr;
    ram_->write(data, base_addr, size);
    /*printf("%0ld: [sim] xrt-mem-write[%d]: addr=0x%lx, size=%ld, data=0x", timestamp, bank_id, base_addr, size);
    for (int i = size-1; i >= 0; --i) {
      printf("%02x", ((const uint8_t*)data)[i]);
    }
    printf(")\n");*/
    return 0;
  }

  int mem_read(uint32_t bank_id, uint64_t addr, uint64_t size, void* data) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (bank_id >= VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
      return -1;
    uint64_t base_addr = bank_id * mem_bank_size_ + addr;
    ram_->read(data, base_addr, size);
    /*printf("%0ld: [sim] xrt-mem-read[%d]: addr=0x%lx, size=%ld, data=0x", timestamp, bank_id, base_addr, size);
    for (int i = size-1; i >= 0; --i) {
      printf("%02x", ((uint8_t*)data)[i]);
    }
    printf(")\n");*/
    return 0;
  }

  int mem_copy(uint32_t bank_id_dest , uint32_t bank_id_src, uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
    std::lock_guard<std::mutex> guard(mutex_);
    if( bank_id_dest >= VX_CFG_PLATFORM_MEMORY_NUM_BANKS || bank_id_src >= VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
      return -1;
    uint64_t dest_base_addr = bank_id_dest * mem_bank_size_ + dest_addr;
    uint64_t src_base_addr = bank_id_src * mem_bank_size_ + src_addr;
    ram_->copy(dest_base_addr, src_base_addr, size);
    return 0;
  }

  // ----- Host memory (XRT host-only BOs; reached by m_axi_host) -----

  int host_mem_alloc(uint64_t size, uint64_t* addr) {
    std::lock_guard<std::mutex> guard(mutex_);
    return host_alloc_->allocate(size, addr);
  }

  int host_mem_free(uint64_t addr) {
    std::lock_guard<std::mutex> guard(mutex_);
    return host_alloc_->release(addr);
  }

  int host_mem_write(uint64_t addr, uint64_t size, const void* data) {
    std::lock_guard<std::mutex> guard(mutex_);
    host_ram_->write(data, addr, size);
    return 0;
  }

  int host_mem_read(uint64_t addr, uint64_t size, void* data) {
    std::lock_guard<std::mutex> guard(mutex_);
    host_ram_->read(data, addr, size);
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
    return 0;
  }

private:

  void reset() {
    this->axi_ctrl_bus_reset();
    this->axi_mem_bus_reset();
    this->axi_host_bus_reset();

    for (auto& reqs : pending_mem_reqs_) {
      reqs.clear();
    }

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      std::queue<mem_req_t*> empty;
      std::swap(dram_queues_[b], empty);
    }

    device_->ap_rst_n = 0;

    for (int i = 0; i < VX_CFG_RESET_DELAY; ++i) {
      device_->ap_clk = 0;
      this->eval();
      device_->ap_clk = 1;
      this->eval();
    }

    device_->ap_rst_n = 1;

    // this AXI device is always ready to accept new requests
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      *m_axi_mem_[b].arready = 1;
      *m_axi_mem_[b].awready = 1;
      *m_axi_mem_[b].wready  = 1;
    }
  }

  void tick() {
    device_->ap_clk = 0;
    this->eval();

    this->axi_mem_bus_eval(0);
    this->axi_host_bus_eval(0);

    device_->ap_clk = 1;
    this->eval();

    this->axi_mem_bus_eval(1);
    this->axi_host_bus_eval(1);

    dram_sim_.tick();

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
      if (!dram_queues_[b].empty()) {
        auto mem_req = dram_queues_[b].front();
        dram_sim_.send_request(mem_req->addr, mem_req->write, [](void* arg)->bool {
          auto orig_req = reinterpret_cast<mem_req_t*>(arg);
          if (orig_req->ready) {
            delete orig_req;
          } else {
            orig_req->ready = true;
          }
          return true;
        }, mem_req);
        dram_queues_[b].pop();
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
    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
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
      m_axi_states_[b].aw_active = false;
    }
  }

  void axi_host_bus_reset() {
    // Ready signals stay constant-high (like the m_axi_mem model): a slave
    // that toggles *ready races the master, which only samples it the cycle
    // after it enters its issue state. Our masters keep one outstanding
    // read/write at a time, so always-ready is correct.
    *m_axi_host_.arready = 1;
    *m_axi_host_.awready = 1;
    *m_axi_host_.wready  = 1;
    *m_axi_host_.rvalid  = 0;
    *m_axi_host_.bvalid  = 0;
    host_rd_active_    = false;
    host_wr_active_    = false;
    host_wr_b_pending_ = false;
    host_rd_rsp_ready_ = false;
    host_wr_rsp_ready_ = false;
  }

  // Burst-capable AXI slave model for the m_axi_host port — the kernel's
  // host-memory master (the Command Processor's host-AXI port). Host memory
  // is reached over the platform slave-bridge, not device DRAM, so this is
  // modeled with minimal latency (no dram_sim timing).
  void axi_host_bus_eval(bool clk) {
    if (!clk) {
      host_rd_rsp_ready_ = *m_axi_host_.rready;
      host_wr_rsp_ready_ = *m_axi_host_.bready;
      return;
    }

    // R channel — retire a presented beat.
    if (*m_axi_host_.rvalid && host_rd_rsp_ready_) {
      *m_axi_host_.rvalid = 0;
      if (host_rd_beat_ >= host_rd_len_)
        host_rd_active_ = false;
      else
        ++host_rd_beat_;
    }
    // B channel — retire the write response.
    if (*m_axi_host_.bvalid && host_wr_rsp_ready_) {
      *m_axi_host_.bvalid = 0;
      host_wr_b_pending_ = false;
    }
    // Accept a read burst (one outstanding at a time).
    if (*m_axi_host_.arvalid && *m_axi_host_.arready && !host_rd_active_) {
      host_rd_active_ = true;
      host_rd_addr_   = uint64_t(*m_axi_host_.araddr);
      host_rd_len_    = *m_axi_host_.arlen;
      host_rd_id_     = *m_axi_host_.arid;
      host_rd_beat_   = 0;
    }
    // Present the next read beat.
    if (host_rd_active_ && !*m_axi_host_.rvalid) {
      uint64_t a = host_rd_addr_
                 + uint64_t(host_rd_beat_) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
      // Host memory is plain process memory (the runtime's host_mem_alloc
      // hands the CP a raw pointer as the cp_addr) — dereference it directly.
      std::memcpy(m_axi_host_.rdata->data(), reinterpret_cast<const void*>(a),
                  VX_CFG_PLATFORM_MEMORY_DATA_SIZE);
      *m_axi_host_.rvalid = 1;
      *m_axi_host_.rid    = host_rd_id_;
      *m_axi_host_.rresp  = 0;
      *m_axi_host_.rlast  = (host_rd_beat_ >= host_rd_len_);
    }
    // Accept a write burst (one outstanding at a time).
    if (*m_axi_host_.awvalid && *m_axi_host_.awready
     && !host_wr_active_ && !host_wr_b_pending_) {
      host_wr_active_ = true;
      host_wr_addr_   = uint64_t(*m_axi_host_.awaddr);
      host_wr_id_     = *m_axi_host_.awid;
      host_wr_beat_   = 0;
    }
    // Accept a write data beat.
    if (host_wr_active_ && *m_axi_host_.wvalid && *m_axi_host_.wready) {
      uint64_t a = host_wr_addr_
                 + uint64_t(host_wr_beat_) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
      auto byteen = *m_axi_host_.wstrb;
      auto data = (const uint8_t*)m_axi_host_.wdata->data();
      for (int i = 0; i < VX_CFG_PLATFORM_MEMORY_DATA_SIZE; ++i) {
        if ((byteen >> i) & 0x1)
          reinterpret_cast<uint8_t*>(a)[i] = data[i];
      }
      if (*m_axi_host_.wlast) {
        host_wr_active_    = false;
        host_wr_b_pending_ = true;
      } else {
        ++host_wr_beat_;
      }
    }
    // Present the write response.
    if (host_wr_b_pending_ && !*m_axi_host_.bvalid) {
      *m_axi_host_.bvalid = 1;
      *m_axi_host_.bid    = host_wr_id_;
      *m_axi_host_.bresp  = 0;
    }
  }

  void axi_mem_bus_eval(bool clk) {
    if (!clk) {
      for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
        m_axi_states_[b].read_rsp_ready = *m_axi_mem_[b].rready;
        m_axi_states_[b].write_rsp_ready = *m_axi_mem_[b].bready;
      }
      return;
    }

    for (int b = 0; b < VX_CFG_PLATFORM_MEMORY_NUM_BANKS; ++b) {
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
          *m_axi_mem_[b].rlast  = mem_rsp->last;
          memcpy(m_axi_mem_[b].rdata->data(), mem_rsp->data.data(), VX_CFG_PLATFORM_MEMORY_DATA_SIZE);
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

      // handle read requests — an AXI burst expands into arlen+1 per-beat
      // responses (one cache line each, INCR addressing).
      if (*m_axi_mem_[b].arvalid && *m_axi_mem_[b].arready) {
        uint32_t len  = *m_axi_mem_[b].arlen;     // beats - 1
        uint64_t base = uint64_t(*m_axi_mem_[b].araddr);
        for (uint32_t beat = 0; beat <= len; ++beat) {
          auto mem_req = new mem_req_t();
          mem_req->tag   = *m_axi_mem_[b].arid;
          mem_req->addr  = base + uint64_t(beat) * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
          ram_->read(mem_req->data.data(), mem_req->addr, VX_CFG_PLATFORM_MEMORY_DATA_SIZE);
          mem_req->write = false;
          mem_req->ready = false;
          mem_req->last  = (beat == len);
          pending_mem_reqs_[b].emplace_back(mem_req);
          dram_queues_[b].push(mem_req);
        }
      }

      // handle write address — latch the burst.
      if (*m_axi_mem_[b].awvalid && *m_axi_mem_[b].awready
       && !m_axi_states_[b].aw_active) {
        m_axi_states_[b].aw_active = true;
        m_axi_states_[b].aw_addr   = uint64_t(*m_axi_mem_[b].awaddr);
        m_axi_states_[b].aw_tag    = *m_axi_mem_[b].awid;
        m_axi_states_[b].aw_beat   = 0;
      }

      // handle write data beats — write one cache line per W beat; the
      // last beat (WLAST) queues a single B response for the burst.
      if (m_axi_states_[b].aw_active
       && *m_axi_mem_[b].wvalid && *m_axi_mem_[b].wready) {
        uint64_t byte_addr = m_axi_states_[b].aw_addr
                           + uint64_t(m_axi_states_[b].aw_beat)
                             * VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
        auto byteen = *m_axi_mem_[b].wstrb;
        auto data = (const uint8_t*)m_axi_mem_[b].wdata->data();
        for (int i = 0; i < VX_CFG_PLATFORM_MEMORY_DATA_SIZE; ++i) {
          if ((byteen >> i) & 0x1) {
            (*ram_)[byte_addr + i] = data[i];
          }
        }
        if (*m_axi_mem_[b].wlast) {
          auto mem_req = new mem_req_t();
          mem_req->tag   = m_axi_states_[b].aw_tag;
          mem_req->addr  = m_axi_states_[b].aw_addr;
          mem_req->write = true;
          mem_req->ready = false;
          mem_req->last  = true;
          pending_mem_reqs_[b].emplace_back(mem_req);
          dram_queues_[b].push(mem_req);
          m_axi_states_[b].aw_active = false;
        } else {
          ++m_axi_states_[b].aw_beat;
        }
      }
    }
  }

  typedef struct {
    bool     read_rsp_ready;
    bool     write_rsp_ready;
    // Write-burst state — latched on AW, advanced one cache line per W beat.
    bool     aw_active;
    uint64_t aw_addr;
    uint32_t aw_beat;
    uint32_t aw_tag;
  } m_axi_state_t;

  typedef struct {
    std::array<uint8_t, VX_CFG_PLATFORM_MEMORY_DATA_SIZE> data;
    uint32_t tag;
    uint64_t addr;
    bool write;
    bool ready;
    bool last;     // last beat of its burst — drives rlast
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
  RAM* host_ram_;
  MemoryAllocator* host_alloc_;
  DramSim dram_sim_;
  uint64_t mem_bank_size_;

  std::future<void> future_;
  bool stop_;

  std::mutex mutex_;

  std::list<mem_req_t*> pending_mem_reqs_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  m_axi_mem_t m_axi_mem_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  MemoryAllocator* mem_alloc_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  m_axi_state_t m_axi_states_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  std::queue<mem_req_t*> dram_queues_[VX_CFG_PLATFORM_MEMORY_NUM_BANKS];

  // m_axi_host AXI slave port + burst state.
  m_axi_mem_t m_axi_host_;
  bool     host_rd_active_;
  uint64_t host_rd_addr_;
  uint32_t host_rd_len_;
  uint32_t host_rd_beat_;
  uint32_t host_rd_id_;
  bool     host_rd_rsp_ready_;
  bool     host_wr_active_;
  uint64_t host_wr_addr_;
  uint32_t host_wr_beat_;
  uint32_t host_wr_id_;
  bool     host_wr_b_pending_;
  bool     host_wr_rsp_ready_;

#ifdef VCD_OUTPUT
  VerilatedVcdC* tfp_;
#endif
#ifdef SAIF_OUTPUT
  VerilatedSaifC* sfp_;
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

int xrt_sim::mem_copy(uint32_t bank_id_dest , uint32_t bank_id_src, uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
  return impl_->mem_copy(bank_id_dest, bank_id_src, dest_addr, src_addr, size);
}

int xrt_sim::host_mem_alloc(uint64_t size, uint64_t* addr) {
  return impl_->host_mem_alloc(size, addr);
}

int xrt_sim::host_mem_free(uint64_t addr) {
  return impl_->host_mem_free(addr);
}

int xrt_sim::host_mem_write(uint64_t addr, uint64_t size, const void* value) {
  return impl_->host_mem_write(addr, size, value);
}

int xrt_sim::host_mem_read(uint64_t addr, uint64_t size, void* value) {
  return impl_->host_mem_read(addr, size, value);
}

int xrt_sim::register_write(uint32_t offset, uint32_t value) {
  return impl_->register_write(offset, value);
}

int xrt_sim::register_read(uint32_t offset, uint32_t* value) {
  return impl_->register_read(offset, value);
}