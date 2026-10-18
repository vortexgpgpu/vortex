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

#include "foldsim_emul.h"

#include "Vfoldsim_afu.h"
#include "Vfoldsim_afu___024root.h"
#include "verilated.h"
#ifdef VM_TRACE
#include "verilated_vcd_c.h"
#endif

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace vortex;

double sc_time_stamp() { return 0; }

namespace {

uint64_t env_or(const char* name, uint64_t value) {
  const char* s = getenv(name);
  return (s != nullptr && s[0] != '\0') ? strtoull(s, nullptr, 0) : value;
}

constexpr uint64_t kPage = 4096;
constexpr uint32_t kBeat = 64;

#ifndef FOLDSIM_MEM_PORTS
#error "FOLDSIM_MEM_PORTS comes from the generated foldsim_kernel.vh"
#endif
static_assert(FOLDSIM_MEM_PORTS >= 1 && FOLDSIM_MEM_PORTS <= 8, "the kernel has 1 to 8 memory ports");

// Device memory: sparse, zero where never written.
class memory_t {
public:
  void write(uint64_t addr, const uint8_t* data, uint64_t size, const uint8_t* strobe = nullptr) {
    for (uint64_t i = 0; i < size; ++i) {
      if (strobe == nullptr || strobe[i])
        page(addr + i)[(addr + i) % kPage] = data[i];
    }
  }
  void read(uint64_t addr, uint8_t* data, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
      auto it = pages_.find((addr + i) / kPage);
      data[i] = (it == pages_.end()) ? 0 : it->second[(addr + i) % kPage];
    }
  }

private:
  std::array<uint8_t, kPage>& page(uint64_t addr) {
    auto& p = pages_[addr / kPage];
    return p;
  }
  std::unordered_map<uint64_t, std::array<uint8_t, kPage>> pages_;
};

} // namespace

class foldsim_emul::Impl {
public:
  Impl()
      : mem_latency_(env_or("FOLDSIM_EMUL_MEM_LATENCY", 100)),
        mmio_cycles_(env_or("FOLDSIM_EMUL_MMIO_CYCLES", 250)) {}

  ~Impl() {
#ifdef VM_TRACE
    if (vcd_)
      vcd_->close();
#endif
    if (top_)
      top_->final();
    if (trace_ != nullptr)
      fclose(trace_);
    if (axi_trace_ != nullptr)
      fclose(axi_trace_);
    if (reg_trace_ != nullptr)
      fclose(reg_trace_);
  }

  int init() {
    // FOLDSIM_EMUL_TRACE=<file>: which classes finish a thread, each kernel
    // cycle, one byte per cycle of up to 8 classes.
    if (const char* path = getenv("FOLDSIM_EMUL_TRACE"))
      trace_ = fopen(path, "wb");
    // FOLDSIM_EMUL_AXI_TRACE=<file>: the kernel's memory port, one line per
    // request and response.
    if (const char* path = getenv("FOLDSIM_EMUL_AXI_TRACE"))
      axi_trace_ = fopen(path, "w");
    // FOLDSIM_EMUL_REG_TRACE=<file>: every register access of the host.
    if (const char* path = getenv("FOLDSIM_EMUL_REG_TRACE"))
      reg_trace_ = fopen(path, "w");
    context_ = std::make_unique<VerilatedContext>();
    top_ = std::make_unique<Vfoldsim_afu>(context_.get());
    bind_ports();
#ifdef VM_TRACE
    // FOLDSIM_EMUL_VCD=<file>: a waveform of the kernel, in a build with --trace,
    // from kernel cycle FOLDSIM_EMUL_VCD_FROM on, FOLDSIM_EMUL_VCD_DEPTH levels
    // of hierarchy deep (2: the kernel's channels).
    vcd_from_ = env_or("FOLDSIM_EMUL_VCD_FROM", 0);
    if (const char* path = getenv("FOLDSIM_EMUL_VCD")) {
      context_->traceEverOn(true);
      vcd_ = std::make_unique<VerilatedVcdC>();
      top_->trace(vcd_.get(), int(env_or("FOLDSIM_EMUL_VCD_DEPTH", 99)));
      vcd_->open(path);
    }
#endif
    top_->ap_rst_n = 0;
    idle_master();
    for (int i = 0; i < 16; ++i)
      tick();
    top_->ap_rst_n = 1;
    tick();
    return 0;
  }

  int reg_write(uint32_t offset, uint32_t value) {
    std::lock_guard<std::mutex> lock(mutex_);
    run(mmio_cycles_);
    top_->s_axi_control_awvalid = 1;
    top_->s_axi_control_awaddr = offset;
    top_->s_axi_control_wvalid = 1;
    top_->s_axi_control_wdata = value;
    top_->s_axi_control_wstrb = 0xf;
    top_->s_axi_control_bready = 1;
    bool aw = false, w = false, b = false;
    for (int i = 0; i < 1000 && !b; ++i) {
      top_->ap_clk = 0;
      top_->eval();
      bool aw_fire = top_->s_axi_control_awvalid && top_->s_axi_control_awready;
      bool w_fire = top_->s_axi_control_wvalid && top_->s_axi_control_wready;
      b = top_->s_axi_control_bvalid;
      edge();
      aw |= aw_fire;
      w |= w_fire;
      top_->s_axi_control_awvalid = !aw;
      top_->s_axi_control_wvalid = !w;
    }
    top_->s_axi_control_bready = 0;
    if (reg_trace_ != nullptr)
      fprintf(reg_trace_, "%lu W %03x %08x\n", (unsigned long)cycle_, offset, value);
    return b ? 0 : -1;
  }

  int reg_read(uint32_t offset, uint32_t* value) {
    std::lock_guard<std::mutex> lock(mutex_);
    run(mmio_cycles_);
    top_->s_axi_control_arvalid = 1;
    top_->s_axi_control_araddr = offset;
    top_->s_axi_control_rready = 1;
    bool r = false;
    for (int i = 0; i < 1000 && !r; ++i) {
      top_->ap_clk = 0;
      top_->eval();
      bool ar_fire = top_->s_axi_control_arvalid && top_->s_axi_control_arready;
      r = top_->s_axi_control_rvalid;
      if (r)
        *value = top_->s_axi_control_rdata;
      edge();
      if (ar_fire)
        top_->s_axi_control_arvalid = 0;
    }
    top_->s_axi_control_rready = 0;
    if (reg_trace_ != nullptr)
      fprintf(reg_trace_, "%lu R %03x %08x\n", (unsigned long)cycle_, offset, *value);
    return r ? 0 : -1;
  }

  int mem_write(uint64_t addr, const void* data, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_.write(addr, static_cast<const uint8_t*>(data), size);
    return 0;
  }

  int mem_read(uint64_t addr, void* data, uint64_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_.read(addr, static_cast<uint8_t*>(data), size);
    return 0;
  }

private:
  // A read, with the data memory held when it took the request: a write
  // that comes after it is not in its response.
  struct request_t {
    uint64_t at;
    std::array<uint8_t, kBeat> data;
  };

  // One of the kernel's device-memory masters: its signals, and what memory
  // has in flight on it.
  struct port_t {
    CData *awvalid, *awready, *wvalid, *wready, *arvalid, *arready;
    CData *rvalid, *rready, *rlast, *rid, *rresp, *bvalid, *bready, *bid, *bresp;
    QData *awaddr, *araddr, *wstrb;
    VlWide<16> *wdata, *rdata;
    std::deque<request_t> reads;
    std::deque<uint64_t> aw;
    std::deque<std::array<uint8_t, 2 * kBeat>> w;
    std::deque<uint64_t> writes;
  };

#define FOLDSIM_PORT(i)                                                                     \
  {                                                                                         \
    port_t& p = ports_[i];                                                                  \
    p.awvalid = &top_->m_axi_mem_##i##_awvalid; p.awready = &top_->m_axi_mem_##i##_awready; \
    p.wvalid = &top_->m_axi_mem_##i##_wvalid;   p.wready = &top_->m_axi_mem_##i##_wready;   \
    p.arvalid = &top_->m_axi_mem_##i##_arvalid; p.arready = &top_->m_axi_mem_##i##_arready; \
    p.rvalid = &top_->m_axi_mem_##i##_rvalid;   p.rready = &top_->m_axi_mem_##i##_rready;   \
    p.rlast = &top_->m_axi_mem_##i##_rlast;     p.rid = &top_->m_axi_mem_##i##_rid;         \
    p.rresp = &top_->m_axi_mem_##i##_rresp;     p.bvalid = &top_->m_axi_mem_##i##_bvalid;   \
    p.bready = &top_->m_axi_mem_##i##_bready;   p.bid = &top_->m_axi_mem_##i##_bid;         \
    p.bresp = &top_->m_axi_mem_##i##_bresp;     p.awaddr = &top_->m_axi_mem_##i##_awaddr;   \
    p.araddr = &top_->m_axi_mem_##i##_araddr;   p.wstrb = &top_->m_axi_mem_##i##_wstrb;     \
    p.wdata = &top_->m_axi_mem_##i##_wdata;     p.rdata = &top_->m_axi_mem_##i##_rdata;     \
  }

  void bind_ports() {
    FOLDSIM_PORT(0)
#if FOLDSIM_MEM_PORTS > 1
    FOLDSIM_PORT(1)
#endif
#if FOLDSIM_MEM_PORTS > 2
    FOLDSIM_PORT(2)
    FOLDSIM_PORT(3)
#endif
#if FOLDSIM_MEM_PORTS > 4
    FOLDSIM_PORT(4)
    FOLDSIM_PORT(5)
    FOLDSIM_PORT(6)
    FOLDSIM_PORT(7)
#endif
  }
#undef FOLDSIM_PORT

  void idle_master() {
    top_->s_axi_control_awvalid = 0;
    top_->s_axi_control_wvalid = 0;
    top_->s_axi_control_bready = 0;
    top_->s_axi_control_arvalid = 0;
    top_->s_axi_control_rready = 0;
  }

  void run(uint64_t cycles) {
    for (uint64_t i = 0; i < cycles; ++i)
      tick();
  }

  void tick() {
    top_->ap_clk = 0;
    top_->eval();
    edge();
  }

  // The memory's side of the kernel's AXI4 ports: every request accepted, and
  // answered in order on its port once its latency has passed.
  void drive_memory() {
    for (auto& p : ports_) {
      *p.awready = 1;
      *p.wready = 1;
      *p.arready = 1;
      bool r = !p.reads.empty() && p.reads.front().at <= cycle_;
      *p.rvalid = r;
      if (r)
        std::memcpy(p.rdata->data(), p.reads.front().data.data(), kBeat);
      *p.rlast = 1;
      *p.rid = 0;
      *p.rresp = 0;
      *p.bvalid = !p.writes.empty() && p.writes.front() <= cycle_;
      *p.bid = 0;
      *p.bresp = 0;
    }
  }

  // What a port did on the edge, as the kernel offered it before the edge.
  struct sample_t {
    bool ar, aw, w, r_fire, b_fire;
    uint64_t araddr, awaddr;
    uint8_t wdata[kBeat], strobe[kBeat];
  };

  // The rising edge: the handshakes as the kernel saw them before it, then
  // the edge itself.
  void edge() {
    drive_memory();
    top_->eval();
    std::array<sample_t, FOLDSIM_MEM_PORTS> smp;
    for (int k = 0; k < FOLDSIM_MEM_PORTS; ++k) {
      auto& p = ports_[k];
      auto& x = smp[k];
      x.ar = *p.arvalid;
      x.araddr = *p.araddr;
      x.aw = *p.awvalid;
      x.awaddr = *p.awaddr;
      x.w = *p.wvalid;
      if (x.w) {
        std::memcpy(x.wdata, p.wdata->data(), kBeat);
        uint64_t s = *p.wstrb;
        for (uint32_t i = 0; i < kBeat; ++i)
          x.strobe[i] = (s >> i) & 1;
      }
      x.r_fire = *p.rvalid && *p.rready;
      x.b_fire = *p.bvalid && *p.bready;
    }

    if (trace_ != nullptr)
      fputc(top_->rootp->foldsim_afu__DOT__profile_finish & 0xff, trace_);
    top_->ap_clk = 1;
    top_->eval();
#ifdef VM_TRACE
    if (vcd_ && cycle_ >= vcd_from_)
      vcd_->dump(cycle_);
#endif
    ++cycle_;

    // Each block of the target's memory belongs to one port, so the order
    // the ports are served in is not visible.
    for (int k = 0; k < FOLDSIM_MEM_PORTS; ++k) {
      auto& p = ports_[k];
      const auto& x = smp[k];
      if (axi_trace_ != nullptr) {
        // Port 0's lines keep the single-port format.
        const char* tag = (k == 0) ? "" : port_tag_[k].c_str();
        if (x.ar)
          fprintf(axi_trace_, "%lu %sAR %lx\n", (unsigned long)cycle_, tag, (unsigned long)x.araddr);
        if (x.aw)
          fprintf(axi_trace_, "%lu %sAW %lx\n", (unsigned long)cycle_, tag, (unsigned long)x.awaddr);
        if (x.w)
          fprintf(axi_trace_, "%lu %sW\n", (unsigned long)cycle_, tag);
        if (x.r_fire)
          fprintf(axi_trace_, "%lu %sR\n", (unsigned long)cycle_, tag);
        if (x.b_fire)
          fprintf(axi_trace_, "%lu %sB\n", (unsigned long)cycle_, tag);
      }
      if (x.ar) {
        request_t req;
        req.at = cycle_ + mem_latency_;
        memory_.read(x.araddr, req.data.data(), kBeat);
        p.reads.push_back(req);
      }
      if (x.r_fire)
        p.reads.pop_front();
      if (x.aw)
        p.aw.push_back(x.awaddr);
      if (x.w) {
        std::array<uint8_t, 2 * kBeat> beat;
        std::memcpy(beat.data(), x.wdata, kBeat);
        std::memcpy(beat.data() + kBeat, x.strobe, kBeat);
        p.w.push_back(beat);
      }
      while (!p.aw.empty() && !p.w.empty()) {
        memory_.write(p.aw.front(), p.w.front().data(), kBeat, p.w.front().data() + kBeat);
        p.aw.pop_front();
        p.w.pop_front();
        p.writes.push_back(cycle_ + mem_latency_);
      }
      if (x.b_fire)
        p.writes.pop_front();
    }
  }

  std::unique_ptr<VerilatedContext> context_;
  std::unique_ptr<Vfoldsim_afu> top_;
#ifdef VM_TRACE
  std::unique_ptr<VerilatedVcdC> vcd_;
  uint64_t vcd_from_ = 0;
#endif
  std::mutex mutex_;
  memory_t memory_;
  uint64_t cycle_ = 0;
  uint64_t mem_latency_, mmio_cycles_;
  FILE* trace_ = nullptr;
  FILE* axi_trace_ = nullptr;
  FILE* reg_trace_ = nullptr;
  std::array<port_t, FOLDSIM_MEM_PORTS> ports_;
  const std::array<std::string, 8> port_tag_ = {"", "1:", "2:", "3:", "4:", "5:", "6:", "7:"};
};

foldsim_emul::foldsim_emul() : impl_(new Impl()) {}

foldsim_emul::~foldsim_emul() { delete impl_; }

int foldsim_emul::init() { return impl_->init(); }

int foldsim_emul::reg_write(uint32_t offset, uint32_t value) {
  return impl_->reg_write(offset, value);
}

int foldsim_emul::reg_read(uint32_t offset, uint32_t* value) {
  return impl_->reg_read(offset, value);
}

int foldsim_emul::mem_write(uint64_t addr, const void* data, uint64_t size) {
  return impl_->mem_write(addr, data, size);
}

int foldsim_emul::mem_read(uint64_t addr, void* data, uint64_t size) {
  return impl_->mem_read(addr, data, size);
}
