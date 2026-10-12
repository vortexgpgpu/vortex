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

#include "maxsim_emul.h"

#include "Vmaxsim_afu.h"
#include "Vmaxsim_afu___024root.h"
#include "verilated.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
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

class maxsim_emul::Impl {
public:
  Impl()
      : mem_latency_(env_or("MAXSIM_EMUL_MEM_LATENCY", 100)),
        mmio_cycles_(env_or("MAXSIM_EMUL_MMIO_CYCLES", 250)) {}

  ~Impl() {
    if (top_)
      top_->final();
    if (trace_ != nullptr)
      fclose(trace_);
  }

  int init() {
    // MAXSIM_EMUL_TRACE=<file>: which classes finish a thread, each kernel
    // cycle, one byte per cycle of up to 8 classes.
    if (const char* path = getenv("MAXSIM_EMUL_TRACE"))
      trace_ = fopen(path, "wb");
    context_ = std::make_unique<VerilatedContext>();
    top_ = std::make_unique<Vmaxsim_afu>(context_.get());
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
  struct request_t {
    uint64_t addr;
    uint64_t at;
  };

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

  // The memory's side of the kernel's AXI4 port: every request accepted, and
  // answered in order once its latency has passed.
  void drive_memory() {
    top_->m_axi_mem_0_awready = 1;
    top_->m_axi_mem_0_wready = 1;
    top_->m_axi_mem_0_arready = 1;
    bool r = !reads_.empty() && reads_.front().at <= cycle_;
    top_->m_axi_mem_0_rvalid = r;
    if (r) {
      uint8_t beat[kBeat];
      memory_.read(reads_.front().addr, beat, kBeat);
      std::memcpy(top_->m_axi_mem_0_rdata.data(), beat, kBeat);
    }
    top_->m_axi_mem_0_rlast = 1;
    top_->m_axi_mem_0_rid = 0;
    top_->m_axi_mem_0_rresp = 0;
    top_->m_axi_mem_0_bvalid = !writes_.empty() && writes_.front() <= cycle_;
    top_->m_axi_mem_0_bid = 0;
    top_->m_axi_mem_0_bresp = 0;
  }

  // The rising edge: the handshakes as the kernel saw them before it, then
  // the edge itself.
  void edge() {
    drive_memory();
    top_->eval();
    bool ar = top_->m_axi_mem_0_arvalid;
    uint64_t araddr = top_->m_axi_mem_0_araddr;
    bool aw = top_->m_axi_mem_0_awvalid;
    uint64_t awaddr = top_->m_axi_mem_0_awaddr;
    bool w = top_->m_axi_mem_0_wvalid;
    uint8_t wdata[kBeat], strobe[kBeat];
    if (w) {
      std::memcpy(wdata, top_->m_axi_mem_0_wdata.data(), kBeat);
      uint64_t s = top_->m_axi_mem_0_wstrb;
      for (uint32_t i = 0; i < kBeat; ++i)
        strobe[i] = (s >> i) & 1;
    }
    bool r_fire = top_->m_axi_mem_0_rvalid && top_->m_axi_mem_0_rready;
    bool b_fire = top_->m_axi_mem_0_bvalid && top_->m_axi_mem_0_bready;

    if (trace_ != nullptr)
      fputc(top_->rootp->maxsim_afu__DOT__profile_finish & 0xff, trace_);
    top_->ap_clk = 1;
    top_->eval();
    ++cycle_;

    if (ar)
      reads_.push_back({araddr, cycle_ + mem_latency_});
    if (r_fire)
      reads_.pop_front();
    if (aw)
      aw_.push_back(awaddr);
    if (w) {
      std::array<uint8_t, 2 * kBeat> beat;
      std::memcpy(beat.data(), wdata, kBeat);
      std::memcpy(beat.data() + kBeat, strobe, kBeat);
      w_.push_back(beat);
    }
    while (!aw_.empty() && !w_.empty()) {
      memory_.write(aw_.front(), w_.front().data(), kBeat, w_.front().data() + kBeat);
      aw_.pop_front();
      w_.pop_front();
      writes_.push_back(cycle_ + mem_latency_);
    }
    if (b_fire)
      writes_.pop_front();
  }

  std::unique_ptr<VerilatedContext> context_;
  std::unique_ptr<Vmaxsim_afu> top_;
  std::mutex mutex_;
  memory_t memory_;
  uint64_t cycle_ = 0;
  uint64_t mem_latency_, mmio_cycles_;
  FILE* trace_ = nullptr;
  std::deque<request_t> reads_;
  std::deque<uint64_t> aw_;
  std::deque<std::array<uint8_t, 2 * kBeat>> w_;
  std::deque<uint64_t> writes_;
};

maxsim_emul::maxsim_emul() : impl_(new Impl()) {}

maxsim_emul::~maxsim_emul() { delete impl_; }

int maxsim_emul::init() { return impl_->init(); }

int maxsim_emul::reg_write(uint32_t offset, uint32_t value) {
  return impl_->reg_write(offset, value);
}

int maxsim_emul::reg_read(uint32_t offset, uint32_t* value) {
  return impl_->reg_read(offset, value);
}

int maxsim_emul::mem_write(uint64_t addr, const void* data, uint64_t size) {
  return impl_->mem_write(addr, data, size);
}

int maxsim_emul::mem_read(uint64_t addr, void* data, uint64_t size) {
  return impl_->mem_read(addr, data, size);
}
