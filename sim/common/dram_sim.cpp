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

#include "dram_sim.h"
#include "util.h"
#include <VX_config.h>
#include <deque>
#include <fstream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNUSED_PARAMETER
DISABLE_WARNING_MISSING_FIELD_INITIALIZERS
#include <base/base.h>
#include <base/request.h>
#include <base/config.h>
#include <frontend/frontend.h>
#include <memory_system/memory_system.h>
DISABLE_WARNING_POP

using namespace vortex;

namespace {

// IPOLY(5) bank hash over 16 banks (Rau, "Pseudo-randomly interleaved memory",
// ISCA 1991), GPGPU-Sim's -dram_bnk_indexing_policy 2: the bank XORed with a
// function of the row, so streams whose bases differ by a power of two land in
// different banks instead of closing each other's rows.
uint32_t ipoly16(uint64_t row, uint32_t bank) {
  auto bit = [row](int i) { return uint32_t(row >> i) & 1u; };
  uint32_t i0 = bit(11) ^ bit(10) ^ bit(9) ^ bit(8) ^ bit(6) ^ bit(4) ^ bit(3) ^ bit(0);
  uint32_t i1 = bit(12) ^ bit(8) ^ bit(7) ^ bit(6) ^ bit(5) ^ bit(3) ^ bit(1) ^ bit(0);
  uint32_t i2 = bit(9) ^ bit(8) ^ bit(7) ^ bit(6) ^ bit(4) ^ bit(2) ^ bit(1);
  uint32_t i3 = bit(10) ^ bit(9) ^ bit(8) ^ bit(7) ^ bit(5) ^ bit(3) ^ bit(2);
  return bank ^ (i0 | (i1 << 1) | (i2 << 2) | (i3 << 3));
}

}

class DramSim::Impl {
private:
  // One request (a block); it completes when all its transactions have.
  struct request_t {
    ResponseCallback callback;
    void* arg;
    uint32_t pending;
  };

  struct mem_tx_t {
    uint64_t addr;
    bool is_write;
    std::shared_ptr<request_t> request;
  };

  struct mem_rsp_t {
    std::shared_ptr<request_t> request;
    uint64_t ready_cycle;
  };

  // GDDR6 holds a fixed 16 Gb per device, so its row count shrinks as the
  // channel count grows (Ramulator only knows the refresh timing of 4/8/16 Gb
  // devices); rows alias above that capacity, which only affects row hits.
  static constexpr uint32_t gddr6_density_mb_ = 16384;

  Config config_;
  uint32_t tx_size_;
  uint32_t row_mask_;
  Ramulator::IFrontEnd* ramulator_frontend_;
  Ramulator::IMemorySystem* ramulator_memorysystem_;
  uint32_t ch_bits_;
  uint64_t cycle_;
  // DRAM ticks owed, in units of 1/core_clock: each core cycle adds
  // dram_clock, each DRAM tick takes core_clock.
  uint64_t clock_credit_;
  // A channel accepts one transaction per DRAM cycle and returns its
  // responses independently of the others; write acknowledgements do not wait
  // behind read data.
  std::vector<std::queue<mem_tx_t>> pending_txs_;
  std::vector<std::deque<mem_rsp_t>> read_rsps_;
  std::vector<std::deque<mem_rsp_t>> write_rsps_;
  std::vector<uint32_t> inflight_;

  // Ramulator address of a transaction: the channel in the low bits, then the
  // fields in the order the RoBaRaCoCh mapper slices them.
  uint64_t dram_addr(uint32_t channel, uint64_t addr) const {
    uint64_t t;
    if (config_.type == Type::GDDR6) {
      // GPGPU-Sim's RTX 3070 layout (-gpgpu_mem_addr_mapping): 32 B bursts,
      // two column bits, one bank bit, four column bits, three bank bits,
      // row; the low bank bit selects the bank group.
      uint64_t col  = ((addr >> 5) & 0x3) | (((addr >> 8) & 0xf) << 2);
      uint64_t row  = (addr >> 15) & row_mask_;
      uint32_t bank = ipoly16(row, uint32_t(((addr >> 7) & 0x1) | (((addr >> 12) & 0x7) << 1)));
      t = row;
      t = (t << 2) | (bank >> 2);
      t = (t << 2) | (bank & 0x3);
      t = (t << 6) | col;
    } else {
      // The A100 HBM interleave: a stream alternates column, pseudo-channel
      // and column again at 64 B granules, then changes bank group every 4 KB
      // and bank every 16 KB, with 2 KB rows. HBM2_A100 uses 2 of the 4 banks
      // of a group (16 per channel), as GPGPU-Sim models the A100; HBM2 all 4.
      uint32_t bank_bits = (config_.type == Type::HBM2_A100) ? 1 : 2;
      uint64_t col  = (((addr >> 8) & 0xf) << 1) | ((addr >> 6) & 0x1);
      uint64_t pc   = (addr >> 7) & 0x1;
      uint64_t bg   = (addr >> 12) & 0x3;
      uint64_t bank = (addr >> 14) & ((1u << bank_bits) - 1);
      uint64_t row  = (addr >> (14 + bank_bits)) & row_mask_;
      t = row;
      t = (t << 2) | bank;
      t = (t << 2) | bg;
      t = (t << 1) | pc;
      t = (t << 5) | col;
    }
    t = (t << ch_bits_) | channel;
    return t * tx_size_;
  }

  static void set_hbm2_device(YAML::Node& dram, const Config& config, uint32_t pow2_channels) {
    dram["impl"] = "HBM2";
    dram["org"]["preset"] = "HBM2_8Gb";
    dram["org"]["density"] = 8192;
    dram["org"]["channel"] = pow2_channels;
    // A transaction is one burst: 2 x 256-bit prefetch = 64 B.
    dram["org"]["channel_width"] = 256;
    dram["timing"]["preset"] = "HBM2_2Gbps";
    if (config.type == Type::HBM2_A100) {
      // GPGPU-Sim SM80_A100 HBM timing in DRAM cycles. One 64 B transaction
      // is two of its 32 B atoms, so the same-bank-group spacing is twice its
      // tCCDL.
      auto timing = dram["timing"];
      timing["nCL"]    = 22;
      timing["nRCDRD"] = 22;
      timing["nRCDWR"] = 22;
      timing["nRP"]    = 22;
      timing["nRAS"]   = 50;
      timing["nRC"]    = 72;
      timing["nWR"]    = 19;
      timing["nRTPS"]  = 7;
      timing["nRTPL"]  = 7;
      timing["nCWL"]   = 4;
      timing["nCCDS"]  = 2;
      timing["nCCDL"]  = 8;
      timing["nRRDS"]  = 7;
      timing["nRRDL"]  = 7;
    }
  }

  static void set_gddr6_device(YAML::Node& dram, const Config& config, uint32_t pow2_channels) {
    // x16 channels with 2 KB pages: 512 columns of 16 bits, 4 bank groups of 4
    // banks; a 32 B transaction is a 16-beat burst (8 prefetched columns over a
    // 32-bit internal channel width).
    dram["impl"] = "GDDR6";
    dram["org"]["dq"] = 16;
    dram["org"]["channel"] = pow2_channels;
    dram["org"]["bankgroup"] = 4;
    dram["org"]["bank"] = 4;
    dram["org"]["column"] = 512;
    dram["org"]["row"] = uint32_t((uint64_t(gddr6_density_mb_) << 20) / (uint64_t(pow2_channels) * 4 * 4 * 512 * 16));
    dram["org"]["density"] = gddr6_density_mb_;
    dram["org"]["channel_width"] = 32;
    // The transfer rate is twice the data clock. Bus timings are in clocks of
    // the 32 B burst; array timings are GPGPU-Sim's RTX 3070 values
    // (-gpgpu_dram_timing_opt) held in nanoseconds; refresh is JEDEC GDDR6.
    auto timing = dram["timing"];
    timing["rate"]   = 2 * config.dram_clock;
    timing["nBL"]    = 8;
    timing["nCCDS"]  = 8;
    timing["nCCDL"]  = 12;
    timing["tCL"]    = 6.8f;
    timing["tRCDRD"] = 6.8f;
    timing["tRCDWD"] = 6.8f;
    timing["tRP"]    = 6.8f;
    timing["tRAS"]   = 15.7f;
    timing["tRC"]    = 22.3f;
    timing["tWR"]    = 6.8f;
    timing["tRTP"]   = 1.2f;
    timing["tCWL"]   = 2.3f;
    timing["tRRDS"]  = 3.5f;
    timing["tRRDL"]  = 3.5f;
    timing["tWTRS"]  = 2.8f;
    timing["tWTRL"]  = 2.8f;
    timing["tFAW"]   = 16.0f;
    timing["tRFC"]   = 120.0f;
    timing["tRFCpb"] = 60.0f;
    timing["tREFI"]  = 1900.0f;
    timing["tRREFD"] = 8.0f;
  }

  // Retires the ready head responses of `rsps`; a request completes with its
  // last transaction.
  void drain(std::deque<mem_rsp_t>& rsps, uint32_t ch) {
    while (!rsps.empty()) {
      auto& rsp = rsps.front();
      if (rsp.ready_cycle > cycle_) {
        break;
      }
      auto& request = *rsp.request;
      if (request.pending == 1 && request.callback && !request.callback(request.arg)) {
        break;
      }
      if (--request.pending == 0) {
        --inflight_[ch];
      }
      rsps.pop_front();
    }
  }

  void handle_pending_responses() {
    for (uint32_t ch = 0; ch < config_.num_channels; ++ch) {
      this->drain(read_rsps_[ch], ch);
      this->drain(write_rsps_[ch], ch);
    }
  }

  void handle_pending_requests() {
    for (uint32_t ch = 0; ch < config_.num_channels; ++ch) {
      auto& txs = pending_txs_[ch];
      if (txs.empty()) {
        continue;
      }
      auto& tx = txs.front();
      auto req_type = tx.is_write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read;
      std::function<void(Ramulator::Request&)> callback = nullptr;
      if (!tx.is_write) {
        callback = [this, ch, request = tx.request](Ramulator::Request& /*dram_req*/) {
          this->read_rsps_[ch].push_back({request, this->cycle_ + 1 + this->config_.latency});
        };
      }
      if (ramulator_frontend_->receive_external_requests(req_type, tx.addr, 0, callback)) {
        // Ramulator does not handle write responses, so we handle the callback ourselves.
        if (tx.is_write) {
          write_rsps_[ch].push_back({tx.request, cycle_});
        }
        txs.pop();
      }
    }
  }

public:
  Impl(const Config& config) : config_(config) {
    uint32_t pow2_channels = 1;
    ch_bits_ = 0;
    while (pow2_channels < config.num_channels) {
      pow2_channels <<= 1;
      ++ch_bits_;
    }

    YAML::Node dram;
    if (config.type == Type::GDDR6) {
      set_gddr6_device(dram, config, pow2_channels);
      tx_size_ = 32;
      row_mask_ = dram["org"]["row"].as<uint32_t>() - 1;
    } else {
      set_hbm2_device(dram, config, pow2_channels);
      tx_size_ = 64;
      row_mask_ = 0x7fff;
    }

    YAML::Node dram_config;
    dram_config["Frontend"]["impl"] = "GEM5";
    dram_config["MemorySystem"]["impl"] = "GenericDRAM";
    dram_config["MemorySystem"]["clock_ratio"] = 1;
    dram_config["MemorySystem"]["DRAM"] = dram;
    dram_config["MemorySystem"]["Controller"]["impl"] = "Generic";
    dram_config["MemorySystem"]["Controller"]["Scheduler"]["impl"] = "FRFCFS";
    // libramulator-vortex: Ramulator's AllBank refresh never fires on rank-less HBM2/GDDR6.
    dram_config["MemorySystem"]["Controller"]["RefreshManager"]["impl"] = "ChannelRefresh";
    dram_config["MemorySystem"]["Controller"]["RowPolicy"]["impl"] = "OpenRowPolicy";
    // Per-command DRAM trace recording is a debugging aid only: it writes an
    // entry for every DRAM command and dominates simulation time (100s of MB
    // of I/O for even tiny workloads). Keep it opt-in via the DRAM_TRACE env
    // var so default simx/rtlsim runs stay fast.
    if (const char* trace_path = getenv("DRAM_TRACE")) {
      YAML::Node draw_plugin;
      draw_plugin["ControllerPlugin"]["impl"] = "TraceRecorder";
      draw_plugin["ControllerPlugin"]["path"] = trace_path[0] ? trace_path : "./trace/ramulator.log";
      dram_config["MemorySystem"]["Controller"]["plugins"].push_back(draw_plugin);
    }
    dram_config["MemorySystem"]["AddrMapper"]["impl"] = "RoBaRaCoCh";

    ramulator_frontend_ = Ramulator::Factory::create_frontend(dram_config);
    ramulator_memorysystem_ = Ramulator::Factory::create_memory_system(dram_config);
    ramulator_frontend_->connect_memory_system(ramulator_memorysystem_);
    ramulator_memorysystem_->connect_frontend(ramulator_frontend_);

    pending_txs_.resize(config.num_channels);
    read_rsps_.resize(config.num_channels);
    write_rsps_.resize(config.num_channels);
    inflight_.resize(config.num_channels, 0);
    this->reset();
  }

  ~Impl() {
    std::ofstream nullstream("ramulator.stats.log");
    auto original_buf = std::cout.rdbuf();
    std::cout.rdbuf(nullstream.rdbuf());
    ramulator_frontend_->finalize();
    ramulator_memorysystem_->finalize();
    std::cout.rdbuf(original_buf);
  }

  void reset() {
    cycle_ = 0;
    clock_credit_ = 0;
    for (auto& q : pending_txs_) {
      q = std::queue<mem_tx_t>();
    }
    for (auto& q : read_rsps_) {
      q.clear();
    }
    for (auto& q : write_rsps_) {
      q.clear();
    }
    std::fill(inflight_.begin(), inflight_.end(), 0);
  }

  void tick() {
    ++cycle_;
    clock_credit_ += config_.dram_clock;
    this->handle_pending_responses();
    while (clock_credit_ >= config_.core_clock) {
      this->handle_pending_requests();
      ramulator_memorysystem_->tick();
      clock_credit_ -= config_.core_clock;
    }
  }

  bool can_accept(uint32_t channel) const {
    return (config_.window == 0) || (inflight_.at(channel) < config_.window);
  }

  void send_request(uint32_t channel, uint64_t addr, bool is_write, ResponseCallback response_cb, void* arg) {
    // A block wider than a burst is several transactions.
    uint32_t n = (config_.block_size > tx_size_) ? (config_.block_size / tx_size_) : 1;
    uint64_t base = addr & ~uint64_t(config_.block_size - 1);
    auto request = std::make_shared<request_t>(request_t{response_cb, arg, n});
    auto& txs = pending_txs_.at(channel);
    for (uint32_t i = 0; i < n; ++i) {
      txs.push({dram_addr(channel, base + i * tx_size_), is_write, request});
    }
    ++inflight_[channel];
  }
};

///////////////////////////////////////////////////////////////////////////////

#define DRAM_STR_(x) #x
#define DRAM_STR(x) DRAM_STR_(x)

DramSim::Config DramSim::platform_config() {
  // The enum's value macro, not its per-value companion: the simulators get
  // raw CONFIGS, where -DVX_CFG_PLATFORM_MEMORY_TYPE=<value> defines no
  // companion.
  const std::string type = DRAM_STR(VX_CFG_PLATFORM_MEMORY_TYPE);
  Config config;
  if (type == "HBM2") {
    config.type = Type::HBM2;
  } else if (type == "HBM2_A100") {
    config.type = Type::HBM2_A100;
  } else if (type == "GDDR6") {
    config.type = Type::GDDR6;
  } else {
    throw std::invalid_argument("dram_sim: unknown VX_CFG_PLATFORM_MEMORY_TYPE " + type);
  }
  config.num_channels = VX_CFG_PLATFORM_MEMORY_NUM_BANKS;
  config.block_size   = VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
  config.core_clock   = VX_CFG_PLATFORM_CLOCK_RATE;
  config.dram_clock   = VX_CFG_PLATFORM_MEMORY_CLOCK_RATE;
  config.latency      = VX_CFG_PLATFORM_MEMORY_LATENCY;
  config.window       = VX_CFG_PLATFORM_MEMORY_WINDOW;
  return config;
}

DramSim::DramSim(const Config& config)
  : impl_(new Impl(config))
{}

DramSim::~DramSim() {
  delete impl_;
}

void DramSim::reset() {
  impl_->reset();
}

void DramSim::tick() {
  impl_->tick();
}

bool DramSim::can_accept(uint32_t channel) const {
  return impl_->can_accept(channel);
}

void DramSim::send_request(uint32_t channel, uint64_t addr, bool is_write, ResponseCallback callback, void* arg) {
  impl_->send_request(channel, addr, is_write, callback, arg);
}
