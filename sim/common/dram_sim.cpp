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
#include "dram_bank_model.h"
#include "util.h"
#include <dram_timing.h>
#include <cstdlib>
#include <cstring>
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

template <size_t B, size_t A>
void set_timing(YAML::Node timing, uint32_t data_rate, const dram_timing::clock_timing_t (&bus)[B],
                const dram_timing::ns_timing_t (&array)[A]) {
  timing["rate"] = data_rate;
  for (auto& t : bus) {
    timing[t.name] = t.clocks;
  }
  for (auto& t : array) {
    timing[t.name] = t.ns;
  }
}

}

// A DRAM backend: Ramulator, or the bank model (sim/common/dram_bank_model.h)
// with VORTEX_DRAM_MODEL=bank -- a choice of simulator, not of machine.
class DramSim::Impl {
public:
  virtual ~Impl() {}
  virtual void reset() = 0;
  virtual void tick() = 0;
  virtual bool can_accept(uint32_t channel) const = 0;
  virtual void send_request(uint32_t channel, uint64_t addr, bool is_write, ResponseCallback cb, void* arg) = 0;
};

namespace {

class BankDram : public DramSim::Impl {
public:
  explicit BankDram(const DramSim::Config& c)
    : sim_(DramBankSim::Config{c.type, c.num_channels, c.block_size, c.core_clock, c.data_rate,
                               c.latency, c.queue_size}) {}
  void reset() override { sim_.reset(); }
  void tick() override { sim_.tick(); }
  bool can_accept(uint32_t channel) const override { return sim_.can_accept(channel); }
  void send_request(uint32_t channel, uint64_t addr, bool is_write, DramSim::ResponseCallback cb,
                    void* arg) override {
    sim_.send_request(channel, addr, is_write, cb, arg);
  }

private:
  DramBankSim sim_;
};

}

class RamulatorDram : public DramSim::Impl {
private:
  using Config = DramSim::Config;
  using Type = DramSim::Type;
  using ResponseCallback = DramSim::ResponseCallback;

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

  // An independent controller and its device. It accepts one transaction per
  // DRAM cycle and returns its responses independently of the other channels;
  // write acknowledgements do not wait behind read data.
  struct channel_t {
    Ramulator::IFrontEnd* frontend;
    Ramulator::IMemorySystem* memory_system;
    std::queue<mem_tx_t> pending_txs;
    std::deque<mem_rsp_t> read_rsps;
    std::deque<mem_rsp_t> write_rsps;
    uint32_t inflight;
  };

  static constexpr uint32_t tx_size_ = dram_timing::kBurstSize;

  // Addresses above a channel's 8 Gb alias, which only affects row hits.
  static constexpr uint64_t channel_size_ = dram_timing::kChannelSize;

  Config config_;
  uint32_t row_visit_bits_;  // bursts per row visit (minimalist open page, 128 B)
  uint32_t bank_bits_;       // pseudo-channel, bank group and bank
  uint32_t column_bits_;     // bursts per row
  std::vector<channel_t> channels_;
  uint64_t cycle_;
  // DRAM ticks owed, in units of 1/(2 x core_clock): each core cycle adds the
  // data rate, each DRAM tick (half the data rate) takes 2 x core_clock.
  uint64_t clock_credit_;

  static void set_hbm_device(YAML::Node dram, const Config& config) {
    // Ramulator's 8 Gb organization: 2 pseudo-channels x 4 bank groups x 4
    // banks, 1 KB rows; a 128-bit channel width with the 2n prefetch makes a
    // transaction one 32 B pseudo-channel burst.
    dram["impl"] = (config.type == Type::HBM3) ? "HBM3" : "HBM2";
    dram["org"]["preset"] = (config.type == Type::HBM3) ? "HBM3_8Gb" : "HBM2_8Gb";
    dram["org"]["density"] = 8192;
    dram["org"]["channel"] = 1;
    dram["org"]["channel_width"] = 128;
    set_timing(dram["timing"], config.data_rate, dram_timing::hbm_bus_timing, dram_timing::hbm_array_timing);
  }

  static void set_gddr6_device(YAML::Node dram, const Config& config) {
    // One x16 channel of a 16 Gb device (8 Gb), 4 bank groups of 4 banks,
    // 2 KB pages. A 32 B burst is 16n prefetch on 16 pins; Ramulator's GDDR6
    // prefetches 8 columns, so its columns are 32 bits.
    dram["impl"] = "GDDR6";
    dram["org"]["dq"] = 32;
    dram["org"]["channel"] = 1;
    dram["org"]["bankgroup"] = 4;
    dram["org"]["bank"] = 4;
    dram["org"]["row"] = 1 << 15;
    dram["org"]["column"] = 512;
    dram["org"]["density"] = 8192;
    dram["org"]["channel_width"] = 32;
    set_timing(dram["timing"], config.data_rate, dram_timing::gddr6_bus_timing, dram_timing::gddr6_array_timing);
  }

  YAML::Node channel_config(uint32_t ch) const {
    YAML::Node config;
    config["Frontend"]["impl"] = "GEM5";
    auto memory_system = config["MemorySystem"];
    memory_system["impl"] = "GenericDRAM";
    memory_system["clock_ratio"] = 1;
    if (config_.type == Type::GDDR6) {
      set_gddr6_device(memory_system["DRAM"], config_);
    } else {
      set_hbm_device(memory_system["DRAM"], config_);
    }
    auto controller = memory_system["Controller"];
    controller["impl"] = "Generic";
    controller["Scheduler"]["impl"] = "FRFCFS";
    // No refresh is modelled: Ramulator's refresh manager refreshes per rank
    // and these devices have no rank level. GPU controllers hide refresh with
    // per-bank refresh, for which Ramulator has no manager (and its HBM REFsb
    // lacks the bank's tRFCsb busy time); all-bank refresh would cost every
    // channel ~9% of its time instead.
    controller["RefreshManager"]["impl"] = "AllBank";
    controller["RowPolicy"]["impl"] = "OpenRowPolicy";
    // Per-command DRAM trace recording is a debugging aid only: it writes an
    // entry for every DRAM command and dominates simulation time (100s of MB
    // of I/O for even tiny workloads). Keep it opt-in via the DRAM_TRACE env
    // var so default simx/rtlsim runs stay fast.
    if (const char* trace_path = getenv("DRAM_TRACE")) {
      std::string path = trace_path[0] ? trace_path : "./trace/ramulator.log";
      YAML::Node trace_plugin;
      trace_plugin["ControllerPlugin"]["impl"] = "TraceRecorder";
      trace_plugin["ControllerPlugin"]["path"] = path + "." + std::to_string(ch);
      controller["plugins"].push_back(trace_plugin);
    }
    // Fields in the order dram_addr lays them out.
    memory_system["AddrMapper"]["impl"] = "RoBaRaCoCh";
    return config;
  }

  // Ramulator address of a transaction. A row visit is four bursts, then the
  // next pseudo-channel, bank group and bank (minimalist open page), and the
  // bank is XORed with the row folded to the bank's width (permutation-based
  // interleaving), so streams a power of two apart open rows in different
  // banks.
  uint64_t dram_addr(uint64_t addr) const {
    uint64_t tx = (addr & (channel_size_ - 1)) / tx_size_;
    uint64_t visit = tx & ((1u << row_visit_bits_) - 1);
    tx >>= row_visit_bits_;
    uint64_t bank = tx & ((1u << bank_bits_) - 1);
    tx >>= bank_bits_;
    uint64_t column = tx & ((1u << (column_bits_ - row_visit_bits_)) - 1);
    uint64_t row = tx >> (column_bits_ - row_visit_bits_);
    for (uint64_t r = row; r != 0; r >>= bank_bits_) {
      bank ^= r & ((1u << bank_bits_) - 1);
    }
    uint64_t t = (row << bank_bits_) | bank;
    t = (t << column_bits_) | (column << row_visit_bits_) | visit;
    return t * tx_size_;
  }

  // Retires the ready head responses of `rsps`; a request completes with its
  // last transaction.
  void drain(std::deque<mem_rsp_t>& rsps, channel_t& channel) {
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
        --channel.inflight;
      }
      rsps.pop_front();
    }
  }

  void issue(channel_t& channel) {
    if (channel.pending_txs.empty()) {
      return;
    }
    auto& tx = channel.pending_txs.front();
    auto req_type = tx.is_write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read;
    std::function<void(Ramulator::Request&)> callback = nullptr;
    if (!tx.is_write) {
      callback = [this, &channel, request = tx.request](Ramulator::Request& /*dram_req*/) {
        channel.read_rsps.push_back({request, this->cycle_ + 1 + this->config_.latency});
      };
    }
    if (channel.frontend->receive_external_requests(req_type, tx.addr, 0, callback)) {
      // Ramulator does not handle write responses, so we handle the callback ourselves.
      if (tx.is_write) {
        channel.write_rsps.push_back({tx.request, cycle_});
      }
      channel.pending_txs.pop();
    }
  }

public:
  RamulatorDram(const Config& config)
    : config_(config)
    , channels_(config.num_channels) {
    auto geometry = dram_timing::geometry(config.type);
    row_visit_bits_ = geometry.log2_visit;
    bank_bits_ = geometry.bank_bits();
    column_bits_ = geometry.log2_row_bursts;
    for (uint32_t ch = 0; ch < config.num_channels; ++ch) {
      auto ramulator_config = this->channel_config(ch);
      auto& channel = channels_[ch];
      channel.frontend = Ramulator::Factory::create_frontend(ramulator_config);
      channel.memory_system = Ramulator::Factory::create_memory_system(ramulator_config);
      channel.frontend->connect_memory_system(channel.memory_system);
      channel.memory_system->connect_frontend(channel.frontend);
    }
    this->reset();
  }

  ~RamulatorDram() override {
    std::ofstream nullstream("ramulator.stats.log");
    auto original_buf = std::cout.rdbuf();
    std::cout.rdbuf(nullstream.rdbuf());
    for (auto& channel : channels_) {
      channel.frontend->finalize();
      channel.memory_system->finalize();
    }
    std::cout.rdbuf(original_buf);
  }

  void reset() override {
    cycle_ = 0;
    clock_credit_ = 0;
    for (auto& channel : channels_) {
      channel.pending_txs = std::queue<mem_tx_t>();
      channel.read_rsps.clear();
      channel.write_rsps.clear();
      channel.inflight = 0;
    }
  }

  void tick() override {
    ++cycle_;
    clock_credit_ += config_.data_rate;
    for (auto& channel : channels_) {
      this->drain(channel.read_rsps, channel);
      this->drain(channel.write_rsps, channel);
    }
    while (clock_credit_ >= 2 * config_.core_clock) {
      for (auto& channel : channels_) {
        this->issue(channel);
        channel.memory_system->tick();
      }
      clock_credit_ -= 2 * config_.core_clock;
    }
  }

  bool can_accept(uint32_t channel) const override {
    return (config_.queue_size == 0) || (channels_.at(channel).inflight < config_.queue_size);
  }

  void send_request(uint32_t channel, uint64_t addr, bool is_write, ResponseCallback response_cb, void* arg) override {
    // A block wider than a burst is several transactions.
    uint32_t n = (config_.block_size > tx_size_) ? (config_.block_size / tx_size_) : 1;
    uint64_t base = addr & ~uint64_t(config_.block_size - 1);
    auto request = std::make_shared<request_t>(request_t{response_cb, arg, n});
    auto& ch = channels_.at(channel);
    for (uint32_t i = 0; i < n; ++i) {
      ch.pending_txs.push({dram_addr(base + i * tx_size_), is_write, request});
    }
    ++ch.inflight;
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
  try {
    config.type = dram_timing::parse_standard(type);
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("dram_sim: unknown VX_CFG_PLATFORM_MEMORY_TYPE " + type);
  }
  config.num_channels = VX_CFG_PLATFORM_MEMORY_NUM_BANKS;
  config.block_size   = VX_CFG_PLATFORM_MEMORY_DATA_SIZE;
  config.core_clock   = VX_CFG_PLATFORM_CLOCK_RATE;
  config.data_rate    = VX_CFG_PLATFORM_MEMORY_DATA_RATE;
  config.latency      = VX_CFG_PLATFORM_MEMORY_LATENCY;
  config.queue_size   = VX_CFG_PLATFORM_MEMORY_QUEUE_SIZE;
  return config;
}

static DramSim::Impl* make_backend(const DramSim::Config& config) {
  const char* model = getenv("VORTEX_DRAM_MODEL");
  if (model == nullptr || model[0] == '\0' || !strcmp(model, "ramulator")) {
    return new RamulatorDram(config);
  }
  if (!strcmp(model, "bank")) {
    return new BankDram(config);
  }
  throw std::invalid_argument(std::string("dram_sim: unknown VORTEX_DRAM_MODEL ") + model);
}

DramSim::DramSim(const Config& config)
  : impl_(make_backend(config))
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
