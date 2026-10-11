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

#include <stdexcept>
#include <vector>

#include "base/base.h"
#include "dram_controller/controller.h"
#include "dram_controller/refresh.h"

namespace Ramulator {

// All-bank refresh every nREFI, one REFab per channel, the channels staggered
// across the interval as independent controllers are. Ramulator's AllBank
// manager refreshes per rank, and HBM2 and GDDR6 have no rank level, so with it
// these devices never refresh.
class ChannelRefresh : public IRefreshManager, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, ChannelRefresh, "ChannelRefresh", "All-bank refresh of a rank-less channel.")
private:
  Clk_t clk_ = 0;
  IDRAMController* ctrl_ = nullptr;
  int num_levels_ = -1;
  int nrefi_ = -1;
  int req_id_ = -1;
  Clk_t next_refresh_ = -1;

public:
  void init() override {
    ctrl_ = cast_parent<IDRAMController>();
  }

  void setup(IFrontEnd* /*frontend*/, IMemorySystem* /*memory_system*/) override {
    auto dram = ctrl_->m_dram;
    num_levels_ = int(dram->m_levels.size());
    nrefi_ = dram->m_timing_vals("nREFI");
    req_id_ = dram->m_requests("all-bank-refresh");
    int num_channels = dram->get_level_size("channel");
    next_refresh_ = nrefi_ + Clk_t(ctrl_->m_channel_id) * nrefi_ / num_channels;
  }

  void tick() override {
    if (++clk_ != next_refresh_) {
      return;
    }
    next_refresh_ += nrefi_;
    std::vector<int> addr_vec(num_levels_, -1);
    addr_vec[0] = ctrl_->m_channel_id;
    Request req(addr_vec, req_id_);
    if (!ctrl_->priority_send(req)) {
      throw std::runtime_error("Failed to send refresh!");
    }
  }
};

}
