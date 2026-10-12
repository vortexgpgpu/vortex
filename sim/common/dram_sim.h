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

#pragma once

#include <cstdint>

namespace vortex {

// Device memory behind the platform's memory banks: one DRAM channel (an
// independent controller) per bank, timed by Ramulator.
class DramSim {
public:
  typedef bool (*ResponseCallback)(void *arg);

  // The JEDEC standards Ramulator models, each at its own organization and
  // timing; the data rate picks the speed.
  enum class Type {
    HBM2,   // HBM2/HBM2E: 128-bit channels of two 64-bit pseudo-channels, 32 B bursts
    HBM3,   // HBM3, a 128-bit channel per pair of 64-bit channels, 32 B bursts
    GDDR6,  // GDDR6 x16 channels, 32 B bursts, 2 KB pages
  };

  struct Config {
    Type     type;
    uint32_t num_channels;  // need not be a power of two
    uint32_t block_size;    // bytes per request
    uint32_t core_clock;    // MHz
    uint32_t data_rate;     // MT/s per pin
    uint32_t latency;       // fixed read latency added in core cycles (PHY, controller, NoC)
    uint32_t queue_size;    // requests a channel holds in flight; 0 = unbounded
  };

  // The memory configured in VX_config.toml.
  static Config platform_config();

  DramSim(const Config& config);
  ~DramSim();

  void reset();

  void tick();

  // False while the channel holds queue_size requests.
  bool can_accept(uint32_t channel) const;

  // addr: byte address within the channel.
  void send_request(uint32_t channel, uint64_t addr, bool is_write, ResponseCallback response_cb, void* arg);

private:
  class Impl;
  Impl* impl_;
};

}
