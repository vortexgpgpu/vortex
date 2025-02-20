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

#include <stdint.h>

namespace vortex {

class DramSim {
public:
  typedef void (*ResponseCallback)(void *arg);

  DramSim(uint32_t num_channels, uint32_t channel_size, float clock_ratio);
  ~DramSim();

  void reset();

  void tick();

  // addr: per-channel block address
  void send_request(uint64_t addr, bool is_write, ResponseCallback response_cb, void* arg);

private:
	class Impl;
	Impl* impl_;
};

}