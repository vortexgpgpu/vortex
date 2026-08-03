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

#include <stdint.h>

namespace vortex {

// Drives a Vortex target compiled by Golden Gate, either as a metasimulation
// or on an FPGA. The two differ only in which platform layer the simulator was
// linked against, so this interface is identical for both.
//
// The simulator owns its own run loop and expects to be entered from main(),
// so it runs on a private thread and these calls are requests handed to it.
// Every call blocks until the simulator has serviced it, which keeps the target
// deterministic: nothing advances except when this interface asks it to.
class firesim_sim {
public:

  firesim_sim();
  virtual ~firesim_sim();

  // Starts the simulator thread and holds the target in reset until it is
  // ready to accept requests. `bitstream` selects the FPGA image and is
  // ignored by a metasimulation.
  int init(const char* bitstream);

  int mem_write(uint64_t addr, uint64_t size, const void* value);

  int mem_read(uint64_t addr, uint64_t size, void* value);

  int dcr_write(uint32_t addr, uint32_t value);

  // The response arrives on a later cycle than the request, so this advances
  // the target until the target asserts it or the bounded wait expires.
  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  // Raises `start` for one target cycle. The kernel runs until `busy` drops.
  int start();

  // Advances the target and reports whether it is still executing. The target
  // only moves while this is being called, so a caller that stops polling
  // stops the simulation rather than leaving it running.
  int is_busy(bool* busy);

  // Polls until the target goes idle. `timeout_ms` bounds wall-clock time, not
  // target cycles; -1 waits indefinitely.
  int ready_wait(int64_t timeout_ms);

private:

  class Impl;
  Impl* impl_;
};

}
