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
#include <VX_config.h>
#include <mem.h>

namespace vortex {

class RAM;
class ProcessorImpl;
class Core;
class Memory;

class Processor {
public:
  Processor();
  ~Processor();

  void attach_ram(RAM* mem);

  void reset();

  int run();

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  // DTM debug entry point: returns cluster[0].socket[0].core[0], or
  // nullptr if the processor has no cores configured.
  Core* get_first_core() const;

  // DTM debug entry point: kick the KMU dispatcher so CTAs start landing
  // when the simulator is ticked manually (debug-mode loop in main.cpp).
  // run() does this internally; debug mode replaces run() with its own
  // tick loop, so it must call this explicitly.
  void start_kmu();

  // True iff any cluster still has work to do (active warps, in-flight
  // CTAs) or any SimChannel still has packets in flight. Mirrors the
  // termination condition used in run().
  bool any_running() const;

  // Single-cycle entry point used by the SST integration
  // (sim/simx/sst/vortex_simulator.cpp). On first call: resets the
  // SimPlatform and the processor, then kicks the KMU. Each subsequent
  // call advances SimPlatform::tick() once and returns whether anything
  // is still running. This lets SST own the clock while v3 owns the
  // datapath.
  bool cycle();

  // Phase 3 SST integration: returns the global DRAM/Memory model so
  // the SST::Component can install its pre-send hook and route a copy
  // of every memory request to an SST memHierarchy link.
  Memory* memsim();

private:
  ProcessorImpl* impl_;
};

}
