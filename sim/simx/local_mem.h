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

#include <simobject.h>
#include "types.h"

namespace vortex {

class LocalMem : public SimObject<LocalMem> {
public:
  struct Config {
    uint32_t capacity;
    uint32_t line_size;
    uint32_t num_reqs;
    uint32_t B; // log2 number of banks
    bool write_reponse;
  };

  struct PerfStats {
    uint64_t reads;
    uint64_t writes;
    uint64_t bank_stalls;

    PerfStats()
      : reads(0)
      , writes(0)
      , bank_stalls(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads += rhs.reads;
      this->writes += rhs.writes;
      this->bank_stalls += rhs.bank_stalls;
      return *this;
    }
  };

  std::vector<SimPort<MemReq>> Inputs;
  std::vector<SimPort<MemRsp>> Outputs;

  LocalMem(const SimContext& ctx, const char* name, const Config& config);
  virtual ~LocalMem();

  void reset();

  void read(void* data, uint64_t addr, uint32_t size);

  void write(const void* data, uint64_t addr, uint32_t size);

  void tick();

  const PerfStats& perf_stats() const;

protected:

  class Impl;
  Impl* impl_;
};

}
