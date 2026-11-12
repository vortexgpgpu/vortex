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

#include "types.h"

namespace vortex {

// Warp-wide local memory. It takes whole warp requests from the LSU blocks,
// one per cycle per engine (two engines with several LSU blocks), and serves
// each in passes: each pass gives every bank to the lowest pending lane that
// maps to it, and every pending read lane of the same word rides along. Lanes
// left over replay in later passes, one every other cycle, and a request
// retires before the next one starts, so the banks see every lane in program
// order.
class LocalMem : public SimObject<LocalMem> {
public:
  // A DMA access covers one full bank row. A row wider than a mem_block
  // arrives as same-cycle block accesses on DMA_PORTS adjacent channels.
  static constexpr uint32_t DMA_ROW_SIZE = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
  static constexpr uint32_t DMA_PORTS =
      (DMA_ROW_SIZE > VX_CFG_MEM_BLOCK_SIZE) ? (DMA_ROW_SIZE / VX_CFG_MEM_BLOCK_SIZE) : 1;

  struct Config {
    uint32_t capacity;
    uint32_t word_size;
    uint32_t num_blocks;  // LSU blocks, one warp request port each
    uint32_t num_lanes;   // lanes of a warp request
    uint32_t B;           // log2 number of banks
    uint32_t dma_clients; // row-wide DMA masters, served round-robin
  };

  struct PerfStats {
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t bank_stalls = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads += rhs.reads;
      this->writes += rhs.writes;
      this->bank_stalls += rhs.bank_stalls;
      return *this;
    }
  };

  std::vector<SimChannel<LsuReq>> ReqIn;
  std::vector<SimChannel<LsuRsp>> RspOut;

  // DMA port: client c uses channels [c * DMA_PORTS, (c + 1) * DMA_PORTS).
  std::vector<SimChannel<MemReq>> DmaInputs;
  std::vector<SimChannel<MemRsp>> DmaOutputs;

  LocalMem(const SimContext& ctx, const char* name, const Config& config);
  virtual ~LocalMem();

  const PerfStats& perf_stats() const;

protected:

  void on_reset();
  void on_tick();

  class Impl;
  Impl* impl_;

  friend class SimObject<LocalMem>;
};

}
