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
#include <vector>
#include <queue>
#include "types.h"
#include "instr_trace.h"

namespace vortex {

class Core;

// VEGETA Tile LSU - Dedicated Load/Store Unit for tile operations
// Handles bulk tile transfers (T-tile: 1KB, M-tile: 128B) via L1 D-Cache
class VegetaLsu : public SimObject<VegetaLsu> {
public:
  // Tile request types
  enum class TileType {
    T_TILE = 0,   // 16x16 float = 1024 bytes
    U_TILE = 1,   // 2 T-tiles = 2048 bytes  
    V_TILE = 2,   // 4 T-tiles = 4096 bytes
    M_TILE = 3    // 16x16 nibbles = 128 bytes (metadata)
  };

  // Tile request structure
  struct TileRequest {
    uint64_t addr;        // Base address
    TileType type;        // Tile type
    uint32_t reg_idx;     // Destination/source register index
    bool is_store;        // true = store, false = load
    uint32_t wid;         // Warp ID
    uint32_t tid;         // Thread ID
  };

  struct PerfStats {
    uint64_t tile_loads;
    uint64_t tile_stores;
    uint64_t cache_requests;
    uint64_t total_bytes;

    PerfStats()
      : tile_loads(0)
      , tile_stores(0)
      , cache_requests(0)
      , total_bytes(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->tile_loads += rhs.tile_loads;
      this->tile_stores += rhs.tile_stores;
      this->cache_requests += rhs.cache_requests;
      this->total_bytes += rhs.total_bytes;
      return *this;
    }
  };

  // Memory request/response ports for L1 cache connection
  std::vector<SimPort<MemReq>> dcache_req_ports;
  std::vector<SimPort<MemRsp>> dcache_rsp_ports;

  VegetaLsu(const SimContext& ctx, const char* name, Core* core, uint32_t num_ports);
  ~VegetaLsu();

  void reset();
  void tick();

  // Load tile from memory to register
  void load_tile(uint64_t addr, TileType type, uint32_t reg_idx, 
                 uint32_t wid, uint32_t tid, void* reg_data);

  // Store tile from register to memory
  void store_tile(uint64_t addr, TileType type, uint32_t reg_idx,
                  uint32_t wid, uint32_t tid, const void* reg_data);

  // Get tile size in bytes
  static uint32_t tile_size(TileType type);

  const PerfStats& perf_stats() const { return perf_stats_; }

private:
  Core* core_;
  uint32_t num_ports_;
  PerfStats perf_stats_;

  // Issue cache-line requests for a tile
  void issue_tile_requests(const TileRequest& req, void* data);
};

} // namespace vortex
