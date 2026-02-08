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

#include "vegeta_lsu.h"
#include "core.h"
#include "debug.h"
#include <cstring>

using namespace vortex;

// Tile sizes in bytes
static constexpr uint32_t T_TILE_SIZE = 1024;  // 16x16 x 4 bytes
static constexpr uint32_t U_TILE_SIZE = 2048;  // 2 T-tiles
static constexpr uint32_t V_TILE_SIZE = 4096;  // 4 T-tiles
static constexpr uint32_t M_TILE_SIZE = 128;   // 16x16 x 0.5 bytes (nibbles)

// Cache line size (from config)
static constexpr uint32_t CACHE_LINE_SIZE = 64;

VegetaLsu::VegetaLsu(const SimContext& ctx, const char* name, Core* core, uint32_t num_ports)
  : SimObject(ctx, name)
  , dcache_req_ports(num_ports, this)
  , dcache_rsp_ports(num_ports, this)
  , core_(core)
  , num_ports_(num_ports)
  , perf_stats_()
{}

VegetaLsu::~VegetaLsu() {}

void VegetaLsu::reset() {
  perf_stats_ = PerfStats();
}

void VegetaLsu::tick() {
  // Process cache responses (for future async implementation)
  for (uint32_t p = 0; p < num_ports_; ++p) {
    auto& rsp_port = dcache_rsp_ports.at(p);
    if (!rsp_port.empty()) {
      // For now, we use synchronous access via core_->dcache_read/write
      rsp_port.pop();
    }
  }
}

uint32_t VegetaLsu::tile_size(TileType type) {
  switch (type) {
    case TileType::T_TILE: return T_TILE_SIZE;
    case TileType::U_TILE: return U_TILE_SIZE;
    case TileType::V_TILE: return V_TILE_SIZE;
    case TileType::M_TILE: return M_TILE_SIZE;
    default: return 0;
  }
}

void VegetaLsu::load_tile(uint64_t addr, TileType type, uint32_t reg_idx,
                          uint32_t wid, uint32_t tid, void* reg_data) {
  __unused(reg_idx, wid, tid);
  
  uint32_t size = tile_size(type);
  uint8_t* data = reinterpret_cast<uint8_t*>(reg_data);
  
  // Align address to cache line boundary for efficient access
  uint64_t aligned_addr = addr & ~(uint64_t)(CACHE_LINE_SIZE - 1);
  uint32_t offset = addr - aligned_addr;
  
  DP(2, "VEGETA_LSU: load_tile type=" << (int)type << " addr=0x" << std::hex << addr 
     << " size=" << std::dec << size << " reg=" << reg_idx);
  
  // Issue cache-line sized requests
  uint32_t bytes_loaded = 0;
  uint64_t current_addr = addr;
  
  while (bytes_loaded < size) {
    uint32_t chunk_size = std::min(size - bytes_loaded, CACHE_LINE_SIZE);
    
    // Use core's dcache_read for each cache line
    core_->dcache_read(data + bytes_loaded, current_addr, chunk_size);
    
    ++perf_stats_.cache_requests;
    bytes_loaded += chunk_size;
    current_addr += chunk_size;
  }
  
  ++perf_stats_.tile_loads;
  perf_stats_.total_bytes += size;
}

void VegetaLsu::store_tile(uint64_t addr, TileType type, uint32_t reg_idx,
                           uint32_t wid, uint32_t tid, const void* reg_data) {
  __unused(reg_idx, wid, tid);
  
  uint32_t size = tile_size(type);
  const uint8_t* data = reinterpret_cast<const uint8_t*>(reg_data);
  
  DP(2, "VEGETA_LSU: store_tile type=" << (int)type << " addr=0x" << std::hex << addr 
     << " size=" << std::dec << size << " reg=" << reg_idx);
  
  // Issue cache-line sized requests
  uint32_t bytes_stored = 0;
  uint64_t current_addr = addr;
  
  while (bytes_stored < size) {
    uint32_t chunk_size = std::min(size - bytes_stored, CACHE_LINE_SIZE);
    
    // Use core's dcache_write for each cache line
    core_->dcache_write(data + bytes_stored, current_addr, chunk_size);
    
    ++perf_stats_.cache_requests;
    bytes_stored += chunk_size;
    current_addr += chunk_size;
  }
  
  ++perf_stats_.tile_stores;
  perf_stats_.total_bytes += size;
}

void VegetaLsu::issue_tile_requests(const TileRequest& req, void* data) {
  if (req.is_store) {
    store_tile(req.addr, req.type, req.reg_idx, req.wid, req.tid, data);
  } else {
    load_tile(req.addr, req.type, req.reg_idx, req.wid, req.tid, data);
  }
}
