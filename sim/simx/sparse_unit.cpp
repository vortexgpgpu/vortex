
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

#include "sparse_unit.h"
#include "sparse_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include "vegeta_lsu.h"
#include <cstring>

using namespace vortex;

namespace vt = vortex::sparse;
using cfg = vt::wmma_config_t<NUM_THREADS>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}


class SparseUnit::Impl {
public:
  Impl(SparseUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
    , tile_reg_file_(8, std::vector<std::vector<typename vt::fp32::dtype>>(16, std::vector<typename vt::fp32::dtype>(16, 0.0f)))
    , metadata_reg_file_(8, std::vector<std::vector<typename vt::uint4::dtype>>(16, std::vector<typename vt::uint4::dtype>(16, 0)))
  {
    // Register file initialized: 8 registers, each 16x16 fp32 elements
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
    // Reset tile register file to zero
    for (auto& reg : tile_reg_file_) {
      for (auto& row : reg) {
        std::fill(row.begin(), row.end(), 0.0f);
      }
    }
    // Reset metadata register file to zero
    for (auto& reg : metadata_reg_file_) {
      for (auto& row : reg) {
        std::fill(row.begin(), row.end(), 0);
      }
    }
  }

  void tick() {
    using ecfg = vortex::vegeta_engine_config_t;
    
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.front();
      int delay = 0;
      #ifdef EXT_VEGETA_ENABLE
      if (std::holds_alternative<VegetaTcuType>(trace->op_type)) {
        auto tcu_type = std::get<VegetaTcuType>(trace->op_type);
        switch (tcu_type) {
        case VegetaTcuType::TILE_GEMM_T:
        // Cycle-accurate pipelined latency:
          // WL (Weight Load) + FF (Feed First) + FS (Feed Second) + DR (Drain) + REDUCE
          delay = ecfg::SINGLE_INSTR_LATENCY;
          break;
        case VegetaTcuType::TILE_GEMM_U:
        // Cycle-accurate pipelined latency:
          // WL (Weight Load) + FF (Feed First) + FS (Feed Second) + DR (Drain) + REDUCE
          delay = ecfg::SINGLE_INSTR_LATENCY;
          break;
        case VegetaTcuType::TILE_GEMM_V:
        // Cycle-accurate pipelined latency:
          // WL (Weight Load) + FF (Feed First) + FS (Feed Second) + DR (Drain) + REDUCE
          delay = ecfg::SINGLE_INSTR_LATENCY;
          break;
        case VegetaTcuType::TILE_GEMM_R:
          // Cycle-accurate pipelined latency:
          // WL (Weight Load) + FF (Feed First) + FS (Feed Second) + DR (Drain) + REDUCE
          delay = ecfg::SINGLE_INSTR_LATENCY;
          break;
        case VegetaTcuType::WMMA:
          // Keep existing WMMA timing for backward compatibility
          delay = 4;
          break;
        default:
          std::abort();
        }
        DT(3, simobject_->name() << ": op=" << tcu_type << ", delay=" << delay << ", " << *trace);
      } else if (std::holds_alternative<VegetaLsuType>(trace->op_type)) {
        auto lsu_type = std::get<VegetaLsuType>(trace->op_type);
        switch (lsu_type) {
        case VegetaLsuType::TILE_LOAD_T:
          // 1KB tile load = TILE_SIZE / MEM_BW cycles
          delay = ecfg::TILE_LOAD_LATENCY;
          break;
        case VegetaLsuType::TILE_LOAD_U:
          // 2KB U-register load = 2 tiles
          delay = 2 * ecfg::TILE_LOAD_LATENCY;
          break;
        case VegetaLsuType::TILE_LOAD_V:
          // 4KB V-register load = 4 tiles  
          delay = 4 * ecfg::TILE_LOAD_LATENCY;
          break;
        case VegetaLsuType::TILE_LOAD_M:
          // 128B metadata load
          delay = 2;
          break;
        case VegetaLsuType::TILE_STORE_T:
          // 1KB tile store
          delay = ecfg::TILE_LOAD_LATENCY;
          break;
        default:
          std::abort();
        }
        DT(3, simobject_->name() << ": op=" << lsu_type << ", delay=" << delay << ", " << *trace);
      } else {
        std::abort();
      }
      #else
      #ifdef EXT_TCU_ENABLE
      auto tcu_type = std::get<TcuType>(trace->op_type);
      switch (tcu_type) {
      case TcuType::WMMA:
        delay = 4;
        break;
      default:
        std::abort();
      }
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      #else
      // SparseUnit without TCU/VEGETA extensions: no valid ops are expected here.
      std::abort();
      #endif // EXT_TCU_ENABLE
      #endif // EXT_VEGETA_ENABLE
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      input.pop();
    }
  }


  // TILE_GEMM_T: Dense tile × Dense tile = Tile (T × T → T)
  // Tiles are 16×16, so this computes: C[16×16] = A[16×16] × B[16×16]
  void tile_gemm_t(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_treg) {
    assert(dst_treg < tile_reg_file_.size() && "Destination tile register out of bounds");
    assert(src1_treg < tile_reg_file_.size() && "Source1 tile register out of bounds");
    assert(src2_treg < tile_reg_file_.size() && "Source2 tile register out of bounds");

    constexpr uint32_t TILE_DIM = 16;
    
    auto& tile_dst = tile_reg_file_[dst_treg];
    const auto& tile_a = tile_reg_file_[src1_treg];
    const auto& tile_b = tile_reg_file_[src2_treg];

    // Matrix multiplication: C[16×16] = A[16×16] × B[16×16]
    // C += A × B (accumulate to existing value)
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
      for (uint32_t j = 0; j < TILE_DIM; ++j) {
        float sum = tile_dst[i][j];  // Accumulate to existing value
        for (uint32_t k = 0; k < TILE_DIM; ++k) {
          sum += tile_a[i][k] * tile_b[k][j];
        }
        tile_dst[i][j] = sum;
      }
    }

    DP(2, "TILE_GEMM_T: dst_t" << dst_treg << " = t" << src1_treg << " × t" << src2_treg);
  }

  // TILE_GEMM_U: Sparse tile (2:4) × Dense tile = Tile (T × U → T)
  void tile_gemm_u(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_ureg, uint32_t meta_reg) {
    assert(dst_treg < tile_reg_file_.size() && "Destination tile register out of bounds");
    assert(src1_treg < tile_reg_file_.size() && "Source1 tile register out of bounds");
    assert(meta_reg < metadata_reg_file_.size() && "Metadata register out of bounds");

    constexpr uint32_t TILE_DIM = 16;
    
    auto& tile_dst = tile_reg_file_[dst_treg];
    const auto& tile_a = tile_reg_file_[src1_treg];  // Sparse tile
    const auto& meta_a = metadata_reg_file_[meta_reg];  // Metadata for sparse tile
    
    // U-register maps to 2 T-registers
    std::vector<uint32_t> src2_tregs = map_ureg_to_treg(src2_ureg);
    
    // For 2:4 sparsity, each 4-element block has 2 non-zero values
    // Metadata byte indicates which 2 positions are non-zero
    // We process 2 T-registers as one U-register
    
    // U-register spans 2 T-registers, giving K dimension of 2*TILE_DIM = 32
    // A is stored in compressed form: 16 values per row representing 32 logical positions
    // Metadata indicates which 2 out of every 4 logical positions are stored
    
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
      for (uint32_t j = 0; j < TILE_DIM; ++j) {
        float sum = tile_dst[i][j];  // Accumulate
        
        // Iterate through compressed A values and map to logical K positions
        uint32_t compressed_idx = 0;  // Index into compressed storage (tile_a)
        
        // Process 8 groups of 4 logical K positions (covering K=0..31)
        for (uint32_t k_grp = 0; k_grp < 8; ++k_grp) {
          uint8_t mask = meta_a[i][k_grp];  // Metadata for this 4-element group
          uint32_t k_base = k_grp * 4;  // Base logical K position for this group
          
          // Check each of the 4 positions in this group
          for (uint32_t offset = 0; offset < 4; ++offset) {
            if (mask & (1u << offset)) {
              // This position is non-zero
              uint32_t k_logical = k_base + offset;  // Logical K position (0-31)
              
              // Access compressed value from tile_a
              float a_val = tile_a[i][compressed_idx];
              
              // Determine which T-register of B to access
              uint32_t treg_idx = (k_logical < TILE_DIM) ? src2_tregs[0] : src2_tregs[1];
              uint32_t k_local = k_logical % TILE_DIM;
              
              sum += a_val * tile_reg_file_[treg_idx][k_local][j];
              
              compressed_idx++;  // Move to next compressed value
            }
          }
        }
        tile_dst[i][j] = sum;
      }
    }

    DP(2, "TILE_GEMM_U: dst_t" << dst_treg << " = t" << src1_treg << "(sparse via m" << meta_reg << ") × u" << src2_ureg);
  }

  // TILE_GEMM_V: Sparse tile (1:4) × Dense tile = Tile (T × V → T)
  void tile_gemm_v(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_vreg, uint32_t meta_reg) {
    assert(dst_treg < tile_reg_file_.size() && "Destination tile register out of bounds");
    assert(src1_treg < tile_reg_file_.size() && "Source1 tile register out of bounds");
    assert(meta_reg < metadata_reg_file_.size() && "Metadata register out of bounds");

    constexpr uint32_t TILE_DIM = 16;
    
    auto& tile_dst = tile_reg_file_[dst_treg];
    const auto& tile_a = tile_reg_file_[src1_treg];  // Sparse tile
    const auto& meta_a = metadata_reg_file_[meta_reg];  // Metadata for sparse tile
    
    // V-register maps to 4 T-registers
    std::vector<uint32_t> src2_tregs = map_vreg_to_treg(src2_vreg);
    
    // For 1:4 sparsity, each 4-element block has 1 non-zero value
    // V-register spans 4 T-registers, giving K dimension of 4*TILE_DIM = 64
    // A is stored in compressed form: 16 values per row representing 64 logical positions
    // Metadata indicates which 1 out of every 4 logical positions is stored
    
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
      for (uint32_t j = 0; j < TILE_DIM; ++j) {
        float sum = tile_dst[i][j];  // Accumulate
        
        // Iterate through compressed A values and map to logical K positions
        uint32_t compressed_idx = 0;  // Index into compressed storage (tile_a)
        
        // Process 16 groups of 4 logical K positions (covering K=0..63)
        for (uint32_t k_grp = 0; k_grp < 16; ++k_grp) {
          uint8_t mask = meta_a[i][k_grp];  // Metadata for this 4-element group
          uint32_t k_base = k_grp * 4;  // Base logical K position for this group
          
          // Check each of the 4 positions in this group
          for (uint32_t offset = 0; offset < 4; ++offset) {
            if (mask & (1u << offset)) {
              // This position is non-zero
              uint32_t k_logical = k_base + offset;  // Logical K position (0-63)
              
              // Access compressed value from tile_a
              float a_val = tile_a[i][compressed_idx];
              
              // Determine which T-register of B to access
              uint32_t treg_idx = src2_tregs[k_logical / TILE_DIM];
              uint32_t k_local = k_logical % TILE_DIM;
              
              sum += a_val * tile_reg_file_[treg_idx][k_local][j];
              
              compressed_idx++;  // Move to next compressed value
            }
          }
        }
        tile_dst[i][j] = sum;
      }
    }

    DP(2, "TILE_GEMM_V: dst_t" << dst_treg << " = t" << src1_treg << "(sparse via m" << meta_reg << ") × v" << src2_vreg);
  }

  // TILE_GEMM_R: Row-wise sparse tile × Dense tile = Tile (T × U → U)
  // ISA: A is 16×32 logical (compressed to 16×16 padded T-tile)
  //      B is 32×16 dense (stored in U-reg = 2 T-regs)
  //      Output is 16×16 (first T-reg of destination U-reg)
  // Metadata: 8 blocks per row × 4 bits/block = 32 bits = 4 bytes per row
  //           Total: 64 bytes mask data + 64 bytes reserved = 128 bytes
  void tile_gemm_r(uint32_t dst_ureg, uint32_t src1_treg, uint32_t src2_ureg, uint32_t meta_reg) {
    assert(src1_treg < tile_reg_file_.size() && "Source1 tile register out of bounds");
    assert(meta_reg < metadata_reg_file_.size() && "Metadata register out of bounds");

    constexpr uint32_t TILE_DIM = 16;
    constexpr uint32_t LOGICAL_K = 32;  // A is 16×32 logical
    
    const auto& tile_a = tile_reg_file_[src1_treg];  // Compressed 16×16 tile
    const auto& meta_a = metadata_reg_file_[meta_reg];  // Metadata for sparse tile
    
    // Both dst and src2 are U-registers (map to 2 T-registers each)
    std::vector<uint32_t> dst_tregs = map_ureg_to_treg(dst_ureg);
    std::vector<uint32_t> src2_tregs = map_ureg_to_treg(src2_ureg);
    
    // Row-wise sparsity: each row of A has 8 blocks of 4 elements (32 total)
    // compressed to 16 values using 2-of-4 sparsity
    for (uint32_t i = 0; i < TILE_DIM; ++i) {
      for (uint32_t j = 0; j < TILE_DIM; ++j) {
        // Destination is first T-reg of U-reg (16×16 output)
        uint32_t dst_treg_idx = dst_tregs[0];
        
        float sum = tile_reg_file_[dst_treg_idx][i][j];  // Accumulate
        
        // Track position in compressed A tile for this row
        uint32_t a_col = 0;
        
        // Process 8 blocks of 4 elements each (K=32 logical)
        for (uint32_t k_blk = 0; k_blk < LOGICAL_K; k_blk += 4) {
          // Metadata layout: meta_a[row][col] stores individual nibbles (uint4)
          // nibble_idx = k_blk / 4 (0..7) directly indexes the metadata column
          uint32_t nibble_idx = k_blk / 4;
          uint8_t mask = meta_a[i][nibble_idx];  // Direct nibble access
          
          for (uint32_t offset = 0; offset < 4; ++offset) {
            if (mask & (1u << offset)) {
              // This position is non-zero in the logical A
              uint32_t k = k_blk + offset;  // Logical K index (0..31)
              
              // B is stored in U-reg (32×16), split into 2 T-regs (rows 0-15 and 16-31)
              uint32_t src2_treg_idx = src2_tregs[k / TILE_DIM];
              uint32_t k_local = k % TILE_DIM;
              
              // Get value from compressed A tile
              sum += tile_a[i][a_col] * tile_reg_file_[src2_treg_idx][k_local][j];
              a_col++;  // Move to next compressed value
            }
          }
        }
        tile_reg_file_[dst_treg_idx][i][j] = sum;
      }
    }

    DP(2, "TILE_GEMM_R: dst_u" << dst_ureg << " = t" << src1_treg << "(sparse via m" << meta_reg << ") × u" << src2_ureg);
  }

  // Map ureg index to tile register indices
  // ureg 0 -> tile reg 0, 1
  // ureg 1 -> tile reg 2, 3
  // ureg 2 -> tile reg 4, 5
  // etc.
  static std::vector<uint32_t> map_ureg_to_treg(uint32_t ureg_idx) {
    std::vector<uint32_t> treg_indices;
    treg_indices.push_back(ureg_idx * 2);
    treg_indices.push_back(ureg_idx * 2 + 1);
    return treg_indices;
  }

  // Map vreg index to tile register indices
  // vreg 0 -> tile reg 0, 1, 2, 3
  // vreg 1 -> tile reg 4, 5, 6, 7
  // etc.
  static std::vector<uint32_t> map_vreg_to_treg(uint32_t vreg_idx) {
    std::vector<uint32_t> treg_indices;
    treg_indices.push_back(vreg_idx * 4);
    treg_indices.push_back(vreg_idx * 4 + 1);
    treg_indices.push_back(vreg_idx * 4 + 2);
    treg_indices.push_back(vreg_idx * 4 + 3);
    return treg_indices;
  }

  void load(const Instr &instr,
            uint32_t wid,
            uint32_t tid,
            const std::vector<reg_data_t> &rs1_data,
            MemTraceData *trace_data) {
    __unused(wid);
    #ifdef EXT_VEGETA_ENABLE
    auto lsu_type = std::get<VegetaLsuType>(instr.getOpType());
    auto lsuArgs = std::get<IntrVegetaLsuArgs>(instr.getArgs());
    uint32_t vd = instr.getDestReg().idx; // DestReg contains the tile register index
    
    // Calculate base address: rs1_data + immediate offset
    uint64_t base_addr = rs1_data.at(tid).i + lsuArgs.offset;

    constexpr uint32_t TILE_DIM = 16;

    switch (lsu_type) {
    case VegetaLsuType::TILE_LOAD_T: {
      // tile_load_t: store in tile register specified by DestReg
      uint32_t tile_reg_idx = vd;
      assert(tile_reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
      auto &tile_reg = tile_reg_file_[tile_reg_idx];
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads

      // Use VegetaLsu for bulk tile load (1KB)
      constexpr uint32_t T_TILE_SIZE = TILE_DIM * TILE_DIM * sizeof(float);
      float tile_buffer[TILE_DIM * TILE_DIM];
      core_->vegeta_lsu()->load_tile(base_addr, VegetaLsu::TileType::T_TILE, 
                                      tile_reg_idx, wid, tid, tile_buffer);
      
      // Copy from linear buffer to 2D tile register
      for (uint32_t row = 0; row < TILE_DIM; ++row) {
        for (uint32_t col = 0; col < TILE_DIM; ++col) {
          tile_reg[row][col] = tile_buffer[row * TILE_DIM + col];
        }
      }
      
      // Record trace for all elements
      constexpr uint32_t ELEMENT_SIZE = sizeof(float);
      for (uint32_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        trace_data->mem_addrs.at(tid).push_back({base_addr + i * ELEMENT_SIZE, ELEMENT_SIZE});
      }
      
      DP(2, "TILE_LOAD_T (via VegetaLsu): wid=" << wid << ", tid=" << tid 
         << ", tile_reg_idx=" << tile_reg_idx << ", base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_U: {
      // tile_load_u: DestReg contains ureg index, map to tile registers
      // ureg 0 -> tile reg 0, 1 (2KB total = 2 T-tiles)
      std::vector<uint32_t> target_tregs = map_ureg_to_treg(vd);
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads
      
      // Use VegetaLsu for bulk U-tile load (2KB)
      constexpr uint32_t T_TILE_ELEMENTS = TILE_DIM * TILE_DIM;
      float tile_buffer[T_TILE_ELEMENTS * 2]; // 2 T-tiles for U-reg
      core_->vegeta_lsu()->load_tile(base_addr, VegetaLsu::TileType::U_TILE, 
                                      vd, wid, tid, tile_buffer);
      
      // Copy from linear buffer to 2D tile registers
      for (uint32_t t = 0; t < target_tregs.size(); ++t) {
        uint32_t treg_idx = target_tregs[t];
        assert(treg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
        auto &tile_reg = tile_reg_file_[treg_idx];
        for (uint32_t row = 0; row < TILE_DIM; ++row) {
          for (uint32_t col = 0; col < TILE_DIM; ++col) {
            tile_reg[row][col] = tile_buffer[t * T_TILE_ELEMENTS + row * TILE_DIM + col];
          }
        }
      }
      
      // Record trace for all elements
      constexpr uint32_t ELEMENT_SIZE = sizeof(float);
      for (uint32_t i = 0; i < T_TILE_ELEMENTS * 2; ++i) {
        trace_data->mem_addrs.at(tid).push_back({base_addr + i * ELEMENT_SIZE, ELEMENT_SIZE});
      }
      
      DP(2, "TILE_LOAD_U (via VegetaLsu): wid=" << wid << ", tid=" << tid 
         << ", ureg_idx=" << vd << ", target_tregs=[" 
         << target_tregs[0] << ", " << target_tregs[1] << "], base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_V: {
      // tile_load_v: DestReg contains vreg index, map to tile registers
      // vreg 0 -> tile reg 0, 1, 2, 3 (4KB total = 4 T-tiles)
      std::vector<uint32_t> target_tregs = map_vreg_to_treg(vd);
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads
      
      // Use VegetaLsu for bulk V-tile load (4KB)
      constexpr uint32_t T_TILE_ELEMENTS = TILE_DIM * TILE_DIM;
      float tile_buffer[T_TILE_ELEMENTS * 4]; // 4 T-tiles for V-reg
      core_->vegeta_lsu()->load_tile(base_addr, VegetaLsu::TileType::V_TILE, 
                                      vd, wid, tid, tile_buffer);
      
      // Copy from linear buffer to 2D tile registers
      for (uint32_t t = 0; t < target_tregs.size(); ++t) {
        uint32_t treg_idx = target_tregs[t];
        assert(treg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
        auto &tile_reg = tile_reg_file_[treg_idx];
        for (uint32_t row = 0; row < TILE_DIM; ++row) {
          for (uint32_t col = 0; col < TILE_DIM; ++col) {
            tile_reg[row][col] = tile_buffer[t * T_TILE_ELEMENTS + row * TILE_DIM + col];
          }
        }
      }
      
      // Record trace for all elements
      constexpr uint32_t ELEMENT_SIZE = sizeof(float);
      for (uint32_t i = 0; i < T_TILE_ELEMENTS * 4; ++i) {
        trace_data->mem_addrs.at(tid).push_back({base_addr + i * ELEMENT_SIZE, ELEMENT_SIZE});
      }
      
      DP(2, "TILE_LOAD_V (via VegetaLsu): wid=" << wid << ", tid=" << tid 
         << ", vreg_idx=" << vd << ", target_tregs=[" 
         << target_tregs[0] << ", " << target_tregs[1] << ", " 
         << target_tregs[2] << ", " << target_tregs[3] << "], base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_M: {
      // tile_load_M: DestReg contains metadata register index, store in that metadata register
      uint32_t meta_reg_idx = vd;
      assert(meta_reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
      auto &metadata_reg = metadata_reg_file_[meta_reg_idx];

      // Use VegetaLsu for bulk M-tile load (128 bytes)
      constexpr uint32_t M_TILE_SIZE = 128;
      uint8_t meta_buffer[M_TILE_SIZE];
      core_->vegeta_lsu()->load_tile(base_addr, VegetaLsu::TileType::M_TILE, 
                                      meta_reg_idx, wid, tid, meta_buffer);
      
      // Parse nibbles from linear buffer into metadata register
      // Each byte stores two uint4 values: upper nibble for col N, lower nibble for col N+1
      for (uint32_t row = 0; row < TILE_DIM; ++row) {
        for (uint32_t col = 0; col < TILE_DIM; col += 2) {
          uint8_t byte = meta_buffer[row * (TILE_DIM / 2) + col / 2];
          metadata_reg[row][col] = (byte >> 4) & 0x0F;
          metadata_reg[row][col + 1] = byte & 0x0F;
        }
      }
      
      // Record trace for all bytes
      for (uint32_t i = 0; i < M_TILE_SIZE; ++i) {
        trace_data->mem_addrs.at(tid).push_back({base_addr + i, 1});
      }

      DP(2, "TILE_LOAD_M (via VegetaLsu): wid=" << wid << ", tid=" << tid 
         << ", metadata_reg_idx=" << meta_reg_idx << ", base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    default:
      std::abort();
    }
    #else
    std::abort(); // EXT_VEGETA_ENABLE required for load operations
    #endif
  }

  void store(const Instr &instr,
             uint32_t wid,
             uint32_t tid,
             const std::vector<reg_data_t> &rs1_data,
             MemTraceData *trace_data) {
    __unused(wid);
    #ifdef EXT_VEGETA_ENABLE
    auto lsuArgs = std::get<IntrVegetaLsuArgs>(instr.getArgs());
    uint32_t vs3 = instr.getSrcReg(1).idx; // Source tile register index
    
    // Calculate base address: rs1_data + immediate offset
    uint64_t base_addr = rs1_data.at(tid).i + lsuArgs.offset;
    base_addr &= 0xFFFFFFFC; // Align to word boundary

    assert(vs3 < tile_reg_file_.size() && "Tile register index out of bounds");
    auto &tile_reg = tile_reg_file_[vs3];
    constexpr uint32_t TILE_DIM = 16;

    // Copy 2D tile register to linear buffer for VegetaLsu
    float tile_buffer[TILE_DIM * TILE_DIM];
    for (uint32_t row = 0; row < TILE_DIM; ++row) {
      for (uint32_t col = 0; col < TILE_DIM; ++col) {
        tile_buffer[row * TILE_DIM + col] = tile_reg[row][col];
      }
    }
    
    // Use VegetaLsu for bulk tile store (1KB)
    core_->vegeta_lsu()->store_tile(base_addr, VegetaLsu::TileType::T_TILE, 
                                     vs3, wid, tid, tile_buffer);
    
    // Record trace for all elements
    constexpr uint32_t ELEMENT_SIZE = sizeof(float);
    for (uint32_t i = 0; i < TILE_DIM * TILE_DIM; ++i) {
      trace_data->mem_addrs.at(tid).push_back({base_addr + i * ELEMENT_SIZE, ELEMENT_SIZE});
    }

    DP(2, "TILE_STORE (via VegetaLsu): wid=" << wid << ", tid=" << tid << ", vs3=" << vs3 
       << ", base_addr=0x" << std::hex << base_addr << std::dec);
    #else
    std::abort(); // EXT_VEGETA_ENABLE required for store operations
    #endif
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

  // Tile register file accessors
  SparseRegFile_t& tile_reg_file() {
    return tile_reg_file_;
  }

  const SparseRegFile_t& tile_reg_file() const {
    return tile_reg_file_;
  }

  // Metadata register file accessors
  std::vector<std::vector<std::vector<typename vt::uint4::dtype>>>& metadata_reg_file() {
    return metadata_reg_file_;
  }

  const std::vector<std::vector<std::vector<typename vt::uint4::dtype>>>& metadata_reg_file() const {
    return metadata_reg_file_;
  }

  // Access a specific tile register (returns reference to 16x32 vector)
  std::vector<std::vector<typename vt::fp32::dtype>>& get_tile_reg(uint32_t reg_idx) {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    return tile_reg_file_[reg_idx];
  }

  const std::vector<std::vector<typename vt::fp32::dtype>>& get_tile_reg(uint32_t reg_idx) const {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    return tile_reg_file_[reg_idx];
  }

  // Access a specific metadata register (returns reference to 16x32 vector)
  std::vector<std::vector<typename vt::uint4::dtype>>& get_metadata_reg(uint32_t reg_idx) {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    return metadata_reg_file_[reg_idx];
  }

  const std::vector<std::vector<typename vt::uint4::dtype>>& get_metadata_reg(uint32_t reg_idx) const {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    return metadata_reg_file_[reg_idx];
  }

  // Access a specific element in a tile register
  typename vt::fp32::dtype& get_tile_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    assert(row < tile_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < tile_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return tile_reg_file_[reg_idx][row][col];
  }

  const typename vt::fp32::dtype& get_tile_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) const {
    assert(reg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
    assert(row < tile_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < tile_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return tile_reg_file_[reg_idx][row][col];
  }

  // Access a specific element in a metadata register
  typename vt::uint4::dtype& get_metadata_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    assert(row < metadata_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < metadata_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return metadata_reg_file_[reg_idx][row][col];
  }

  const typename vt::uint4::dtype& get_metadata_reg_element(uint32_t reg_idx, uint32_t row, uint32_t col) const {
    assert(reg_idx < metadata_reg_file_.size() && "Metadata register index out of bounds");
    assert(row < metadata_reg_file_[reg_idx].size() && "Row index out of bounds");
    assert(col < metadata_reg_file_[reg_idx][row].size() && "Column index out of bounds");
    return metadata_reg_file_[reg_idx][row][col];
  }

private:

  SparseUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
  SparseRegFile_t tile_reg_file_;  // 8 registers, each 16x16 fp32 elements
  std::vector<std::vector<std::vector<typename vt::uint4::dtype>>> metadata_reg_file_;  // 8 registers, each 16x16 uint4 elements
};

SparseUnit::SparseUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<SparseUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
{}

SparseUnit::~SparseUnit() {
  delete impl_;
}

void SparseUnit::reset() {
  impl_->reset();
}

void SparseUnit::tick() {
  impl_->tick();
}

const SparseUnit::PerfStats &SparseUnit::perf_stats() const {
	return impl_->perf_stats();
}

void SparseUnit::load(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, MemTraceData* trace_data) {
  impl_->load(instr, wid, tid, rs1_data, trace_data);
}

void SparseUnit::store(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, MemTraceData* trace_data) {
  impl_->store(instr, wid, tid, rs1_data, trace_data);
}

void SparseUnit::tile_gemm_t(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_treg) {
  impl_->tile_gemm_t(dst_treg, src1_treg, src2_treg);
}

void SparseUnit::tile_gemm_u(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_ureg, uint32_t meta_reg) {
  impl_->tile_gemm_u(dst_treg, src1_treg, src2_ureg, meta_reg);
}

void SparseUnit::tile_gemm_v(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_vreg, uint32_t meta_reg) {
  impl_->tile_gemm_v(dst_treg, src1_treg, src2_vreg, meta_reg);
}

void SparseUnit::tile_gemm_r(uint32_t dst_ureg, uint32_t src1_treg, uint32_t src2_ureg, uint32_t meta_reg) {
  impl_->tile_gemm_r(dst_ureg, src1_treg, src2_ureg, meta_reg);
}