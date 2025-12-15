
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
#include <cstring>

using namespace vortex;

namespace vt = vortex::sparse;
using cfg = vt::wmma_config_t<NUM_THREADS>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

template <typename It, typename Ot>
struct FMA {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static otype eval(itype a, itype b, otype c) {
    return static_cast<otype>(a) * static_cast<otype>(b) + c;
  }
};

template <>
struct FMA<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xc = rv_htof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoh_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xc = rv_btof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftob_s(xd, 0, nullptr);
    return xh;
  }
};

template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
  static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
  auto acc = bit_cast<otype>(c_val);
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
    auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA<It, Ot>::eval(a[i], b[i], acc);
    }
  }
  return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::int4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        if (a_val & 0x8) {
          a_val |= 0xFFFFFFF0;
        }
        if (b_val & 0x8) {
          b_val |= 0xFFFFFFF0;
        }
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::uint4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

// Sparse FEDP: uses metadata to select which values from fragB to use
// fragA is sparse (2:4), fragB is dense
// metadata contains bitmasks indicating which 2 of 4 positions are non-zero
template <typename It, typename Ot>
struct SparseFEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, const uint32_t* metadata) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "SparseFEDP: tcK * i_ratio must be <= 32");
    auto acc = bit_cast<otype>(c_val);
    
    constexpr uint32_t regs_per_block = (i_ratio == 2) ? 2 : 4;
    
    for (uint32_t z = 0; z < cfg::tcK; z += regs_per_block) {
      uint32_t block_idx = z / regs_per_block;
      uint32_t meta = (block_idx < 8) ? metadata[block_idx] : 0;
      uint8_t meta_byte = meta & 0xFF;
      
      for (uint32_t pos = 0; pos < 4; ++pos) {
        if (meta_byte & (1u << pos)) {
          uint32_t reg_idx = z + (pos / i_ratio);
          uint32_t elem_idx = pos % i_ratio;
          
          if (reg_idx < cfg::tcK) {
            auto a = reinterpret_cast<const itype *>(&a_row[reg_idx].u32);
            auto b = reinterpret_cast<const itype *>(&b_col[reg_idx].u32);
            acc = FMA<It, Ot>::eval(a[elem_idx], b[elem_idx], acc);
          }
        }
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

template <>
struct SparseFEDP<vt::fp32, vt::fp32> {
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, const uint32_t* metadata) {
    __unused(metadata);
    auto acc = bit_cast<float>(c_val);
    
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a_val = bit_cast<float>(a_row[z].u32);
      auto b_val = bit_cast<float>(b_col[z].u32);
      acc = FMA<vt::fp32, vt::fp32>::eval(a_val, b_val, acc);
    }
    
    return bit_cast<uint32_t>(acc);
  }
};

using PFN_FEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t);
using PFN_SparseFEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t, const uint32_t*);

static PFN_SparseFEDP select_SparseFEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp32::id:
      return SparseFEDP<vt::fp32, vt::fp32>::eval;
    case vt::fp16::id:
      return SparseFEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return SparseFEDP<vt::bf16, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported sparse mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return SparseFEDP<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported sparse mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return SparseFEDP<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported sparse mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported sparse output type: " << OT << "!" << std::endl;
    std::abort();
  }
}

static PFN_FEDP select_FEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::int32::id:
    switch (IT) {
    case vt::int8::id:
      return FEDP<vt::int8, vt::int32>::eval;
    case vt::uint8::id:
      return FEDP<vt::uint8, vt::int32>::eval;
    case vt::int4::id:
      return FEDP<vt::int4, vt::int32>::eval;
    case vt::uint4::id:
      return FEDP<vt::uint4, vt::int32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported output type: " << OT << "!" << std::endl;
    std::abort();
  }
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
        case VegetaTcuType::TILE_GEMM_U:
        case VegetaTcuType::TILE_GEMM_V:
        case VegetaTcuType::TILE_GEMM_R:
        case VegetaTcuType::WMMA:
          delay = 4;
          break;
        default:
          std::abort();
        }
        DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      } else if (std::holds_alternative<VegetaLsuType>(trace->op_type)) {
        auto lsu_type = std::get<VegetaLsuType>(trace->op_type);
        switch (lsu_type) {
        case VegetaLsuType::TILE_LOAD_T:
        case VegetaLsuType::TILE_LOAD_U:
        case VegetaLsuType::TILE_LOAD_V:
        case VegetaLsuType::TILE_LOAD_M:
        case VegetaLsuType::TILE_STORE_T:
          delay = 2;
          break;
        default:
          std::abort();
        }
        DT(3, simobject_->name() << ": op=" << lsu_type << ", " << *trace);
      } else {
        std::abort();
      }
      #else
      auto tcu_type = std::get<TcuType>(trace->op_type);
      switch (tcu_type) {
      case TcuType::WMMA:
        delay = 4;
        break;
      default:
        std::abort();
      }
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      #endif
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      input.pop();
    }
  }

  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data,
            const uint32_t* metadata) {
    __unused(trace_data);

    // Use provided metadata from integer registers 0-7 for sparse fragA
    // If metadata is null, use zeros (dense mode fallback)
    uint32_t meta[8] = {0};
    if (metadata != nullptr) {
      for (uint32_t i = 0; i < 8; ++i) {
        meta[i] = metadata[i];
      }
    }
    
    // Use sparse FEDP for sparse-dense GEMM
    auto sparse_fedp = select_SparseFEDP(fmt_s, fmt_d);

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        
        uint32_t idx = i * cfg::tcN + j;
        if (idx >= rs3_data.size() || idx >= rd_data.size()) {
          std::cout << "Error: index out of bounds in sparse_unit wmma: idx=" << idx 
                    << ", rs3_data.size()=" << rs3_data.size() 
                    << ", rd_data.size()=" << rd_data.size() << std::endl;
          std::abort();
        }
        
        auto c_val = rs3_data.at(idx).u32;
        
        // Map metadata from fragment registers to K dimension registers
        uint32_t meta_for_k[8] = {0};
        for (uint32_t z = 0; z < cfg::tcK && z < 8; ++z) {
          uint32_t frag_reg_idx = a_off + i * cfg::tcK + z;
          if (frag_reg_idx < 8) {
            meta_for_k[z] = meta[frag_reg_idx];
          }
        }
        
        // Perform sparse-dense FEDP: fragA is sparse, fragB is dense
        auto d_val = sparse_fedp(a_row, b_col, c_val, meta_for_k);
        rd_data.at(idx).u64 = nan_box(d_val);

        DTH(3, "SparseFEDP: wid=" << wid << ", i=" << i << ", j=" << j << ", m=" << step_m << ", n=" << step_n << ", a_row={" << std::hex);
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << a_row[q].u32);
        }
        DTN(3, "}, b_col={");
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << b_col[q].u32);
        }
        DTN(3, "}, metadata={");
        for (uint32_t q = 0; q < 8 && q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << meta_for_k[q]);
        }
        DTN(3, "}, c_val=0x" << c_val << ", d_val=0x" << d_val << std::dec << std::endl);
      }
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
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype); // 4 bytes for fp32
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads

      // Load tile from memory: 16 rows x 16 columns = 256 fp32 elements = 1024 bytes
      for (uint32_t row = 0; row < TILE_DIM; ++row) {
        for (uint32_t col = 0; col < TILE_DIM; ++col) {
          uint64_t mem_addr = base_addr + (row * TILE_DIM + col) * ELEMENT_SIZE;
          uint32_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
          
          // Interpret as float and store in tile register
          float value;
          std::memcpy(&value, &mem_data, ELEMENT_SIZE);
          tile_reg[row][col] = value;
        }
      }
      
      DP(2, "TILE_LOAD_T: wid=" << wid << ", tid=" << tid 
         << ", tile_reg_idx=" << tile_reg_idx << ", base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_U: {
      // tile_load_u: DestReg contains ureg index, map to tile registers
      // ureg 0 -> tile reg 0, 1
      std::vector<uint32_t> target_tregs = map_ureg_to_treg(vd);
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype);
      
      uint64_t current_addr = base_addr;
      for (uint32_t treg_idx : target_tregs) {
        assert(treg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
        auto &tile_reg = tile_reg_file_[treg_idx];
        
        // Load tile from memory: 16 rows x 16 columns = 256 fp32 elements = 1024 bytes
        for (uint32_t row = 0; row < TILE_DIM; ++row) {
          for (uint32_t col = 0; col < TILE_DIM; ++col) {
            uint64_t mem_addr = current_addr + (row * TILE_DIM + col) * ELEMENT_SIZE;
            uint32_t mem_data = 0;
            core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
            trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
            
            float value;
            std::memcpy(&value, &mem_data, ELEMENT_SIZE);
            tile_reg[row][col] = value;
          }
        }
        current_addr += TILE_DIM * TILE_DIM * ELEMENT_SIZE; // Move to next tile (1KB)
      }
      
      DP(2, "TILE_LOAD_U: wid=" << wid << ", tid=" << tid 
         << ", ureg_idx=" << vd << ", target_tregs=[" 
         << target_tregs[0] << ", " << target_tregs[1] << "], base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    case VegetaLsuType::TILE_LOAD_V: {
      // tile_load_v: DestReg contains vreg index, map to tile registers
      // vreg 0 -> tile reg 0, 1, 2, 3
      std::vector<uint32_t> target_tregs = map_vreg_to_treg(vd);
      base_addr &= 0xFFFFFFFC; // Align to word boundary for fp32 loads
      constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype);
      
      uint64_t current_addr = base_addr;
      for (uint32_t treg_idx : target_tregs) {
        assert(treg_idx < tile_reg_file_.size() && "Tile register index out of bounds");
        auto &tile_reg = tile_reg_file_[treg_idx];
        
        // Load tile from memory: 16 rows x 16 columns = 256 fp32 elements = 1024 bytes
        for (uint32_t row = 0; row < TILE_DIM; ++row) {
          for (uint32_t col = 0; col < TILE_DIM; ++col) {
            uint64_t mem_addr = current_addr + (row * TILE_DIM + col) * ELEMENT_SIZE;
            uint32_t mem_data = 0;
            core_->dcache_read(&mem_data, mem_addr, ELEMENT_SIZE);
            trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
            
            float value;
            std::memcpy(&value, &mem_data, ELEMENT_SIZE);
            tile_reg[row][col] = value;
          }
        }
        current_addr += TILE_DIM * TILE_DIM * ELEMENT_SIZE; // Move to next tile (1KB)
      }
      
      DP(2, "TILE_LOAD_V: wid=" << wid << ", tid=" << tid 
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

      // Load metadata from memory: 16 rows x 16 columns = 256 uint4 elements = 128 bytes
      // Each byte stores two uint4 values: upper nibble for col N, lower nibble for col N+1
      for (uint32_t row = 0; row < TILE_DIM; ++row) {
        for (uint32_t col = 0; col < TILE_DIM; col += 2) {
          uint64_t mem_addr = base_addr + (row * (TILE_DIM / 2) + col / 2);
          uint8_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, 1);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, 1});
          
          // Upper nibble for col N, lower nibble for col N+1
          metadata_reg[row][col] = (mem_data >> 4) & 0x0F;
          metadata_reg[row][col + 1] = mem_data & 0x0F;
        }
      }

      DP(2, "TILE_LOAD_M: wid=" << wid << ", tid=" << tid 
         << ", metadata_reg_idx=" << meta_reg_idx << ", base_addr=0x" << std::hex << base_addr << std::dec);
      break;
    }
    default:
      std::abort();
    }
  }

  void store(const Instr &instr,
             uint32_t wid,
             uint32_t tid,
             const std::vector<reg_data_t> &rs1_data,
             MemTraceData *trace_data) {
    __unused(wid);

    auto lsuArgs = std::get<IntrVegetaLsuArgs>(instr.getArgs());
    uint32_t vs3 = instr.getSrcReg(1).idx; // Source tile register index
    
    // Calculate base address: rs1_data + immediate offset
    uint64_t base_addr = rs1_data.at(tid).i + lsuArgs.offset;
    base_addr &= 0xFFFFFFFC; // Align to word boundary

    assert(vs3 < tile_reg_file_.size() && "Tile register index out of bounds");
    auto &tile_reg = tile_reg_file_[vs3];
    constexpr uint32_t TILE_DIM = 16;
    constexpr uint32_t ELEMENT_SIZE = sizeof(typename vt::fp32::dtype); // 4 bytes for fp32

    // Store tile to memory: 16 rows x 16 columns = 256 fp32 elements = 1024 bytes
    for (uint32_t row = 0; row < TILE_DIM; ++row) {
      for (uint32_t col = 0; col < TILE_DIM; ++col) {
        uint64_t mem_addr = base_addr + (row * TILE_DIM + col) * ELEMENT_SIZE;
        float value = tile_reg[row][col];
        uint32_t mem_data = 0;
        std::memcpy(&mem_data, &value, ELEMENT_SIZE);
        core_->dcache_write(&mem_data, mem_addr, ELEMENT_SIZE);
        trace_data->mem_addrs.at(tid).push_back({mem_addr, ELEMENT_SIZE});
      }
    }

    DP(2, "TILE_STORE: wid=" << wid << ", tid=" << tid << ", vs3=" << vs3 
       << ", base_addr=0x" << std::hex << base_addr << std::dec);
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

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
  default:
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

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

void SparseUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data,
                      const uint32_t* metadata) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data, metadata);
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