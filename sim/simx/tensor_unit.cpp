
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

#include "tensor_unit.h"
#include "tensor_cfg.h"
#include "sparse_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include <bitset>
#include <iomanip>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

// Sparse WMMA support (uses vortex::sparse config)
namespace sv = vortex::sparse;
using scfg = sv::wmma_config_t<NUM_THREADS>;



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

// Sparse FMA specializations (vortex::sparse formats use same encoding as tensor formats)
template <>
struct FMA<sv::fp16, sv::fp32> {
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
struct FMA<sv::fp16, sv::fp16> {
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
struct FMA<sv::bf16, sv::fp32> {
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
struct FMA<sv::bf16, sv::bf16> {
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

// FEDP variant that only consumes the first `regs_used` A/B registers.
// This is used to model sparse WMMA where the hardware custom op only reads
// 2 (1:4) or 4 (2:4) registers from A (and correspondingly only needs that
// many effectual B inputs after metadata-based masking).
template <typename It, typename Ot>
struct FEDP_Regs {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t regs_used) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP_Regs: tcK * i_ratio must be <= 32");
    auto acc = bit_cast<otype>(c_val);
    uint32_t z_max = std::min<uint32_t>(regs_used, scfg::tcK);
    for (uint32_t z = 0; z < z_max; ++z) {
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

using PFN_FEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t);

static PFN_FEDP select_FEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp32::id:
      return FEDP<vt::fp32, vt::fp32>::eval;
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

using PFN_FEDP_REGS = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t, uint32_t);

static PFN_FEDP_REGS select_FEDP_regs(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp32::id:
      return FEDP_Regs<vt::fp32, vt::fp32>::eval;
    case vt::fp16::id:
      return FEDP_Regs<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return FEDP_Regs<vt::bf16, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported mma format (regs): " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP_Regs<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported mma format (regs): " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return FEDP_Regs<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported mma format (regs): " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported output type (regs): " << OT << "!" << std::endl;
    std::abort();
  }
}

// Sparse FEDP: uses metadata to select which values from fragB to use
// fragA is sparse (1:4 or 2:4), fragB is dense
// metadata contains bitmasks indicating which positions are non-zero
// sparsity_degree: 1 for 1:4 sparsity (1 of 4 non-zero), 2 for 2:4 sparsity (2 of 4 non-zero)
template <typename It, typename Ot>
struct SparseFEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, const uint32_t* metadata, uint32_t sparsity_degree) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "SparseFEDP: tcK * i_ratio must be <= 32");
    auto acc = bit_cast<otype>(c_val);
    
    constexpr uint32_t regs_per_block = (i_ratio == 2) ? 2 : 4;
    
    // Verify sparsity_degree is valid
    if (sparsity_degree != 1 && sparsity_degree != 2) {
      std::cout << "Error: invalid sparsity_degree=" << sparsity_degree << ", must be 1 or 2" << std::endl;
      std::abort();
    }
    
    // Track compressed index for sparse values
    uint32_t compressed_idx = 0;
    
    for (uint32_t z = 0; z < scfg::tcK; z += regs_per_block) {
      uint32_t block_idx = z / regs_per_block;
      uint32_t meta = (block_idx < 8) ? metadata[block_idx] : 0;
      uint8_t meta_byte = meta & 0xFF;
      
      // Count set bits in metadata to verify sparsity_degree
      uint32_t bits_set = 0;
      for (uint32_t pos = 0; pos < 4; ++pos) {
        if (meta_byte & (1u << pos)) {
          bits_set++;
        }
      }
      
      // For 1:4 sparsity, expect 1 bit set per 4-element block
      // For 2:4 sparsity, expect 2 bits set per 4-element block
      // (Note: bits_set may vary in practice, but we process all set bits)
      
      for (uint32_t pos = 0; pos < 4; ++pos) {
        if (meta_byte & (1u << pos)) {
          uint32_t reg_idx = z + (pos / i_ratio);
          uint32_t elem_idx = pos % i_ratio;
          
          if (reg_idx < scfg::tcK) {
            // For sparse formats, a_row contains compressed values
            // We need to index into the compressed storage
            uint32_t a_compressed_reg = compressed_idx / i_ratio;
            uint32_t a_compressed_elem = compressed_idx % i_ratio;
            
            if (a_compressed_reg < scfg::tcK) {
              auto a = reinterpret_cast<const itype *>(&a_row[a_compressed_reg].u32);
              auto b = reinterpret_cast<const itype *>(&b_col[reg_idx].u32);
              acc = FMA<It, Ot>::eval(a[a_compressed_elem], b[elem_idx], acc);
            }
            compressed_idx++;
          }
        }
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

void sparse_wmma(uint32_t fmt_s,
  uint32_t fmt_d,
  uint32_t step_m,
  uint32_t step_n,
  uint32_t step_k,
  const std::vector<reg_data_t>& rs1_data,
  const std::vector<reg_data_t>& rs2_data,
  const std::vector<reg_data_t>& rs3_data,
  std::vector<reg_data_t>& rd_data,
  const uint32_t* metadata_words,
  uint32_t sparsity_degree) {

  // Derive element width from source format
  uint32_t element_bits = 32;
  switch (fmt_s) {
  case sv::fp16::id:
  case sv::bf16::id: element_bits = 16; break;
  case sv::fp32::id:
  default:           element_bits = 32; break;
  }

  // I_RATIO: number of elements packed per 32-bit register
  const uint32_t I_RATIO = 32u / element_bits;

  // Metadata is provided as one word per lane (selected in execute.cpp).
  const uint32_t meta_regs_per_thread = 1u;

  uint32_t meta_bits_per_step = scfg::tcK;
  if (I_RATIO == 2u && sparsity_degree == 1u) {
    meta_bits_per_step = scfg::tcK * I_RATIO;
  }
  const uint32_t bit_offset = step_k * meta_bits_per_step;

  auto fedp_regs = select_FEDP_regs(fmt_s, fmt_d);

  uint32_t a_off = (step_m % scfg::a_sub_blocks) * scfg::a_block_size;
  uint32_t b_off = (step_n % scfg::b_sub_blocks) * scfg::b_block_size;

  // Debug: dump metadata words
  DTH(3, "SparseWMMA: m=" << step_m << ", n=" << step_n << ", k=" << step_k
  << ", bit_offset=" << bit_offset
  << ", metadata_words={");
  for (uint32_t idx = 0; idx < scfg::tcM * scfg::tcN; ++idx) {
  if (idx) DTN(3, ", ");
  DTN(3, "[");
  for (uint32_t r = 0; r < meta_regs_per_thread; ++r) {
  if (r) DTN(3, ",");
  DTN(3, "0b" << std::bitset<8>(metadata_words[idx * meta_regs_per_thread + r] & 0xFFu));
  }
  DTN(3, "]");
  }
  DTN(3, "}" << std::endl);

  for (uint32_t i = 0; i < scfg::tcM; ++i) {
  for (uint32_t j = 0; j < scfg::tcN; ++j) {
  auto b_col = rs2_data.data() + b_off + j * scfg::tcK;

  uint32_t idx = i * scfg::tcN + j;
  if (idx >= rs3_data.size() || idx >= rd_data.size()) {
  std::cout << "Error: index out of bounds in sparse_wmma: idx=" << idx
    << ", rs3_data.size()=" << rs3_data.size()
    << ", rd_data.size()=" << rd_data.size() << std::endl;
  std::abort();
  }

  auto c_val = rs3_data.at(idx).u32;

  // Get metadata word.
  // For fp16 2:4 with one metadata reg/thread, metadata is selected by (i,k):
  //   k=0 -> lanes 0,2 for i=0,1
  //   k=1 -> lanes 1,3 for i=0,1
  uint32_t meta_lane_idx = idx;
  if (I_RATIO == 2u && sparsity_degree == 2u) {
    meta_lane_idx = i * scfg::tcN + step_k;
  }
  const uint32_t meta_word = metadata_words[meta_lane_idx];

  // Extract active mask for this k-step.
  // fp32: tcK bits per step (existing behavior).
  // fp16: one nibble (4 bits) per step, because each step spans tcK packed
  // registers and each packed register holds two fp16 elements.
  uint32_t active_mask = 0u;
  uint32_t compressed_idx = 0u;
  if (I_RATIO == 2u) {
    // fp16 2:4: consume metadata in pair order per k-step:
    // k=0 -> bits [3:2], k=1 -> bits [1:0].
    const uint32_t meta4 = meta_word & 0xFu;
    const uint32_t pair_shift = (step_k == 0u) ? 2u : 0u;
    active_mask = (meta4 >> pair_shift) & 0x3u;
    if (step_k == 1u) {
      compressed_idx = __builtin_popcount((meta4 >> 2u) & 0x3u);
    }
  } else {
    const uint32_t mask_width = meta_bits_per_step;
    const uint32_t step_bit_offset = bit_offset;
    const uint32_t bit_mask = (1u << mask_width) - 1u;
    active_mask = (meta_word >> step_bit_offset) & bit_mask;

    // Count set bits in metadata below the current step window.
    const uint32_t lower_mask = (step_bit_offset == 0u)
      ? 0u
      : ((1u << step_bit_offset) - 1u);
    compressed_idx = __builtin_popcount(meta_word & lower_mask);
  }
  const uint32_t a_thread_base = i * scfg::tcN; // row i starts at thread i*tcN
  
  std::array<reg_data_t, scfg::tcK> a_row_masked{};
  if (I_RATIO == 1u) {
      for (uint32_t z = 0; z < scfg::tcK; ++z) {
          if (active_mask & (1u << z)) {
              // compressed_idx selects which thread within row i's group
              a_row_masked[z] = rs1_data[a_off + a_thread_base + compressed_idx];
              ++compressed_idx;
          }
      }
  } else {
      const uint32_t elem_mask = (1u << element_bits) - 1u;
      if (I_RATIO == 2u) {
              const uint32_t src_idx = a_off + a_thread_base + step_k;
              const uint32_t src_packed = (src_idx < rs1_data.size())
                                        ? rs1_data[src_idx].u32
                                        : 0u;
              const uint32_t lo = src_packed & elem_mask;
              const uint32_t hi = (src_packed >> element_bits) & elem_mask;
              const uint32_t meta4 = meta_word & 0xFu;
              uint32_t val_rank = 0u;
              for (uint32_t pos = 0; pos < 4u; ++pos) {
                if (meta4 & (1u << pos)) {
                  const uint32_t val = (val_rank == 0u) ? lo : hi;
                  const uint32_t dst_reg = pos / 2u;          // 0 -> a_row[0], 1 -> a_row[1]
                  const uint32_t dst_lane = pos % 2u;         // low/high 16b within packed reg
                  a_row_masked[dst_reg].u32 |= (val << (dst_lane * element_bits));
                  ++val_rank;
                }
              }
      } else {
          for (uint32_t z = 0; z < scfg::tcK; ++z) {
              uint32_t packed = 0u;
              for (uint32_t e = 0; e < I_RATIO; ++e) {
                  const uint32_t bit_pos = z * I_RATIO + e;
                  if (active_mask & (1u << bit_pos)) {
                      const uint32_t elem = (rs1_data[a_off + a_thread_base + compressed_idx].u32
                                             >> (e * element_bits)) & elem_mask;
                      packed |= elem << (e * element_bits);
                      ++compressed_idx;
                  }
              }
              a_row_masked[z].u32 = packed;
          }
      }
  }
  const reg_data_t* a_row_for_fedp = a_row_masked.data();

  auto d_val = fedp_regs(a_row_for_fedp, b_col, c_val, scfg::tcK);
  rd_data.at(idx).u64 = nan_box(d_val);

  // Debug output
  auto print_u32 = [&](uint32_t fmt, uint32_t x) {
  __unused(x);
  if (fmt == sv::fp32::id || fmt == vt::fp32::id) {
    DTN(3, std::setprecision(8) << bit_cast<float>(x));
  } else {
    DTN(3, "0x" << std::hex << x << std::dec);
  }
  };
  __unused(print_u32);

  DTH(3, "SparseFEDP: i=" << i << ", j=" << j
  << ", m=" << step_m << ", n=" << step_n << ", k=" << step_k
  << ", active_mask=0b" << std::bitset<4>(active_mask)
  << ", a_row={");
  for (uint32_t q = 0; q < scfg::tcK; ++q) {
  if (q) DTN(3, ", ");
  print_u32(fmt_s, a_row_for_fedp[q].u32);
  }
  DTN(3, "}, b_col={");
  for (uint32_t q = 0; q < scfg::tcK; ++q) {
  if (q) DTN(3, ", ");
  print_u32(fmt_s, b_col[q].u32);
  }
  DTN(3, "}, meta_word=0b" << std::bitset<4>(meta_word & 0xFu));
  DTN(3, ", sparsity_degree=" << sparsity_degree);
  DTN(3, ", c_val="); print_u32(fmt_d, c_val);
  DTN(3, ", d_val="); print_u32(fmt_d, d_val);
  DTN(3, std::endl);
  }
  }
}

// Member wrapper for ABI compatibility; forwards to free function.
void TensorUnit::sparse_wmma(uint32_t fmt_s,
                             uint32_t fmt_d,
                             uint32_t step_m,
                             uint32_t step_n,
                             uint32_t step_k,
                             const std::vector<reg_data_t>& rs1_data,
                             const std::vector<reg_data_t>& rs2_data,
                             const std::vector<reg_data_t>& rs3_data,
                             std::vector<reg_data_t>& rd_data,
                             const uint32_t* metadata_words,
                             uint32_t sparsity_degree) {
  ::sparse_wmma(fmt_s, fmt_d, step_m, step_n, step_k,
                rs1_data, rs2_data, rs3_data,
                rd_data, metadata_words, sparsity_degree);
}

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
  {
    //--
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.front();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      int delay = 0;
      switch (tcu_type) {
      case TcuType::WMMA:
        delay = 4;
        break;
      default:
        std::abort();
      }
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
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
            ExeTraceData* trace_data) {
    __unused(wid);
    __unused(trace_data);

    auto fedp = select_FEDP(fmt_s, fmt_d);

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;
        auto d_val = fedp(a_row, b_col, c_val);
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(d_val);

        DTH(3, "FEDP: wid=" << wid << ", i=" << i << ", j=" << j << ", m=" << step_m << ", n=" << step_n << ", a_row={" << std::hex);
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << a_row[q].u32);
        }
        DTN(3, "}, b_col={");
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << b_col[q].u32);
        }
        DTN(3, "}, c_val=0x" << c_val << ", d_val=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.sparsity_degree), ""};
  default:
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<TensorUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
{}

TensorUnit::~TensorUnit() {
  delete impl_;
}

void TensorUnit::reset() {
  impl_->reset();
}

void TensorUnit::tick() {
  impl_->tick();
}

const TensorUnit::PerfStats &TensorUnit::perf_stats() const {
	return impl_->perf_stats();
}

void TensorUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}