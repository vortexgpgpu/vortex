
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

#include <VX_types.h>
#include "tcu_unit.h"
#include "tensor_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include "scheduler.h"
#include "local_mem.h"
#include "processor_impl.h"
#include "mem/memory.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg    = vt::wmma_config_t<VX_CFG_NUM_THREADS>;
using wg_cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::fp32, vt::fp32>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

static inline uint8_t unpack_u8(uint32_t word, uint32_t idx) {
  return (word >> (idx * 8)) & 0xffu;
}

// FMA<It, Ot>: fused multiply-add returning an Ot-typed accumulator (bit-packed in uint32).
// Widens narrow inputs/accumulator to fp32, performs mul+add, rounds once to Ot.
template <typename It, typename Ot>
struct FMA {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(itype a, itype b, uint32_t c) {
    otype fa = static_cast<otype>(a);
    otype fb = static_cast<otype>(b);
    otype fc = bit_cast<otype>(c);
    return bit_cast<uint32_t>(fa * fb + fc);
  }
};

// -- fp16 inputs --
template <> struct FMA<vt::fp16, vt::fp32> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_htof_s(a, 0, nullptr);
    auto fb = rv_htof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::fp16, vt::fp16> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_htof_s(a, 0, nullptr);
    auto fb = rv_htof_s(b, 0, nullptr);
    auto fc = rv_htof_s(uint16_t(c), 0, nullptr);
    return rv_ftoh_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- bf16 inputs --
template <> struct FMA<vt::bf16, vt::fp32> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_btof_s(a, 0, nullptr);
    auto fb = rv_btof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::bf16, vt::bf16> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_btof_s(a, 0, nullptr);
    auto fb = rv_btof_s(b, 0, nullptr);
    auto fc = rv_btof_s(uint16_t(c), 0, nullptr);
    return rv_ftob_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- fp8 inputs --
template <> struct FMA<vt::fp8, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e4m3tof_s(a, 0, nullptr);
    auto fb = rv_e4m3tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::fp8, vt::fp8> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e4m3tof_s(a, 0, nullptr);
    auto fb = rv_e4m3tof_s(b, 0, nullptr);
    auto fc = rv_e4m3tof_s(uint8_t(c), 0, nullptr);
    return rv_ftoe4m3_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- bf8 inputs --
template <> struct FMA<vt::bf8, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e5m2tof_s(a, 0, nullptr);
    auto fb = rv_e5m2tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::bf8, vt::bf8> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e5m2tof_s(a, 0, nullptr);
    auto fb = rv_e5m2tof_s(b, 0, nullptr);
    auto fc = rv_e5m2tof_s(uint8_t(c), 0, nullptr);
    return rv_ftoe5m2_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- tf32 inputs --
template <> struct FMA<vt::tf32, vt::fp32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto fa = rv_tf32tof_s(a, 0, nullptr);
    auto fb = rv_tf32tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::tf32, vt::tf32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto fa = rv_tf32tof_s(a, 0, nullptr);
    auto fb = rv_tf32tof_s(b, 0, nullptr);
    auto fc = rv_tf32tof_s(c, 0, nullptr);
    return rv_ftotf32_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// Generic FEDP: universal rule keyed on output width.
//   * Wide Ot (fp32): accumulate Σ(a_k*b_k) in fp32, add c_val last.
//   * Narrow Ot (fp16/bf16/fp8/bf8/…): chain FMA<It,Ot> so the accumulator is
//     rounded to Ot each step.
template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
    if constexpr (std::is_same_v<Ot, vt::fp32>) {
      uint32_t acc = 0;
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
        auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
        uint32_t prod = 0;
        for (uint32_t i = 0; i < i_ratio; ++i) {
          prod = FMA<It, vt::fp32>::eval(a[i], b[i], prod);
        }
        acc = rv_fadd_s(prod, acc, 0, nullptr);
      }
      return rv_fadd_s(c_val, acc, 0, nullptr);
    } else {
      uint32_t acc = c_val;
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
        auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
        for (uint32_t i = 0; i < i_ratio; ++i) {
          acc = FMA<It, Ot>::eval(a[i], b[i], acc);
        }
      }
      return acc;
    }
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
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::fp32>::eval;
    case vt::fp8::id:
      return FEDP<vt::fp8, vt::fp32>::eval;
    case vt::bf8::id:
      return FEDP<vt::bf8, vt::fp32>::eval;
    case vt::tf32::id:
      return FEDP<vt::tf32, vt::fp32>::eval;
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
  case vt::fp8::id:
    switch (IT) {
    case vt::fp8::id:
      return FEDP<vt::fp8, vt::fp8>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf8::id:
    switch (IT) {
    case vt::bf8::id:
      return FEDP<vt::bf8, vt::bf8>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::tf32::id:
    switch (IT) {
    case vt::tf32::id:
      return FEDP<vt::tf32, vt::tf32>::eval;
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

// Format-agnostic sparse gather: for each bword, iterate over its elem_count packed
// elements, collect those flagged by lo_mask/hi_mask respectively.
static inline uint32_t gather_sparse(uint32_t bword0, uint32_t bword1,
                                     uint32_t lo_mask, uint32_t hi_mask,
                                     uint32_t elem_bits) {
  uint32_t elem_count = 32 / elem_bits;
  uint32_t elem_mask  = (elem_bits < 32) ? ((1u << elem_bits) - 1u) : ~0u;
  assert((uint32_t)(__builtin_popcount(lo_mask) + __builtin_popcount(hi_mask)) == elem_count &&
         "gather_sparse: total selected elements must equal elem_count");
  uint32_t out = 0, k = 0;
  for (uint32_t i = 0; i < elem_count; ++i) {
    if (lo_mask & (1u << i))
      out |= ((bword0 >> (i * elem_bits)) & elem_mask) << (k++ * elem_bits);
  }
  for (uint32_t i = 0; i < elem_count; ++i) {
    if (hi_mask & (1u << i))
      out |= ((bword1 >> (i * elem_bits)) & elem_mask) << (k++ * elem_bits);
  }
  return out;
}

static inline uint32_t meta_num_cols(uint32_t fmt_s) {
  return vt::sparse_meta_num_cols(fmt_s, VX_CFG_NUM_THREADS);
}

static inline uint32_t meta_row_width(uint32_t elem_bits) {
  // Each K-step uses (32/elem_bits) meta bits per half (lo and hi), 2 halves per row.
  return cfg::tcK * 2 * (32 / elem_bits);
}

class TcuUnit::Impl {
public:

  struct lmem_desc_t {
    uint64_t base = 0;
    uint32_t ldm = 0;
    bool col_major = false;
  };

  Impl(TcuUnit* simobject, Core* core)
    : simobject_(simobject)
    , core_(core)
    , sparse_meta_(VX_CFG_NUM_WARPS, std::vector<uint32_t>(kMetaBanks * kMaxMetaCols, 0))
  #ifdef VX_CFG_TCU_MX_ENABLE
    , mx_meta_a_(VX_CFG_NUM_WARPS, std::vector<uint32_t>(VX_CFG_NUM_THREADS, 0))
    , mx_meta_b_(VX_CFG_NUM_WARPS, std::vector<uint32_t>(VX_CFG_NUM_THREADS, 0))
  #endif
    , perf_stats_()
  {
    exec_done_.fill(false);
    wgmma_planned_warps_.fill(0);
    in_wgmma_.fill(false);
  }

  ~Impl() {}

  void reset() {
    perf_stats_ = PerfStats();
    for (auto& sparse_meta : sparse_meta_) {
      std::fill(sparse_meta.begin(), sparse_meta.end(), 0);
    }
  #ifdef VX_CFG_TCU_MX_ENABLE
    for (auto& mx_meta : mx_meta_a_) {
      std::fill(mx_meta.begin(), mx_meta.end(), 0);
    }
    for (auto& mx_meta : mx_meta_b_) {
      std::fill(mx_meta.begin(), mx_meta.end(), 0);
    }
  #endif
    exec_done_.fill(false);
    wgmma_planned_warps_.fill(0);
    in_wgmma_.fill(false);
    cta_owner_a_.fill(-1);
    cta_owner_b_ = -1;
    cur_block_ = 0;
  }

  void tick() {
  #ifdef VX_CFG_TCU_WGMMA_ENABLE
    // Q-warp lock-step probe.
    // Pass 1 — identify active WGMMA blocks and prime each one's plan() on
    // first uop. WMMA and TCU_LD blocks are unaffected (no Q-coupling).
    uint32_t wgmma_active = 0;
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      if (input.empty()) continue;
      auto trace = input.peek();
      if (!tcu_is_wgmma(std::get<TcuType>(trace->op_type))) continue;
      wgmma_active |= (1u << b);

      uint32_t wid = trace->wid;
      uint64_t wid_bit = (uint64_t(1) << wid);
      if (wgmma_planned_warps_.at(b) & wid_bit) continue;
      auto& instr = *trace->instr_ptr;
      auto tpuArgs = std::get<IntrTcuArgs>(instr.get_args());
      if (!(tpuArgs.step_m == 0 && tpuArgs.step_n == 0 && tpuArgs.step_k == 0)) {
        // Non-first uop arrived without a prior plan: first uop already drained.
        // Mark planned and continue (descriptors persist in lmem_desc_[wid]).
        wgmma_planned_warps_.at(b) |= wid_bit;
        continue;
      }
      auto& rs1_data = trace->src_data[0];
      auto& rs2_data = trace->src_data[1];
      uint32_t a_desc = rs1_data.empty() ? 0 : rs1_data.at(0).u32;
      uint32_t b_desc = rs2_data.empty() ? 0 : rs2_data.at(0).u32;

      // CTA-overlap fence — defer this block's WGMMA if any other block
      // is mid-flight with a different CTA. The shared B buffer assumes
      // single-CTA occupancy across all blocks.
      int32_t new_cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
      bool block_other_cta_inflight = false;
      for (uint32_t k = 0; k < VX_CFG_NUM_TCU_BLOCKS; ++k) {
        if (k == b) continue;
        if (in_wgmma_.at(k) && cta_owner_a_.at(k) != new_cta) {
          block_other_cta_inflight = true;
          break;
        }
      }
      if (block_other_cta_inflight) {
        wgmma_active &= ~(1u << b);
        continue;
      }

      // Drop the shared B buffer only when no other block is mid-WGMMA —
      // otherwise we'd evict their resident bytes mid-flight.
      bool any_in_wgmma = false;
      for (auto v : in_wgmma_) any_in_wgmma = any_in_wgmma || v;
      auto& tbuf = simobject_->tbuf();
      if (!any_in_wgmma) {
        tbuf->invalidate_b();
        cta_owner_b_ = -1;
      }
      // Only drop the per-block A buffer when no warp is currently in flight.
      if (!in_wgmma_.at(b)) {
        tbuf->invalidate_a(b);
      }
      this->plan_wgmma_lines(b, wid, a_desc, b_desc, tpuArgs,
                             std::get<TcuType>(trace->op_type) == TcuType::WGMMA_SP);
      if (tbuf->ready_a(b) && tbuf->ready_b()) {
        ++perf_stats_.tbuf_cache_hits;
      }
      in_wgmma_.at(b) = true;
      wgmma_planned_warps_.at(b) |= wid_bit;
      cta_owner_a_.at(b) = new_cta;
      if (cta_owner_b_ == -1) cta_owner_b_ = new_cta;
    }

    // Pass 2 — all active WGMMA blocks must have A/B operands resident
    // before any of them advances.
    if (wgmma_active != 0) {
      uint32_t ready_mask = 0;
      auto& tbuf = simobject_->tbuf();
      for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
        if (!((wgmma_active >> b) & 1u)) continue;
        auto trace = simobject_->Inputs.at(b).peek();
        auto tpuArgs = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
        bool a_ok = !tpuArgs.is_a_smem || tbuf->ready_a(b);
        bool b_ok = tbuf->ready_b();
        if (a_ok && b_ok) ready_mask |= (1u << b);
      }
      if (ready_mask != wgmma_active) {
        ++perf_stats_.tbuf_stalls;
        return; // hold all blocks; per-block dispatch deferred to next tick
      }
    }
  #endif

    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      if (input.empty())
        continue;
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);

    #ifdef VX_CFG_TCU_WGMMA_ENABLE
      // CTA-overlap fence deferred this block — skip until pass 1 plans it.
      if (tcu_is_wgmma(tcu_type) &&
          !(wgmma_planned_warps_.at(b) & (uint64_t(1) << trace->wid)))
        continue;
    #endif

      // Execute once per trace; results persist across backpressure retries
      // via exec_done_[b].
      if (!exec_done_.at(b)) {
        auto& instr = *trace->instr_ptr;
        auto tpuArgs = std::get<IntrTcuArgs>(instr.get_args());
        uint32_t wid = trace->wid;
        uint32_t num_threads = VX_CFG_NUM_THREADS;
        auto& rs1_data = trace->src_data[0];
        auto& rs2_data = trace->src_data[1];
        auto& rs3_data = trace->src_data[2];
        trace->dst_data.assign(num_threads, reg_data_t{});
        auto& rd_data = trace->dst_data;

        switch (tcu_type) {
        case TcuType::WMMA:
        case TcuType::WMMA_SP:
          this->wmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d,
                     tpuArgs.step_m, tpuArgs.step_n, tpuArgs.step_k,
                     rs1_data, rs2_data, rs3_data, rd_data,
                     tcu_is_sparse(tcu_type));
          break;
      #ifdef VX_CFG_TCU_WGMMA_ENABLE
        case TcuType::WGMMA:
        case TcuType::WGMMA_SP: {
          uint32_t a_desc = rs1_data.empty() ? 0 : rs1_data.at(0).u32;
          uint32_t b_desc = rs2_data.empty() ? 0 : rs2_data.at(0).u32;
          cur_block_ = b;
          // CTA lockstep invariant: no block may execute a WGMMA uop for a
          // different cta_id while another block is mid-WGMMA.
          {
            int32_t this_cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
            for (uint32_t k = 0; k < VX_CFG_NUM_TCU_BLOCKS; ++k) {
              if (k == b) continue;
              if (in_wgmma_.at(k) && cta_owner_a_.at(k) != this_cta) {
                std::cerr << "*** TCU CTA lockstep violation: block " << b
                          << " executing WGMMA cta_id=" << this_cta
                          << " while block " << k << " holds cta_id="
                          << cta_owner_a_.at(k) << std::endl;
                std::abort();
              }
            }
          }
          this->wgmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d,
                      tpuArgs.step_m, tpuArgs.step_n, tpuArgs.step_k,
                      a_desc, b_desc, rs1_data, rs3_data, rd_data,
                      tcu_is_sparse(tcu_type),
                      tpuArgs.cd_nregs, tpuArgs.is_a_smem);
        } break;
      #endif
      #ifdef VX_CFG_TCU_META_ENABLE
        case TcuType::TCU_LD: {
          // rs1 is a full-width address (.u64); use u64 to avoid truncation on XLEN=64.
          uint64_t base_addr = rs1_data.empty() ? 0 : rs1_data.at(0).u64;
          this->tcu_ld(wid, tpuArgs.fmt_s, tpuArgs.fmt_d, base_addr);
        } break;
      #endif
        default:
          std::abort();
        }
        exec_done_.at(b) = true;
      }

      int delay = 0;
      switch (tcu_type) {
      case TcuType::WMMA:
      case TcuType::WMMA_SP:
      case TcuType::WGMMA:
      case TcuType::WGMMA_SP:
        delay = 4;
        break;
    #ifdef VX_CFG_TCU_META_ENABLE
      case TcuType::TCU_LD:
        delay = 4;
        break;
    #endif
      default:
        std::abort();
      }
      if (simobject_->Outputs.at(b).try_send(trace, 2 + delay)) {
        exec_done_.at(b) = false;
      #ifdef VX_CFG_TCU_WGMMA_ENABLE
        // Clear this warp's plan bit on its last uop so the next WGMMA
        // re-decodes descriptors. Block stays in_wgmma_ until all warps drain.
        if (tcu_is_wgmma(tcu_type) && trace->instr_ptr->get_fu_unlock()) {
          uint64_t wid_bit = (uint64_t(1) << trace->wid);
          wgmma_planned_warps_.at(b) &= ~wid_bit;
          if (wgmma_planned_warps_.at(b) == 0) {
            in_wgmma_.at(b) = false;
            cta_owner_a_.at(b) = -1;
          }
        }
      #endif
        DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
        input.pop();
      }
    }
  }

  // Plan all line addresses required for the current WGMMA's A, B and
  // sparse-metadata tiles into the per-role caches inside TcuTbuf.
  // Lines already resident or in-flight are skipped (additive plan).
  void plan_wgmma_lines(uint32_t b, uint32_t wid,
                        uint32_t a_desc, uint32_t b_desc,
                        const IntrTcuArgs& args, bool is_sparse) {
    uint32_t fmt_s = args.fmt_s;
    bool is_a_smem = args.is_a_smem;
    uint32_t e_bits = elem_bits(fmt_s);
    if (e_bits < 8) return;
    uint32_t e_bytes = e_bits / 8;
    // NRC: cd_nregs 0/1/2 → 8/16/32; xtileN = NRC * NT / xtileM.
    uint32_t nrc      = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
    uint32_t xtile_n  = (nrc * VX_CFG_NUM_THREADS) / wg_cfg::xtileM;

    lmem_desc_t sd_a{}, sd_b{};
    if (is_a_smem) {
      sd_a = {uint64_t(VX_MEM_LMEM_BASE_ADDR) + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
      lmem_desc_[wid][0] = sd_a;
    }
    sd_b = {uint64_t(VX_MEM_LMEM_BASE_ADDR) + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};
    lmem_desc_[wid][1] = sd_b;

    // tileK = xtileK × ratio (ratio = 32/e_bits); sparse compresses K on A only.
    uint32_t ratio  = 32 / e_bits;
    uint32_t tile_k = uint32_t(wg_cfg::xtileK) * ratio;
    uint32_t a_k    = is_sparse ? (tile_k / 2) : tile_k;

    // ldm==0 → block-major layout; ldm!=0 → row-major (stride in elements).
    uint32_t k_blk_dim   = cfg::tcK * ratio;
    uint32_t a_blk_elems = cfg::tcM * k_blk_dim;
    uint32_t b_blk_elems = k_blk_dim * cfg::tcN;
    uint32_t n_steps     = xtile_n / cfg::tcN;

    auto& tbuf = simobject_->tbuf();

    // Plan A lines (SS mode only): xtileM rows × a_k columns.
    if (is_a_smem) {
      bool a_block_major = (sd_a.ldm == 0);
      std::vector<uint64_t> a_lines;
      a_lines.reserve(uint32_t(wg_cfg::xtileM) * a_k);
      for (uint32_t r = 0; r < wg_cfg::xtileM; ++r) {
        for (uint32_t c = 0; c < a_k; ++c) {
          uint64_t elem_off;
          if (a_block_major) {
            uint32_t m_blk = r / cfg::tcM;
            uint32_t i_in  = r % cfg::tcM;
            uint32_t k_blk = c / k_blk_dim;
            uint32_t k_in  = c % k_blk_dim;
            elem_off = (k_blk * wg_cfg::m_steps + m_blk) * a_blk_elems
                     + i_in * k_blk_dim + k_in;
          } else {
            elem_off = uint64_t(r) * sd_a.ldm + c;
          }
          uint64_t addr = sd_a.base + elem_off * e_bytes;
          a_lines.push_back(addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1));
        }
      }
      tbuf->plan_a(b, a_lines);
      // Sparse metadata is preloaded into sparse_meta_ via TCU_LD;
      // no metadata lines are planned through tbuf here.
    }

    // Plan B lines: always dense in K, tileK rows × xtileN columns.
    //   ldm == 0 → block-major; ldm != 0 → K-major (smem[n*ldm + k]).
    bool b_block_major = (sd_b.ldm == 0);
    std::vector<uint64_t> b_lines;
    b_lines.reserve(tile_k * xtile_n);
    for (uint32_t r = 0; r < tile_k; ++r) {
      for (uint32_t c = 0; c < xtile_n; ++c) {
        uint64_t elem_off;
        if (b_block_major) {
          uint32_t k_blk = r / k_blk_dim;
          uint32_t r_in  = r % k_blk_dim;
          uint32_t n_blk = c / cfg::tcN;
          uint32_t n_in  = c % cfg::tcN;
          // Within-block layout: N outer, K inner.
          elem_off = (k_blk * n_steps + n_blk) * b_blk_elems
                   + n_in * k_blk_dim + r_in;
        } else {
          elem_off = uint64_t(c) * sd_b.ldm + r;
        }
        uint64_t addr = sd_b.base + elem_off * e_bytes;
        b_lines.push_back(addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1));
      }
    }
    tbuf->plan_b(b_lines);
  }

#ifdef VX_CFG_TCU_META_ENABLE
  // TCU_LD — warp-level metadata load. selector[4] chooses sparse or MX SRAM.
  void tcu_ld(uint32_t wid,
              uint32_t fmt_s,
              uint32_t selector,
              uint64_t base_addr) {
    (void)fmt_s;
    auto& lmem = *core_->local_mem();
    auto* memsim = core_->processor()->memsim();
  #ifdef VX_CFG_TCU_MX_ENABLE
    if (selector & 0x10) {
      auto& dst = (selector & 1) ? mx_meta_b_.at(wid) : mx_meta_a_.at(wid);
      for (uint32_t lane = 0; lane < VX_CFG_NUM_THREADS; ++lane) {
        uint64_t addr = base_addr + uint64_t(lane) * 4;
        dst.at(lane) = (get_addr_type(addr) == AddrType::Shared)
                     ? lmem.read_word(addr)
                     : memsim->read_word(addr);
      }
      return;
    }
  #endif
  #ifdef VX_CFG_TCU_SPARSE_ENABLE
    uint32_t slot_idx = selector & 0xf;
    // Map lane T to its (bank, col) metadata cell using the host pack layout:
    //   flat_store = slot*cols_per_load + T/BPS
    //   col  = flat_store / stores_per_col
    //   bank = (flat_store % stores_per_col)*BPS + T%BPS
    // This formula covers NT < kMetaBanks (stores_per_col > 1) as well as NT >= kMetaBanks.
    constexpr uint32_t PWD = kMetaBanks;
    constexpr uint32_t BPS = cfg::banks_per_store;
    constexpr uint32_t SPC = cfg::stores_per_col;
    constexpr uint32_t CPL = cfg::meta_cols_per_load;
    for (uint32_t T = 0; T < VX_CFG_NUM_THREADS; ++T) {
      uint32_t store_in_load   = T / BPS;
      uint32_t thread_in_store = T % BPS;
      uint32_t flat_store      = slot_idx * CPL + store_in_load;
      uint32_t col             = flat_store / SPC;
      uint32_t store_in_col    = flat_store % SPC;
      uint32_t bank            = store_in_col * BPS + thread_in_store;
      if (bank >= PWD || col >= kMaxMetaCols)
        continue;
      // base_addr is pre-advanced per slot by the caller; read lane T at base_addr + T*4.
      uint64_t word_idx = T;
      uint64_t addr = base_addr + word_idx * 4;
      // Route to shared memory or device memory based on address type.
      uint32_t word = (get_addr_type(addr) == AddrType::Shared)
                    ? lmem.read_word(addr)
                    : memsim->read_word(addr);
      sparse_meta_.at(wid).at(bank * kMaxMetaCols + col) = word;
      // Trace: META_SRAM write (CSV: wid,bank,col,addr,value).
      if (const char* p = std::getenv("VORTEX_TCU_TRACE")) {
        if (p[0] == '1') {
          fprintf(stderr, "META_TRC,%u,%u,%u,0x%lx,0x%08x\n",
                  wid, bank, col, (unsigned long)addr, word);
        }
      }
    }
  #else
    __unused(selector);
  #endif
  }
#endif

  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            uint32_t step_k,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            bool is_sparse) {
    if (is_sparse) {
      if (!vt::sparse_format_supported(fmt_s)) {
        std::cout << "Error: WMMA_SP unsupported input format: "
                  << vt::fmt_string(fmt_s) << " (id=" << fmt_s
                  << ")." << std::endl;
        std::abort();
      }
      if ((VX_CFG_NUM_THREADS % cfg::b_block_size_sp) != 0) {
        std::cout << "Error: VX_CFG_NUM_THREADS must be divisible by sparse B block size" << std::endl;
        std::abort();
      }
    }

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = is_sparse
                   ? (step_n % cfg::b_sub_blocks_sp) * cfg::b_block_size_sp
                   : (step_n % cfg::b_sub_blocks)    * cfg::b_block_size;

    // Prepare A tile [tcM][tcK]
    reg_data_t a_tile[cfg::tcM * cfg::tcK];
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        a_tile[i * cfg::tcK + z] = rs1_data.at(a_off + i * cfg::tcK + z);
      }
    }

    // Prepare B tile [tcM][tcN][tcK]
    reg_data_t b_tile[cfg::tcM * cfg::tcN * cfg::tcK];
    if (is_sparse) {
      constexpr uint32_t kCompression = 2;
      uint32_t ebits = elem_bits(fmt_s);
      uint32_t meta_bits = 32 / ebits;
      uint32_t bank = step_m * (cfg::k_steps / 2) + step_k;
      auto meta_bit = [&](uint32_t bit_idx) {
        return (sparse_meta_.at(wid).at(bank * kMaxMetaCols + bit_idx / 32) >> (bit_idx % 32)) & 1u;
      };
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        uint32_t row_base = i * meta_row_width(ebits);
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t j_sp = cfg::sym_sparse ? (j % (cfg::tcN / 2)) : j;
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            uint32_t b_idx = b_off + j_sp * cfg::tcK * kCompression + z * kCompression;
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < meta_bits; ++b) {
              lo |= meta_bit(row_base + meta_bits * z + b) << b;
              hi |= meta_bit(row_base + meta_bits * (cfg::tcK + z) + b) << b;
            }
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z].u32 =
                gather_sparse(rs2_data.at(b_idx).u32, rs2_data.at(b_idx + 1).u32, lo, hi, ebits);
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z] = rs2_data.at(b_off + j * cfg::tcK + z);
          }
        }
      }
    }

    fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d,
              a_tile, b_tile, rs3_data, rd_data, is_sparse, true);
  }

  void wgmma(uint32_t wid,
             uint32_t fmt_s,
             uint32_t fmt_d,
             uint32_t step_m,
             uint32_t step_n,
             uint32_t step_k,
             uint32_t a_desc,
             uint32_t b_desc,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             bool is_sparse,
             uint32_t cd_nregs,
             uint32_t is_a_smem) {
    __unused(cd_nregs);
    if (is_sparse && !vt::sparse_format_supported(fmt_s)) {
      std::cout << "Error: WGMMA_SP unsupported input format: "
                << vt::fmt_string(fmt_s) << " (id=" << fmt_s << ")" << std::endl;
      std::abort();
    }

    uint32_t ratio   = elem_ratio(fmt_s);
    uint32_t k_words = cfg::tcK;
    uint32_t e_bytes = elem_bits(fmt_s) / 8;

    // Decode smem descriptors (B always from smem, A optionally).
    lmem_desc_t sd_a, sd_b;
    if (step_k == 0 && step_m == 0 && step_n == 0) {
      if (is_a_smem) {
        sd_a = {uint64_t(VX_MEM_LMEM_BASE_ADDR) + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
        lmem_desc_[wid][0] = sd_a;
      }
      sd_b = {uint64_t(VX_MEM_LMEM_BASE_ADDR) + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};
      lmem_desc_[wid][1] = sd_b;
    } else {
      sd_a = lmem_desc_[wid][0];
      sd_b = lmem_desc_[wid][1];
    }
    // load_lmem_word distinguishes A from B by descriptor base.
    cur_a_desc_base_ = is_a_smem ? sd_a.base : ~uint64_t(0);
    // NRC: cd_nregs 0/1/2 → 8/16/32; xtileN = NRC * NT / xtileM.
    {
      uint32_t nrc = (cd_nregs == 0) ? 8 : (cd_nregs == 1) ? 16 : 32;
      cur_xtile_n_ = (nrc * VX_CFG_NUM_THREADS) / wg_cfg::xtileM;
    }

    // Prepare A tile [tcM][tcK]
    reg_data_t a_tile[cfg::tcM * cfg::tcK];
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      uint32_t a_row_idx = step_m * cfg::tcM + i;
      for (uint32_t z = 0; z < k_words; ++z) {
        if (is_a_smem) {
          uint32_t k_elem = (step_k * k_words + z) * ratio;
          a_tile[i * cfg::tcK + z].u32 = load_lmem_word(sd_a, a_row_idx, k_elem, fmt_s, false);
        } else {
          a_tile[i * cfg::tcK + z] = rs1_data.at(i * cfg::tcK + z);
        }
      }
    }

    // Prepare B tile [tcM][tcN][tcK]
    reg_data_t b_tile[cfg::tcM * cfg::tcN * cfg::tcK];
    if (is_sparse) {
      uint32_t ebits       = elem_bits(fmt_s);
      uint32_t rtl_i_ratio = 32 / ebits;
      uint32_t meta_row_w  = k_words * 2 * rtl_i_ratio;
      // Bank encoding {step_m, step_k_half}: m=1 → bank (cfg::k_steps/2).
      uint32_t wg_bank = step_m * (cfg::k_steps / 2) + step_k;
      // Metadata is preloaded into sparse_meta_ via TCU_LD before dispatch.
      auto meta_bit_wg = [&](uint32_t bit_idx) -> uint32_t {
        uint32_t word_idx = wg_bank * kMaxMetaCols + bit_idx / 32;
        uint32_t word_val = sparse_meta_.at(wid).at(word_idx);
        // Trace: META_RD (CSV: wid,step_m,step_k,bank,value).
        if (const char* p = std::getenv("VORTEX_TCU_TRACE")) {
          if (p[0] == '1' && (bit_idx % 32) == 0) {
            fprintf(stderr, "META_RD,%u,%u,%u,%u,0x%08x\n",
                    wid, step_m, step_k, wg_bank, word_val);
          }
        }
        return (word_val >> (bit_idx % 32)) & 1u;
      };
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        uint32_t row_base = i * meta_row_w;
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t b_col_idx = step_n * cfg::tcN + j;
          for (uint32_t z = 0; z < k_words; ++z) {
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < rtl_i_ratio; ++b) {
              lo |= meta_bit_wg(row_base + rtl_i_ratio * z              + b) << b;
              hi |= meta_bit_wg(row_base + rtl_i_ratio * (k_words + z) + b) << b;
            }
            constexpr uint32_t kCompression = 2;
            uint32_t k_elem_b0 = (step_k * k_words + z) * ratio * kCompression;
            uint32_t bword0 = load_lmem_word(sd_b, k_elem_b0,         b_col_idx, fmt_s, true);
            uint32_t bword1 = load_lmem_word(sd_b, k_elem_b0 + ratio, b_col_idx, fmt_s, true);
            uint32_t gathered = gather_sparse(bword0, bword1, lo, hi, ebits);
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z].u32 = gathered;
            // Trace: B-gather (CSV: wid,step_m,step_n,i,lane,bword0,bword1,lo,hi,gathered).
            if (const char* p = std::getenv("VORTEX_TCU_TRACE")) {
              if (p[0] == '1') {
                fprintf(stderr, "GATHER,%u,%u,%u,%u,%u,0x%08x,0x%08x,%u,%u,0x%08x\n",
                        wid, step_m, step_n, i, j*cfg::tcK+z,
                        bword0, bword1, lo, hi, gathered);
              }
            }
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t b_col_idx = step_n * cfg::tcN + j;
          for (uint32_t z = 0; z < k_words; ++z) {
            uint32_t k_elem = (step_k * k_words + z) * ratio;
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z].u32 =
                load_lmem_word(sd_b, k_elem, b_col_idx, fmt_s, true);
          }
        }
      }
    }

    fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d,
              a_tile, b_tile, rs3_data, rd_data, is_sparse, false);
    __unused(b_desc);
  }

  const PerfStats& perf_stats() const {
    // lmem_reads: total MemReq traffic from TcuTbuf (abuf + bbuf).
    perf_stats_.lmem_reads = simobject_->tbuf()->reads();
    return perf_stats_;
  }

private:
  uint32_t elem_bits(uint32_t fmt_s) const {
    switch (fmt_s) {
    case vt::fp32::id:
    case vt::tf32::id:
    case vt::int32::id:
      return 32;
    case vt::fp16::id:
    case vt::bf16::id:
      return 16;
    case vt::fp8::id:
    case vt::bf8::id:
    case vt::int8::id:
    case vt::uint8::id:
    case vt::mxfp8::id:
    case vt::mxbf8::id:
    case vt::mxint8::id:
      return 8;
    case vt::int4::id:
    case vt::uint4::id:
    case vt::mxfp4::id:
    case vt::nvfp4::id:
      return 4;
    default:
      std::abort();
    }
  }

  uint32_t elem_ratio(uint32_t fmt_s) const {
    return 32 / elem_bits(fmt_s);
  }

  uint32_t mx_tile_scale_blocks(uint32_t fmt_s) const {
    uint32_t logical_tile_k = cfg::k_steps * cfg::tcK * elem_ratio(fmt_s);
    uint32_t block_size = vt::mx_scale_block_size(fmt_s);
    return std::max(1u, logical_tile_k / block_size);
  }

  static uint8_t meta_byte(const std::vector<uint32_t>& words, uint32_t index) {
    return (words.at(index / 4) >> (8 * (index & 0x3))) & 0xff;
  }

  static int32_t trunc_shift(int32_t value, int32_t shift) {
    if (shift >= 0)
      return value << shift;
    uint32_t amount = static_cast<uint32_t>(-shift);
    uint32_t magnitude = value < 0 ? static_cast<uint32_t>(-value) : static_cast<uint32_t>(value);
    int32_t scaled = static_cast<int32_t>(magnitude >> amount);
    return value < 0 ? -scaled : scaled;
  }

  uint32_t eval_mx_fedp(uint32_t wid, uint32_t fmt_s, uint32_t fmt_d,
                        uint32_t step_m, uint32_t step_n, uint32_t step_k,
                        uint32_t i, uint32_t j, const reg_data_t* a_row,
                        const reg_data_t* b_col, uint32_t c_val, bool is_sparse) const {
#ifndef VX_CFG_TCU_MX_ENABLE
    __unused(wid, fmt_s, fmt_d, step_m, step_n, step_k, i, j, a_row, b_col, is_sparse);
    return c_val;
#else
    uint32_t ratio = elem_ratio(fmt_s);
    uint32_t scale_blocks_k = mx_tile_scale_blocks(fmt_s);
    uint32_t block_size = vt::mx_scale_block_size(fmt_s);
    auto scale_a = [&](uint32_t elem_k) {
      uint32_t row = step_m * cfg::tcM + i;
      return meta_byte(mx_meta_a_.at(wid), row * scale_blocks_k + elem_k / block_size);
    };
    auto scale_b = [&](uint32_t elem_k) {
      uint32_t col = step_n * cfg::tcN + j;
      return meta_byte(mx_meta_b_.at(wid), col * scale_blocks_k + elem_k / block_size);
    };

    if (fmt_d == vt::fp32::id) {
      uint32_t acc = c_val;
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        for (uint32_t e = 0; e < ratio; ++e) {
          uint32_t sparse_ratio = is_sparse ? 2 : 1;
          uint32_t elem_k = ((step_k * cfg::tcK + z) * ratio + e) * sparse_ratio;
          uint32_t xa, xb;
          if (fmt_s == vt::mxfp8::id) {
            xa = rv_mxfp8tof_s((a_row[z].u32 >> (8 * e)) & 0xff, scale_a(elem_k), 0, nullptr);
            xb = rv_mxfp8tof_s((b_col[z].u32 >> (8 * e)) & 0xff, scale_b(elem_k), 0, nullptr);
          } else if (fmt_s == vt::mxbf8::id) {
            xa = rv_mxbf8tof_s((a_row[z].u32 >> (8 * e)) & 0xff, scale_a(elem_k), 0, nullptr);
            xb = rv_mxbf8tof_s((b_col[z].u32 >> (8 * e)) & 0xff, scale_b(elem_k), 0, nullptr);
          } else if (fmt_s == vt::mxfp4::id) {
            xa = rv_mxfp4tof_s((a_row[z].u32 >> (4 * e)) & 0xf, scale_a(elem_k), 0, nullptr);
            xb = rv_mxfp4tof_s((b_col[z].u32 >> (4 * e)) & 0xf, scale_b(elem_k), 0, nullptr);
          } else if (fmt_s == vt::nvfp4::id) {
            xa = rv_nvfp4tof_s((a_row[z].u32 >> (4 * e)) & 0xf, scale_a(elem_k), 0, nullptr);
            xb = rv_nvfp4tof_s((b_col[z].u32 >> (4 * e)) & 0xf, scale_b(elem_k), 0, nullptr);
          } else {
            std::abort();
          }
          acc = rv_fadd_s(rv_fmul_s(xa, xb, 0, nullptr), acc, 0, nullptr);
        }
      }
      return acc;
    }

    if (fmt_s == vt::mxint8::id && fmt_d == vt::int32::id) {
      int32_t acc = bit_cast<int32_t>(c_val);
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        for (uint32_t e = 0; e < ratio; ++e) {
          uint32_t elem_k = ((step_k * cfg::tcK + z) * ratio + e) * (is_sparse ? 2 : 1);
          auto a = static_cast<int8_t>((a_row[z].u32 >> (8 * e)) & 0xff);
          auto b = static_cast<int8_t>((b_col[z].u32 >> (8 * e)) & 0xff);
          int32_t shift = int32_t(scale_a(elem_k)) + int32_t(scale_b(elem_k)) - 266;
          acc += trunc_shift(int32_t(a) * int32_t(b), shift);
        }
      }
      return bit_cast<uint32_t>(acc);
    }

    std::cout << "Error: unsupported MX mma format: " << fmt_s << " -> " << fmt_d << "!" << std::endl;
    std::abort();
#endif
  }

  // Gather one 32-bit operand word from a TCU line cache.
  // `read_line` is supplied by the caller and routes to the right per-role
  // buffer inside TcuTbuf (A → read_a, B → read_b). For sub-32-bit formats,
  // packs `ratio = 32 / e_bits` adjacent elements along the K direction
  // (col for A, row for B via pack_along_row).
  template <typename ReadLine>
  uint32_t gather_word(ReadLine read_line, const lmem_desc_t& desc,
                       uint32_t row, uint32_t col,
                       uint32_t fmt_s, bool pack_along_row) const {
    uint32_t e_bits  = elem_bits(fmt_s);
    uint32_t ratio   = (e_bits >= 32) ? 1 : (32 / e_bits);
    uint32_t e_bytes = (e_bits >= 8)  ? (e_bits / 8) : 1;
    uint32_t result = 0;
    for (uint32_t r = 0; r < ratio; ++r) {
      uint32_t cur_row = pack_along_row ? (row + r) : row;
      uint32_t cur_col = pack_along_row ? col       : (col + r);
      uint64_t byte_addr;
      if (desc.ldm == 0) {
        // Block-major SMEM. K dimension is along col for A (pack_along_row
        // false) and along row for B (pack_along_row true).
        uint32_t k_blk_dim = cfg::tcK * ratio;
        if (pack_along_row) {
          // B: r is K coord, c is N coord. Within-block layout: N outer, K inner.
          uint32_t k_blk = cur_row / k_blk_dim;
          uint32_t r_in  = cur_row % k_blk_dim;
          uint32_t n_blk = cur_col / cfg::tcN;
          uint32_t n_in  = cur_col % cfg::tcN;
          uint32_t b_blk_elems = k_blk_dim * cfg::tcN;
          uint32_t n_steps     = cur_xtile_n_ / cfg::tcN;
          uint64_t off = (k_blk * n_steps + n_blk) * b_blk_elems
                       + n_in * k_blk_dim + r_in;
          byte_addr = desc.base + off * e_bytes;
        } else {
          // A: r is M coord, c is K coord.
          uint32_t m_blk = cur_row / cfg::tcM;
          uint32_t i_in  = cur_row % cfg::tcM;
          uint32_t k_blk = cur_col / k_blk_dim;
          uint32_t k_in  = cur_col % k_blk_dim;
          uint32_t a_blk_elems = cfg::tcM * k_blk_dim;
          uint64_t off = (k_blk * wg_cfg::m_steps + m_blk) * a_blk_elems
                       + i_in * k_blk_dim + k_in;
          byte_addr = desc.base + off * e_bytes;
        }
      } else if (desc.col_major) {
        byte_addr = desc.base + (uint64_t(cur_col) * desc.ldm + cur_row) * e_bytes;
      } else if (pack_along_row) {
        // B: K-major (N-outer K-inner). cur_row=K, cur_col=N;
        // ldm = stride in elements between N rows.
        byte_addr = desc.base + (uint64_t(cur_col) * desc.ldm + cur_row) * e_bytes;
      } else {
        // A: row-major (M-outer K-inner). cur_row=M, cur_col=K.
        byte_addr = desc.base + (uint64_t(cur_row) * desc.ldm + cur_col) * e_bytes;
      }
      auto line = read_line(byte_addr);
      if (!line) {
        std::cout << "Error: TCU buffer miss at 0x" << std::hex << byte_addr
                  << std::dec << std::endl;
        std::abort();
      }
      uint32_t off = byte_addr & (VX_CFG_MEM_BLOCK_SIZE - 1);
      if (e_bits == 32) {
        result = (uint32_t((*line)[off    ]))        |
                 (uint32_t((*line)[off + 1]) << 8)   |
                 (uint32_t((*line)[off + 2]) << 16)  |
                 (uint32_t((*line)[off + 3]) << 24);
        return result;
      } else if (e_bits == 16) {
        uint32_t val = uint32_t((*line)[off]) | (uint32_t((*line)[off + 1]) << 8);
        result |= (val & 0xFFFF) << (r * 16);
      } else if (e_bits == 8) {
        result |= uint32_t((*line)[off]) << (r * 8);
      } else {
        // 4-bit (int4/uint4) not supported.
        std::cout << "Error: TCU 4-bit operand gather not supported" << std::endl;
        std::abort();
      }
    }
    return result;
  }

  // Routes A reads through the current block's A buffer; B through the shared B buffer.
  uint32_t load_lmem_word(const lmem_desc_t& desc, uint32_t row, uint32_t col,
                          uint32_t fmt_s, bool pack_along_row) const {
    auto& tbuf = simobject_->tbuf();
    if (desc.base == cur_a_desc_base_) {
      uint32_t b = cur_block_;
      return gather_word([&](uint64_t addr) { return tbuf->read_a(b, addr); },
                         desc, row, col, fmt_s, pack_along_row);
    }
    return gather_word([&](uint64_t addr) { return tbuf->read_b(addr); },
                       desc, row, col, fmt_s, pack_along_row);
  }

  static constexpr uint32_t kSparseKSteps = cfg::k_steps / 2;
  static constexpr uint32_t kMetaBanks = cfg::m_steps * kSparseKSteps;
  static constexpr uint32_t kMaxMetaCols = VX_CFG_NUM_THREADS / 2;

  // FEDP tile computation for both WMMA and WGMMA.
  // a_tile: [tcM][tcK] flat array of pre-loaded A operand words.
  // b_tile: [tcM][tcN][tcK] flat array of pre-loaded B operand words
  //         (for dense, each i-slice is identical; for sparse, already gathered).
  void fedp_tile(uint32_t wid,
                 uint32_t step_m, uint32_t step_n, uint32_t step_k,
                 uint32_t fmt_s, uint32_t fmt_d,
                 const reg_data_t* a_tile,
                 const reg_data_t* b_tile,
                 const std::vector<reg_data_t>& rs3_data,
                 std::vector<reg_data_t>& rd_data,
                 bool is_sparse,
                 bool allow_mx) {
    if (!allow_mx && vt::mx_scale_format(fmt_s)) {
      std::cout << "Error: MX formats are supported only by WMMA." << std::endl;
      std::abort();
    }
    PFN_FEDP fedp = vt::mx_scale_format(fmt_s) ? nullptr : select_FEDP(fmt_s, fmt_d);

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto t = i * cfg::tcN + j;
        auto c_val = rs3_data.at(t).u32;
        auto a_row = &a_tile[i * cfg::tcK];
        auto b_col = &b_tile[(i * cfg::tcN + j) * cfg::tcK];

        uint32_t d_val = (allow_mx && vt::mx_scale_format(fmt_s))
                       ? eval_mx_fedp(wid, fmt_s, fmt_d, step_m, step_n, step_k, i, j, a_row, b_col, c_val, is_sparse)
                       : fedp(a_row, b_col, c_val);

        rd_data.at(t).u64 = nan_box(d_val);
        DTH(3, simobject_->name() << " FEDP"
            << ": wid=" << wid << ", i=" << i << ", j=" << j
            << ", m=" << step_m << ", n=" << step_n
            << ", k=" << step_k << std::hex
            << ", a0=0x" << a_row[0].u32 << ", b0=0x" << b_col[0].u32
            << ", c=0x" << c_val << ", d=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  TcuUnit*   simobject_;
  Core*         core_;
  std::vector<std::vector<uint32_t>> sparse_meta_;
#ifdef VX_CFG_TCU_MX_ENABLE
  std::vector<std::vector<uint32_t>> mx_meta_a_;
  std::vector<std::vector<uint32_t>> mx_meta_b_;
#endif
  std::unordered_map<uint32_t, lmem_desc_t[2]> lmem_desc_;
  mutable PerfStats perf_stats_;
  // Per-block guard: execute already happened for this trace; reset on pop().
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> exec_done_;
  // Per-block bitmask of warp IDs with planned WGMMA lines; cleared on fu_unlock.
  std::array<uint64_t, VX_CFG_NUM_TCU_BLOCKS> wgmma_planned_warps_;
  // True while a block is between its first and last WGMMA uop.
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> in_wgmma_;
  // Current block index, set before delegating to wgmma().
  uint32_t cur_block_ = 0;
  // A-descriptor base for the current wgmma(); distinguishes A from B in load_lmem_word.
  uint64_t cur_a_desc_base_ = ~uint64_t(0);
  // xtileN for the active WGMMA (derived from NRC).
  uint32_t cur_xtile_n_ = 8;
  // CTA owner per block's A buffer and the shared B buffer (-1 = unowned).
  std::array<int32_t, VX_CFG_NUM_TCU_BLOCKS> cta_owner_a_{};
  int32_t cta_owner_b_ = -1;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t TcuUnit::op_string(TcuType tcu_type, IntrTcuArgs args) {
  bool is_sparse = tcu_is_sparse(tcu_type);
  switch (tcu_type) {
  case TcuType::WMMA:
  case TcuType::WMMA_SP:
    return {"WMMA." + std::string(is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  case TcuType::WGMMA:
  case TcuType::WGMMA_SP: {
    uint32_t nrc = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
    std::string src_mode = std::string(args.is_a_smem ? "S" : "R") + "S";
    return {"WGMMA." + std::string(is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(nrc) + "." + src_mode
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
  }
  case TcuType::TCU_LD:
    return {"TCU_LD." + std::string((args.fmt_d & 0x10) ? "MX." : "SP.")
             + std::string(vt::fmt_string(args.fmt_s)) + ".slot" + std::to_string(args.fmt_d & 0xf), ""};
  default:
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

uint32_t TcuUopGen::uop_count(const Instr& instr) {
  if (instr.get_fu_type() != FUType::TCU)
    return 1;

  auto tcu_type = std::get<TcuType>(instr.get_op_type());
#ifdef VX_CFG_TCU_WGMMA_ENABLE
  auto args = std::get<IntrTcuArgs>(instr.get_args());
#endif

  if (tcu_is_wmma(tcu_type)) {
    using wmma = vt::wmma_config_t<VX_CFG_NUM_THREADS>;
    bool is_sparse = tcu_is_sparse(tcu_type);
    // Metadata SRAM is filled by a preceding TCU_LD instruction.
    uint32_t k_count = is_sparse ? (wmma::k_steps / 2) : wmma::k_steps;
    uint32_t mma_steps = (wmma::sym_sparse && is_sparse)
                       ? (wmma::m_steps * wmma::n_steps * wmma::k_steps)
                       : (wmma::m_steps * wmma::n_steps * k_count);
    return mma_steps;
  }

#ifdef VX_CFG_TCU_WGMMA_ENABLE
  if (tcu_is_wgmma(tcu_type)) {
    bool is_sparse = tcu_is_sparse(tcu_type);
    uint32_t nrc = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
    uint32_t k_count = is_sparse ? (wg_cfg::k_steps / 2) : wg_cfg::k_steps;
    uint32_t mma_uops = k_count * nrc;
    return mma_uops;
  }
#endif

  return 1;
}

///////////////////////////////////////////////////////////////////////////////

Instr::Ptr TcuUopGen::get(const Instr& macro_instr, uint32_t uop_index) {
  auto tcu_type = std::get<TcuType>(macro_instr.get_op_type());
  auto args = std::get<IntrTcuArgs>(macro_instr.get_args());
  uint64_t parent_uuid = macro_instr.get_uuid();
  uint32_t total = uop_count(macro_instr);

  // Compute UUID for this micro-op
  uint32_t uuid_hi = (parent_uuid >> 32) & 0xffffffff;
  uint32_t uuid_lo = parent_uuid & 0xffffffff;
  uint32_t steps_shift = (total > 1) ? (32 - log2ceil(total)) : 0;
  uint64_t uop_uuid = (static_cast<uint64_t>(uuid_hi) << 32) | ((uop_index << steps_shift) | uuid_lo);

  auto uop_instr = std::allocate_shared<Instr>(pool_, uop_uuid, FUType::TCU);
  uop_instr->set_parent_uuid(parent_uuid);

  if (tcu_is_wmma(tcu_type)) {
    using wmma = vt::wmma_config_t<VX_CFG_NUM_THREADS>;
    constexpr uint32_t rc_base = 0, ra_base = 10;
    constexpr uint32_t rb_base = (wmma::NRB == 4) ? 28 : 24;
    bool is_sparse = tcu_is_sparse(tcu_type);
    uint32_t fmt_s = args.fmt_s;
    uint32_t fmt_d = args.fmt_d;

    {
      // MMA uops.
      uint32_t mma_idx = uop_index;
      uint32_t k_count = is_sparse ? (wmma::k_steps / 2) : wmma::k_steps;

      if (wmma::sym_sparse && is_sparse) {
        // Symmetric-sparse: flatten (m, n, k) into a single counter
        constexpr uint32_t lg_n = (wmma::n_steps > 1) ? log2ceil(wmma::n_steps) : 0;
        constexpr uint32_t lg_k = (wmma::k_steps > 1) ? log2ceil(wmma::k_steps) : 0;
        constexpr uint32_t step_bits = lg_n + lg_k;
        constexpr uint32_t step_mask = step_bits ? ((1u << step_bits) - 1) : 0;
        constexpr uint32_t sym_mask_lo = []() {
          uint32_t mask = 0;
          for (uint32_t lane = 0; lane < VX_CFG_NUM_THREADS; ++lane)
            if ((lane % wmma::tcN) < (wmma::tcN / 2)) mask |= (1u << lane);
          return mask;
        }();
        constexpr uint32_t all_lanes = (VX_CFG_NUM_THREADS == 32) ? 0xffffffffu : ((1u << VX_CFG_NUM_THREADS) - 1);

        uint32_t n_sp = step_bits ? (mma_idx & step_mask) : 0;
        uint32_t m_sp = mma_idx >> step_bits;
        // n_sp encodes both actual N step (high bits) and lo/hi half (low lg_k bits).
        // Extract just the N step for accum indexing; n_sp is still used for B register selection.
        uint32_t actual_n = lg_k ? (n_sp >> lg_k) : n_sp;
        uint32_t reg_rs3 = rc_base + (mma_idx >> 1);
        uop_instr->set_op_type(TcuType::WMMA_SP);
        uop_instr->set_args(IntrTcuArgs{0, 0, fmt_s, fmt_d, m_sp, actual_n, 0, 0, 0});
        uop_instr->set_dest_reg(reg_rs3, RegType::Float);
        uop_instr->set_src_reg(0, ra_base + m_sp, RegType::Float);
        uop_instr->set_src_reg(1, rb_base + n_sp, RegType::Float);
        uop_instr->set_src_reg(2, reg_rs3, RegType::Float);
        // Symmetric sparse: no k-loop, always wb=1 and always reads C from RF
        uop_instr->set_tmask(ThreadMask(VX_CFG_NUM_THREADS, (mma_idx & 1) ? (all_lanes & ~sym_mask_lo) : sym_mask_lo));
      } else {
        // Standard k-major triple loop (dense or non-sym sparse)
        uint32_t b_sub = is_sparse ? wmma::b_sub_blocks_sp : wmma::b_sub_blocks;
        uint32_t mn = wmma::m_steps * wmma::n_steps;
        uint32_t k = mma_idx / mn;
        uint32_t rem = mma_idx % mn;
        uint32_t m = rem / wmma::n_steps;
        uint32_t n = rem % wmma::n_steps;
        // Bank-conflict-free register offset formulas (0 stalls).
        // Permutes A, B, C offsets so all three operands always land in
        // different GPR banks for every uop.
        uint32_t reg_rs1, reg_rs2, reg_rs3;
        if (is_sparse) {
          // Sparse non-sym: A=identity, B and C permuted for 0 stalls.
          // B={n[hi], ~(n[0]^k), ~k}, C={n[hi], m, ~(m^n[0])}
          uint32_t n_hi = n >> 1;
          uint32_t n_lo = n & 1;
          uint32_t a_off = (m / wmma::a_sub_blocks) * k_count + k;
          uint32_t b_off = (n_hi << 2) | ((1 - (n_lo ^ k)) << 1) | (1 - k);
          uint32_t c_off = (n_hi << 2) | (m << 1) | (1 - (m ^ n_lo));
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + b_off;
          reg_rs3 = rc_base + c_off;
        } else if (b_sub == 1) {
          // Dense Pattern A (NT=4,16,64): m_steps=4, k_steps=2, n_steps=2
          // A={m[0], ~m[hi], k}, B={n^k, ~k}, C={m[0], ~m[hi], XNOR(m[hi],n)}
          uint32_t m_hi = m >> 1;
          uint32_t m_lo = m & 1;
          uint32_t a_off = (m_lo << 2) | ((1 - m_hi) << 1) | k;
          uint32_t b_off = ((n ^ k) << 1) | (1 - k);
          uint32_t c_off = (m_lo << 2) | ((1 - m_hi) << 1) | (1 - (m_hi ^ n));
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + b_off;
          reg_rs3 = rc_base + c_off;
        } else {
          // Dense Pattern B (NT=8,32): m_steps=2, k_steps=4, n_steps=4, b_sub=2
          // A={k[0], ~m, m^k[hi]}, B={k[0], k[hi]^np, ~np}, C={n[0], ~m, n[hi]}
          uint32_t k_hi = k >> 1;
          uint32_t k_lo = k & 1;
          uint32_t n_pair = n >> 1;
          uint32_t n_lo = n & 1;
          uint32_t a_off = (k_lo << 2) | ((1 - m) << 1) | (m ^ k_hi);
          uint32_t b_off = (k_lo << 2) | ((k_hi ^ n_pair) << 1) | (1 - n_pair);
          uint32_t c_off = (n_lo << 2) | ((1 - m) << 1) | n_pair;
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + b_off;
          reg_rs3 = rc_base + c_off;
        }
        uop_instr->set_op_type(is_sparse ? TcuType::WMMA_SP : TcuType::WMMA);
        uop_instr->set_args(IntrTcuArgs{0, 0, fmt_s, fmt_d, m, n, k, 0, 0});
        uop_instr->set_dest_reg(reg_rs3, RegType::Float);
        uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
        uop_instr->set_src_reg(1, reg_rs2, RegType::Float);
        uop_instr->set_src_reg(2, reg_rs3, RegType::Float);
      }
    }
  }
#ifdef VX_CFG_TCU_WGMMA_ENABLE
  else if (tcu_is_wgmma(tcu_type)) {
    constexpr uint32_t m_steps = wg_cfg::m_steps;
    constexpr uint32_t k_steps = wg_cfg::k_steps;
    uint32_t fmt_s = args.fmt_s;
    uint32_t fmt_d = args.fmt_d;
    bool is_sparse = tcu_is_sparse(tcu_type);
    bool is_a_smem = args.is_a_smem;
    uint32_t cd_nregs = args.cd_nregs;
    uint32_t k_count = is_sparse ? (k_steps / 2) : k_steps;
    constexpr uint32_t a0 = 10, a1 = 11;

    {
      // MMA phase
      uint32_t mma_idx = uop_index;
      uint32_t ra_base = is_a_smem ? 10 : 24;

      // Loop order: m (inner) -> n (middle) -> k (outer). K-outer maximizes
      // per-block A-buffer reuse: each A_w[m,k] is consumed across the entire
      // (n,m) inner sweep, and each shared B[k,n] is consumed for m_steps
      // consecutive uops.
      uint32_t mn = total / k_count;
      uint32_t k = mma_idx / mn;
      uint32_t rem = mma_idx % mn;
      uint32_t n = rem / m_steps;
      uint32_t m = rem % m_steps;
      uint32_t r = n * m_steps + m;

      uop_instr->set_op_type(is_sparse ? TcuType::WGMMA_SP : TcuType::WGMMA);
      bool first = (uop_index == 0);
      bool last  = (uop_index == (total - 1));
      uop_instr->set_args(IntrTcuArgs{is_a_smem ? 1u : 0u, cd_nregs,
                                     fmt_s, fmt_d, m, n, k, first ? 1u : 0u, last ? 1u : 0u});
      uop_instr->set_dest_reg(r, RegType::Float);
      if (mma_idx == 0) {
        if (is_a_smem) {
          uop_instr->set_src_reg(0, a0, RegType::Integer);
        } else {
          uint32_t rs1_off = m * k_count + k;
          uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
        }
        uop_instr->set_src_reg(1, a1, RegType::Integer);
      } else if (!is_a_smem) {
        uint32_t rs1_off = m * k_count + k;
        uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
      }
      uop_instr->set_src_reg(2, r, RegType::Float);
    }
    // fu_lock on first uop, fu_unlock on last uop
    uop_instr->set_fu_lock(uop_index == 0);
    uop_instr->set_fu_unlock(uop_index == (total - 1));
  }
#endif

  return uop_instr;
}

///////////////////////////////////////////////////////////////////////////////

TcuUnit::TcuUnit(const SimContext &ctx, const char* name, Core* core)
	: FuncUnit(ctx, name, core)
	, impl_(new Impl(this, core))
{
  char sname[128];
  snprintf(sname, sizeof(sname), "%s-tbuf", name);
  tbuf_ = TcuTbuf::Create(sname);
}

TcuUnit::~TcuUnit() {
  delete impl_;
}

TcuTbuf::Ptr& TcuUnit::tbuf() {
  return tbuf_;
}

void TcuUnit::on_reset() {
  impl_->reset();
}

void TcuUnit::on_tick() {
  impl_->tick();
}

const TcuUnit::PerfStats &TcuUnit::perf_stats() const {
	return impl_->perf_stats();
}

void TcuUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      uint32_t step_k,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      bool is_sparse) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, step_k,
              rs1_data, rs2_data, rs3_data,
              rd_data,
              is_sparse);
}

void TcuUnit::wgmma(uint32_t wid,
                       uint32_t fmt_s,
                       uint32_t fmt_d,
                       uint32_t step_m,
                       uint32_t step_n,
                       uint32_t step_k,
                       uint32_t a_desc,
                       uint32_t b_desc,
                       const std::vector<reg_data_t>& rs1_data,
                       const std::vector<reg_data_t>& rs3_data,
                       std::vector<reg_data_t>& rd_data,
                       bool is_sparse,
                       uint32_t cd_nregs,
                       uint32_t is_a_smem) {
  impl_->wgmma(wid, fmt_s, fmt_d, step_m, step_n, step_k, a_desc, b_desc,
               rs1_data, rs3_data, rd_data,
               is_sparse, cd_nregs, is_a_smem);
}
