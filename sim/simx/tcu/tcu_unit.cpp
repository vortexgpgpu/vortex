
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
#include "fragment_layout.h"
#include <rvfloats.h>
#include <fedp.h>
#include "core.h"
#include "scheduler.h"
#include "local_mem.h"
#include "processor_impl.h"
#include "mem/memory.h"
#include "stack_interleave.h"
#include <cstring>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <type_traits>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg    = vt::wmma_config_t<VX_CFG_NUM_THREADS>;
using wg_cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::fp32, vt::fp32>;
static constexpr uint32_t kFedpWords = wg_cfg::fedpK;
#ifdef VX_CFG_TCU_FEDP2K
static constexpr bool kFedp2K = true;
#else
static constexpr bool kFedp2K = false;
#endif
// A tensor op's sparse step covers fedpK compressed A words (TMEM builds with
// FEDP2K), as Blackwell's sparse tcgen05.mma; a warp WGMMA_SP uop covers tcK.
static constexpr bool kSpFullK = (wg_cfg::sp_k_mult > 1);
// Dense FEDP2K RS uops read A from both rs1 and rs2, so they carry the B
// descriptor in a leading setup uop.
static constexpr bool wgmma_needs_setup(bool is_a_smem, bool is_sparse) {
  return kFedp2K && !is_a_smem && !is_sparse;
}

#if defined(VX_CFG_TCU_TMEM_ENABLE) && !defined(VX_CFG_TCU_WGMMA_ENABLE)
#error "VX_CFG_TCU_TMEM_ENABLE requires VX_CFG_TCU_WGMMA_ENABLE"
#endif
#if !defined(VX_CFG_TCU_WMMA_ENABLE) && !defined(VX_CFG_TCU_TMEM_ENABLE)
#error "disabling VX_CFG_TCU_WMMA_ENABLE requires VX_CFG_TCU_TMEM_ENABLE"
#endif

// Dot-product pipeline depth of the configured tensor-PE type.
static constexpr uint32_t kFedpLatency = VX_CFG_TCU_LATENCY;
// End-to-end MMA uop cost: dispatch plus the dot-product pipeline (an
// unblocked result bypasses the landing queue).
static constexpr uint32_t kMmaLatency = 1 + kFedpLatency;
// The FEDP array holds tcM x tcN / kPeRatio dot products: a compute uop
// occupies it for kPeRatio cycles, one output slice per cycle.
static constexpr uint32_t kPeRatio = VX_CFG_TCU_PE_RATIO;
static_assert(kPeRatio != 0 && (kPeRatio & (kPeRatio - 1)) == 0
           && ((cfg::tcM * cfg::tcN) % kPeRatio) == 0,
              "VX_CFG_TCU_PE_RATIO must be a power of two that divides tcM x tcN");
#ifdef VX_CFG_TCU_WGMMA_ENABLE
static_assert(VX_CFG_LMEM_ENABLED, "TCU_WGMMA_ENABLE requires LMEM_ENABLE");
static_assert(VX_CFG_LMEM_LOG_SIZE <= vt::smem_desc_offset_bits,
              "LMEM exceeds the smem descriptor offset field");
#endif

static constexpr uint64_t kLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// WGMMA tile-buffer geometry in 32-bit words.
static constexpr uint32_t kLmemBanks     = VX_CFG_LMEM_NUM_BANKS;
static constexpr uint32_t kBankRowWords  = kLmemBanks * (VX_CFG_XLEN / 32);
#ifdef VX_CFG_TCU_WGMMA_ENABLE
// A slice-aligned row-major K-slice then never straddles two bank rows.
static_assert((kBankRowWords % kFedpWords) == 0 && (kBankRowWords % (2 * cfg::tcK)) == 0,
              "a WGMMA K-slice must divide an LMEM bank row");
#endif
static constexpr uint32_t kBBlockWords  = kFedpWords * cfg::tcN;
static constexpr uint32_t kBBlockWordsSp = cfg::tcK * cfg::tcN * 2;
// One bank row holds kBSubBlocks consecutive dense B blocks.
static constexpr uint32_t kBSubBlocks =
    (VX_CFG_NUM_THREADS > kBBlockWords) ? (VX_CFG_NUM_THREADS / kBBlockWords) : 1;
// A sparse B block spanning two 32-bit bank rows takes two reads.
static constexpr bool kBSparseTwoFetch =
    (kBBlockWordsSp == 2 * kLmemBanks) && (VX_CFG_XLEN == 32);
// A block-major A stripe is m_steps blocks, each padded to whole bank rows.
static constexpr uint32_t a_stripe_reads(uint32_t block_words) {
  return (wg_cfg::m_steps * ((block_words + kLmemBanks - 1) / kLmemBanks) * kLmemBanks
          + kBankRowWords - 1) / kBankRowWords;
}
static constexpr uint32_t kAStripeReads   = a_stripe_reads(cfg::tcM * kFedpWords);
static constexpr uint32_t kAStripeReadsSp = a_stripe_reads(cfg::tcM * cfg::tcK);

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

static inline uint8_t unpack_u8(uint32_t word, uint32_t idx) {
  return (word >> (idx * 8)) & 0xffu;
}

// FMA<It, Ot>: one multiply-add step into an Ot-typed accumulator (bit-packed
// in uint32): fp32 for float inputs, int32 for integer inputs.
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
// -- bf16 inputs --
template <> struct FMA<vt::bf16, vt::fp32> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_btof_s(a, 0, nullptr);
    auto fb = rv_btof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
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
// -- bf8 inputs --
template <> struct FMA<vt::bf8, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e5m2tof_s(a, 0, nullptr);
    auto fb = rv_e5m2tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
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
// Float source formats the native TFR datapath reduces (exponent, mantissa bits).
template <typename It> struct TfrFormat { static constexpr bool valid = false; };
template <> struct TfrFormat<vt::fp16> { static constexpr bool valid = true; static constexpr int eb = 5, sb = 10; };
template <> struct TfrFormat<vt::bf16> { static constexpr bool valid = true; static constexpr int eb = 8, sb = 7; };
template <> struct TfrFormat<vt::fp8>  { static constexpr bool valid = true; static constexpr int eb = 4, sb = 3; };
template <> struct TfrFormat<vt::bf8>  { static constexpr bool valid = true; static constexpr int eb = 5, sb = 2; };
template <> struct TfrFormat<vt::tf32> { static constexpr bool valid = true; static constexpr int eb = 8, sb = 10; };

// Generic FEDP, keyed on the output format.
//   * fp32 Ot on the native TFR datapath: its fixed-point reduction —
//     products aligned to the largest exponent in a 25-bit window, summed as
//     integers, rounded once (the TfrFedp model the RTL is checked against).
//   * fp32 Ot otherwise: accumulate Σ(a_k*b_k) in fp32 per register,
//     the register sums in order, c_val last (the DPI datapath's order).
//   * int32 Ot (integer inputs): exact integer accumulation.
template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    return eval_n(a_row, b_col, c_val, cfg::tcK);
  }

  static uint32_t eval_n(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t k_words) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
#ifdef VX_CFG_TCU_TYPE_TFR
    if constexpr (std::is_same_v<Ot, vt::fp32> && TfrFormat<It>::valid) {
      std::vector<uint32_t> a(k_words), b(k_words);
      for (uint32_t z = 0; z < k_words; ++z) {
        a[z] = a_row[z].u32;
        b[z] = b_col[z].u32;
      }
      TfrFedp fedp(TfrFormat<It>::eb, TfrFormat<It>::sb);
      float c;
      std::memcpy(&c, &c_val, sizeof(c));
      float d = fedp(a.data(), b.data(), c, k_words);
      uint32_t d_bits;
      std::memcpy(&d_bits, &d, sizeof(d_bits));
      return d_bits;
    } else
#endif
    if constexpr (std::is_same_v<Ot, vt::fp32>) {
      uint32_t acc = 0;
      for (uint32_t z = 0; z < k_words; ++z) {
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
      static_assert(std::is_same_v<Ot, vt::int32>, "TCU output type must be fp32 (int32 for integer inputs)");
      uint32_t acc = c_val;
      for (uint32_t z = 0; z < k_words; ++z) {
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
    return eval_n(a_row, b_col, c_val, cfg::tcK);
  }

  static uint32_t eval_n(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t k_words) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < k_words; ++z) {
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
    return eval_n(a_row, b_col, c_val, cfg::tcK);
  }

  static uint32_t eval_n(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t k_words) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < k_words; ++z) {
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

using PFN_FEDP_N = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t, uint32_t);

static PFN_FEDP_N select_FEDP_N(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp32>::eval_n;
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::fp32>::eval_n;
    case vt::fp8::id:
      return FEDP<vt::fp8, vt::fp32>::eval_n;
    case vt::bf8::id:
      return FEDP<vt::bf8, vt::fp32>::eval_n;
    case vt::tf32::id:
      return FEDP<vt::tf32, vt::fp32>::eval_n;
    default:
      break;
    }
    break;
  case vt::int32::id:
    switch (IT) {
    case vt::int8::id:
      return FEDP<vt::int8, vt::int32>::eval_n;
    case vt::uint8::id:
      return FEDP<vt::uint8, vt::int32>::eval_n;
    case vt::int4::id:
      return FEDP<vt::int4, vt::int32>::eval_n;
    case vt::uint4::id:
      return FEDP<vt::uint4, vt::int32>::eval_n;
    default:
      break;
    }
    break;
  default:
    break;
  }
  std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
  std::abort();
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
    , sparse_meta_(kMetaSlots, std::vector<uint32_t>(kMetaBanks * kMaxMetaCols, 0))
  #ifdef VX_CFG_TCU_MX_ENABLE
    , mx_meta_a_(VX_CFG_NUM_WARPS, std::vector<uint32_t>(VX_CFG_NUM_THREADS, 0))
    , mx_meta_b_(VX_CFG_NUM_WARPS, std::vector<uint32_t>(VX_CFG_NUM_THREADS, 0))
  #endif
    , perf_stats_()
  {
    exec_done_.fill(false);
    tj_fresh_.fill(true);
    tj_hold_until_.fill(0);
    slices_.fill(0);
    in_wgmma_.fill(false);
    wgmma_desc_.fill({0, 0});
  #ifdef VX_CFG_TCU_TMEM_ENABLE
    tmem_.assign(kTmemLanes * kTmemCols, reg_data_t{});
  #endif
  }

  ~Impl() {}

  void reset() {
    // perf_stats_ survives: the MPM counters run from device reset.
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
    slices_.fill(0);
    for (auto& due : result_due_) {
      due.clear();
    }
    in_wgmma_.fill(false);
    lmem_desc_.clear();
    wgmma_desc_.fill({0, 0});
    cta_owner_a_.fill(-1);
  #ifdef VX_CFG_TCU_WGMMA_ENABLE
    abuf_.fill(abuf_slot_t{});
    bbuf_ = bbuf_slot_t{};
  #endif
    cur_block_ = 0;
  #ifdef VX_CFG_TCU_TMEM_ENABLE
    std::fill(tmem_.begin(), tmem_.end(), reg_data_t{});
    tmem_used_ = 0;
    tmem_owner_.fill(-1);
    ops_.clear();
    run_valid_ = false;
    commits_.clear();
    step_wb_.clear();
    spans_.clear();
    steps_issued_ = 0;
    steps_done_ = 0;
    blk_port_ops_.fill(0);
    tmem_rd_taken_ = false;
  #endif
  #ifdef TCU_META_ENABLE
    agu_.fill(agu_state_t{});
    agu_issue_rr_ = 0;
  #endif
  }

#ifdef TCU_META_ENABLE
  // Each TCU block's AGU tracks one metadata load and writes each returned
  // lane straight into the block's metadata SRAM.
  static constexpr uint32_t kAguBeats =
      (VX_CFG_NUM_THREADS + VX_CFG_NUM_LSU_LANES - 1) / VX_CFG_NUM_LSU_LANES;

  struct agu_state_t {
    bool busy = false;
    bool complete = false;
    instr_trace_t* trace = nullptr;
    uint32_t wid = 0;
    uint32_t selector = 0;
    uint32_t next_beat = 0;
    uint32_t rsp_count = 0;
    std::array<uint64_t, VX_CFG_NUM_THREADS> addrs{};
  };
  std::array<agu_state_t, VX_CFG_NUM_TCU_BLOCKS> agu_;
  uint32_t agu_issue_rr_ = 0;

  void agu_start(uint32_t block, uint32_t wid, uint32_t selector,
                 uint64_t base_addr, instr_trace_t* trace) {
    assert((base_addr & 3) == 0 && "metadata base must be word-aligned");
    auto& agu = agu_.at(block);
    agu = agu_state_t{};
    agu.busy     = true;
    agu.trace    = trace;
    agu.wid      = wid;
    agu.selector = selector;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      agu.addrs.at(t) = base_addr + uint64_t(t) * 4;
      // Metadata is read as a linear warp-wide tile, which an interleaved
      // per-thread stack cannot provide.
      if (StackInterleave::contains(agu.addrs.at(t))) {
        std::cout << "Error: TCU metadata inside the thread stack window: addr=0x"
                  << std::hex << agu.addrs.at(t) << std::dec << std::endl;
        std::abort();
      }
    }
  }

  LsuReq agu_make_req(uint32_t block, agu_state_t& agu) {
    LsuReq req(VX_CFG_NUM_LSU_LANES);
    req.op   = MemOp::LD;
    req.tag  = block * kAguBeats + agu.next_beat;
    req.wid  = agu.wid;
    req.cid  = agu.trace->cid;
    req.uuid = agu.trace->uuid;
    for (uint32_t i = 0; i < VX_CFG_NUM_LSU_LANES; ++i) {
      uint32_t idx = agu.next_beat * VX_CFG_NUM_LSU_LANES + i;
      if (idx >= VX_CFG_NUM_THREADS) {
        break;
      }
      req.mask.set(i);
      req.addrs.at(i) = agu.addrs.at(idx);
      req.tids.at(i)  = idx % VX_CFG_NUM_THREADS;
    }
    ++agu.next_beat;
    return req;
  }

  void agu_receive(const LsuRsp& rsp) {
    uint32_t block = (uint32_t)rsp.tag / kAguBeats;
    uint32_t beat = (uint32_t)rsp.tag % kAguBeats;
    auto& agu = agu_.at(block);
    for (uint32_t lane = 0; lane < rsp.mask.size(); ++lane) {
      if (!rsp.mask.test(lane)) {
        continue;
      }
      uint32_t idx = beat * VX_CFG_NUM_LSU_LANES + lane;
      assert(idx < VX_CFG_NUM_THREADS && rsp.data.at(lane));
      uint32_t off = agu.addrs.at(idx) & (VX_CFG_MEM_BLOCK_SIZE - 1);
      uint32_t word = 0;
      std::memcpy(&word, rsp.data.at(lane)->data() + off, 4);
      this->agu_write(agu, idx, word);
      ++agu.rsp_count;
    }
    if (agu.busy && !agu.complete && agu.rsp_count == VX_CFG_NUM_THREADS) {
      agu.complete = true;
    }
  }

  // A load's first request goes out in the cycle the AGU takes the load: the
  // channel's one-cycle delay is the AGU's busy register, whose output drives
  // the LSU scheduler's request queue directly.
  void agu_issue() {
    if (TCU_AGU_PORTS == 1) {
      // every block shares one port: round-robin one request per cycle
      if (!simobject_->agu_req_out.at(0).full()) {
        for (uint32_t offset = 0; offset < VX_CFG_NUM_TCU_BLOCKS; ++offset) {
          uint32_t block = (agu_issue_rr_ + offset) % VX_CFG_NUM_TCU_BLOCKS;
          auto& agu = agu_.at(block);
          if (!agu.busy || agu.next_beat >= kAguBeats) {
            continue;
          }
          simobject_->agu_req_out.at(0).send(this->agu_make_req(block, agu));
          agu_issue_rr_ = (block + 1) % VX_CFG_NUM_TCU_BLOCKS;
          break;
        }
      }
    } else {
      // each block issues on its own port
      for (uint32_t block = 0; block < VX_CFG_NUM_TCU_BLOCKS; ++block) {
        auto& agu = agu_.at(block);
        if (!agu.busy || agu.next_beat >= kAguBeats
         || simobject_->agu_req_out.at(block).full()) {
          continue;
        }
        simobject_->agu_req_out.at(block).send(this->agu_make_req(block, agu));
      }
    }
  }

  void agu_collect() {
    for (uint32_t p = 0; p < TCU_AGU_PORTS; ++p) {
      auto& rsp_in = simobject_->agu_rsp_in.at(p);
      if (rsp_in.empty()) {
        continue;
      }
      this->agu_receive(rsp_in.peek());
      rsp_in.pop();
    }
  }

  // A TCU_LD retires once its words are deposited into the metadata SRAM;
  // the AGU takes the next load the cycle after.
  void agu_retire(uint64_t now) {
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& agu = agu_.at(b);
      if (!agu.complete)
        continue;
      // The result is offered in the cycle its last fragment lands, as the
      // AGU's done flag; the lane gather adds the one register to commit.
      if (simobject_->Outputs.at(b).try_send(agu.trace, 0)) {
        result_due_.at(b).push_back(now);
        DT(3, simobject_->name() << " execute: op=" << TcuType::TCU_LD << ", " << *agu.trace);
        agu = agu_state_t{};
      }
    }
  }

  // Write lane T's fetched word into the selected metadata SRAM.
  void agu_write(const agu_state_t& agu, uint32_t T, uint32_t word) {
    uint32_t wid = agu.wid;
  #ifdef VX_CFG_TCU_MX_ENABLE
    if (agu.selector & 0x10) {
      auto& dst = (agu.selector & 1) ? mx_meta_b_.at(wid) : mx_meta_a_.at(wid);
      dst.at(T) = word;
      return;
    }
  #endif
  #ifdef VX_CFG_TCU_SPARSE_ENABLE
    uint32_t slot_idx = agu.selector & 0xf;
    // Map lane T to its (bank, col) metadata cell using the host pack layout:
    //   flat_store = slot*cols_per_load + T/BPS
    //   col  = flat_store / stores_per_col
    //   bank = (flat_store % stores_per_col)*BPS + T%BPS
    // Covers NT < kMetaBanks (stores_per_col > 1) as well as NT >= kMetaBanks.
    constexpr uint32_t PWD = kMetaBanks;
    constexpr uint32_t BPS = cfg::banks_per_store;
    constexpr uint32_t SPC = cfg::stores_per_col;
    constexpr uint32_t CPL = cfg::meta_cols_per_load;
    uint32_t store_in_load   = T / BPS;
    uint32_t thread_in_store = T % BPS;
    uint32_t flat_store      = slot_idx * CPL + store_in_load;
    uint32_t col             = flat_store / SPC;
    uint32_t store_in_col    = flat_store % SPC;
    uint32_t bank            = store_in_col * BPS + thread_in_store;
    if (bank >= PWD || col >= kMaxMetaCols) {
      return;
    }
    sparse_meta_.at(wid).at(bank * kMaxMetaCols + col) = word;
    // Trace: META_SRAM write (CSV: wid,bank,col,addr,value).
    if (const char* p = std::getenv("VORTEX_TCU_TRACE")) {
      if (p[0] == '1') {
        fprintf(stderr, "META_TRC,%u,%u,%u,0x%lx,0x%08x\n",
                wid, bank, col, (unsigned long)agu.addrs.at(T), word);
      }
    }
  #else
    __unused(T, word);
  #endif
  }
#endif // TCU_META_ENABLE

  void tick() {
    cyc_ = {};
  #ifdef TCU_META_ENABLE
    this->agu_collect();
  #endif
  #ifdef VX_CFG_TCU_WGMMA_ENABLE
    uint32_t wgmma_ready = this->tbuf_step();
    uint32_t setup_fired = 0;
  #endif
  #ifdef VX_CFG_TCU_TMEM_ENABLE
    this->job_step();
    tj_port_busy_ = false;
  #endif

    uint64_t now = SimPlatform::instance().cycles();
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      // A result completing in a cycle the result port refuses it holds
      // admission for the cycle. The port refuses while the lane-gather
      // buffer is full and, in VX_tcu_block, while the AGU's TCU_LD result
      // takes it.
      auto& due = result_due_.at(b);
      due.erase(std::remove_if(due.begin(), due.end(), [now](uint64_t t) { return t < now; }), due.end());
      bool completing = std::any_of(due.begin(), due.end(), [now](uint64_t t) { return t == now; });
      bool result_ready = true;
      auto& stall_in = simobject_->result_stall_in.at(b);
      while (!stall_in.empty()) {
        result_ready &= (stall_in.peek() != now);
        stall_in.pop();
      }
    #ifdef TCU_META_ENABLE
      result_ready &= !agu_.at(b).complete;
    #endif
      auto& input = simobject_->Inputs.at(b);
      if (input.empty()) {
      #ifdef VX_CFG_TCU_TMEM_ENABLE
        tj_fresh_.at(b) = true;
      #endif
        continue;
      }
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      auto tpuArgs = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());

    #ifdef TCU_META_ENABLE
      // A TCU_LD leaves the input for the AGU once the AGU is free, so the ops
      // behind it run while its metadata is in flight; it retires from the AGU.
      if (tcu_type == TcuType::TCU_LD) {
        if (!agu_.at(b).busy) {
          // rs1 is a full-width address (.u64); use u64 to avoid truncation on XLEN=64.
          auto& rs1_data = trace->src_data[0];
          uint64_t base_addr = rs1_data.empty() ? 0 : rs1_data.at(0).u64;
          this->agu_start(b, trace->wid, tpuArgs.fmt_d, base_addr, trace);
          input.pop();
        }
        continue;
      }
    #endif

      if (completing && !result_ready) {
        continue;
      }

    #ifdef VX_CFG_TCU_TMEM_ENABLE
      // TMEM and tc ops pass a registered select stage: one reaching an idle
      // input is taken the cycle after; one queued behind follows back to back.
      bool is_tj_op = tcu_is_tj_op(tcu_type);
      if (is_tj_op) {
        if (tj_fresh_.at(b)) {
          tj_fresh_.at(b) = false;
          tj_hold_until_.at(b) = now + 1;
        }
        if (now < tj_hold_until_.at(b)) {
          continue;
        }
      }
    #else
      constexpr bool is_tj_op = false;
    #endif

    #ifdef VX_CFG_TCU_WGMMA_ENABLE
      // A WGMMA compute uop fires once its tile-buffer operands are ready.
      if (tcu_is_wgmma(tcu_type) && !tpuArgs.is_setup_uop
       && !exec_done_.at(b) && !((wgmma_ready >> b) & 1u)) {
        cyc_.opnd = true;
        continue;
      }
    #endif

      // Execute once per trace; results persist across backpressure retries
      // via exec_done_[b].
      if (!exec_done_.at(b)) {
        uint32_t wid = trace->wid;
        uint32_t num_threads = VX_CFG_NUM_THREADS;
        auto& rs1_data = trace->src_data[0];
        auto& rs2_data = trace->src_data[1];
        auto& rs3_data = trace->src_data[2];
        trace->dst_data.assign(num_threads, reg_data_t{});
        auto& rd_data = trace->dst_data;

        if (!is_tj_op && !vt::tcu_output_valid(tpuArgs.fmt_s, tpuArgs.fmt_d)) {
          std::cerr << "*** TCU output format " << tpuArgs.fmt_d << " for input format " << tpuArgs.fmt_s
                    << " is not supported: the TCU returns fp32 (int32 for integer inputs)" << std::endl;
          std::abort();
        }

        switch (tcu_type) {
        case TcuType::WMMA:
        case TcuType::WMMA_SP:
          this->wmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d,
                     tpuArgs.step_m, tpuArgs.step_n, tpuArgs.step_k,
                     rs1_data, rs2_data, rs3_data, rd_data,
                     tcu_is_sparse(tcu_type));
          ++cyc_.busy_blocks;
          break;
      #ifdef VX_CFG_TCU_WGMMA_ENABLE
        case TcuType::WGMMA:
        case TcuType::WGMMA_SP: {
          uint32_t a_desc = rs1_data.empty() ? 0 : rs1_data.at(0).u32;
          uint32_t b_desc = rs2_data.empty() ? 0 : rs2_data.at(0).u32;
          cur_block_ = b;
          int32_t this_cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
          // CTA lockstep invariant: no block may execute a WGMMA uop for a
          // different cta_id while another block is mid-WGMMA.
          if (!tpuArgs.is_setup_uop) {
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
                      a_desc, b_desc, rs1_data, rs2_data, rs3_data, rd_data,
                      tcu_is_sparse(tcu_type),
                      tpuArgs.cd_nregs, tpuArgs.is_a_smem, tpuArgs.is_setup_uop);
          if (tpuArgs.is_setup_uop) {
            setup_fired |= (1u << b);
          } else {
            ++cyc_.busy_blocks;
            // The block owns its CTA from its first compute uop to its last.
            if (tpuArgs.is_first_uop) {
              in_wgmma_.at(b) = true;
              cta_owner_a_.at(b) = this_cta;
            }
            if (tpuArgs.is_last_uop) {
              in_wgmma_.at(b) = false;
              cta_owner_a_.at(b) = -1;
            }
          }
        } break;
      #endif
      #ifdef VX_CFG_TCU_TMEM_ENABLE
        case TcuType::TMEM_LD:
        case TcuType::TMEM_ST: {
          // A warp's transfers go to its own block's TMEM lanes, through the
          // block's port. taddr = {lane [31:16], column [15:0]}; the lane is
          // the block's first. Lane per thread: thread t holds lane
          // + t % xtileM and register r column + r * kTmemThreadsPerLane
          // + t / xtileM. Fragment: register r is the fragment of M step
          // r % m_steps, columns + (r / m_steps) * tcN: thread t holds lane
          // + m * tcM + t / tcN, column + t % tcN (the WGMMA accumulator layout).
          uint32_t taddr = rs1_data.at(0).u;
          uint32_t r = tpuArgs.tmem_idx;
          uint32_t lane0 = taddr >> 16;
          uint32_t col0 = taddr & 0xffff;
          uint32_t cols = kTmemThreadsPerLane;
          if (tpuArgs.tmem_row) {
            lane0 += (r % wg_cfg::m_steps) * wg_cfg::tcM;
            col0 += (r / wg_cfg::m_steps) * wg_cfg::tcN;
            cols = wg_cfg::tcN;
          } else {
            col0 += r * kTmemThreadsPerLane;
          }
          bool is_ld = (tcu_type == TcuType::TMEM_LD);
          if ((taddr >> 16) != b * wg_cfg::xtileM || ((taddr & 0xffff) % kFragCols) != 0
           || col0 + cols > kTmemCols) {
            std::cout << "Error: tmem." << (is_ld ? "ld" : "st") << " of lane " << (taddr >> 16)
                      << " column " << (taddr & 0xffff) << " from a warp of block " << b
                      << " (its lanes start at " << b * wg_cfg::xtileM << ", columns are whole groups of "
                      << kFragCols << " below " << kTmemCols << ")" << std::endl;
            std::abort();
          }
          // A register is one read or write: of both M steps' rows (lane per
          // thread) or of one (fragment).
          uint32_t port_ops = 1;
          // A step's D write-back has the write port before a tmem.st.
          if (!is_ld && port_ops != 0 && this->job_writing(now)) {
            ++perf_stats_.tmem_stalls;
            continue;
          }
          if (port_ops != 0) {
            if (blk_port_ops_.at(b) + 1 < port_ops) {
              ++blk_port_ops_.at(b);
              if (is_ld) {
                tmem_rd_taken_ = true;
              }
              ++perf_stats_.tmem_stalls;
              continue;
            }
            blk_port_ops_.at(b) = 0;
            if (is_ld) {
              tmem_rd_taken_ = true;
            }
          }
          int32_t cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
          this->contract_check(is_ld ? "tmem.ld" : "tmem.st", cta, col0, col0 + cols, !is_ld);
          ++perf_stats_.tmem_xfers;
          if (is_ld) {
            rd_data.resize(VX_CFG_NUM_THREADS);
          }
          for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
            auto& cell = tpuArgs.tmem_row
                       ? this->tmem_at(lane0 + t / wg_cfg::tcN, col0 + t % wg_cfg::tcN)
                       : this->tmem_at(lane0 + t % wg_cfg::xtileM, col0 + t / wg_cfg::xtileM);
            if (is_ld) {
              rd_data.at(t).u64 = nan_box(cell.u32);
            } else {
              cell.u64 = nan_box(rs2_data.at(t).u32);
            }
          }
        } break;
        case TcuType::TMEM_ALLOC:
        case TcuType::TMEM_DEALLOC: {
          if (tj_port_busy_) {
            continue;
          }
          uint32_t t0    = scalar_thread(trace);
          uint32_t col   = rs1_data.at(t0).u;
          uint32_t ncols = (tcu_type == TcuType::TMEM_ALLOC) ? col : rs2_data.at(t0).u;
          int32_t cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
          if (ncols < kAllocCols || ncols > kTmemCols || (ncols & (ncols - 1)) != 0) {
            std::cout << "Error: TMEM allocation of " << ncols << " columns (a power of two from "
                      << kAllocCols << " to " << kTmemCols << ")" << std::endl;
            std::abort();
          }
          uint32_t k = ncols / kAllocCols;
          uint32_t mask = (k == 32) ? ~0u : ((1u << k) - 1);
          if (tcu_type == TcuType::TMEM_ALLOC) {
            // Waits until a run fits; the launch budget keeps the resident
            // CTAs' allocations within TMEM.
            uint32_t first = this->alloc_fit(k);
            if (first == ~0u) {
              ++perf_stats_.tmem_stalls;
              continue;
            }
            tmem_used_ |= mask << first;
            for (uint32_t i = first; i < first + k; ++i) {
              tmem_owner_.at(i) = cta;
            }
            // The column goes to thread 0 only, as in the RTL; the other
            // threads get a poison value, an address no transfer accepts.
            rd_data.resize(VX_CFG_NUM_THREADS);
            for (auto& rd : rd_data) {
              rd.u = ~0u;
            }
            rd_data.at(0).u = first * kAllocCols;
          } else {
            uint32_t first = col / kAllocCols;
            bool owned = (col % ncols) == 0 && (tmem_used_ & (mask << first)) == (mask << first);
            for (uint32_t i = first; owned && i < first + k; ++i) {
              owned = (tmem_owner_.at(i) == cta);
            }
            if (!owned) {
              std::cout << "Error: TMEM dealloc of " << ncols << " columns at column " << col
                        << " that the CTA does not own" << std::endl;
              std::abort();
            }
            this->contract_check("tmem.dealloc", cta, col, col + ncols, true);
            tmem_used_ &= ~(mask << first);
          }
          tj_port_busy_ = true;
        } break;
        case TcuType::TC_MMA:
        case TcuType::TC_CP:
        case TcuType::TC_COMMIT: {
          // One op per cycle into the in-order queue: scalars only.
          if (tj_port_busy_ || (tcu_type == TcuType::TC_COMMIT ? !this->tc_commit_ready()
                                                               : ops_.size() >= kOpQueue)) {
            continue;
          }
          tj_port_busy_ = true;
          tj_t job;
          job.cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
          job.wid = wid;
          job.seq = tj_seq_++;
          uint32_t t0 = scalar_thread(trace);
          uint32_t s1 = rs1_data.at(t0).u;
          uint32_t s2 = rs2_data.at(t0).u;
          if (tcu_type == TcuType::TC_COMMIT) {
            this->tc_commit(job.cta, wid, s1, s2);
            break;
          } else if (tcu_type == TcuType::TC_CP) {
            job.kind   = tc_kind_t::CP;
            job.taddr  = s1 & 0xffff;
            job.cols   = s1 >> 16;
            job.a_base = s2;
            this->copy_check(job);
            this->span_post(job);
          } else {
            job.kind = tc_kind_t::MMA;
            this->job_decode(job, s1, s2, rs3_data.at(t0).u);
            this->span_post(job);
          }
          ops_.push_back(job);
        } break;
      #endif
        default:
          std::abort();
        }
        exec_done_.at(b) = true;
      }

      // A compute uop holds its input while its leading output slices go
      // through the FEDP array, under the admission conditions of its fire,
      // and leaves with the last one.
      if (kPeRatio > 1 && !is_tj_op && !tpuArgs.is_setup_uop && slices_.at(b) + 1 < kPeRatio) {
      #ifdef VX_CFG_TCU_WGMMA_ENABLE
        bool tbuf_ready = !tcu_is_wgmma(tcu_type) || ((wgmma_ready >> b) & 1u);
      #else
        bool tbuf_ready = true;
      #endif
        if (tbuf_ready && !simobject_->Outputs.at(b).full()) {
          ++slices_.at(b);
        }
        continue;
      }

      uint32_t delay = kMmaLatency;
    #ifdef VX_CFG_TCU_TMEM_ENABLE
      if (is_tj_op) {
        delay = (tcu_type == TcuType::TMEM_LD) ? (kTmemReadLatency + 1) : 1;
      }
    #endif
      if (simobject_->Outputs.at(b).try_send(trace, delay)) {
        due.push_back(now + delay);
        exec_done_.at(b) = false;
        slices_.at(b) = 0;
        DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
        input.pop();
      }
    }

  #ifdef TCU_META_ENABLE
    this->agu_issue();
    this->agu_retire(now);
  #endif

  #ifdef VX_CFG_TCU_WGMMA_ENABLE
    // A setup uop drops its block's A stripe and the shared B row at the end
    // of the cycle; uops firing alongside it still read the old contents.
    if (setup_fired != 0) {
      auto& tbuf = simobject_->tbuf();
      for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
        if ((setup_fired >> b) & 1u) {
          abuf_.at(b) = abuf_slot_t{};
          tbuf->invalidate_a(b);
        }
      }
      bbuf_ = bbuf_slot_t{};
      tbuf->invalidate_b();
    }
  #endif

    if (cyc_.busy_blocks != 0) {
      perf_stats_.mma_busy += cyc_.busy_blocks;
    } else if (cyc_.opnd) {
      ++perf_stats_.opnd_waits;
    } else if (cyc_.other) {
      ++perf_stats_.job_other;
    } else {
      ++perf_stats_.idle;
    }
  }

  // The tile buffers cannot fetch a descriptor that breaks the alignment rule,
  // so SimX must not read it from its exact address either.
  static void check_desc(uint32_t desc, bool is_b, bool is_sparse) {
    uint32_t slice_words = vt::wgmma_slice_words(is_b, is_sparse, cfg::tcK, kFedpWords);
    auto fault = vt::smem_desc_check(desc, slice_words, kBankRowWords * 4);
    if (fault == vt::smem_desc_fault::none) {
      return;
    }
    std::cerr << "*** WGMMA " << (is_b ? "B" : "A") << " descriptor 0x" << std::hex << desc << std::dec
              << " (offset " << vt::smem_desc_offset(desc) << " B, stride " << vt::smem_desc_stride(desc) << " B): ";
    switch (fault) {
    case vt::smem_desc_fault::block_base:
      std::cerr << "a block-major base must be aligned to an LMEM bank row (" << (kBankRowWords * 4) << " B)";
      break;
    case vt::smem_desc_fault::row_base:
      std::cerr << "a row-major base must be a multiple of the " << (slice_words * 4) << "-byte K-slice";
      break;
    default:
      std::cerr << "a row-major stride must be a multiple of the " << (slice_words * 4) << "-byte K-slice";
      break;
    }
    std::cerr << std::endl;
    std::abort();
  }

  // Decode a warp's WGMMA descriptors into LMEM base and row stride.
  void decode_descs(uint32_t wid, uint32_t fmt_s, bool is_a_smem, bool is_sparse) {
    uint32_t e_bits = elem_bits(fmt_s);
    uint32_t a_desc = wgmma_desc_[wid][0];
    uint32_t b_desc = wgmma_desc_[wid][1];
    if (is_a_smem) {
      check_desc(a_desc, false, is_sparse);
      lmem_desc_[wid][0] = {uint64_t(VX_MEM_LMEM_BASE_ADDR) + vt::smem_desc_offset(a_desc), vt::smem_desc_stride(a_desc) * 8 / e_bits, false};
    }
    check_desc(b_desc, true, is_sparse);
    lmem_desc_[wid][1] = {uint64_t(VX_MEM_LMEM_BASE_ADDR) + vt::smem_desc_offset(b_desc), vt::smem_desc_stride(b_desc) * 8 / e_bits, false};
  }

#ifdef VX_CFG_TCU_WGMMA_ENABLE
  // A block's A buffer: one k-stripe.
  struct abuf_slot_t {
    bool     valid = false;
    bool     fetching = false;
    bool     first_refetched = false;
    uint32_t step_k = 0;
  };

  // The shared B buffer and its refill key.
  struct bbuf_slot_t {
    bool     valid = false;
    bool     fetching = false;
    bool     first_refetched = false;
    bool     kmajor = false;
    bool     sparse = false;
    uint64_t row = 0;       // block-major: LMEM address of the bank row
    int32_t  cta = -1;      // K-major: the (cta, step_k, step_n) block
    uint32_t step_k = 0;
    uint32_t step_n = 0;

    bool same_row(const bbuf_slot_t& key) const {
      if (kmajor != key.kmajor || sparse != key.sparse) {
        return false;
      }
      if (kmajor) {
        return cta == key.cta && step_k == key.step_k && step_n == key.step_n;
      }
      return row == key.row;
    }
  };

  // One block's WGMMA compute uop as the tile buffers see it.
  struct tbuf_req_t {
    bool     valid = false;
    uint32_t wid = 0;
    int32_t  cta = -1;
    bool     is_first_uop = false;
    bool     is_sparse = false;
    bool     a_is_smem = false;
    uint32_t step_m = 0;
    uint32_t step_n = 0;
    uint32_t step_k = 0;
    uint32_t cd_nregs = 0;
    uint32_t xtile_n = 0;
    uint32_t fmt_s = 0;
    uint32_t desc_b = 0;

    // The A buffer forces a refetch on a WGMMA's first compute uop.
    bool first_compute() const {
      return step_m == 0 && step_n == 0 && step_k == 0;
    }
  };

  // Evaluate the A/B tile buffers for the uops at the head of each block
  // and return the blocks whose WGMMA compute uop may fire.
  //
  // The B buffer holds one bank row and refills when the key of the lowest
  // block with a uop changes; a block whose (desc_b, step_k, step_n) differs
  // from that block's waits. Each A buffer holds its block's k-stripe.
  // Refills start from the buffer state at the beginning of the cycle, and a
  // refill that completes this cycle serves from the next.
  uint32_t tbuf_step() {
    auto& tbuf = simobject_->tbuf();

    std::array<tbuf_req_t, VX_CFG_NUM_TCU_BLOCKS> reqs;
    int32_t rep = -1;
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      // A uop requests its tiles for as long as it holds the block's input,
      // through its last output slice and any wait for the result side.
      if (input.empty()) {
        continue;
      }
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      if (!tcu_is_wgmma(tcu_type)) {
        continue;
      }
      auto tpuArgs = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
      if (tpuArgs.is_setup_uop) {
        continue;
      }
      uint32_t wid = trace->wid;
      int32_t cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
      // CTA lockstep: wait while another block is mid-WGMMA for another CTA.
      bool cta_conflict = false;
      for (uint32_t k = 0; k < VX_CFG_NUM_TCU_BLOCKS; ++k) {
        if (k != b && in_wgmma_.at(k) && cta_owner_a_.at(k) != cta) {
          cta_conflict = true;
          break;
        }
      }
      if (cta_conflict) {
        continue;
      }

      bool is_sparse = (tcu_type == TcuType::WGMMA_SP);
      bool needs_setup = wgmma_needs_setup(tpuArgs.is_a_smem, is_sparse);
      if (tpuArgs.is_first_uop && !needs_setup) {
        // The first compute uop carries the live descriptors.
        if (tpuArgs.is_a_smem) {
          wgmma_desc_[wid][0] = trace->src_data.at(0).at(0).u32;
        }
        wgmma_desc_[wid][1] = trace->src_data.at(1).at(0).u32;
        this->decode_descs(wid, tpuArgs.fmt_s, tpuArgs.is_a_smem, is_sparse);
      }

      auto& req = reqs.at(b);
      req.valid        = true;
      req.wid          = wid;
      req.cta          = cta;
      req.is_first_uop = tpuArgs.is_first_uop;
      req.is_sparse    = is_sparse;
      req.a_is_smem    = tpuArgs.is_a_smem;
      req.step_m       = tpuArgs.step_m;
      req.step_n       = tpuArgs.step_n;
      req.step_k       = tpuArgs.step_k;
      req.cd_nregs     = tpuArgs.cd_nregs;
      req.xtile_n      = xtile_n_of(tpuArgs.cd_nregs);
      req.fmt_s        = tpuArgs.fmt_s;
      req.desc_b       = wgmma_desc_[wid][1];
      if (rep < 0) {
        rep = b;
      }
    }

  #ifdef VX_CFG_TCU_TMEM_ENABLE
    if (rep >= 0 && (run_valid_ || !ops_.empty())) {
      std::cout << "Error: WGMMA issued while a tensor job owns the tile buffers" << std::endl;
      std::abort();
    }
  #endif

    // Shared B buffer, keyed by the lowest block with a uop.
    bool bbuf_ready = true;
    if (rep >= 0) {
      auto& r = reqs.at(rep);
      bbuf_slot_t key = this->bbuf_key(r);
      bool resident = bbuf_.valid && bbuf_.same_row(key)
                   && (!key.kmajor || !r.is_first_uop || bbuf_.first_refetched);
      if (resident) {
        ++perf_stats_.tbuf_cache_hits;
      } else {
        bbuf_ready = false;
        ++perf_stats_.tbuf_stalls;
        if (!bbuf_.fetching) {
          bool first_refetched = bbuf_.first_refetched;
          bbuf_ = key;
          bbuf_.fetching = true;
          bbuf_.first_refetched = first_refetched;
          DT(3, simobject_->name() << " bbuf: alloc desc_b=0x" << std::hex << r.desc_b
                << std::dec << ", sparse=" << r.is_sparse << ", step_k=" << r.step_k
                << ", step_n=" << r.step_n << ", row=0x" << std::hex << key.row
                << std::dec << ", wid=" << r.wid);
          tbuf->fill_b(this->b_refill(r));
        }
      }
    }

    // Per-block A buffers.
    std::array<bool, VX_CFG_NUM_TCU_BLOCKS> abuf_ready;
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& r = reqs.at(b);
      auto& slot = abuf_.at(b);
      abuf_ready.at(b) = true;
      if (!r.valid || !r.a_is_smem) {
        continue;
      }
      bool resident = slot.valid && slot.step_k == r.step_k
                   && (!r.first_compute() || slot.first_refetched);
      if (resident) {
        continue;
      }
      abuf_ready.at(b) = false;
      ++perf_stats_.tbuf_stalls;
      if (!slot.fetching) {
        slot.valid    = false;
        slot.fetching = true;
        slot.step_k   = r.step_k;
        tbuf->fill_a(b, this->a_refill(r));
      }
    }

    uint32_t ready_mask = 0;
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& r = reqs.at(b);
      if (!r.valid) {
        continue;
      }
      auto& p = reqs.at(rep);
      bool key_match = (vt::smem_desc_offset(r.desc_b) == vt::smem_desc_offset(p.desc_b))
                    && r.step_k == p.step_k
                    && r.step_n == p.step_n
                    && r.cd_nregs == p.cd_nregs;
      if (abuf_ready.at(b) && bbuf_ready && key_match) {
        ready_mask |= (1u << b);
      }
    }

    // Buffer state for the next cycle.
    if (bbuf_.fetching && !tbuf->filling_b()) {
      bbuf_.fetching = false;
      bbuf_.valid = true;
      if (rep >= 0 && reqs.at(rep).is_first_uop) {
        bbuf_.first_refetched = true;
      }
    } else if (rep >= 0 && bbuf_ready && !reqs.at(rep).is_first_uop) {
      bbuf_.first_refetched = false;
    }
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& r = reqs.at(b);
      auto& slot = abuf_.at(b);
      if (slot.fetching && !tbuf->filling_a(b)) {
        slot.fetching = false;
        slot.valid = true;
        if (r.valid && r.first_compute()) {
          slot.first_refetched = true;
        }
      } else if (r.valid && abuf_ready.at(b) && !r.first_compute()) {
        slot.first_refetched = false;
      }
    }

    return ready_mask;
  }

  static uint32_t xtile_n_of(uint32_t cd_nregs) {
    // NRC: cd_nregs 0/1/2 → 8/16/32; xtileN = NRC * NT / xtileM.
    uint32_t nrc = (cd_nregs == 0) ? 8 : (cd_nregs == 1) ? 16 : 32;
    return (nrc * VX_CFG_NUM_THREADS) / wg_cfg::xtileM;
  }

  // Refill key of the B row serving `req`. Block-major B is keyed by the
  // bank row holding the (step_k, step_n) block; K-major B by the block.
  bbuf_slot_t bbuf_key(const tbuf_req_t& req) {
    const auto& sd_b = lmem_desc_[req.wid][1];
    bbuf_slot_t key;
    key.kmajor = (sd_b.ldm != 0);
    key.sparse = req.is_sparse;
    if (key.kmajor) {
      key.cta    = req.cta;
      key.step_k = req.step_k;
      key.step_n = req.step_n;
    } else {
      uint32_t n_steps = req.xtile_n / cfg::tcN;
      uint32_t blk = req.step_k * n_steps + req.step_n;
      uint32_t blk_bytes = req.is_sparse ? (kBBlockWordsSp * 4) : (kBBlockWords * 4);
      uint32_t blks_per_row = req.is_sparse ? 1 : kBSubBlocks;
      key.row = sd_b.base + uint64_t(blk / blks_per_row) * blks_per_row * blk_bytes;
    }
    return key;
  }

  // LMEM reads that refill the B buffer for `req`: one bank row for a
  // block-major row (two when a sparse block spans two rows) and one per
  // N-row of the block in K-major layout.
  std::vector<TcuTbuf::LmemRead> b_refill(const tbuf_req_t& req) {
    const auto& sd_b = lmem_desc_[req.wid][1];
    uint32_t e_bits  = elem_bits(req.fmt_s);
    uint32_t ratio   = 32 / e_bits;
    uint32_t xtile_n = req.xtile_n;
    uint32_t n_steps = xtile_n / cfg::tcN;
    uint32_t k_words = req.is_sparse ? wg_cfg::sp_k_words : kFedpWords;
    bool kmajor = (sd_b.ldm != 0);

    // (step_k, step_n) blocks this refill brings in.
    uint32_t blk = req.step_k * n_steps + req.step_n;
    uint32_t blks = (kmajor || req.is_sparse) ? 1 : kBSubBlocks;
    uint32_t first_blk = blk - (blk % blks);

    std::vector<uint64_t> lines;
    for (uint32_t bi = first_blk; bi < first_blk + blks; ++bi) {
      uint32_t step_k = bi / n_steps;
      uint32_t step_n = bi % n_steps;
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        uint32_t col = step_n * cfg::tcN + j;
        for (uint32_t z = 0; z < k_words; ++z) {
          uint32_t k_elem = (step_k * k_words + z) * ratio * (req.is_sparse ? 2 : 1);
          uint32_t k_elems = req.is_sparse ? (2 * ratio) : ratio;
          for (uint32_t e = 0; e < k_elems; ++e) {
            uint64_t off = elem_offset(sd_b, k_elem + e, col, e_bits, xtile_n,
                                       true, req.is_sparse, false);
            lines.push_back((sd_b.base + off * e_bits / 8) & kLineMask);
          }
        }
      }
    }

    uint32_t num_reads = (req.is_sparse && kBSparseTwoFetch) ? 2 : 1;
    if (kmajor) {
      num_reads = cfg::tcN;
    }
#ifdef VX_CFG_TCU_TMEM_ENABLE
    if (kmajor) {
      // The bbuf reads consecutive N-rows that share an LMEM bank row at once.
      constexpr uint64_t kBankRowBytes = uint64_t(kLmemBanks) * (VX_CFG_XLEN / 8);
      uint32_t k_elem = req.step_k * k_words * ratio * (req.is_sparse ? 2 : 1);
      uint64_t prev_row = ~uint64_t(0);
      num_reads = 0;
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        uint64_t off = elem_offset(sd_b, k_elem, req.step_n * cfg::tcN + j, e_bits, xtile_n,
                                   true, req.is_sparse, false);
        uint64_t row = (sd_b.base + off * e_bits / 8) / kBankRowBytes;
        num_reads += (row != prev_row);
        prev_row = row;
      }
    }
#endif
    return split_reads(std::move(lines), num_reads);
  }

  // LMEM reads that refill a block's A buffer with the k-stripe of `req`:
  // one per bank row of a block-major stripe, one per row in row-major.
  std::vector<TcuTbuf::LmemRead> a_refill(const tbuf_req_t& req) {
    const auto& sd_a = lmem_desc_[req.wid][0];
    uint32_t e_bits  = elem_bits(req.fmt_s);
    uint32_t ratio   = 32 / e_bits;
    uint32_t xtile_n = req.xtile_n;
    // A full-K sparse step's compressed A has the dense shape (fedpK words).
    bool sparse_a    = req.is_sparse && !kSpFullK;
    uint32_t k_words = sparse_a ? cfg::tcK : kFedpWords;

    std::vector<uint64_t> lines;
    for (uint32_t row = 0; row < wg_cfg::m_steps * cfg::tcM; ++row) {
      for (uint32_t z = 0; z < k_words; ++z) {
        uint32_t k_elem = (req.step_k * k_words + z) * ratio;
        for (uint32_t e = 0; e < ratio; ++e) {
          uint64_t off = elem_offset(sd_a, row, k_elem + e, e_bits, xtile_n,
                                     false, false, sparse_a);
          lines.push_back((sd_a.base + off * e_bits / 8) & kLineMask);
        }
      }
    }

    uint32_t num_reads = (sd_a.ldm != 0) ? (wg_cfg::m_steps * cfg::tcM)
                       : sparse_a ? kAStripeReadsSp : kAStripeReads;
    return split_reads(std::move(lines), num_reads);
  }

  // Spread a refill's distinct lines over the hardware's read count.
  static std::vector<TcuTbuf::LmemRead> split_reads(std::vector<uint64_t> lines,
                                                    uint32_t num_reads) {
    std::sort(lines.begin(), lines.end());
    lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
    uint32_t n = lines.size();
    if (n == 0) {
      return {};
    }
    std::vector<TcuTbuf::LmemRead> reads(num_reads);
    for (uint32_t i = 0; i < num_reads; ++i) {
      if (n >= num_reads) {
        reads.at(i).assign(lines.begin() + (uint64_t(i) * n) / num_reads,
                           lines.begin() + (uint64_t(i + 1) * n) / num_reads);
      } else {
        reads.at(i).push_back(lines.at(i % n));
      }
    }
    return reads;
  }

#ifdef VX_CFG_TCU_TMEM_ENABLE
  // Tensor memory: VX_CFG_NUM_TCU_BLOCKS * xtileM lanes by kTmemCols 32-bit
  // columns. Block b owns lanes [b*xtileM, (b+1)*xtileM), the M rows it
  // computes, and holds them fragment-major: a row of its storage is one C/D
  // fragment, tcM lanes by tcN columns, and the two M steps of a column group
  // sit in two RAMs, so an MMA step reads and writes one row of one RAM and a
  // thread-per-row transfer reads both RAMs once.
  //
  // tc.mma, tc.cp and tc.commit enter one in-order queue per core, with their
  // scalars only. The engine runs the head op: a tc.mma is one K step, walked
  // (m, n) on all blocks in lockstep with C read from and D written back to
  // TMEM; a tc.cp copies whole LMEM bank rows of pre-packed data through the
  // tile-buffer port. A tc.commit at the head marks the steps issued so far
  // and leaves; it arrives on its barrier once they have been written back.
  // tmem.ld/st go to the issuing warp's own block. Their ordering with the
  // MMAs is the software's (tc.commit, a barrier wait, tmem::wait_st); the
  // contract checker below reports a transfer that breaks it.
  static constexpr uint32_t kTmemLanes = VX_CFG_NUM_TCU_BLOCKS * wg_cfg::xtileM;
  static constexpr uint32_t kTmemCols  = VX_CFG_TCU_TMEM_COLS;
  static constexpr uint32_t kTmemThreadsPerLane = VX_CFG_NUM_THREADS / wg_cfg::xtileM;
  static constexpr uint32_t kFragCols  = wg_cfg::tcN;   // columns of a fragment row
  static_assert(VX_CFG_NUM_THREADS % wg_cfg::xtileM == 0, "TMEM needs NUM_THREADS to be a multiple of WGMMA M");
  static_assert(wg_cfg::tcM * wg_cfg::tcN == VX_CFG_NUM_THREADS, "C fragment must span the warp");
  static_assert(kPeRatio == 1, "TMEM requires VX_CFG_TCU_PE_RATIO=1");
  static_assert(VX_CFG_FETCH_WIDTH == VX_CFG_ISSUE_WIDTH, "TMEM needs one launch lane per TCU block");
  static_assert(kTmemCols % kFragCols == 0, "TMEM columns are whole fragment groups");
  static_assert(kFedpWords % kFragCols == 0, "a K step of A from TMEM is whole fragment groups");
  static constexpr uint32_t kTmemReadLatency = 1;
  static constexpr uint32_t kOpQueue     = 4;    // tc.mma / tc.cp / tc.commit
  static constexpr uint32_t kOpCommits   = 2;    // commits an op carries
  static constexpr uint32_t kCommitQueue = 4;    // commits waiting for their steps
  static constexpr uint32_t kAllocCols   = 32;
  static constexpr uint32_t kAllocChunks = kTmemCols / kAllocCols;
  static_assert(kTmemCols % kAllocCols == 0 && kAllocChunks <= 32, "TMEM allocation granularity");
  // tc.cp moves whole LMEM bank rows.
  static constexpr uint32_t kCpRowWords = kLmemBanks * (VX_CFG_XLEN / 32);
  static_assert(kCpRowWords % VX_CFG_NUM_THREADS == 0, "a bank row holds whole fragment rows");
  static constexpr uint32_t kJobKeyBase = 0x10000;   // lmem_desc_ key of block 0

  // Instruction descriptor of a tc.mma (rs1): fmt_s [4:0], sparse [5],
  // a_tmem [6], acc_init [7], n_frag - 1 [14:8], D fragment group [22:15],
  // side-operand column [31:23] (sparse metadata).
  struct idesc_t {
    uint32_t fmt_s, n_frag, d_col, side_col;
    bool sparse, a_tmem, acc_init;
  };
  static idesc_t idesc_decode(uint32_t w) {
    idesc_t d;
    d.fmt_s    = w & 0x1f;
    d.sparse   = (w >> 5) & 1;
    d.a_tmem   = (w >> 6) & 1;
    d.acc_init = (w >> 7) & 1;
    d.n_frag   = ((w >> 8) & 0x7f) + 1;
    d.d_col    = ((w >> 15) & 0xff) * kFragCols;
    d.side_col = w >> 23;
    return d;
  }

  reg_data_t& tmem_at(uint32_t lane, uint32_t col) {
    return tmem_.at(lane * kTmemCols + col);
  }

  const reg_data_t& tmem_at(uint32_t lane, uint32_t col) const {
    return tmem_.at(lane * kTmemCols + col);
  }

  enum class tc_kind_t { MMA, CP, COMMIT };

  struct tj_t {
    tc_kind_t  kind = tc_kind_t::MMA;
    int32_t    cta = -1;
    uint32_t   wid = 0;
    uint64_t   seq = 0;
    // tc.mma
    uint32_t   fmt_s = 0;
    uint32_t   fmt_d = 0;
    uint32_t   n_frag = 0;
    bool       acc_init = false;
    bool       sparse = false;
    bool       a_tmem = false;     // a_base is A's first TMEM column
    uint32_t   taddr = 0;          // D's first column (tc.cp: the first column written)
    uint32_t   side = 0;           // metadata column
    uint32_t   a_base = 0;         // shared-memory byte offset, or TMEM column
    uint32_t   b_base = 0;
    uint32_t   lda = 0;            // leading bytes
    uint32_t   ldb = 0;
    // tc.cp: cols columns of every lane from a_base
    uint32_t   cols = 0;
    // tc.commit, or the commits that follow the op: they arrive once its
    // steps (and every earlier op's) are written back.
    uint32_t   ncommit = 0;
    uint32_t   bar[2] = {};
    uint32_t   cnt[2] = {};
    // MMA walk
    uint32_t   n = 0;
    uint32_t   m = 0;
    uint64_t   start_at = 0;
    uint32_t   pro_done = 0;
    bool       a_valid = false;
    bool       a_fetching = false;
    bool       b_valid = false;
    bool       b_fetching = false;
    uint32_t   b_n = 0;
    bool       copying = false;
  };

  // An op whose TMEM columns are not settled yet, for the contract checker:
  // D (or tc.cp's columns) [d_lo, d_hi), A [a_lo, a_hi), metadata [s_lo, s_hi).
  struct tc_span_t {
    int32_t  cta;
    uint64_t seq;
    uint32_t d_lo, d_hi, a_lo, a_hi, s_lo, s_hi;
    uint64_t steps = ~uint64_t(0); // steps_issued_ after its last step; ~0 while it runs
  };

  struct tc_wb_t {
    uint64_t due;
    uint32_t m, col0;
  };

  struct tc_commit_t {
    uint32_t bar, cnt, wid;
    uint64_t target;
  };

  // Columns of one K step's metadata per lane: a lane holds, from bit 0, the
  // VX_tcu_sp_mux rows of the step's lower then upper TC_K words.
  uint32_t job_meta_cols(const tj_t& job) const {
    return std::max(1u, kFedpWords * 2 * elem_ratio(job.fmt_s) / 32);
  }

  static uint32_t scalar_thread(const instr_trace_t* trace) {
    if (!trace->tmask.test(0)) {
      std::cout << "Error: tensor-memory op without thread 0" << std::endl;
      std::abort();
    }
    return 0;
  }

  static void tc_error(const char* op, int32_t cta, const char* err) {
    std::cout << "Error: " << op << " (cta=" << cta << "): " << err << std::endl;
    std::abort();
  }

  void job_decode(tj_t& job, uint32_t w, uint32_t a, uint32_t b) {
    auto d = idesc_decode(w);
    job.fmt_s    = d.fmt_s;
    job.fmt_d    = vt::tcu_output_fmt(d.fmt_s);
    job.sparse   = d.sparse;
    job.a_tmem   = d.a_tmem;
    job.acc_init = d.acc_init;
    job.n_frag   = d.n_frag;
    job.taddr    = d.d_col;
    job.side     = d.side_col;
    job.a_base   = d.a_tmem ? a : vt::smem_desc_offset(a);
    job.lda      = d.a_tmem ? 0 : vt::smem_desc_stride(a);
    job.b_base   = vt::smem_desc_offset(b);
    job.ldb      = vt::smem_desc_stride(b);

    uint32_t d_end = job.taddr + job.n_frag * wg_cfg::tcN;
    auto bad_columns = [&](uint32_t base, uint32_t cols) {
      return (base % kFragCols) != 0 || base + cols > kTmemCols
          || (base < d_end && job.taddr < base + cols);
    };
    const char* err = nullptr;
    if (job.sparse && !vt::sparse_format_supported(job.fmt_s)) {
      err = "sparse MMA with an unsupported input format";
    } else if (job.sparse && bad_columns(job.side, kFragCols)) {
      err = "metadata columns misaligned, past TMEM or overlapping D";
    } else if (job.a_tmem && bad_columns(job.a_base, kFedpWords)) {
      err = "A columns misaligned, past TMEM or overlapping D";
    } else if (!job.a_tmem && job.lda == 0) {
      err = "A must be row-major (a nonzero leading dimension)";
    } else if (job.ldb == 0) {
      err = "B must be K-major (a nonzero leading dimension)";
    } else if (d_end > kTmemCols) {
      err = "D columns exceed TMEM";
    } else if (!vt::tcu_output_valid(job.fmt_s, job.fmt_d)) {
      err = "unsupported input format";
    }
    if (err) {
      tc_error("tc.mma", job.cta, err);
    }
    DT(3, simobject_->name() << " tc-mma: cta=" << job.cta << ", n_frag=" << job.n_frag
          << ", taddr=" << job.taddr << ", acc_init=" << job.acc_init
          << ", a_base=0x" << std::hex << job.a_base << ", b_base=0x" << job.b_base
          << ", lda=0x" << job.lda << ", ldb=0x" << job.ldb << std::dec);
  }

  // tc.cp source: the pre-packed fragment rows of columns [taddr, taddr + cols)
  // in storage order, column group by column group, block by block, M step by M
  // step, NT words each (word i*tcN + j is lane m*tcM + i, column j of the group).
  void copy_check(const tj_t& job) {
    const char* err = nullptr;
    uint32_t words = job.cols * kTmemLanes;
    if (job.cols == 0 || (job.cols % kFragCols) != 0 || (job.taddr % kFragCols) != 0) {
      err = "copy columns must be whole fragment groups";
    } else if ((job.a_base % (kCpRowWords * 4)) != 0 || (words % kCpRowWords) != 0) {
      err = "copy source must be whole LMEM bank rows";
    } else if (job.taddr + job.cols > kTmemCols) {
      err = "copy columns exceed TMEM";
    }
    if (err) {
      tc_error("tc.cp", job.cta, err);
    }
    DT(3, simobject_->name() << " tc-cp: cta=" << job.cta << ", cols=" << job.cols
          << ", taddr=" << job.taddr << ", src=0x" << std::hex << job.a_base << std::dec);
  }

  // One read per LMEM bank row.
  std::vector<TcuTbuf::LmemRead> copy_reads(const tj_t& job) const {
    std::vector<TcuTbuf::LmemRead> reads;
    uint32_t rows = job.cols * kTmemLanes / kCpRowWords;
    for (uint32_t r = 0; r < rows; ++r) {
      uint64_t addr = uint64_t(VX_MEM_LMEM_BASE_ADDR) + job.a_base + uint64_t(r) * kCpRowWords * 4;
      TcuTbuf::LmemRead read;
      for (uint32_t off = 0; off < kCpRowWords * 4; off += VX_CFG_MEM_BLOCK_SIZE) {
        read.push_back((addr + off) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1));
      }
      std::sort(read.begin(), read.end());
      read.erase(std::unique(read.begin(), read.end()), read.end());
      reads.push_back(std::move(read));
    }
    return reads;
  }

  void copy_write(const tj_t& job) {
    auto& tbuf = simobject_->tbuf();
    uint32_t words = job.cols * kTmemLanes;
    for (uint32_t w = 0; w < words; ++w) {
      uint32_t frag = w / VX_CFG_NUM_THREADS;
      uint32_t t    = w % VX_CFG_NUM_THREADS;
      uint32_t m    = frag % wg_cfg::m_steps;
      uint32_t b    = (frag / wg_cfg::m_steps) % VX_CFG_NUM_TCU_BLOCKS;
      uint32_t g    = frag / (wg_cfg::m_steps * VX_CFG_NUM_TCU_BLOCKS);
      uint32_t lane = b * wg_cfg::xtileM + m * wg_cfg::tcM + t / wg_cfg::tcN;
      uint32_t col  = job.taddr + g * kFragCols + t % wg_cfg::tcN;
      uint64_t addr = uint64_t(VX_MEM_LMEM_BASE_ADDR) + job.a_base + uint64_t(w) * 4;
      auto line = tbuf->read_c(addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1));
      if (!line) {
        std::cout << "Error: tc.cp source line 0x" << std::hex << addr << std::dec << " was not read" << std::endl;
        std::abort();
      }
      uint32_t word;
      std::memcpy(&word, line->data() + (addr & (VX_CFG_MEM_BLOCK_SIZE - 1)), 4);
      this->tmem_at(lane, col).u64 = nan_box(word);
    }
  }

  // MX scales of a tensor op's step, kMxSf bytes per A row and B column (scale
  // s covers the step's K elements from s * step_elems / kMxSf), as the RTL
  // reads them. A: lane l, columns side + s/4, byte s%4. B: byte
  // q = col * kMxSf + s of a dense stream of NT*4-byte rows; row r is M step 1
  // of block r % NUM_TCU_BLOCKS at group side/TC_N + 1 + r / NUM_TCU_BLOCKS.
  static constexpr uint32_t kMxSf = vt::mx_max_fedp_sf(kFedpWords);
  struct job_mx_t {
    bool     active = false;
    uint32_t side = 0, m = 0, n = 0, block = 0;
  };
  job_mx_t job_mx_;

  uint32_t tmem_scale_a(uint32_t i, uint32_t s) const {
    uint32_t lane = job_mx_.block * wg_cfg::xtileM + job_mx_.m * cfg::tcM + i;
    uint32_t word = this->tmem_at(lane, job_mx_.side + s / 4).u32;
    return (word >> (8 * (s % 4))) & 0xff;
  }

  uint32_t tmem_scale_b(uint32_t j, uint32_t s) const {
    constexpr uint32_t kRowBytes = VX_CFG_NUM_THREADS * 4;
    uint32_t q    = (job_mx_.n * cfg::tcN + j) * kMxSf + s;
    uint32_t r    = q / kRowBytes;
    uint32_t o    = q % kRowBytes;
    uint32_t blk  = r % VX_CFG_NUM_TCU_BLOCKS;
    uint32_t g    = job_mx_.side / cfg::tcN + 1 + r / VX_CFG_NUM_TCU_BLOCKS;
    uint32_t w    = o / 4;
    uint32_t lane = blk * wg_cfg::xtileM + cfg::tcM + w / cfg::tcN;
    uint32_t word = this->tmem_at(lane, g * cfg::tcN + w % cfg::tcN).u32;
    return (word >> (8 * (o % 4))) & 0xff;
  }

  // Latch the op's metadata from TMEM into each block's job metadata slot, in
  // the bank layout mma_lmem reads: one fragment group, both M steps, once.
  void job_load_meta(const tj_t& job) {
    uint32_t row_w = cfg::tcK * 2 * elem_ratio(job.fmt_s);
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& meta = sparse_meta_.at(this->meta_slot(kJobKeyBase + b));
      std::fill(meta.begin(), meta.end(), 0);
      for (uint32_t lane = 0; lane < wg_cfg::xtileM; ++lane) {
        uint32_t m = lane / cfg::tcM;
        uint32_t i = lane % cfg::tcM;
        for (uint32_t h = 0; h < kFedpWords / cfg::tcK; ++h) {
          for (uint32_t bit = 0; bit < row_w; ++bit) {
            uint32_t src = h * row_w + bit;
            uint32_t word = this->tmem_at(b * wg_cfg::xtileM + lane, job.side + src / 32).u32;
            uint32_t dst = i * row_w + bit;
            uint32_t bank = m * (cfg::k_steps / 2) + h;
            meta.at(bank * kMaxMetaCols + dst / 32) |= ((word >> (src % 32)) & 1u) << (dst % 32);
          }
        }
      }
    }
  }

  // Point each block's LMEM descriptors at the op's K step.
  void job_set_descs(const tj_t& job) {
    uint32_t e_bits = elem_bits(job.fmt_s);
    uint64_t a_addr = uint64_t(VX_MEM_LMEM_BASE_ADDR) + job.a_base;
    uint64_t b_addr = uint64_t(VX_MEM_LMEM_BASE_ADDR) + job.b_base;
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& d = lmem_desc_[kJobKeyBase + b];
      d[0] = {a_addr + uint64_t(b) * wg_cfg::xtileM * job.lda, job.lda * 8 / e_bits, false};
      d[1] = {b_addr, job.ldb * 8 / e_bits, false};
    }
  }

  // A dense K-major block's row reads stay in flight together.
  static uint32_t job_b_inflight(const tj_t& job) {
    return (!job.sparse && job.ldb != 0) ? cfg::tcN : 1;
  }

  tbuf_req_t job_req(const tj_t& job, uint32_t b, uint32_t n = 0) const {
    tbuf_req_t req;
    req.valid     = true;
    req.wid       = kJobKeyBase + b;
    req.cta       = job.cta;
    req.a_is_smem = !job.a_tmem;
    req.is_sparse = job.sparse;
    req.step_m    = job.m;
    req.step_n    = n;
    req.xtile_n   = job.n_frag * cfg::tcN;
    req.fmt_s     = job.fmt_s;
    return req;
  }

  // TMEM reads ahead of an MMA's first step: the metadata row, then A, one
  // fragment group (both M steps) per read.
  uint32_t job_pro_reads(const tj_t& job) const {
    bool side = job.sparse || vt::mx_scale_format(job.fmt_s);
    return (side ? 1 : 0) + (job.a_tmem ? kFedpWords / kFragCols : 0);
  }

  // D write-backs land in order, kFedpLatency after their step.
  void steps_retire(uint64_t now) {
    while (!step_wb_.empty() && step_wb_.front().due < now) {
      step_wb_.pop_front();
      ++steps_done_;
    }
  }

  bool job_writing(uint64_t now) const {
    return !step_wb_.empty() && step_wb_.front().due == now;
  }

  // Consecutive MMAs on the same accumulators: a step waits while its C
  // fragment's D write-back is still in flight.
  bool row_hazard(uint32_t m, uint32_t col0) const {
    for (auto& wb : step_wb_) {
      if (wb.m == m && wb.col0 == col0) {
        return true;
      }
    }
    return false;
  }

  // ---- contract checker ----

  void span_settle() {
    while (!spans_.empty() && spans_.front().steps <= steps_done_) {
      spans_.pop_front();
    }
  }

  // A transfer or dealloc of the CTA's columns [lo, hi) while an op of the CTA
  // that reads or writes them is unsettled breaks the tcgen05 ordering contract.
  void contract_check(const char* op, int32_t cta, uint32_t lo, uint32_t hi, bool is_write) {
    this->span_settle();
    for (auto& s : spans_) {
      if (s.cta != cta) {
        continue;
      }
      bool d_hit = lo < s.d_hi && s.d_lo < hi;
      bool r_hit = (lo < s.a_hi && s.a_lo < hi) || (lo < s.s_hi && s.s_lo < hi);
      if (d_hit || (is_write && r_hit)) {
        std::cout << "Error: TMEM contract violation: " << op << " (cta=" << cta << ") of columns ["
                  << lo << ", " << hi << ") while tensor op " << s.seq
                  << " still uses them; wait for its tc.commit barrier (and tmem::wait_st before an MMA reads stored columns)" << std::endl;
        std::abort();
      }
    }
  }

  void span_post(const tj_t& job) {
    tc_span_t s{job.cta, job.seq, 0, 0, 0, 0, 0, 0};
    if (job.kind == tc_kind_t::CP) {
      s.d_lo = job.taddr;
      s.d_hi = job.taddr + job.cols;
    } else {
      s.d_lo = job.taddr;
      s.d_hi = job.taddr + job.n_frag * wg_cfg::tcN;
      if (job.a_tmem) {
        s.a_lo = job.a_base;
        s.a_hi = job.a_base + kFedpWords;
      }
      if (job.sparse) {
        s.s_lo = job.side;
        s.s_hi = job.side + kFragCols;
      }
    }
    spans_.push_back(s);
  }

  void span_ran(uint64_t seq) {
    for (auto& s : spans_) {
      if (s.seq == seq) {
        s.steps = steps_issued_;
      }
    }
  }

  // ---- engine ----

  // One commit arrival per cycle, the oldest first, once its steps are written.
  void job_bar_op() {
    if (commits_.empty() || commits_.front().target > steps_done_) {
      return;
    }
    auto& c = commits_.front();
    DT(3, simobject_->name() << " tc-commit: bar=0x" << std::hex << c.bar << std::dec);
    core_->barrier_arrive(bar_decode_id(c.bar, VX_CFG_NUM_BARRIERS), c.cnt, c.wid, false);
    commits_.pop_front();
  }

  // An op leaving the engine posts its commits, marking the steps issued so far.
  bool job_post_commits(const tj_t& job) {
    if (commits_.size() + job.ncommit > kCommitQueue) {
      return false;
    }
    for (uint32_t c = 0; c < job.ncommit; ++c) {
      commits_.push_back({job.bar[c], job.cnt[c], job.wid, steps_issued_});
    }
    return true;
  }

  // A tc.commit rides on the youngest op not yet out of the engine, or posts
  // at once when there is none; else it queues as an op of its own.
  // The running op takes one only while it is a tc.mma short of its last step.
  bool run_open() const {
    return run_valid_ && run_.kind == tc_kind_t::MMA
        && !(run_.m + 1 == wg_cfg::m_steps && run_.n + 1 == run_.n_frag);
  }

  void tc_commit(int32_t cta, uint32_t wid, uint32_t bar, uint32_t cnt) {
    tj_t* young = !ops_.empty() ? &ops_.back() : (this->run_open() ? &run_ : nullptr);
    if (young != nullptr && young->ncommit < kOpCommits) {
      young->bar[young->ncommit] = bar;
      young->cnt[young->ncommit] = cnt;
      ++young->ncommit;
      return;
    }
    tj_t job;
    job.kind = tc_kind_t::COMMIT;
    job.cta = cta;
    job.wid = wid;
    job.seq = tj_seq_++;
    job.ncommit = 1;
    job.bar[0] = bar;
    job.cnt[0] = cnt;
    ops_.push_back(job);
  }

  bool tc_commit_ready() const {
    tj_t const* young = !ops_.empty() ? &ops_.back() : (this->run_open() ? &run_ : nullptr);
    return (young != nullptr && young->ncommit < kOpCommits) || ops_.size() < kOpQueue;
  }

  void job_step() {
    uint64_t now = SimPlatform::instance().cycles();
    this->steps_retire(now);
    this->job_bar_op();

    if (!commits_.empty() || (run_valid_ && run_.kind == tc_kind_t::CP)) {
      cyc_.other = true;
    }

    // A tmem.ld has priority over the running op's TMEM reads; the op loses
    // the cycle after the transfer's.
    bool rd_taken = tmem_rd_taken_;
    tmem_rd_taken_ = false;

    if (run_valid_) {
      if (run_.kind == tc_kind_t::CP) {
        this->job_copy(now);
      } else {
        this->job_run(run_, rd_taken);
      }
      if (run_valid_) {
        return;
      }
    }
    // The datapath is free: the head op takes it in the same cycle.
    if (ops_.empty()) {
      return;
    }
    auto& head = ops_.front();
    if (head.kind == tc_kind_t::COMMIT) {
      if (this->job_post_commits(head)) {
        ops_.pop_front();
      }
      return;
    }
    run_ = head;
    run_valid_ = true;
    ops_.pop_front();
    cyc_.other = true;
    if (run_.kind == tc_kind_t::CP) {
      DT(3, simobject_->name() << " tc-cp-run: cta=" << run_.cta);
      run_.start_at = now + 1;
    } else {
      DT(3, simobject_->name() << " tc-run: cta=" << run_.cta);
      run_.start_at = now + 1;
      this->job_set_descs(run_);
    }
  }

  // The copy's rows land in TMEM once read; the op then leaves the engine.
  // It starts once the steps in flight are written back, as its rows share
  // the write port with them.
  void job_copy(uint64_t now) {
    auto& tbuf = simobject_->tbuf();
    if (!run_.copying) {
      if (now >= run_.start_at && steps_done_ == steps_issued_) {
        tbuf->fill_c(this->copy_reads(run_));
        run_.copying = true;
      }
      return;
    }
    if (tbuf->filling_c()) {
      return;
    }
    if (!this->job_post_commits(run_)) {
      return;
    }
    this->copy_write(run_);
    this->span_ran(run_.seq);
    run_valid_ = false;
  }

  // One MMA step per cycle on all blocks. A step's A stripes (first step) and
  // B block are fetched when it is presented.
  void job_run(tj_t& job, bool rd_taken) {
    auto& tbuf = simobject_->tbuf();
    uint64_t now = SimPlatform::instance().cycles();
    if (now < job.start_at) {
      cyc_.other = true;
      return;
    }

    // Prologue: metadata and A come over the TMEM read port, all blocks in
    // parallel, ahead of the first step.
    uint32_t pro_reads = this->job_pro_reads(job);
    if (job.pro_done < pro_reads) {
      if (!rd_taken && ++job.pro_done == pro_reads) {
        if (job.sparse) {
          this->job_load_meta(job);
        }
        job.a_valid = job.a_tmem;
      }
      ++perf_stats_.tbuf_stalls;
      cyc_.opnd = true;
      return;
    }

    bool landed = false;
    if (!job.a_valid) {
      if (!job.a_fetching) {
        for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
          tbuf->fill_a(b, this->a_refill(this->job_req(job, b)));
        }
        job.a_fetching = true;
      } else {
        bool filling = false;
        for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
          filling |= tbuf->filling_a(b);
        }
        if (!filling) {
          job.a_fetching = false;
          job.a_valid = true;
          landed = true;
        }
      }
    }

    // B: the step's N block, refetched when the step moves to the next one.
    if (job.b_fetching) {
      if (!tbuf->filling_b()) {
        job.b_fetching = false;
        job.b_valid = true;
        landed = (job.b_n == job.n);
      }
    } else if (!job.b_valid || job.b_n != job.n) {
      tbuf->fill_b(this->b_refill(this->job_req(job, 0, job.n)), this->job_b_inflight(job));
      job.b_fetching = true;
      job.b_valid = false;
      job.b_n = job.n;
    }
    bool b_ready = job.b_valid && job.b_n == job.n;

    // A step presents the cycle after its operands land.
    if (!job.a_valid || !b_ready || landed) {
      ++perf_stats_.tbuf_stalls;
      cyc_.opnd = true;
      return;
    }
    uint32_t col0 = job.taddr + job.n * wg_cfg::tcN;
    bool zero_c = job.acc_init;
    bool last_step = (job.m + 1 == wg_cfg::m_steps) && (job.n + 1 == job.n_frag);
    if (rd_taken || (!zero_c && this->row_hazard(job.m, col0))
     || (last_step && commits_.size() + job.ncommit > kCommitQueue)) {
      cyc_.other = true;
      return;
    }

    // Thread t of fragment (m, n) is M row m*tcM + t/tcN, N column n*tcN + t%tcN.
    uint32_t xtile_n = job.n_frag * cfg::tcN;
    static const RegDataVec kNone;
    RegDataVec c_frag(VX_CFG_NUM_THREADS);
    RegDataVec d_frag(VX_CFG_NUM_THREADS);
    RegDataVec a_lo(VX_CFG_NUM_THREADS);
    RegDataVec a_hi(VX_CFG_NUM_THREADS);
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      uint32_t lane0 = b * wg_cfg::xtileM + job.m * wg_cfg::tcM;
      if (job.a_tmem) {
        // The operand layout of the register-A path: rs1 lower, rs2 upper K words.
        for (uint32_t i = 0; i < cfg::tcM; ++i) {
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            a_lo.at(i * cfg::tcK + z) = this->tmem_at(lane0 + i, job.a_base + z);
            a_hi.at(i * cfg::tcK + z) = this->tmem_at(lane0 + i, job.a_base + cfg::tcK + z);
          }
        }
      }
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        c_frag.at(t) = zero_c ? reg_data_t{}
                              : this->tmem_at(lane0 + t / wg_cfg::tcN, col0 + t % wg_cfg::tcN);
      }
      cur_block_ = b;
      job_mx_ = {vt::mx_scale_format(job.fmt_s), job.side, job.m, job.n, b};
      this->mma_lmem(kJobKeyBase + b, job.fmt_s, job.fmt_d, job.m, job.n, 0,
                     job.a_tmem ? a_lo : kNone, job.a_tmem ? a_hi : kNone, c_frag, d_frag,
                     job.sparse, !job.a_tmem, xtile_n);
      for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
        this->tmem_at(lane0 + t / wg_cfg::tcN, col0 + t % wg_cfg::tcN) = d_frag.at(t);
      }
    }
    ++perf_stats_.tj_mma_uops;
    ++steps_issued_;
    step_wb_.push_back({now + kFedpLatency, job.m, col0});
    cyc_.busy_blocks += VX_CFG_NUM_TCU_BLOCKS;

    if (++job.m < wg_cfg::m_steps) {
      return;
    }
    job.m = 0;
    if (++job.n < job.n_frag) {
      return;
    }
    // Last step issued: the op leaves the engine and posts its commits; its
    // write-backs settle it.
    this->job_post_commits(job);
    this->span_ran(job.seq);
    run_valid_ = false;
  }

  // ---- allocation ----

  // The lowest free run of `chunks` chunks aligned to its size, or ~0.
  uint32_t alloc_fit(uint32_t chunks) const {
    uint32_t mask = (chunks == 32) ? ~0u : ((1u << chunks) - 1);
    for (uint32_t i = 0; i + chunks <= kAllocChunks; i += chunks) {
      if ((tmem_used_ & (mask << i)) == 0) {
        return i;
      }
    }
    return ~0u;
  }

  // A CTA's last warp has retired: free the columns it still owns.
  void cta_exit(int32_t cta) {
    for (uint32_t i = 0; i < kAllocChunks; ++i) {
      if (((tmem_used_ >> i) & 1u) && tmem_owner_.at(i) == cta) {
        tmem_used_ &= ~(1u << i);
      }
    }
  }
#endif // VX_CFG_TCU_TMEM_ENABLE
#endif // VX_CFG_TCU_WGMMA_ENABLE


  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            uint32_t step_k,
            const RegDataVec& rs1_data,
            const RegDataVec& rs2_data,
            const RegDataVec& rs3_data,
            RegDataVec& rd_data,
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

    // WMMA occupies the low half of a widened FEDP and zeros the unused half.
    reg_data_t a_tile[cfg::tcM * kFedpWords] = {};
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        a_tile[i * kFedpWords + z] = rs1_data.at(a_off + i * cfg::tcK + z);
      }
    }

    reg_data_t b_tile[cfg::tcM * cfg::tcN * kFedpWords] = {};
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
            b_tile[(i * cfg::tcN + j) * kFedpWords + z].u32 =
                gather_sparse(rs2_data.at(b_idx).u32, rs2_data.at(b_idx + 1).u32, lo, hi, ebits);
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            b_tile[(i * cfg::tcN + j) * kFedpWords + z] = rs2_data.at(b_off + j * cfg::tcK + z);
          }
        }
      }
    }

    fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d,
              a_tile, b_tile, rs3_data, rd_data,
              is_sparse, kFedpWords, cfg::tcK,
              cfg::k_steps * cfg::tcK);
  }

  void wgmma(uint32_t wid,
             uint32_t fmt_s,
             uint32_t fmt_d,
             uint32_t step_m,
             uint32_t step_n,
             uint32_t step_k,
             uint32_t a_desc,
             uint32_t b_desc,
             const RegDataVec& rs1_data,
             const RegDataVec& rs2_data,
             const RegDataVec& rs3_data,
             RegDataVec& rd_data,
             bool is_sparse,
             uint32_t cd_nregs,
             uint32_t is_a_smem,
             uint32_t is_setup_uop) {
    if (is_sparse && !vt::sparse_format_supported(fmt_s)) {
      std::cout << "Error: WGMMA_SP unsupported input format: "
                << vt::fmt_string(fmt_s) << " (id=" << fmt_s << ")" << std::endl;
      std::abort();
    }

    // NRC: cd_nregs 0/1/2 → 8/16/32; xtileN = NRC * NT / xtileM.
    uint32_t nrc = (cd_nregs == 0) ? 8 : (cd_nregs == 1) ? 16 : 32;
    uint32_t xtile_n = (nrc * VX_CFG_NUM_THREADS) / wg_cfg::xtileM;

    // Decode smem descriptors (B always from smem, A optionally).
    if (is_setup_uop) {
      wgmma_desc_[wid][0] = a_desc;
      wgmma_desc_[wid][1] = b_desc;
      this->decode_descs(wid, fmt_s, is_a_smem, is_sparse);
      cur_xtile_n_ = xtile_n;
      return;
    }

    this->mma_lmem(wid, fmt_s, fmt_d, step_m, step_n, step_k,
                   rs1_data, rs2_data, rs3_data, rd_data,
                   is_sparse, is_a_smem, xtile_n);
    __unused(b_desc);
  }

  // One WGMMA compute uop with B (and A when is_a_smem) read through the tile
  // buffers. `key` selects the decoded LMEM descriptors.
  void mma_lmem(uint32_t key,
                uint32_t fmt_s,
                uint32_t fmt_d,
                uint32_t step_m,
                uint32_t step_n,
                uint32_t step_k,
                const RegDataVec& rs1_data,
                const RegDataVec& rs2_data,
                const RegDataVec& rs3_data,
                RegDataVec& rd_data,
                bool is_sparse,
                bool is_a_smem,
                uint32_t xtile_n) {
    uint32_t wid     = key;
    uint32_t ratio   = elem_ratio(fmt_s);
    uint32_t k_words = is_sparse ? wg_cfg::sp_k_words : kFedpWords;
    lmem_desc_t sd_a = lmem_desc_[key][0];
    lmem_desc_t sd_b = lmem_desc_[key][1];
    if (is_sparse && kSpFullK && sd_b.ldm == 0) {
      // A block-major sparse B block holds tcK K words, not the fedpK a
      // full-K sparse step consumes.
      std::cout << "Error: a full-K sparse step requires a K-major B descriptor (ldm != 0)" << std::endl;
      std::abort();
    }
    // load_lmem_word distinguishes A from B by descriptor base.
    cur_a_desc_base_ = is_a_smem ? sd_a.base : ~uint64_t(0);
    cur_is_sparse_ = is_sparse;
    cur_xtile_n_ = xtile_n;

    reg_data_t a_tile[cfg::tcM * kFedpWords] = {};
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      uint32_t a_row_idx = step_m * cfg::tcM + i;
      for (uint32_t z = 0; z < k_words; ++z) {
        if (is_a_smem) {
          uint32_t k_elem = (step_k * k_words + z) * ratio;
          a_tile[i * k_words + z].u32 =
              load_lmem_word(sd_a, a_row_idx, k_elem, fmt_s, false, false);
        } else {
          if (z < cfg::tcK) {
            a_tile[i * k_words + z] = rs1_data.at(i * cfg::tcK + z);
          } else {
            a_tile[i * k_words + z] = rs2_data.at(i * cfg::tcK + (z - cfg::tcK));
          }
        }
      }
    }

    reg_data_t b_tile[cfg::tcM * cfg::tcN * kFedpWords] = {};
    if (is_sparse) {
      uint32_t ebits       = elem_bits(fmt_s);
      uint32_t rtl_i_ratio = 32 / ebits;
      // One metadata bank per tcK compressed words: bank {step_m,
      // step_k_half}, m=1 -> bank (cfg::k_steps/2). A full-K sparse step spans
      // two banks (its upper FEDP half reads {step_m, 1}).
      uint32_t meta_row_w  = cfg::tcK * 2 * rtl_i_ratio;
      uint32_t wg_bank = 0;
      // Metadata is preloaded into sparse_meta_ via TCU_LD before dispatch.
      auto meta_bit_wg = [&](uint32_t bit_idx) -> uint32_t {
        uint32_t word_idx = wg_bank * kMaxMetaCols + bit_idx / 32;
        uint32_t word_val = sparse_meta_.at(this->meta_slot(wid)).at(word_idx);
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
            uint32_t zz = z % cfg::tcK;
            wg_bank = step_m * (cfg::k_steps / 2) + step_k + z / cfg::tcK;
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < rtl_i_ratio; ++b) {
              lo |= meta_bit_wg(row_base + rtl_i_ratio * zz              + b) << b;
              hi |= meta_bit_wg(row_base + rtl_i_ratio * (cfg::tcK + zz) + b) << b;
            }
            constexpr uint32_t kCompression = 2;
            uint32_t k_elem_b0 = (step_k * k_words + z) * ratio * kCompression;
            uint32_t bword0 = load_lmem_word(sd_b, k_elem_b0,         b_col_idx, fmt_s, true, true);
            uint32_t bword1 = load_lmem_word(sd_b, k_elem_b0 + ratio, b_col_idx, fmt_s, true, true);
            uint32_t gathered = gather_sparse(bword0, bword1, lo, hi, ebits);
            b_tile[(i * cfg::tcN + j) * k_words + z].u32 = gathered;
            // Trace: B-gather (CSV: wid,step_m,step_n,i,lane,bword0,bword1,lo,hi,gathered).
            if (const char* p = std::getenv("VORTEX_TCU_TRACE")) {
              if (p[0] == '1') {
                fprintf(stderr, "GATHER,%u,%u,%u,%u,%u,0x%08x,0x%08x,%u,%u,0x%08x\n",
                        wid, step_m, step_n, i, j*k_words+z,
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
            b_tile[(i * cfg::tcN + j) * k_words + z].u32 =
                load_lmem_word(sd_b, k_elem, b_col_idx, fmt_s, true, false);
          }
        }
      }
    }

    fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d,
              a_tile, b_tile, rs3_data, rd_data,
              is_sparse, k_words, k_words,
              wg_cfg::k_steps * wg_cfg::fedpK);
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

  uint32_t mx_tile_scale_blocks(uint32_t fmt_s, uint32_t tile_k_words) const {
    uint32_t logical_tile_k = tile_k_words * elem_ratio(fmt_s);
    uint32_t block_size = vt::mx_scale_block_size(fmt_s);
    return std::max(1u, (logical_tile_k + block_size - 1) / block_size);
  }

  static uint8_t meta_byte(const std::vector<uint32_t>& words, uint32_t index) {
    return (words.at(index / 4) >> (8 * (index & 0x3))) & 0xff;
  }

  uint32_t eval_mx_fedp(uint32_t wid, uint32_t fmt_s, uint32_t fmt_d,
                        uint32_t step_m, uint32_t step_n, uint32_t step_k,
                        uint32_t i, uint32_t j, const reg_data_t* a_row,
                        const reg_data_t* b_col, uint32_t c_val, bool is_sparse,
                        uint32_t fedp_words, uint32_t tile_k_words) const {
#ifndef VX_CFG_TCU_MX_ENABLE
    __unused(wid, fmt_s, fmt_d, step_m, step_n, step_k, i, j, a_row, b_col,
             is_sparse, fedp_words, tile_k_words);
    return c_val;
#else
    uint32_t ratio = elem_ratio(fmt_s);
    uint32_t scale_blocks_k = mx_tile_scale_blocks(fmt_s, tile_k_words);
    uint32_t block_size = vt::mx_scale_block_size(fmt_s);
#ifdef VX_CFG_TCU_TMEM_ENABLE
    // A tensor op's step reads its scales from TMEM (tmem_scale_a/b).
    bool job_mx = (wid >= kJobKeyBase) && job_mx_.active;
    uint32_t step_elems = fedp_words * ratio * (is_sparse ? 2 : 1);
#endif
    auto scale_a = [&](uint32_t elem_k) -> uint32_t {
#ifdef VX_CFG_TCU_TMEM_ENABLE
      if (job_mx) {
        return this->tmem_scale_a(i, elem_k * kMxSf / step_elems);
      }
#endif
      uint32_t row = step_m * cfg::tcM + i;
      return meta_byte(mx_meta_a_.at(wid), row * scale_blocks_k + elem_k / block_size);
    };
    auto scale_b = [&](uint32_t elem_k) -> uint32_t {
#ifdef VX_CFG_TCU_TMEM_ENABLE
      if (job_mx) {
        return this->tmem_scale_b(j, elem_k * kMxSf / step_elems);
      }
#endif
      uint32_t col = step_n * cfg::tcN + j;
      return meta_byte(mx_meta_b_.at(wid), col * scale_blocks_k + elem_k / block_size);
    };

    if (fmt_d == vt::fp32::id) {
#ifdef VX_CFG_TCU_TYPE_TFR
      // The native datapath: the TfrFedp model, with each lane's (16 bits of
      // a register) scales taken at its first element.
      {
        int eb = 0, sb = 0;
        bool e4m3_scale = false;
        if (fmt_s == vt::mxfp8::id) {
          eb = 4; sb = 3;
        } else if (fmt_s == vt::mxbf8::id) {
          eb = 5; sb = 2;
        } else if (fmt_s == vt::mxfp4::id) {
          eb = 2; sb = 1;
        } else if (fmt_s == vt::nvfp4::id) {
          eb = 2; sb = 1; e4m3_scale = true;
        } else {
          std::abort();
        }
        uint32_t sparse_ratio = is_sparse ? 2 : 1;
        std::vector<uint32_t> a(fedp_words), b(fedp_words);
        std::vector<uint8_t> sf_a(2 * fedp_words), sf_b(2 * fedp_words);
        for (uint32_t z = 0; z < fedp_words; ++z) {
          a[z] = a_row[z].u32;
          b[z] = b_col[z].u32;
        }
        for (uint32_t lane = 0; lane < 2 * fedp_words; ++lane) {
          uint32_t z = lane / 2;
          uint32_t e = (lane % 2) * (ratio / 2);
          uint32_t elem_k = ((step_k * fedp_words + z) * ratio + e) * sparse_ratio;
          sf_a[lane] = uint8_t(scale_a(elem_k));
          sf_b[lane] = uint8_t(scale_b(elem_k));
        }
        TfrFedp fedp(eb, sb);
        float c;
        std::memcpy(&c, &c_val, sizeof(c));
        float d = fedp(a.data(), b.data(), c, fedp_words, sf_a.data(), sf_b.data(), e4m3_scale);
        uint32_t d_bits;
        std::memcpy(&d_bits, &d, sizeof(d_bits));
        return d_bits;
      }
#endif
      uint32_t acc = c_val;
      for (uint32_t z = 0; z < fedp_words; ++z) {
        for (uint32_t e = 0; e < ratio; ++e) {
          uint32_t sparse_ratio = is_sparse ? 2 : 1;
          uint32_t elem_k = ((step_k * fedp_words + z) * ratio + e) * sparse_ratio;
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

    std::cout << "Error: unsupported MX mma format: " << fmt_s << " -> " << fmt_d << "!" << std::endl;
    std::abort();
#endif
  }

  // Element offset of one operand element from its descriptor base. A is
  // (row=M, col=K) and B is (row=K, col=N); `pack_along_row` selects B.
  static uint64_t elem_offset(const lmem_desc_t& desc, uint32_t row, uint32_t col,
                              uint32_t e_bits, uint32_t xtile_n,
                              bool pack_along_row, bool sparse_b,
                              bool sparse_a_layout) {
    uint32_t ratio = (e_bits >= 32) ? 1 : (32 / e_bits);
    uint64_t elem_off;
    if (desc.ldm == 0) {
      // Block-major SMEM. K dimension is along col for A (pack_along_row
      // false) and along row for B (pack_along_row true).
      uint32_t k_blk_dim = (sparse_a_layout && !pack_along_row)
                         ? (cfg::tcK * ratio)
                         : (kFedpWords * ratio);
      if (pack_along_row && sparse_b) {
        // Sparse B in flat (candidate-pair) layout: block-contiguous
        // K-word-major / N-inner order [kw_in*tcN + n_in].
        uint32_t b_tcK_words = cfg::tcK * 2;
        uint32_t k_word = row / ratio;
        uint32_t elem   = row % ratio;
        uint32_t k_blk  = k_word / b_tcK_words;
        uint32_t kw_in  = k_word % b_tcK_words;
        uint32_t n_blk  = col / cfg::tcN;
        uint32_t n_in   = col % cfg::tcN;
        uint32_t blk_words = cfg::tcN * b_tcK_words;
        uint32_t n_steps   = xtile_n / cfg::tcN;
        uint64_t word_off  = (k_blk * n_steps + n_blk) * blk_words
                           + (kw_in * cfg::tcN + n_in);
        elem_off = word_off * ratio + elem;
      } else if (pack_along_row) {
        // Dense B (block-major): r is K coord, c is N coord; N outer, K inner.
        uint32_t k_blk = row / k_blk_dim;
        uint32_t r_in  = row % k_blk_dim;
        uint32_t n_blk = col / cfg::tcN;
        uint32_t n_in  = col % cfg::tcN;
        uint32_t b_blk_elems = k_blk_dim * cfg::tcN;
        uint32_t n_steps     = xtile_n / cfg::tcN;
        elem_off = (k_blk * n_steps + n_blk) * b_blk_elems
                 + n_in * k_blk_dim + r_in;
      } else {
        // A: r is M coord, c is K coord.
        uint32_t m_blk = row / cfg::tcM;
        uint32_t i_in  = row % cfg::tcM;
        uint32_t k_blk = col / k_blk_dim;
        uint32_t k_in  = col % k_blk_dim;
        uint32_t a_blk_elems = cfg::tcM * k_blk_dim;
        elem_off = (k_blk * wg_cfg::m_steps + m_blk) * a_blk_elems
                 + i_in * k_blk_dim + k_in;
      }
    } else if (desc.col_major) {
      elem_off = uint64_t(col) * desc.ldm + row;
    } else if (pack_along_row) {
      // B: K-major (N-outer K-inner). row=K, col=N;
      // ldm = stride in elements between N rows.
      elem_off = uint64_t(col) * desc.ldm + row;
    } else {
      // A: row-major (M-outer K-inner). row=M, col=K.
      elem_off = uint64_t(row) * desc.ldm + col;
    }
    return elem_off;
  }

  // Gather one 32-bit operand word from a TCU line cache.
  // `read_line` is supplied by the caller and routes to the right per-role
  // buffer inside TcuTbuf (A → read_a, B → read_b). For sub-32-bit formats,
  // packs `ratio = 32 / e_bits` adjacent elements along the K direction
  // (col for A, row for B via pack_along_row).
  template <typename ReadLine>
  uint32_t gather_word(ReadLine read_line, const lmem_desc_t& desc,
                       uint32_t row, uint32_t col,
                       uint32_t fmt_s, bool pack_along_row, bool sparse_b,
                       bool sparse_a_layout) const {
    uint32_t e_bits  = elem_bits(fmt_s);
    uint32_t ratio   = (e_bits >= 32) ? 1 : (32 / e_bits);
    uint32_t result = 0;
    for (uint32_t r = 0; r < ratio; ++r) {
      uint32_t cur_row = pack_along_row ? (row + r) : row;
      uint32_t cur_col = pack_along_row ? col       : (col + r);
      uint64_t elem_off = elem_offset(desc, cur_row, cur_col, e_bits, cur_xtile_n_,
                                      pack_along_row, sparse_b, sparse_a_layout);
      uint64_t byte_addr = desc.base + elem_off * e_bits / 8;
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
      } else if (e_bits == 4) {
        uint32_t val = (uint32_t((*line)[off]) >> (4 * (elem_off & 1))) & 0xf;
        result |= val << (r * 4);
      } else {
        std::abort();
      }
    }
    return result;
  }

  // Routes A reads through the current block's A buffer; B through the shared B buffer.
  // sparse_b selects the flat candidate-pair B layout.
  uint32_t load_lmem_word(const lmem_desc_t& desc, uint32_t row, uint32_t col,
                          uint32_t fmt_s, bool pack_along_row, bool sparse_b) const {
    auto& tbuf = simobject_->tbuf();
    if (desc.base == cur_a_desc_base_) {
      uint32_t b = cur_block_;
      return gather_word([&](uint64_t addr) { return tbuf->read_a(b, addr); },
                         desc, row, col, fmt_s, pack_along_row, false,
                         cur_is_sparse_ && !kSpFullK);
    }
    return gather_word([&](uint64_t addr) { return tbuf->read_b(addr); },
                       desc, row, col, fmt_s, pack_along_row, sparse_b, false);
  }

  static constexpr uint32_t kSparseKSteps = cfg::k_steps / 2;
  static constexpr uint32_t kMetaBanks = cfg::m_steps * kSparseKSteps;
  static constexpr uint32_t kMaxMetaCols = VX_CFG_NUM_THREADS / 2;
#ifdef VX_CFG_TCU_TMEM_ENABLE
  // Tensor jobs hold one K-tile's metadata per block, after the warps' slots.
  static constexpr uint32_t kMetaSlots = VX_CFG_NUM_WARPS + VX_CFG_NUM_TCU_BLOCKS;
  uint32_t meta_slot(uint32_t key) const {
    return (key >= kJobKeyBase) ? (VX_CFG_NUM_WARPS + key - kJobKeyBase) : key;
  }
#else
  static constexpr uint32_t kMetaSlots = VX_CFG_NUM_WARPS;
  uint32_t meta_slot(uint32_t key) const { return key; }
#endif

  // FEDP tile computation for both WMMA and WGMMA.
  // a_tile: [tcM][k_words] flat array of pre-loaded A operand words.
  // b_tile: [tcM][tcN][k_words] flat array of pre-loaded B operand words
  //         (for dense, each i-slice is identical; for sparse, already gathered).
  void fedp_tile(uint32_t wid,
                 uint32_t step_m, uint32_t step_n, uint32_t step_k,
                 uint32_t fmt_s, uint32_t fmt_d,
                 const reg_data_t* a_tile,
                 const reg_data_t* b_tile,
                 const RegDataVec& rs3_data,
                 RegDataVec& rd_data,
                 bool is_sparse,
                 uint32_t k_words = cfg::tcK,
                 uint32_t fedp_words = cfg::tcK,
                 uint32_t tile_k_words = cfg::k_steps * cfg::tcK) {
    PFN_FEDP_N fedp = vt::mx_scale_format(fmt_s) ? nullptr : select_FEDP_N(fmt_s, fmt_d);

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto t = i * cfg::tcN + j;
        auto c_val = rs3_data.at(t).u32;
        auto a_row = &a_tile[i * k_words];
        auto b_col = &b_tile[(i * cfg::tcN + j) * k_words];

        uint32_t d_val = vt::mx_scale_format(fmt_s)
                       ? eval_mx_fedp(wid, fmt_s, fmt_d, step_m, step_n, step_k,
                                      i, j, a_row, b_col, c_val, is_sparse,
                                      fedp_words, tile_k_words)
                       : fedp(a_row, b_col, c_val, k_words);

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
  std::array<std::array<uint32_t, 2>, VX_CFG_NUM_WARPS> wgmma_desc_;
  mutable PerfStats perf_stats_;
  // This cycle's MMA activity, for the busy/wait breakdown.
  struct {
    uint32_t busy_blocks;
    bool opnd, other;
  } cyc_ = {};
  // Per-block guard: execute already happened for this trace; reset on pop().
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> exec_done_;
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> tj_fresh_;
  std::array<uint64_t, VX_CFG_NUM_TCU_BLOCKS> tj_hold_until_;
  // Output slices of the head uop already through the FEDP array.
  std::array<uint32_t, VX_CFG_NUM_TCU_BLOCKS> slices_;
  // Cycles at which each block's results sent so far become visible.
  std::array<std::vector<uint64_t>, VX_CFG_NUM_TCU_BLOCKS> result_due_;
  // True while a block is between its first and last WGMMA uop.
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> in_wgmma_;
  // Current block index, set before delegating to wgmma().
  uint32_t cur_block_ = 0;
  // B buffer slot read by the current MMA.
  // A-descriptor base for the current wgmma(); distinguishes A from B in load_lmem_word.
  uint64_t cur_a_desc_base_ = ~uint64_t(0);
  bool cur_is_sparse_ = false;
  // xtileN for the active WGMMA (derived from NRC).
  uint32_t cur_xtile_n_ = 8;
  // CTA each block holds while mid-WGMMA (-1 = none).
  std::array<int32_t, VX_CFG_NUM_TCU_BLOCKS> cta_owner_a_{};
#ifdef VX_CFG_TCU_WGMMA_ENABLE
  std::array<abuf_slot_t, VX_CFG_NUM_TCU_BLOCKS> abuf_;
  bbuf_slot_t bbuf_;
#endif
#ifdef VX_CFG_TCU_TMEM_ENABLE
  std::vector<reg_data_t> tmem_;  // [lane][column]
  std::deque<tj_t> ops_;          // posted tc.mma / tc.cp / tc.commit, in order
  tj_t run_;                      // the op the engine runs
  bool run_valid_ = false;
  std::deque<tc_commit_t> commits_;
  std::deque<tc_wb_t> step_wb_;   // in-flight steps' D write-backs, in order
  std::deque<tc_span_t> spans_;   // unsettled ops, for the contract checker
  uint64_t steps_issued_ = 0;
  uint64_t steps_done_ = 0;
  bool tj_port_busy_ = false;
  bool tmem_rd_taken_ = false;
  std::array<uint32_t, VX_CFG_NUM_TCU_BLOCKS> blk_port_ops_{};  // port cycles of the head transfer so far
  uint64_t tj_seq_ = 0;
  std::array<int32_t, kAllocChunks> tmem_owner_{};
  uint32_t tmem_used_ = 0;  // allocated kAllocCols-column chunks, bit per chunk
#endif
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
  case TcuType::TMEM_LD:
    return {std::string(args.tmem_row ? "TMEM_LD.F." : "TMEM_LD.") + std::to_string(args.tmem_n) + "." + std::to_string(args.tmem_idx), ""};
  case TcuType::TMEM_ST:
    return {std::string(args.tmem_row ? "TMEM_ST.F." : "TMEM_ST.") + std::to_string(args.tmem_n) + "." + std::to_string(args.tmem_idx), ""};
  case TcuType::TMEM_ALLOC:
    return {"TMEM_ALLOC", ""};
  case TcuType::TMEM_DEALLOC:
    return {"TMEM_DEALLOC", ""};
  case TcuType::TC_MMA:
    return {"TC_MMA", ""};
  case TcuType::TC_COMMIT:
    return {"TC_COMMIT", ""};
  case TcuType::TC_CP:
    return {"TC_CP", ""};
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

#ifdef VX_CFG_TCU_TMEM_ENABLE
  if (tcu_is_tmem(tcu_type)) {
    return std::get<IntrTcuArgs>(instr.get_args()).tmem_n;
  }
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
    bool needs_setup = wgmma_needs_setup(args.is_a_smem, is_sparse);
    uint32_t nrc = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
    uint32_t k_count = is_sparse ? std::max(1u, wg_cfg::k_steps / 2) : wg_cfg::k_steps;
    uint32_t mma_uops = k_count * nrc;
    return mma_uops + needs_setup;
  }
#endif

  return 1;
}

///////////////////////////////////////////////////////////////////////////////

Instr::Ptr TcuUopGen::get(const Instr& macro_instr, uint32_t uop_index) {
  auto tcu_type = std::get<TcuType>(macro_instr.get_op_type());
  auto args = std::get<IntrTcuArgs>(macro_instr.get_args());
  uint64_t parent_uuid = macro_instr.get_uuid();
  uint64_t uop_uuid = make_uop_uuid(parent_uuid, uop_index);

  auto uop_instr = std::allocate_shared<Instr>(pool_, uop_uuid, FUType::TCU);
  uop_instr->set_parent_uuid(parent_uuid);
  uop_instr->set_fcsr_use(macro_instr.fcsr_reads(), macro_instr.fcsr_writes());

#ifdef VX_CFG_TCU_TMEM_ENABLE
  if (tcu_is_tmem(tcu_type)) {
    // Uop i moves FP register (f + i) to or from its TMEM columns.
    uint32_t freg = (tcu_type == TcuType::TMEM_LD) ? args.fmt_d : args.fmt_s;
    if (freg + args.tmem_n > 32) {
      std::cout << "Error: TMEM transfer of " << args.tmem_n << " registers from f"
                << freg << " exceeds the register file" << std::endl;
      std::abort();
    }
    IntrTcuArgs uop_args = args;
    uop_args.tmem_idx = uop_index;
    uop_instr->set_op_type(tcu_type);
    uop_instr->set_args(uop_args);
    uop_instr->set_src_reg(0, macro_instr.get_src_reg(0).idx, RegType::Integer);
    if (tcu_type == TcuType::TMEM_LD) {
      uop_instr->set_dest_reg(freg + uop_index, RegType::Float);
    } else {
      uop_instr->set_src_reg(1, freg + uop_index, RegType::Float);
    }
    return uop_instr;
  }
#endif

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

        uint32_t n_sp = step_bits ? (mma_idx & step_mask) : 0;
        uint32_t m_sp = mma_idx >> step_bits;
        // n_sp encodes both actual N step (high bits) and lo/hi half (low lg_k bits).
        // Extract just the N step for accum indexing; n_sp is still used for B register selection.
        uint32_t actual_n = lg_k ? (n_sp >> lg_k) : n_sp;
        uint32_t reg_rs3 = rc_base + (mma_idx >> 1);
        uop_instr->set_op_type(TcuType::WMMA_SP);
        uop_instr->set_args(IntrTcuArgs{0, 0, fmt_s, fmt_d, m_sp, actual_n, 0, 0, 0, 0, 0, 0, 0});
        uop_instr->set_dest_reg(reg_rs3, RegType::Float);
        uop_instr->set_src_reg(0, ra_base + m_sp, RegType::Float);
        uop_instr->set_src_reg(1, rb_base + n_sp, RegType::Float);
        uop_instr->set_src_reg(2, reg_rs3, RegType::Float);
        // Symmetric sparse: no k-loop, always wb=1 and always reads C from RF.
        // Even uops run the low half of each block's columns, odd uops the high half.
        ThreadMask tmask(VX_CFG_NUM_THREADS);
        bool hi_half = (mma_idx & 1);
        for (uint32_t lane = 0; lane < VX_CFG_NUM_THREADS; ++lane) {
          if (((lane % wmma::tcN) >= (wmma::tcN / 2)) == hi_half) {
            tmask.set(lane);
          }
        }
        uop_instr->set_tmask(tmask);
      } else {
        // Standard k-major triple loop (dense or non-sym sparse)
        uint32_t mn = wmma::m_steps * wmma::n_steps;
        uint32_t k = mma_idx / mn;
        uint32_t rem = mma_idx % mn;
        uint32_t m = rem / wmma::n_steps;
        uint32_t n = rem % wmma::n_steps;
        uint32_t reg_rs1, reg_rs2, reg_rs3;
        if (is_sparse) {
          // Sparse non-sym: A=identity, B and C permuted.
          uint32_t a_off = (m / wmma::a_sub_blocks) * k_count + k;
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + frag::rb_off_sp(n, k);
          reg_rs3 = rc_base + frag::rc_off_sp(m, n);
        } else {
          reg_rs1 = ra_base + frag::ra_off(m, k);
          reg_rs2 = rb_base + frag::rb_off(n, k);
          reg_rs3 = rc_base + frag::rc_off(m, n);
        }
        uop_instr->set_op_type(is_sparse ? TcuType::WMMA_SP : TcuType::WMMA);
        uop_instr->set_args(IntrTcuArgs{0, 0, fmt_s, fmt_d, m, n, k, 0, 0, 0, 0, 0, 0});
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
    constexpr uint32_t a_reg_k_steps = 2;
    uint32_t fmt_s = args.fmt_s;
    uint32_t fmt_d = args.fmt_d;
    bool is_sparse = tcu_is_sparse(tcu_type);
    bool is_a_smem = args.is_a_smem;
    bool needs_setup = wgmma_needs_setup(is_a_smem, is_sparse);
    uint32_t cd_nregs = args.cd_nregs;
    uint32_t k_count = is_sparse ? std::max(1u, k_steps / 2) : k_steps;
    uint32_t total = uop_count(macro_instr);
    constexpr uint32_t a0 = 10, a1 = 11;

    {
      uop_instr->set_op_type(is_sparse ? TcuType::WGMMA_SP : TcuType::WGMMA);

      if (needs_setup && uop_index == 0) {
        uop_instr->set_args(IntrTcuArgs{is_a_smem ? 1u : 0u, cd_nregs,
                                       fmt_s, fmt_d, 0, 0, 0, 1, 0, 1, 0, 0, 0});
        uop_instr->set_src_reg(0, a0, RegType::Integer);
        uop_instr->set_src_reg(1, a1, RegType::Integer);
        uop_instr->set_fu_lock(true);
        uop_instr->set_fu_unlock(false);
        return uop_instr;
      }

      // MMA phase
      uint32_t mma_idx = uop_index - needs_setup;
      uint32_t ra_base = is_a_smem ? 10 : 24;

      // Loop order: m (inner) -> n (middle) -> k (outer). K-outer maximizes
      // per-block A-buffer reuse: each A_w[m,k] is consumed across the entire
      // (n,m) inner sweep, and each shared B[k,n] is consumed for m_steps
      // consecutive uops.
      uint32_t compute_total = total - needs_setup;
      uint32_t mn = compute_total / k_count;
      uint32_t k = mma_idx / mn;
      uint32_t rem = mma_idx % mn;
      uint32_t n = rem / m_steps;
      uint32_t m = rem % m_steps;
      uint32_t r = n * m_steps + m;

      bool first = (mma_idx == 0);
      bool last  = (uop_index == (total - 1));
      uop_instr->set_args(IntrTcuArgs{is_a_smem ? 1u : 0u, cd_nregs,
                                     fmt_s, fmt_d, m, n, k, first ? 1u : 0u, last ? 1u : 0u, 0, 0, 0, 0});
      uop_instr->set_dest_reg(r, RegType::Float);
      if (!is_a_smem) {
        uint32_t rs1_off = is_sparse ? m : (m * a_reg_k_steps + k);
        uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
        if (needs_setup)
          uop_instr->set_src_reg(1, ra_base + rs1_off + 1, RegType::Float);
      } else if (first) {
        uop_instr->set_src_reg(0, a0, RegType::Integer);
      }
      if (first && !needs_setup)
        uop_instr->set_src_reg(1, a1, RegType::Integer);
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
	: FuncUnit(ctx, name, core, 1u << log2ceil(VX_CFG_TCU_LATENCY + 1)) // landing-queue depth bounds results awaiting the consumer
#ifdef TCU_META_ENABLE
	, agu_req_out(TCU_AGU_PORTS, this)
	, agu_rsp_in(TCU_AGU_PORTS, this)
#endif
#ifdef VX_CFG_TCU_TMEM_ENABLE
	, cta_exit_in(this)
#endif
	, result_stall_in(VX_CFG_NUM_TCU_BLOCKS, this)
	, impl_(new Impl(this, core))
{}

TcuUnit::~TcuUnit() {
  delete impl_;
}

void TcuUnit::set_tbuf(TcuTbuf::Ptr tbuf) {
  tbuf_ = std::move(tbuf);
}

const TcuTbuf::Ptr& TcuUnit::tbuf() const {
  return tbuf_;
}

void TcuUnit::on_reset() {
  impl_->reset();
}

void TcuUnit::on_tick() {
#ifdef VX_CFG_TCU_TMEM_ENABLE
  while (!cta_exit_in.empty()) {
    impl_->cta_exit(cta_exit_in.peek());
    cta_exit_in.pop();
  }
#endif
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
                      const RegDataVec& rs1_data,
                      const RegDataVec& rs2_data,
                      const RegDataVec& rs3_data,
                      RegDataVec& rd_data,
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
                    const RegDataVec& rs1_data,
                    const RegDataVec& rs2_data,
                    const RegDataVec& rs3_data,
                    RegDataVec& rd_data,
                    bool is_sparse,
                    uint32_t cd_nregs,
                    uint32_t is_a_smem,
                    uint32_t is_setup_uop) {
  impl_->wgmma(wid, fmt_s, fmt_d, step_m, step_n, step_k, a_desc, b_desc,
               rs1_data, rs2_data, rs3_data, rd_data,
               is_sparse, cd_nregs, is_a_smem, is_setup_uop);
}
