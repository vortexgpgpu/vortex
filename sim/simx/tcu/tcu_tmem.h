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
#include "tensor_cfg.h"
#include <unordered_set>

#ifdef VX_CFG_TCU_TMEM_ENABLE

namespace vortex {

// Tensor memory: the UMMA accumulator store, plus its column allocator.
//
// TMEM in Blackwell is architecturally per-SM, not per-tensor-core. Living one
// level down inside TcuUnit is correct because Vortex instantiates one TcuUnit
// per Core; if that ever changes, this object should move up to Core.
class TcuTmem {
public:
  static constexpr uint32_t kCols = VX_CFG_TCU_TMEM_COLS;

  // TMEM geometry only depends on NT. Element types are placeholders and must
  // not be used for anything type-dependent.
  using wg_cfg = vortex::tensor::wgmma_config_t<VX_CFG_NUM_THREADS,
                                                vortex::tensor::fp32,
                                                vortex::tensor::fp32>;
  static constexpr uint32_t kLanes = wg_cfg::xtileM * VX_CFG_NUM_TCU_BLOCKS;
  static_assert(kLanes <= 128, "TMEM lanes exceed cap");

  // One physical bank per TCU block
  static constexpr uint32_t kBanks = VX_CFG_NUM_TCU_BLOCKS;

  // Live-allocation CAM and its free list are both NUM_ENTRIES deep.
  static constexpr uint32_t kWarpgroupSize = kBanks;  // == WARPGROUP_SIZE
  static constexpr uint32_t kMaxConcurrentCtas = VX_CFG_NUM_WARPS / kWarpgroupSize;
  static constexpr uint32_t kAllocEntries = kMaxConcurrentCtas + 1;

  static constexpr uint32_t kNoGrant = uint32_t(-1);

  struct PerfStats {
    uint64_t bank_stalls = 0;    // cycles a requester lost its bank to a conflict
    uint64_t hazard_stalls = 0;  // cycles a read was held by the RAW interlock
  };

  TcuTmem();

  void reset();

  // First-fit free-list column allocator. Multiple warpgroups (different
  // CTAs) can hold disjoint, concurrently-live allocations.
  //
  // CTA-scoped idempotency: every warp of a CTA can call alloc()
  // independently and get the same handle back, so the kernel needs no
  // elected-thread + shared-memory broadcast. The first call for a given
  // cta_id allocates; later calls from sibling warps return the cached handle.
  uint32_t alloc(uint32_t ncols, int32_t cta_id);

  // Mirrors alloc(): the range is only actually freed once every warp of the
  // CTA has called dealloc.
  void dealloc(uint32_t handle, int32_t cta_id, uint32_t wid, uint32_t cta_size);

  // Whether a fresh ALLOC would have to stall this cycle. A repeat request from
  // a CTA that already holds an allocation always succeeds, while a fresh one
  // needs both a large-enough free range and a free CAM slot.
  bool alloc_would_stall(uint32_t ncols, int32_t cta_id) const;

  // One ALLOC/DEALLOC is granted per cycle across every block, DEALLOC ahead of
  // ALLOC. Fixed priority, lowest block index first. Returns the granted block,
  // or kNoGrant.
  static uint32_t mgmt_grant(const std::bitset<kBanks>& alloc_ready,
                             const std::bitset<kBanks>& dealloc_ready);

  // Live-allocation width in columns, or 0 if `handle` is not live.
  uint32_t alloc_ncols(uint32_t handle) const;

  // Bounds-checked element access.
  uint32_t read(uint32_t lane, uint32_t col) const;
  void write(uint32_t lane, uint32_t col, uint32_t value);

  const PerfStats& perf_stats() const { return perf_stats_; }

  // Geometry
  static constexpr uint32_t kBankLanes = wg_cfg::xtileM;  // == TCU_WG_TILE_M
  static constexpr uint32_t kWordCols = wg_cfg::tcN;      // == TCU_TC_N
  // The read port serves three requesters (accumulator read, TMEM_LD, abuf
  // A-stripe fill); the write port serves two.
  static constexpr uint32_t kArbRdW = 3 * kBanks;         // == ARB_RD_W
  static constexpr uint32_t kArbWrW = 2 * kBanks;         // == ARB_WR_W

  static uint32_t bank_of(uint32_t lane_base) { return lane_base / kBankLanes; }
  static uint32_t word_of(uint32_t col_base) { return col_base / kWordCols; }

  // One requester's bid for one cycle. `valid` false means no request.
  struct BankReq {
    bool valid = false;
    uint32_t lane_base = 0;
    uint32_t col_base = 0;
  };

  // Read pool: compute [0, kBanks), ldst [kBanks, 2*kBanks), a_fill
  // [2*kBanks, 3*kBanks). Write pool: compute, then ldst.
  struct CycleReqs {
    std::array<BankReq, kBanks> compute_rd{};  // UMMA accumulator read
    std::array<BankReq, kBanks> ldst_rd{};     // TMEM_LD
    std::array<BankReq, kBanks> a_fill_rd{};   // abuf A-stripe fill (A-from-TMEM)
    std::array<BankReq, kBanks> compute_wr{};  // UMMA accumulator writeback
    std::array<BankReq, kBanks> ldst_wr{};     // TMEM_ST
  };

  // Read grants are the previous arb_arbitrate()'s, registered. Write grants
  // are the current arb_arbitrate()'s. An ungranted requester must retry
  // without popping.
  struct Grants {
    std::bitset<kBanks> compute_rd;
    std::bitset<kBanks> ldst_rd;
    std::bitset<kBanks> a_fill_rd;
    std::bitset<kBanks> compute_wr;
    std::bitset<kBanks> ldst_wr;
  };

  // Arbitration is split so the consumer can close the RTL's combinational
  // loop: tmem_rd_valid depends on ~umma_rd_won, which depends on the grant
  // published this cycle. The caller therefore decodes addresses, publishes,
  // decides validity from the published grants, then arbitrates. The caller
  // owns the compute-side sticky latches. Only the TMEM_ST latch lives on this
  // side, and arb_arbitrate() applies it itself.
  void arb_publish(const CycleReqs& reqs);
  void arb_arbitrate(const CycleReqs& reqs);

  const Grants& grants() const { return grants_; }

  // A granted TMEM_ST whose result cannot retire this cycle keeps its request
  // asserted.
  bool won_ldst_wr(uint32_t block) const { return ldst_wr_won_.test(block); }

  // Drop the latch.
  void clear_ldst_wr_win(uint32_t block) { ldst_wr_won_.reset(block); }

private:
  void validate_lane_col(uint32_t lane, uint32_t col) const;

  std::array<std::array<uint32_t, kCols>, kLanes> data_{};

  // Free-list allocator state: {start_col, ncols} ranges, and handle->ncols
  // for active allocations.
  std::vector<std::pair<uint32_t, uint32_t>> free_{{0, kCols}};
  std::unordered_map<uint32_t, uint32_t> allocs_;

  // CTA-scoped idempotent-alloc bookkeeping.
  std::unordered_map<int32_t, uint32_t> cta_handle_;
  // Distinct wids that have called dealloc for this CTA.
  std::unordered_map<int32_t, std::unordered_set<uint32_t>> cta_dealloc_warps_;

  // Arbitration state. One round-robin arbiter per bank per direction.
  std::vector<Arbiter> rd_arb_;
  std::vector<Arbiter> wr_arb_;

  // Per-bank one-hot read grant, registered one cycle (bank_rd_grant_onehot_d).
  std::array<std::bitset<kArbRdW>, kBanks> rd_grant_onehot_d_{};

  std::bitset<kBanks> ldst_wr_won_;  // == ldst_wr_won_r
  Grants grants_{};

  PerfStats perf_stats_;
};

// $clog2 semantics: rounds up
constexpr uint32_t tmem_clog2(uint32_t x) {
  uint32_t r = 0;
  while ((1u << r) < x) ++r;
  return r;
}

// In-flight TMEM write tracker
//
// One per TCU block, owned by TcuUnit. It lives in this header only because it
// shares TMEM's addressing.
//
// A UMMA uop must not read an accumulator tile that an earlier
// in-flight uop still owes a write to, or it would accumulate onto a stale
// value. The uop loop order (k outer, n middle, m inner) means consecutive
// uops of one warp address different tiles, so this should never fire.
class TcuTmemWrTrack {
public:
  // LANDQ_SIZE = MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY), where
  // PIPE_LATENCY = FEDP_LATENCY + 1. The credit bound caps
  // admitted-but-unretired uops at the same number, so it cannot overflow.
  static constexpr uint32_t kPipeLatency = 1 + VX_CFG_TCU_LATENCY;
  static constexpr uint32_t kDepth = 1u << tmem_clog2(kPipeLatency);

  void reset();

  // Whether a UMMA uop addressing this tile must stall. Only entries that owe
  // TMEM a write can collide, and the comparison is on the whole tile origin.
  bool hazard(uint32_t lane_base, uint32_t col_base) const;

  // Admission. Every non-setup uop takes an entry; `is_umma` records whether it
  // owes TMEM a write, since only those can cause a hazard.
  //
  // SimX has no landing queue and executes the whole uop at admission, so the
  // entry is released by cycle. `retire_cycle` is admission + the uop's
  // latency, which ignores output back-pressure.
  void push(bool is_umma, uint32_t lane_base, uint32_t col_base, uint64_t retire_cycle);

  // Release every entry whose retirement cycle has arrived.
  void retire(uint64_t cycle);

  uint32_t size() const { return size_; }

private:
  struct Entry {
    bool valid = false;
    bool is_umma = false;
    uint32_t lane_base = 0;
    uint32_t col_base = 0;
    uint64_t retire_cycle = 0;
  };

  std::array<Entry, kDepth> entries_{};
  uint32_t head_ = 0;
  uint32_t tail_ = 0;
  uint32_t size_ = 0;
};

} // namespace vortex

#endif // VX_CFG_TCU_TMEM_ENABLE
