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

#include "tcu_tmem.h"

#ifdef VX_CFG_TCU_TMEM_ENABLE

#include "debug.h"
#include <algorithm>
#include <iostream>

using namespace vortex;

TcuTmem::TcuTmem() {
  rd_arb_.reserve(kBanks);
  wr_arb_.reserve(kBanks);
  // Accepted divergence from RTL: RoundRobinArbiter resets last_grant_ to 0 and
  // so begins its scan at index 1, while VX_rr_arbiter resets reqs_mask to
  // all-ones and grants the lowest set request. The two pick different winners
  // on the first contended cycle after a reset, and agree from then on.
  for (uint32_t r = 0; r < kBanks; ++r) {
    rd_arb_.emplace_back(ArbiterType::RoundRobin, kArbRdW);
    wr_arb_.emplace_back(ArbiterType::RoundRobin, kArbWrW);
  }
  this->reset();
}

void TcuTmem::reset() {
  for (auto& row : data_) row.fill(0);
  free_.assign(1, {0, kCols});
  allocs_.clear();
  cta_handle_.clear();
  cta_dealloc_warps_.clear();
  for (uint32_t r = 0; r < kBanks; ++r) {
    rd_arb_.at(r).reset();
    wr_arb_.at(r).reset();
    rd_grant_onehot_d_.at(r).reset();
  }
  ldst_wr_won_.reset();
  grants_ = Grants{};
  perf_stats_ = PerfStats{};
}

///////////////////////////////////////////////////////////////////////////////

uint32_t TcuTmem::alloc(uint32_t ncols, int32_t cta_id) {
  auto cta_it = cta_handle_.find(cta_id);
  if (cta_it != cta_handle_.end()) {
    uint32_t handle = cta_it->second;
    if (allocs_.at(handle) != ncols) {
      std::cout << "Error: TMEM_ALLOC ncols mismatch for cta_id=" << cta_id
                << " (existing=" << allocs_.at(handle) << ", requested=" << ncols
                << ") — one CTA can only hold one live allocation in this PoC" << std::endl;
      std::abort();
    }
    return handle;
  }

  if (allocs_.size() >= kAllocEntries) {
    std::cout << "Error: TMEM_ALLOC exhausted the allocator CAM (" << kAllocEntries
              << " entries) — caller must gate on alloc_would_stall()" << std::endl;
    std::abort();
  }
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < ncols) continue;
    uint32_t handle = it->first;
    if (it->second == ncols) {
      free_.erase(it);
    } else {
      it->first  += ncols;
      it->second -= ncols;
    }
    allocs_[handle] = ncols;
    for (uint32_t c = handle; c < handle + ncols; ++c)
      for (auto& row : data_)
        row[c] = 0;
    cta_handle_[cta_id] = handle;
    return handle;
  }
  std::cout << "Error: TMEM allocation failed (ncols=" << ncols
            << ", no free range large enough)" << std::endl;
  std::abort();
}

void TcuTmem::dealloc(uint32_t handle, int32_t cta_id, uint32_t wid, uint32_t cta_size) {
  auto it = allocs_.find(handle);
  if (it == allocs_.end()) {
    std::cout << "Error: TMEM_DEALLOC unknown handle " << handle << std::endl;
    std::abort();
  }
  auto& dealloc_warps = cta_dealloc_warps_[cta_id];
  dealloc_warps.insert(wid);
  if (dealloc_warps.size() < cta_size) {
    return; // other warps of this CTA still hold the allocation open
  }

  free_.push_back({handle, it->second});
  allocs_.erase(it);
  cta_handle_.erase(cta_id);
  cta_dealloc_warps_.erase(cta_id);
  // Coalesce adjacent free ranges to keep the allocator from fragmenting.
  std::sort(free_.begin(), free_.end());
  for (size_t i = 0; i + 1 < free_.size();) {
    if (free_[i].first + free_[i].second == free_[i + 1].first) {
      free_[i].second += free_[i + 1].second;
      free_.erase(free_.begin() + i + 1);
    } else {
      ++i;
    }
  }
  if (free_.size() > kAllocEntries) {
    std::cout << "Error: TMEM free list exceeded " << kAllocEntries
              << " entries (allocator undersized)" << std::endl;
    std::abort();
  }
}

bool TcuTmem::alloc_would_stall(uint32_t ncols, int32_t cta_id) const {
  // A request wider than the whole array can never be satisfied, so it is an
  // error.
  if (ncols > kCols) {
    std::cout << "Error: TMEM_ALLOC request (ncols=" << ncols
              << ") exceeds kCols=" << kCols << " — unsatisfiable" << std::endl;
    std::abort();
  }
  if (cta_handle_.count(cta_id))
    return false;  // repeat request from a sibling warp, already satisfied
  if (allocs_.size() >= kAllocEntries)
    return true;   // no free CAM slot
  for (auto& range : free_) {
    if (range.second >= ncols)
      return false;
  }
  return true;     // no free range large enough
}

uint32_t TcuTmem::mgmt_grant(const std::bitset<kBanks>& alloc_ready,
                             const std::bitset<kBanks>& dealloc_ready) {
  for (uint32_t b = 0; b < kBanks; ++b) {
    if (dealloc_ready.test(b))
      return b;
  }
  for (uint32_t b = 0; b < kBanks; ++b) {
    if (alloc_ready.test(b))
      return b;
  }
  return kNoGrant;
}

uint32_t TcuTmem::alloc_ncols(uint32_t handle) const {
  auto it = allocs_.find(handle);
  return (it == allocs_.end()) ? 0 : it->second;
}

///////////////////////////////////////////////////////////////////////////////

void TcuTmem::validate_lane_col(uint32_t lane, uint32_t col) const {
  if (lane >= kLanes) {
    std::cout << "Error: TMEM lane " << lane << " exceeds kLanes=" << kLanes << std::endl;
    std::abort();
  }
  for (auto& kv : allocs_) {
    if (col >= kv.first && col < kv.first + kv.second) return;
  }
  std::cout << "Error: TMEM column " << col << " not within any active allocation" << std::endl;
  std::abort();
}

uint32_t TcuTmem::read(uint32_t lane, uint32_t col) const {
  this->validate_lane_col(lane, col);
  return data_.at(lane).at(col);
}

void TcuTmem::write(uint32_t lane, uint32_t col, uint32_t value) {
  this->validate_lane_col(lane, col);
  data_.at(lane).at(col) = value;
}

///////////////////////////////////////////////////////////////////////////////
// Bank arbitration timing model

namespace {

// Set one requester class's slice of a bank's request vector. `base` is the
// class's pool offset, `suppress` drops a requester that already holds its
// grant this cycle.
void add_class(BitVector<>& req_vec, uint32_t bank, uint32_t base,
               const std::array<TcuTmem::BankReq, TcuTmem::kBanks>& reqs,
               const std::bitset<TcuTmem::kBanks>& suppress) {
  for (uint32_t bi = 0; bi < TcuTmem::kBanks; ++bi) {
    if (reqs.at(bi).valid && 
        TcuTmem::bank_of(reqs.at(bi).lane_base) == bank &&
        !suppress.test(bi)) {
      req_vec.set(base + bi);
    }
  }
}

// req_vec = {a_fill, ldst, compute}: compute at bi, ldst at kBanks + bi,
// a_fill at 2*kBanks + bi.
BitVector<> build_rd_req_vec(uint32_t bank,
                             const TcuTmem::CycleReqs& reqs,
                             const std::bitset<TcuTmem::kBanks>& ldst_suppress,
                             const std::bitset<TcuTmem::kBanks>& a_fill_suppress) {
  static const std::bitset<TcuTmem::kBanks> kNone;
  BitVector<> req_vec(TcuTmem::kArbRdW);
  add_class(req_vec, bank, 0, reqs.compute_rd, kNone);
  add_class(req_vec, bank, TcuTmem::kBanks, reqs.ldst_rd, ldst_suppress);
  add_class(req_vec, bank, 2 * TcuTmem::kBanks, reqs.a_fill_rd, a_fill_suppress);
  return req_vec;
}

// req_vec = {ldst, compute}.
BitVector<> build_wr_req_vec(uint32_t bank,
                             const TcuTmem::CycleReqs& reqs,
                             const std::bitset<TcuTmem::kBanks>& ldst_suppress) {
  static const std::bitset<TcuTmem::kBanks> kNone;
  BitVector<> req_vec(TcuTmem::kArbWrW);
  add_class(req_vec, bank, 0, reqs.compute_wr, kNone);
  add_class(req_vec, bank, TcuTmem::kBanks, reqs.ldst_wr, ldst_suppress);
  return req_vec;
}

} // namespace

void TcuTmem::arb_publish(const CycleReqs& reqs) {
  // Routed through this cycle's bank decode. A read grant is not qualified by
  // rd_valid, so a requester that posted nothing can observe a stale grant and
  // must check its own validity. The write grant is valid-qualified.
  grants_.compute_rd.reset();
  grants_.ldst_rd.reset();
  grants_.a_fill_rd.reset();
  for (uint32_t bi = 0; bi < kBanks; ++bi) {
    if (rd_grant_onehot_d_.at(bank_of(reqs.compute_rd.at(bi).lane_base)).test(bi)) {
      grants_.compute_rd.set(bi);
    }
    if (rd_grant_onehot_d_.at(bank_of(reqs.ldst_rd.at(bi).lane_base)).test(kBanks + bi)) {
      grants_.ldst_rd.set(bi);
    }
    if (rd_grant_onehot_d_.at(bank_of(reqs.a_fill_rd.at(bi).lane_base)).test(2 * kBanks + bi)) {
      grants_.a_fill_rd.set(bi);
    }
  }
}

void TcuTmem::arb_arbitrate(const CycleReqs& reqs) {
  // A TMEM_LD that already has its registered grant has its data, so it drops
  // its request.
  std::array<std::bitset<kArbRdW>, kBanks> rd_grant_onehot{};
  for (uint32_t r = 0; r < kBanks; ++r) {
    auto req_vec = build_rd_req_vec(r, reqs, grants_.ldst_rd, grants_.a_fill_rd);
    if (req_vec.count() > 1) {
      ++perf_stats_.bank_stalls;
    }
    uint32_t winner = rd_arb_.at(r).grant(req_vec);
    if (winner != -1u) {
      rd_grant_onehot.at(r).set(winner);
    }
  }

  grants_.compute_wr.reset();
  grants_.ldst_wr.reset();
  for (uint32_t r = 0; r < kBanks; ++r) {
    auto req_vec = build_wr_req_vec(r, reqs, ldst_wr_won_);
    if (req_vec.count() > 1) {
      ++perf_stats_.bank_stalls;
    }
    uint32_t winner = wr_arb_.at(r).grant(req_vec);
    if (winner == -1u)
      continue;
    // Writes are consumed the cycle they are granted, so route immediately.
    if (winner < kBanks) {
      grants_.compute_wr.set(winner);
    } else {
      uint32_t bi = winner - kBanks;
      grants_.ldst_wr.set(bi);
      ldst_wr_won_.set(bi);
    }
  }

  rd_grant_onehot_d_ = rd_grant_onehot;
}

///////////////////////////////////////////////////////////////////////////////

void TcuTmemWrTrack::reset() {
  entries_.fill(Entry{});
  head_ = 0;
  tail_ = 0;
  size_ = 0;
}

bool TcuTmemWrTrack::hazard(uint32_t lane_base, uint32_t col_base) const {
  for (auto& e : entries_) {
    if (e.valid && e.is_umma && e.lane_base == lane_base && e.col_base == col_base) {
      return true;
    }
  }
  return false;
}

void TcuTmemWrTrack::push(bool is_umma, uint32_t lane_base, uint32_t col_base,
                          uint64_t retire_cycle) {
  if (size_ >= kDepth) {
    std::cout << "Error: TMEM write tracker overflow (depth " << kDepth << ")" << std::endl;
    std::abort();
  }
  entries_.at(tail_) = Entry{true, is_umma, lane_base, col_base, retire_cycle};
  tail_ = (tail_ + 1) % kDepth;
  ++size_;
}

void TcuTmemWrTrack::retire(uint64_t cycle) {
  // In-order. An entry behind an unretired one cannot be released early.
  while (size_ != 0 && entries_.at(head_).retire_cycle <= cycle) {
    entries_.at(head_).valid = false;
    head_ = (head_ + 1) % kDepth;
    --size_;
  }
}

#endif // VX_CFG_TCU_TMEM_ENABLE
