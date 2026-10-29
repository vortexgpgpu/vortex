// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TcuTmem bank-arbitration model tests.
//
// Bank conflicts are structurally unreachable through the normal compute path,
// because a block's bank index is its warp's cta_rank, so the four blocks of a
// warpgroup always target four different banks. These tests drive
// the arbiter directly so the contended cases actually execute.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "tcu_tmem.h"

using namespace vortex;

namespace {

uint32_t g_checks = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      std::cerr << "FAILED: " << __FILE__ << ":" << __LINE__ << ": " << #cond   \
                << std::endl;                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

constexpr uint32_t kBanks = TcuTmem::kBanks;
constexpr uint32_t kBankLanes = TcuTmem::kBankLanes;
constexpr uint32_t kWordCols = TcuTmem::kWordCols;

// A compute request from block `b` against its own bank, as the UMMA path
// produces it: lane_base = cta_rank * TCU_WG_TILE_M.
TcuTmem::BankReq own_bank_req(uint32_t b, uint32_t col_base = 0) {
  return TcuTmem::BankReq{true, b * kBankLanes, col_base};
}

// A request aimed at `bank` regardless of which block issues it. This is the
// TMEM_LD/ST case: its lane_base comes from a software-computed address, so
// nothing ties it to the issuing block's own bank.
TcuTmem::BankReq foreign_bank_req(uint32_t bank, uint32_t col_base = 0) {
  return TcuTmem::BankReq{true, bank * kBankLanes, col_base};
}

void step(TcuTmem& tmem, const TcuTmem::CycleReqs& reqs) {
  tmem.arb_publish(reqs);
  tmem.arb_arbitrate(reqs);
}

///////////////////////////////////////////////////////////////////////////////

void test_geometry() {
  // rd_bank = lane_base[LANE_BITS-1:ROW_IDX_W], i.e. lane_base / TCU_WG_TILE_M
  // (VX_tcu_tmem.sv g_compute_decode).
  for (uint32_t b = 0; b < kBanks; ++b) {
    for (uint32_t row = 0; row < kBankLanes; ++row) {
      CHECK(TcuTmem::bank_of(b * kBankLanes + row) == b);
    }
  }
  // rd_word = col_base[COL_BITS-1:COL_SEL_W], i.e. col_base / TCU_TC_N.
  CHECK(TcuTmem::word_of(0) == 0);
  CHECK(TcuTmem::word_of(kWordCols - 1) == 0);
  CHECK(TcuTmem::word_of(kWordCols) == 1);
  CHECK(TcuTmem::word_of(3 * kWordCols + 1) == 3);

  // Every block of a warpgroup decodes to a distinct bank.
  for (uint32_t a = 0; a < kBanks; ++a) {
    for (uint32_t b = a + 1; b < kBanks; ++b) {
      CHECK(TcuTmem::bank_of(a * kBankLanes) != TcuTmem::bank_of(b * kBankLanes));
    }
  }
}

// Every block reads its own bank. No conflicts, and every block is granted.
void test_no_conflict_across_banks() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  for (uint32_t b = 0; b < kBanks; ++b) {
    reqs.compute_rd.at(b) = own_bank_req(b);
  }

  step(tmem, reqs);
  CHECK(tmem.perf_stats().bank_stalls == 0);
  // Read grants are registered, so nothing is visible on the granting step.
  CHECK(tmem.grants().compute_rd.none());

  step(tmem, reqs);
  CHECK(tmem.perf_stats().bank_stalls == 0);
  for (uint32_t b = 0; b < kBanks; ++b) {
    CHECK(tmem.grants().compute_rd.test(b));
  }
}

// bank_rd_grant_onehot_d: the bank's rdata reflects the address presented one
// cycle earlier, so the grant is delayed to match (VX_tcu_tmem.sv
// bank_rd_grant_onehot_d). Writes are unaffected by OUT_REG and are granted in
// the same cycle (g_wr_route).
void test_read_grant_is_registered_write_is_not() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.compute_rd.at(0) = own_bank_req(0);
  reqs.compute_wr.at(0) = own_bank_req(0);

  step(tmem, reqs);
  CHECK(!tmem.grants().compute_rd.test(0));  // one cycle late
  CHECK(tmem.grants().compute_wr.test(0));   // same cycle

  step(tmem, reqs);
  CHECK(tmem.grants().compute_rd.test(0));
}

// Two requesters on one bank: exactly one stall counted, one winner, and loser
// wins next cycle.
void test_read_conflict_and_rotation() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.compute_rd.at(0) = own_bank_req(0);       // pool index 0
  reqs.ldst_rd.at(0) = foreign_bank_req(0);      // pool index kBanks

  // Cycle 0: both bid. One stall, one winner.
  step(tmem, reqs);
  CHECK(tmem.perf_stats().bank_stalls == 1);

  // Cycle 1: ldst's win surfaces and its request drops out of the pool. Compute
  // bids alone and no second conflict is counted.
  step(tmem, reqs);
  CHECK(tmem.grants().ldst_rd.test(0));
  CHECK(!tmem.grants().compute_rd.test(0));
  CHECK(tmem.perf_stats().bank_stalls == 1);

  // Cycle 2: compute's win surfaces. ldst's grant has aged out of the register
  // so it bids again.
  step(tmem, reqs);
  CHECK(tmem.grants().compute_rd.test(0));
  CHECK(tmem.perf_stats().bank_stalls == 2);
}

// bank_rd_conflict = $countones(req_vec) > 1 looks only at how many
// requesters hit the bank, never at whether they want the same word.
void test_conflict_ignores_address_equality() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.compute_rd.at(0) = own_bank_req(0, 0);
  reqs.ldst_rd.at(0) = foreign_bank_req(0, 0);  // same bank AND same word

  step(tmem, reqs);
  CHECK(tmem.perf_stats().bank_stalls == 1);
}

// Read and write pools are arbitrated independently, so a bank can serve one
// read and one write in the same cycle. Conflicts in both directions are
// counted separately: tmem_bank_stalls accumulates
// $countones(bank_rd_conflict) + $countones(bank_wr_conflict) per cycle.
void test_read_and_write_pools_are_independent() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.compute_rd.at(0) = own_bank_req(0);
  reqs.compute_wr.at(0) = own_bank_req(0);
  step(tmem, reqs);
  CHECK(tmem.perf_stats().bank_stalls == 0);  // one read + one write, no contention

  TcuTmem tmem2;
  TcuTmem::CycleReqs both;
  both.compute_rd.at(0) = own_bank_req(0);
  both.ldst_rd.at(0) = foreign_bank_req(0);
  both.compute_wr.at(0) = own_bank_req(0);
  both.ldst_wr.at(0) = foreign_bank_req(0);
  step(tmem2, both);
  CHECK(tmem2.perf_stats().bank_stalls == 2);  // one read conflict + one write
}

// ldst_wr_won_r: a granted TMEM_ST whose result cannot retire keeps
// mgmt_valid asserted. Without the latch the arbiter would regrant it every
// cycle.
void test_tmem_st_sticky_win() {
  TcuTmem tmem;
  TcuTmem::CycleReqs st_only;
  st_only.ldst_wr.at(0) = foreign_bank_req(0);

  step(tmem, st_only);
  CHECK(tmem.grants().ldst_wr.test(0));
  CHECK(tmem.won_ldst_wr(0));
  CHECK(tmem.perf_stats().bank_stalls == 0);

  // The store has not retired, so it is still asserting. A compute write now
  // bids for the same bank: the latch must keep the store out of the pool.
  TcuTmem::CycleReqs contended = st_only;
  contended.compute_wr.at(0) = own_bank_req(0);

  step(tmem, contended);
  CHECK(tmem.perf_stats().bank_stalls == 0);   // suppressed, so no conflict
  CHECK(!tmem.grants().ldst_wr.test(0));       // not regranted
  CHECK(tmem.grants().compute_wr.test(0));     // compute takes the bank
  CHECK(tmem.won_ldst_wr(0));                  // latch still held

  // Retiring the op clears the latch and the store competes again.
  tmem.clear_ldst_wr_win(0);
  CHECK(!tmem.won_ldst_wr(0));
  step(tmem, contended);
  CHECK(tmem.perf_stats().bank_stalls == 1);
}

// A write grant is qualified by wr_valid (g_wr_route), so a block that is not
// requesting can never be told it won the write port.
void test_write_grant_requires_valid() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;  // nothing valid
  step(tmem, reqs);
  CHECK(tmem.grants().compute_wr.none());
  CHECK(tmem.grants().ldst_wr.none());
  CHECK(tmem.perf_stats().bank_stalls == 0);
}

// reset() must return the arbiters to their post-reset rotation.
void test_reset_restores_rotation() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.compute_rd.at(0) = own_bank_req(0);
  reqs.ldst_rd.at(0) = foreign_bank_req(0);

  step(tmem, reqs);   // ldst wins, rotation advances
  step(tmem, reqs);   // compute wins
  CHECK(tmem.perf_stats().bank_stalls == 1);

  tmem.reset();
  CHECK(tmem.perf_stats().bank_stalls == 0);

  // Post-reset the same requester wins first again
  step(tmem, reqs);
  step(tmem, reqs);
  CHECK(tmem.grants().ldst_rd.test(0));
  CHECK(!tmem.grants().compute_rd.test(0));
}

///////////////////////////////////////////////////////////////////////////////

void test_allocator() {
  TcuTmem tmem;

  // CTA-scoped idempotency: sibling warps of one CTA get the same handle.
  uint32_t h0 = tmem.alloc(32, /*cta_id*/ 0);
  CHECK(tmem.alloc(32, 0) == h0);
  CHECK(tmem.alloc_ncols(h0) == 32);

  // A different CTA gets a disjoint range.
  uint32_t h1 = tmem.alloc(32, 1);
  CHECK(h1 != h0);
  CHECK(h1 >= h0 + 32 || h0 >= h1 + 32);

  // Not-live handles report zero width rather than aborting, which is what
  // UMMA's per-uop range check relies on.
  CHECK(tmem.alloc_ncols(h0 + 1) == 0);

  // A range is only freed once every warp of the CTA has released it.
  tmem.dealloc(h0, 0, /*wid*/ 0, /*cta_size*/ 2);
  CHECK(tmem.alloc_ncols(h0) == 32);  // still held by the sibling warp
  tmem.dealloc(h0, 0, /*wid*/ 0, /*cta_size*/ 2);
  CHECK(tmem.alloc_ncols(h0) == 32);  // same warp twice must not count twice
  tmem.dealloc(h0, 0, /*wid*/ 1, /*cta_size*/ 2);
  CHECK(tmem.alloc_ncols(h0) == 0);

  tmem.dealloc(h1, 1, 0, 1);

  // Coalescing: after freeing everything, the whole array must be allocatable
  // as one range again.
  uint32_t whole = tmem.alloc(TcuTmem::kCols, 2);
  CHECK(tmem.alloc_ncols(whole) == TcuTmem::kCols);
  tmem.dealloc(whole, 2, 0, 1);
}

// One ALLOC/DEALLOC is granted per cycle across all blocks, DEALLOC ahead of
// ALLOC, both classes pre-masked by result_ready.
void test_mgmt_arbitration() {
  std::bitset<kBanks> none;

  CHECK(TcuTmem::mgmt_grant(none, none) == TcuTmem::kNoGrant);

  // Lowest requesting block wins within a class.
  std::bitset<kBanks> allocs;
  for (uint32_t b = 0; b < kBanks; ++b) allocs.set(b);
  CHECK(TcuTmem::mgmt_grant(allocs, none) == 0);

  std::bitset<kBanks> alloc_hi;
  alloc_hi.set(kBanks - 1);
  CHECK(TcuTmem::mgmt_grant(alloc_hi, none) == kBanks - 1);

  // DEALLOC outranks ALLOC even when the ALLOC is on a lower block.
  if (kBanks > 1) {
    std::bitset<kBanks> alloc0, dealloc_hi;
    alloc0.set(0);
    dealloc_hi.set(kBanks - 1);
    CHECK(TcuTmem::mgmt_grant(alloc0, dealloc_hi) == kBanks - 1);
  }

  // A block masked out by result_ready simply is not in the input, so the
  // arbiter falls through to the next one that is.
  if (kBanks > 1) {
    std::bitset<kBanks> alloc_not_block0;
    for (uint32_t b = 1; b < kBanks; ++b) alloc_not_block0.set(b);
    CHECK(TcuTmem::mgmt_grant(alloc_not_block0, none) == 1);
  }
}

// A fresh ALLOC that has no large-enough free range or no free CAM slot stalls
// and retries.
void test_alloc_backpressure() {
  TcuTmem tmem;

  // Fill the CAM.
  uint32_t per_cta = TcuTmem::kCols / (TcuTmem::kAllocEntries + 1);
  CHECK(per_cta > 0);
  for (uint32_t i = 0; i < TcuTmem::kAllocEntries; ++i) {
    CHECK(!tmem.alloc_would_stall(per_cta, (int32_t)i));
    tmem.alloc(per_cta, (int32_t)i);
  }

  // One more CTA has nowhere to go, even though columns remain free.
  CHECK(tmem.alloc_would_stall(per_cta, (int32_t)TcuTmem::kAllocEntries));
  // A repeat request from a CTA already holding an allocation still succeeds,
  // doesn't need new CAM slot.
  CHECK(!tmem.alloc_would_stall(per_cta, 0));
  CHECK(tmem.alloc(per_cta, 0) == tmem.alloc(per_cta, 0));

  // Freeing one slot reopens the door.
  tmem.dealloc(tmem.alloc(per_cta, 0), 0, 0, 1);
  CHECK(!tmem.alloc_would_stall(per_cta, (int32_t)TcuTmem::kAllocEntries));

  // A fresh CTA with a free slot but no range wide enough also stalls.
  TcuTmem tight;
  tight.alloc(TcuTmem::kCols, 0);
  CHECK(tight.alloc_would_stall(1, 1));
}

///////////////////////////////////////////////////////////////////////////////
// RAW interlock

void test_interlock_basics() {
  TcuTmemWrTrack track;

  // Nothing in flight, nothing blocked.
  CHECK(!track.hazard(0, 0));
  CHECK(track.size() == 0);

  // A UMMA uop in flight blocks a later uop addressing the same tile origin.
  track.push(/*is_umma*/ true, /*lane_base*/ 8, /*col_base*/ 4, /*retire*/ 10);
  CHECK(track.size() == 1);
  CHECK(track.hazard(8, 4));
  CHECK(!track.hazard(8, 5));
  CHECK(!track.hazard(0, 4));
  CHECK(!track.hazard(9, 4));

  // The entry is released once its retirement cycle arrives.
  track.retire(9);
  CHECK(track.hazard(8, 4));
  track.retire(10);
  CHECK(!track.hazard(8, 4));
  CHECK(track.size() == 0);
}

// Entries are taken by every non-setup FEDP uop, but only those that owe TMEM
// a write can collide.
void test_interlock_ignores_non_umma() {
  TcuTmemWrTrack track;
  track.push(/*is_umma*/ false, 8, 4, 10);
  CHECK(track.size() == 1);   // occupies depth
  CHECK(!track.hazard(8, 4)); // but cannot cause a hazard
}

// Release is in order. an entry behind an unretired one cannot leave early.
void test_interlock_retires_in_order() {
  TcuTmemWrTrack track;
  track.push(true, 0, 0, /*retire*/ 20);  // retires late
  track.push(true, 8, 0, /*retire*/ 10);  // would retire sooner

  track.retire(15);
  CHECK(track.size() == 2);     // head is not ready, so neither leaves
  CHECK(track.hazard(0, 0));
  CHECK(track.hazard(8, 0));

  track.retire(20);
  CHECK(track.size() == 0);     // head clears, and the one behind follows
  CHECK(!track.hazard(0, 0));
  CHECK(!track.hazard(8, 0));
}

// The tracker is a circular buffer LANDQ_SIZE deep. Filling and draining it
// repeatedly must not leave a stale entry behind.
void test_interlock_wraps() {
  TcuTmemWrTrack track;
  CHECK(TcuTmemWrTrack::kDepth >= TcuTmemWrTrack::kPipeLatency);

  for (uint32_t round = 0; round < 3; ++round) {
    for (uint32_t i = 0; i < TcuTmemWrTrack::kDepth; ++i) {
      track.push(true, i * kBankLanes, /*col_base*/ 0, /*retire*/ 100 + round);
    }
    CHECK(track.size() == TcuTmemWrTrack::kDepth);
    for (uint32_t i = 0; i < TcuTmemWrTrack::kDepth; ++i) {
      CHECK(track.hazard(i * kBankLanes, 0));
    }
    track.retire(100 + round);
    CHECK(track.size() == 0);
    for (uint32_t i = 0; i < TcuTmemWrTrack::kDepth; ++i) {
      CHECK(!track.hazard(i * kBankLanes, 0));
    }
  }
}

void test_interlock_reset() {
  TcuTmemWrTrack track;
  track.push(true, 8, 4, 10);
  CHECK(track.hazard(8, 4));
  track.reset();
  CHECK(track.size() == 0);
  CHECK(!track.hazard(8, 4));
}

void test_storage_roundtrip() {
  TcuTmem tmem;
  uint32_t h = tmem.alloc(kWordCols * 2, 0);
  for (uint32_t lane = 0; lane < TcuTmem::kLanes; ++lane) {
    tmem.write(lane, h, 0xA5000000u | lane);
  }
  for (uint32_t lane = 0; lane < TcuTmem::kLanes; ++lane) {
    CHECK(tmem.read(lane, h) == (0xA5000000u | lane));
  }
  // alloc() zeroes the range it hands out, so a fresh column reads back 0.
  CHECK(tmem.read(0, h + 1) == 0);
  tmem.dealloc(h, 0, 0, 1);
}

} // namespace

// A-from-TMEM adds a third read requester on the same bank read port. It must
// arbitrate against compute and ldst, and drop its request once its registered
// grant lands (it has its word).
void test_a_fill_is_a_third_read_requester() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.a_fill_rd.at(0) = own_bank_req(0);

  // Uncontended: registered grant, so one cycle late, exactly like compute.
  step(tmem, reqs);
  CHECK(!tmem.grants().a_fill_rd.test(0));
  CHECK(tmem.perf_stats().bank_stalls == 0);
  step(tmem, reqs);
  CHECK(tmem.grants().a_fill_rd.test(0));
}

// Three requesters on one bank is a 2-count conflict then a 1-count conflict,
// and all three eventually win -- nothing is starved by the round-robin.
void test_a_fill_contends_with_compute_and_ldst() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.compute_rd.at(0) = own_bank_req(0);       // pool 0
  reqs.ldst_rd.at(0)    = foreign_bank_req(0);   // pool kBanks
  reqs.a_fill_rd.at(0)  = own_bank_req(0);       // pool 2*kBanks

  uint32_t compute_wins = 0, ldst_wins = 0, a_fill_wins = 0;
  for (uint32_t c = 0; c < 8; ++c) {
    step(tmem, reqs);
    if (tmem.grants().compute_rd.test(0)) ++compute_wins;
    if (tmem.grants().ldst_rd.test(0))    ++ldst_wins;
    if (tmem.grants().a_fill_rd.test(0))  ++a_fill_wins;
  }
  // Every class gets served; none is locked out by pool position.
  CHECK(compute_wins > 0);
  CHECK(ldst_wins > 0);
  CHECK(a_fill_wins > 0);
  // One grant per cycle per bank, so wins cannot exceed the cycles stepped.
  CHECK(compute_wins + ldst_wins + a_fill_wins <= 8);
}

// A-fill is bank-local: its lane_base is cta_rank*xtileM, so block b always
// targets bank b and distinct blocks never collide -- the same structural
// property the compute read has.
void test_a_fill_across_banks_never_conflicts() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  for (uint32_t b = 0; b < kBanks; ++b) {
    reqs.a_fill_rd.at(b) = own_bank_req(b);
  }
  step(tmem, reqs);
  step(tmem, reqs);
  for (uint32_t b = 0; b < kBanks; ++b) {
    CHECK(tmem.grants().a_fill_rd.test(b));
  }
  CHECK(tmem.perf_stats().bank_stalls == 0);
}

// The write pool is unchanged by A-fill: it has no write requester, so a
// compute write and a TMEM_ST on one bank still conflict exactly once.
void test_a_fill_does_not_touch_the_write_pool() {
  TcuTmem tmem;
  TcuTmem::CycleReqs reqs;
  reqs.a_fill_rd.at(0)  = own_bank_req(0);
  reqs.compute_wr.at(0) = own_bank_req(0);
  reqs.ldst_wr.at(0)    = foreign_bank_req(0);

  step(tmem, reqs);
  // One write conflict. The read side has a lone a_fill bid, so no read stall.
  CHECK(tmem.perf_stats().bank_stalls == 1);
}

int main() {
  std::cout << "TcuTmem: kBanks=" << kBanks
            << ", kBankLanes=" << kBankLanes
            << ", kWordCols=" << kWordCols
            << ", kLanes=" << TcuTmem::kLanes
            << ", kCols=" << TcuTmem::kCols
            << ", kArbRdW=" << TcuTmem::kArbRdW
            << ", kArbWrW=" << TcuTmem::kArbWrW
            << ", wr_track depth=" << TcuTmemWrTrack::kDepth << std::endl;

  test_geometry();
  test_no_conflict_across_banks();
  test_read_grant_is_registered_write_is_not();
  test_read_conflict_and_rotation();
  test_a_fill_is_a_third_read_requester();
  test_a_fill_contends_with_compute_and_ldst();
  test_a_fill_across_banks_never_conflicts();
  test_a_fill_does_not_touch_the_write_pool();
  test_conflict_ignores_address_equality();
  test_read_and_write_pools_are_independent();
  test_tmem_st_sticky_win();
  test_write_grant_requires_valid();
  test_reset_restores_rotation();
  test_allocator();
  test_mgmt_arbitration();
  test_alloc_backpressure();
  test_interlock_basics();
  test_interlock_ignores_non_umma();
  test_interlock_retires_in_order();
  test_interlock_wraps();
  test_interlock_reset();
  test_storage_roundtrip();

  std::cout << g_checks << " checks PASSED!" << std::endl;
  return 0;
}
