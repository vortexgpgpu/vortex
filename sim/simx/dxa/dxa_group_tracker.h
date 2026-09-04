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

#pragma once

#include <VX_config.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vortex {

// Functional mirror of VX_dxa_group_tracker's SOURCE_CONSUMED state.
//
// The implementation deliberately has no global free list. Operation
// contexts are direct-indexed physical slots statically partitioned by warp;
// the same physical array can of course be implemented as one SRAM addressed
// by {wid, local_slot}. A warp owns one open group and an ordered ring of
// sealed boundaries. A boundary stores only the number of source events still
// pending, never a bitmap of historical operations.
class DxaGroupTracker {
public:
  static constexpr uint32_t kSeqBits = 8;
  static constexpr uint32_t kEpochBits = 2;

  enum class IssueResult {
    Tracked,
    Backpressured,
    IgnoredPoisoned,
  };

  enum class CommitResult {
    Accepted,
    Backpressured,
    IgnoredPoisoned,
  };

  enum class CompletionResult {
    Applied,
    Stale,
    Duplicate,
    Invalid,
  };

  struct OperationToken {
    uint32_t epoch = 0;
    uint8_t group_seq = 0;
    // op_id is {generation, physical_context_slot}. The physical slot is
    // global only for transport; allocation is restricted to this warp's
    // statically assigned partition.
    uint32_t op_id = 0;
  };

  explicit DxaGroupTracker(
      uint32_t num_warps = VX_CFG_NUM_WARPS,
      uint32_t num_contexts = VX_CFG_DXA_GROUP_CONTEXTS,
      uint32_t ring_depth = VX_CFG_DXA_GROUP_DEPTH,
      uint32_t generation_bits = VX_CFG_DXA_GROUP_CTX_GEN_BITS)
      : issuers_(num_warps)
      , contexts_(num_contexts)
      , ring_depth_(ring_depth)
      , generation_bits_(generation_bits)
      , contexts_per_warp_(num_warps ? num_contexts / num_warps : 0) {
    if (num_warps == 0)
      throw std::invalid_argument("DXA group tracker needs at least one warp");
    if (num_contexts == 0 || !is_power_of_two(num_contexts))
      throw std::invalid_argument("DXA group contexts must be a power of two");
    if (num_contexts % num_warps != 0 || contexts_per_warp_ == 0
        || !is_power_of_two(contexts_per_warp_)) {
      throw std::invalid_argument(
          "DXA group contexts must be evenly power-of-two partitioned per warp");
    }
    if (ring_depth < 2 || !is_power_of_two(ring_depth)
        || ring_depth > 32 || ring_depth >= (1u << (kSeqBits - 1)))
      throw std::invalid_argument("DXA READ-group ring depth must be a power-of-two >= 2");
    if (generation_bits < 4 || generation_bits > 24)
      throw std::invalid_argument("DXA context generation must be 4..24 bits");

    while ((1u << context_index_bits_) < num_contexts)
      ++context_index_bits_;
    for (auto& issuer : issuers_)
      issuer.ring.resize(ring_depth_);
  }

  void reset() {
    for (auto& issuer : issuers_) {
      // A reset is a lifetime boundary, not a legal way to recycle the same
      // completion identity.  Keep the epoch moving across reset so a
      // delayed SOURCE_CONSUMED token from the previous run cannot mutate a
      // freshly allocated context.  (The RTL reset path also flushes the
      // transport; this guard makes the functional model robust when a test
      // deliberately injects a late response.)
      const uint32_t next_epoch =
          (issuer.epoch + 1u) & ((1u << kEpochBits) - 1u);
      issuer = Issuer{};
      issuer.epoch = next_epoch;
      issuer.ring.resize(ring_depth_);
    }
    for (auto& context : contexts_) {
      // Preserve a tombstone generation as an additional defense for the
      // (eventual) epoch wrap.  The next issue increments it again before
      // making the slot live.
      const uint32_t next_generation =
          (context.generation + 1u) & generation_mask();
      const uint32_t owner_wid = context.wid;
      const uint32_t owner_epoch = context.epoch;
      const uint8_t owner_seq = context.group_seq;
      context = Context{};
      context.seen = true;
      context.done = true;
      context.generation = next_generation;
      context.wid = owner_wid;
      context.epoch = owner_epoch;
      context.group_seq = owner_seq;
    }
    groups_committed_ = 0;
    source_consumed_ = 0;
    max_committed_depth_ = 0;
    ring_full_stalls_ = 0;
    context_stalls_ = 0;
  }

  // Retire at most one completed boundary per warp per tick. This mirrors
  // the bounded hardware retirement port and preserves FIFO ordering even
  // when operations complete out of order.
  void tick() {
    for (auto& issuer : issuers_) {
      if (issuer.read_head == issuer.sealed_tail)
        continue;
      auto& head = issuer.ring[ring_slot(issuer.read_head)];
      if (head.live && head.remaining == 0) {
        head = Boundary{};
        issuer.read_head = next_seq(issuer.read_head);
      }
    }
  }

  IssueResult issue(uint32_t wid, OperationToken* token) {
    auto& issuer = issuers_.at(wid);
    if (issuer.poisoned)
      return IssueResult::IgnoredPoisoned;

    // A new open group occupies the next boundary slot. Existing open groups
    // may continue accepting operations while older sealed groups drain; only
    // this warp is backpressured when its ring is full.
    if (!issuer.open_valid && committed_depth(wid) >= ring_depth_) {
      ++ring_full_stalls_;
      return IssueResult::Backpressured;
    }

    const uint32_t base = context_base(wid);
    uint32_t slot = base;
    while (slot < base + contexts_per_warp_ && contexts_[slot].live)
      ++slot;
    if (slot == base + contexts_per_warp_) {
      ++context_stalls_;
      return IssueResult::Backpressured;
    }

    auto& context = contexts_[slot];
    context.generation = (context.generation + 1u) & generation_mask();
    context.live = true;
    context.seen = true;
    context.done = false;
    context.wid = wid;
    context.epoch = issuer.epoch;
    context.group_seq = issuer.sealed_tail;
    if (!issuer.open_valid) {
      issuer.open_valid = true;
      issuer.open_slot = ring_slot(issuer.sealed_tail);
    }
    ++issuer.open_remaining;

    if (token) {
      token->epoch = issuer.epoch;
      token->group_seq = issuer.sealed_tail;
      token->op_id = (context.generation << context_index_bits_) | slot;
    }
    return IssueResult::Tracked;
  }

  CommitResult commit(uint32_t wid) {
    auto& issuer = issuers_.at(wid);
    if (issuer.poisoned)
      return CommitResult::IgnoredPoisoned;
    // An empty commit has no source lifetime to preserve.  If every physical
    // boundary row is occupied, accept it as a completed no-op rather than
    // making the warp wait for storage that this commit does not need.  A
    // non-empty open group still requires a boundary row and is backpressured.
    if (issuer.open_valid && committed_depth(wid) >= ring_depth_) {
      ++ring_full_stalls_;
      return CommitResult::Backpressured;
    }

    // There is no boundary state to retain for an empty commit.  Once the
    // committed ring is full, accepting this no-op without advancing the
    // sequence is the only way to preserve the "empty commit is trivially
    // complete" rule without overwriting the oldest live row.
    if (!issuer.open_valid && committed_depth(wid) >= ring_depth_) {
      ++groups_committed_;
      return CommitResult::Accepted;
    }

    auto& boundary = issuer.ring[ring_slot(issuer.sealed_tail)];
    if (boundary.live)
      throw std::logic_error("DXA group commit overwrote a live boundary");
    boundary.live = true;
    boundary.remaining = issuer.open_valid ? issuer.open_remaining : 0;
    issuer.open_remaining = 0;
    issuer.open_valid = false;
    issuer.open_slot = 0;
    issuer.sealed_tail = next_seq(issuer.sealed_tail);
    ++groups_committed_;
    const auto depth = committed_depth(wid);
    if (depth > max_committed_depth_)
      max_committed_depth_ = depth;
    return CommitResult::Accepted;
  }

  CompletionResult complete(uint32_t wid, uint32_t epoch,
                            uint8_t group_seq, uint32_t op_id) {
    auto& issuer = issuers_.at(wid);
    if (epoch != issuer.epoch)
      return CompletionResult::Stale;

    const uint32_t slot = op_id & context_index_mask();
    const uint32_t generation = (op_id >> context_index_bits_)
                              & generation_mask();
    const uint32_t base = context_base(wid);
    if (slot < base || slot >= base + contexts_per_warp_)
      return CompletionResult::Invalid;

    auto& context = contexts_[slot];
    if (context.seen && context.generation != generation)
      return CompletionResult::Stale;

    const bool owner_match = context.seen
                          && context.generation == generation
                          && context.wid == wid
                          && context.epoch == epoch
                          && context.group_seq == group_seq;
    if (owner_match && context.done)
      return CompletionResult::Duplicate;
    if (!owner_match || !context.live)
      return CompletionResult::Invalid;

    if (issuer.open_valid && group_seq == issuer.sealed_tail) {
      if (issuer.open_remaining == 0)
        return CompletionResult::Invalid;
      --issuer.open_remaining;
    } else {
      const uint32_t live_span = seq_distance(issuer.sealed_tail,
                                              issuer.read_head);
      const uint32_t group_span = seq_distance(group_seq, issuer.read_head);
      auto& boundary = issuer.ring[ring_slot(group_seq)];
      if (group_span >= live_span || !boundary.live
          || boundary.remaining == 0)
        return CompletionResult::Invalid;
      --boundary.remaining;
    }

    context.live = false;
    context.done = true;
    ++source_consumed_;
    return CompletionResult::Applied;
  }

  // A wait snapshots only committed groups at decode/accept time. Later
  // commits do not move an already-issued wait's target; an open group is not
  // part of this depth calculation.
  bool wait_satisfied(uint32_t wid, uint32_t n,
                      uint8_t snapshot) const {
    const auto& issuer = issuers_.at(wid);
    const uint32_t required = seq_distance(snapshot, issuer.read_head);
    return required > ring_depth_ || required <= n;
  }

  uint8_t wait_snapshot(uint32_t wid) const {
    return issuers_.at(wid).sealed_tail;
  }

  void poison(uint32_t wid) {
    issuers_.at(wid).poisoned = true;
  }

  bool advance_epoch(uint32_t wid) {
    auto& issuer = issuers_.at(wid);
    if (!drained(wid))
      return false;
    // `drained()` intentionally treats a source-complete, uncommitted open
    // group as drainable: there is no SMEM lifetime left to protect.  Its
    // identity must nevertheless be discarded before this physical warp slot
    // is assigned to a new CTA, otherwise the first issue in the new epoch
    // would be merged into the old open sequence.
    issuer.open_valid = false;
    issuer.open_slot = 0;
    issuer.open_remaining = 0;
    issuer.sealed_tail = 0;
    issuer.read_head = 0;
    for (auto& boundary : issuer.ring)
      boundary = Boundary{};
    issuer.epoch = (issuer.epoch + 1u) & ((1u << kEpochBits) - 1u);
    issuer.poisoned = false;
    return true;
  }

  bool drained(uint32_t wid) const {
    const auto& issuer = issuers_.at(wid);
    // A completed but uncommitted open group has no source lifetime left and
    // may be discarded when a warp/CTA is killed. Pending operations and
    // sealed boundaries still prevent lifetime advancement.
    if (issuer.read_head != issuer.sealed_tail
        || issuer.open_remaining != 0)
      return false;
    const uint32_t base = context_base(wid);
    for (uint32_t slot = base; slot < base + contexts_per_warp_; ++slot) {
      if (contexts_[slot].live)
        return false;
    }
    return true;
  }

  uint32_t open_remaining(uint32_t wid) const {
    return issuers_.at(wid).open_remaining;
  }

  uint32_t num_warps() const {
    return static_cast<uint32_t>(issuers_.size());
  }

  // Decode the transport portion of an operation token for diagnostics and
  // for the future destination-completion domain.  The tracker itself still
  // performs the owner/generation checks before mutating state.
  uint32_t op_slot(uint32_t op_id) const {
    return op_id & context_index_mask();
  }

  uint32_t op_generation(uint32_t op_id) const {
    return (op_id >> context_index_bits_) & generation_mask();
  }

  bool open_valid(uint32_t wid) const {
    return issuers_.at(wid).open_valid;
  }

  uint8_t sealed_tail(uint32_t wid) const {
    return issuers_.at(wid).sealed_tail;
  }

  uint8_t read_head(uint32_t wid) const {
    return issuers_.at(wid).read_head;
  }

  uint32_t committed_depth(uint32_t wid) const {
    const auto& issuer = issuers_.at(wid);
    return seq_distance(issuer.sealed_tail, issuer.read_head);
  }

  uint32_t epoch(uint32_t wid) const {
    return issuers_.at(wid).epoch;
  }

  uint64_t groups_committed() const { return groups_committed_; }
  uint64_t source_consumed() const { return source_consumed_; }
  uint32_t max_committed_depth() const { return max_committed_depth_; }
  uint64_t ring_full_stalls() const { return ring_full_stalls_; }
  uint64_t context_stalls() const { return context_stalls_; }

private:
  struct Boundary {
    bool live = false;
    uint32_t remaining = 0;
  };

  struct Issuer {
    uint8_t sealed_tail = 0;
    uint8_t read_head = 0;
    uint32_t open_remaining = 0;
    uint32_t open_slot = 0;
    uint32_t epoch = 0;
    bool open_valid = false;
    bool poisoned = false;
    std::vector<Boundary> ring;
  };

  struct Context {
    bool live = false;
    bool seen = false;
    bool done = false;
    uint32_t generation = 0;
    uint32_t wid = 0;
    uint32_t epoch = 0;
    uint8_t group_seq = 0;
  };

  static bool is_power_of_two(uint32_t value) {
    return value && ((value & (value - 1u)) == 0);
  }

  static uint8_t next_seq(uint8_t seq) {
    return static_cast<uint8_t>(seq + 1u);
  }

  static uint32_t seq_distance(uint8_t newer, uint8_t older) {
    return static_cast<uint8_t>(newer - older);
  }

  uint32_t ring_slot(uint8_t seq) const {
    return uint32_t(seq) & (ring_depth_ - 1u);
  }

  uint32_t context_base(uint32_t wid) const {
    return wid * contexts_per_warp_;
  }

  uint32_t context_index_mask() const {
    return contexts_.size() - 1u;
  }

  uint32_t generation_mask() const {
    return (1u << generation_bits_) - 1u;
  }

  std::vector<Issuer> issuers_;
  std::vector<Context> contexts_;
  uint32_t ring_depth_;
  uint32_t generation_bits_;
  uint32_t contexts_per_warp_;
  uint32_t context_index_bits_ = 0;
  uint64_t groups_committed_ = 0;
  uint64_t source_consumed_ = 0;
  uint32_t max_committed_depth_ = 0;
  uint64_t ring_full_stalls_ = 0;
  uint64_t context_stalls_ = 0;
};

} // namespace vortex
