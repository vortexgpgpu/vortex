// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <VX_config.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vortex {

class DxaGroupTracker {
public:
  enum class IssueResult { Tracked, Backpressured, OwnerClosed };
  struct OperationToken { uint32_t group_id = 0; };

  explicit DxaGroupTracker(uint32_t num_warps = VX_CFG_NUM_WARPS,
                           uint32_t depth = VX_CFG_DXA_GROUP_DEPTH,
                           uint32_t max_pending = VX_CFG_DXA_GROUP_MAX_PENDING)
      : issuers_(num_warps), depth_(depth), max_pending_(max_pending) {
    if (!num_warps || depth < 2 || depth > 32 || (depth & (depth - 1)) || !max_pending) {
      throw std::invalid_argument("invalid DXA source-group dimensions");
    }
    reset();
  }

  void reset() {
    for (auto& issuer : issuers_) {
      issuer = Issuer{};
      issuer.pending.resize(depth_);
    }
    groups_committed_ = source_consumed_ = ring_full_stalls_ = counter_full_stalls_ = 0;
    max_committed_depth_ = 0;
  }

  void tick() {
    for (auto& issuer : issuers_) {
      if (issuer.head != issuer.tail && issuer.pending[slot(issuer.head)] == 0) {
        issuer.head = advance(issuer.head);
      }
    }
  }

  IssueResult issue(uint32_t wid, OperationToken* token) {
    auto& issuer = issuers_.at(wid);
    if (issuer.closed) {
      return IssueResult::OwnerClosed;
    }
    if (!issuer.open && committed_depth(wid) == depth_) {
      ++ring_full_stalls_;
      return IssueResult::Backpressured;
    }
    auto& pending = issuer.pending[slot(issuer.tail)];
    if (pending == max_pending_) {
      ++counter_full_stalls_;
      return IssueResult::Backpressured;
    }
    issuer.open = true;
    ++pending;
    if (token) {
      token->group_id = slot(issuer.tail);
    }
    return IssueResult::Tracked;
  }

  void commit(uint32_t wid) {
    auto& issuer = issuers_.at(wid);
    if (issuer.closed || !issuer.open) {
      return;
    }
    // First issue reserved this row, so commit never allocates another row.
    if (committed_depth(wid) >= depth_) {
      throw std::logic_error("DXA open group lost its reserved row");
    }
    issuer.open = false;
    issuer.tail = advance(issuer.tail);
    ++groups_committed_;
    max_committed_depth_ = std::max(max_committed_depth_, committed_depth(wid));
  }

  void complete(uint32_t wid, uint32_t group_id) {
    auto& issuer = issuers_.at(wid);
    const uint32_t distance = (group_id - slot(issuer.head)) & (depth_ - 1);
    const bool committed = distance < committed_depth(wid);
    const bool open = issuer.open && group_id == slot(issuer.tail);
    // The transport owes exactly one completion per accepted operation.
    if (group_id >= depth_ || (!committed && !open) || issuer.pending[group_id] == 0) {
      throw std::logic_error("DXA source completion outside a live nonzero group");
    }
    --issuer.pending[group_id];
    ++source_consumed_;
  }

  bool wait_satisfied(uint32_t wid, uint32_t n) const {
    return committed_depth(wid) <= n;
  }

  void close_owner(uint32_t wid) { issuers_.at(wid).closed = true; }

  bool reinitialize_owner(uint32_t wid) {
    if (!drained(wid)) {
      return false;
    }
    auto& issuer = issuers_.at(wid);
    issuer = Issuer{};
    issuer.pending.resize(depth_);
    return true;
  }

  bool drained(uint32_t wid) const {
    const auto& issuer = issuers_.at(wid);
    return issuer.head == issuer.tail && (!issuer.open || issuer.pending[slot(issuer.tail)] == 0);
  }

  uint32_t num_warps() const { return issuers_.size(); }
  bool open_valid(uint32_t wid) const { return issuers_.at(wid).open; }
  uint32_t open_gid(uint32_t wid) const { return slot(issuers_.at(wid).tail); }
  uint32_t open_remaining(uint32_t wid) const {
    const auto& issuer = issuers_.at(wid);
    return issuer.open ? issuer.pending[slot(issuer.tail)] : 0;
  }
  uint32_t remaining(uint32_t wid, uint32_t gid) const { return issuers_.at(wid).pending.at(gid); }
  uint32_t read_head(uint32_t wid) const { return issuers_.at(wid).head; }
  uint32_t sealed_tail(uint32_t wid) const { return issuers_.at(wid).tail; }
  uint32_t committed_depth(uint32_t wid) const {
    const auto& issuer = issuers_.at(wid);
    return (issuer.tail - issuer.head) & (2 * depth_ - 1);
  }
  uint64_t groups_committed() const { return groups_committed_; }
  uint64_t source_consumed() const { return source_consumed_; }
  uint32_t max_committed_depth() const { return max_committed_depth_; }
  uint64_t ring_full_stalls() const { return ring_full_stalls_; }
  uint64_t counter_full_stalls() const { return counter_full_stalls_; }

private:
  struct Issuer {
    uint32_t head = 0;
    uint32_t tail = 0;
    bool open = false;
    bool closed = false;
    std::vector<uint32_t> pending;
  };

  uint32_t slot(uint32_t pointer) const { return pointer & (depth_ - 1); }
  uint32_t advance(uint32_t pointer) const { return (pointer + 1) & (2 * depth_ - 1); }

  std::vector<Issuer> issuers_;
  uint32_t depth_;
  uint32_t max_pending_;
  uint64_t groups_committed_ = 0;
  uint64_t source_consumed_ = 0;
  uint32_t max_committed_depth_ = 0;
  uint64_t ring_full_stalls_ = 0;
  uint64_t counter_full_stalls_ = 0;
};

} // namespace vortex
