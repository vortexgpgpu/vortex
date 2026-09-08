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

#include <array>
#include <cstdlib>
#include <iostream>

#include "dxa_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
DxaUnit::DxaUnit(Core* core, SimChannel<DxaReq>& req_out,
                 SimChannel<DxaReadCompletion>& completion_in)
  : core_(core)
  , req_out_(req_out)
  , completion_in_(completion_in)
  , tracker_(VX_CFG_NUM_WARPS)
  , parked_waits_(VX_CFG_NUM_WARPS)
{}
#endif

void DxaUnit::reset() {
#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
  // The tracker and parked waits are helper state (DxaUnit itself is not a
  // SimObject), so they do not get reset by SimPlatform automatically.  A
  // parked trace belongs to the previous instruction pool/lifetime and must
  // be discarded before the scheduler starts dispatching the next image.
  tracker_.reset();
  for (auto& wait : parked_waits_)
    wait = ParkedWait{};
#endif
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  // Reset also flushes the worker transport; no pre-reset event may survive.
  while (!completion_in_.empty())
    completion_in_.pop();
#endif
}

void DxaUnit::tick() {
#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
  std::array<uint8_t, VX_CFG_NUM_WARPS> head_before{};
  for (uint32_t wid = 0; wid < tracker_.num_warps(); ++wid)
    head_before[wid] = tracker_.read_head(wid);
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  // The RTL core endpoint accepts one worker completion beat per cycle.
  if (!completion_in_.empty()) {
    const auto completion = completion_in_.peek();
    tracker_.complete(completion.wid, completion.group_id);
    DT(4, "SOURCE_CONSUMED core=" << core_->id()
       << ", wid=" << completion.wid << ", gid=" << completion.group_id
       << ", pending=" << tracker_.remaining(completion.wid, completion.group_id)
       << ", depth=" << tracker_.committed_depth(completion.wid));
    completion_in_.pop();
  }
#endif

  tracker_.tick();
  for (uint32_t wid = 0; wid < tracker_.num_warps(); ++wid) {
    const uint8_t before = head_before[wid];
    const uint8_t after = tracker_.read_head(wid);
    if (before != after) {
      // tick() retires at most one row per warp. The retired sequence is the
      // old head, before the modulo pointer advances.
      DT(4, "GROUP_SOURCE_RETIRE core=" << core_->id()
         << " wid=" << wid << " gid=" << unsigned(before & (VX_CFG_DXA_GROUP_DEPTH - 1))
         << " depth=" << tracker_.committed_depth(wid));
    }
  }
  for (uint32_t wid = 0; wid < parked_waits_.size(); ++wid) {
    auto& wait = parked_waits_[wid];
    if (wait.trace && !wait.ready && tracker_.wait_satisfied(wid, wait.n)) {
      wait.ready = true;
      DT(3, "WAIT_READ_WAKE core=" << core_->id() << " wid=" << wid
         << ", N=" << wait.n);
    }
  }
#endif
}

#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
bool DxaUnit::peek_ready_wait(uint32_t* wid, uint32_t* block_id,
                              instr_trace_t** trace) const {
  for (uint32_t i = 0; i < parked_waits_.size(); ++i) {
    const auto& wait = parked_waits_[i];
    if (!wait.ready)
      continue;
    *wid = i;
    *block_id = wait.block_id;
    *trace = wait.trace;
    return true;
  }
  return false;
}

void DxaUnit::pop_ready_wait(uint32_t wid) {
  auto& wait = parked_waits_.at(wid);
  if (!wait.ready || !wait.trace)
    std::abort();
  wait = ParkedWait{};
}

void DxaUnit::close_owner(uint32_t wid) {
  tracker_.close_owner(wid);
}

bool DxaUnit::drained(uint32_t wid) const {
  return tracker_.drained(wid);
}

bool DxaUnit::reinitialize_owner(uint32_t wid) {
  return tracker_.reinitialize_owner(wid);
}
#endif

instr_trace_t* DxaUnit::process(instr_trace_t* trace, uint32_t block_id,
                                bool* release_warp, bool* parked) {
  *parked = false;
#ifndef VX_CFG_EXT_DXA_GROUP_ENABLE
  // The legacy ISSUE-only build has no parked wait path, so the SFU block
  // selector is intentionally unused there.
  (void)block_id;
  (void)release_warp;
#endif
#ifdef VX_CFG_EXT_DXA_GROUP_ENABLE
  const auto dxa_type = std::get<DxaType>(trace->op_type);

  if (dxa_type == DxaType::COMMIT_GROUP) {
    const bool had_open = tracker_.open_valid(trace->wid);
    const uint32_t commit_gid = tracker_.open_gid(trace->wid);
    const uint32_t pending = tracker_.open_remaining(trace->wid);
    tracker_.commit(trace->wid);
    // DT() is compiled out in normal builds; keep the state snapshots useful
    // for trace builds without triggering -Werror in trace-off builds.
    (void)had_open;
    (void)commit_gid;
    (void)pending;
    DT(4, "GROUP_COMMIT core=" << core_->id() << " wid=" << trace->wid
       << " gid=" << commit_gid
       << " pending=" << pending << " empty=" << !had_open
       << " depth=" << tracker_.committed_depth(trace->wid));
    return trace;
  }

  if (dxa_type == DxaType::WAIT_READ) {
    auto args = std::get<IntrDxaArgs>(trace->instr_ptr->get_args());
    if (tracker_.wait_satisfied(trace->wid, args.uimm5)) {
      *release_warp = true;
      DT(4, "WAIT_READ_PASS core=" << core_->id() << " wid=" << trace->wid
         << " N=" << args.uimm5
         << " depth=" << tracker_.committed_depth(trace->wid));
    } else {
      auto& wait = parked_waits_.at(trace->wid);
      if (wait.trace)
        std::abort();
      wait.trace = trace;
      wait.block_id = block_id;
      wait.n = args.uimm5;
      *parked = true;
      DT(3, "WAIT_READ_STALL core=" << core_->id() << " wid=" << trace->wid
         << " N=" << args.uimm5
         << " depth=" << tracker_.committed_depth(trace->wid));
    }
    return trace;
  }
#endif

#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  const bool is_s2g = dxa_type == DxaType::ISSUE_S2G;
#else
  const bool is_s2g = false;
#endif

  // A retried request must not increment the group counter twice.
  if (req_out_.full())
    return nullptr;

#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  // G2S does not participate in source groups.
  DxaGroupTracker::OperationToken token{};
  if (is_s2g) {
    const bool opening = !tracker_.open_valid(trace->wid);
    auto result = tracker_.issue(trace->wid, &token);
    if (result == DxaGroupTracker::IssueResult::Backpressured) {
      DT(3, "GROUP_BACKPRESSURE core=" << core_->id() << " wid=" << trace->wid
         << " depth=" << tracker_.committed_depth(trace->wid)
         << " pending=" << tracker_.open_remaining(trace->wid));
      return nullptr;
    }
    if (result == DxaGroupTracker::IssueResult::OwnerClosed)
      return trace;
    (void)opening;
    if (opening) {
      DT(4, "GROUP_OPEN core=" << core_->id() << " wid=" << trace->wid
         << " gid=" << unsigned(token.group_id)
         << " depth=" << tracker_.committed_depth(trace->wid));
    }
    DT(4, "GROUP_ISSUE core=" << core_->id() << " wid=" << trace->wid
       << " gid=" << unsigned(token.group_id)
       << " pending=" << tracker_.open_remaining(trace->wid)
       << " depth=" << tracker_.committed_depth(trace->wid));
  }
#endif

  // 4-lane wgather encoding:
  //   Lane 0: rs1=smem_addr, rs2=coord2
  //   Lane 1: rs1=meta,      rs2=coord3
  //   Lane 2: rs1=coord0,    rs2=coord4
  //   Lane 3: rs1=coord1,    rs2=cta_mask
  auto& rs1 = trace->src_data[0];
  auto& rs2 = trace->src_data[1];

  uint64_t smem_addr = static_cast<uint64_t>(rs1.at(0).u);
  uint32_t meta      = rs1.at(1).u;
  uint32_t coords[5] = {
    static_cast<uint32_t>(rs1.at(2).u),
    static_cast<uint32_t>(rs1.at(3).u),
    static_cast<uint32_t>(rs2.at(0).u),
    static_cast<uint32_t>(rs2.at(1).u),
    static_cast<uint32_t>(rs2.at(2).u),
  };
  uint32_t cta_mask  = rs2.at(3).u;
  uint32_t desc_slot = meta & 0x0fu;
  uint32_t raw_bar   = (meta >> 4) & 0x07ffffffu;

  DxaReq req{};
  req.core      = core_;
  req.uuid      = trace->uuid;
  req.wid       = trace->wid;
  req.direction = is_s2g ? DxaDirection::S2G : DxaDirection::G2S;
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  req.group_id  = token.group_id;
#endif
  req.desc_slot = desc_slot;
  // Keep raw bar_id; multicast offset arithmetic relies on encoded form
  // (cta_no in low 8 bits → bar_id + cta_idx targets next CTA's same bar).
  // Release call site decodes via bar_decode_id().
  req.bar_id    = raw_bar;
  req.cta_mask  = cta_mask;
  req.smem_addr = smem_addr;
  for (int i = 0; i < 5; ++i) req.coords[i] = coords[i];

  // Barrier pre-registration is the kernel's responsibility via
  // vx_barrier_expect_tx(). The DXA pipeline only emits release events on
  // completion; pre-registration happens explicitly per-CTA so multicast
  // destinations correctly wait.

  req_out_.send(req);
  DT(4, "dxa-unit submit: core=" << core_->id() << ", wid=" << trace->wid
     << ", dir=" << (is_s2g ? "s2g" : "g2s")
     << ", slot=" << desc_slot << ", bar=" << raw_bar
     << ", cta_mask=0x" << std::hex << cta_mask << std::dec);
  return trace;
}
