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

#include "rtu_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"
#include "rtu_core.h"  // §8.6 async pool: allocate_slot / free_slot
#include <cstring>

using namespace vortex;

namespace {
inline float bits_to_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}
inline uint32_t float_to_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));
  return bits;
}
} // namespace

RtuUnit::RtuUnit(Core* core, SimChannel<RtuReq>& req_out)
  : regfile_(VX_CFG_NUM_WARPS)
  , core_(core)
  , req_out_(req_out)
{
  for (auto& w : regfile_) {
    for (auto& l : w) {
      l.fill(0);
    }
  }
}

instr_trace_t* RtuUnit::process_set(instr_trace_t* trace) {
  // Single-slot write: rs1 -> regfile[wid][lane][slot].
  auto& instr = *trace->instr_ptr;
  auto args = std::get<IntrRtuArgs>(instr.get_args());
  uint32_t slot = args.slot;
  if (slot >= VX_RT_SLOT_COUNT) {
    return trace;
  }
  auto& wregs = regfile_.at(trace->wid);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    auto& lregs = wregs.at(t);
    lregs[slot] = static_cast<uint32_t>(trace->src_data[0].at(t).u);
  }
  return trace;
}

instr_trace_t* RtuUnit::process_get(instr_trace_t* trace) {
  auto& instr = *trace->instr_ptr;
  auto args = std::get<IntrRtuArgs>(instr.get_args());
  uint32_t slot = args.slot;
  if (slot >= VX_RT_SLOT_COUNT) {
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (trace->tmask.test(t)) trace->dst_data[t].u = 0;
    }
    return trace;
  }
  auto& wregs = regfile_.at(trace->wid);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    trace->dst_data[t].u = wregs.at(t).at(slot);
  }
  return trace;
}

instr_trace_t* RtuUnit::process_trace(instr_trace_t* trace, uint32_t block_id) {
  // §8.6 async ray pool: pre-allocate a slot in RtuCore's pool BEFORE
  // sending the request. The slot index is the handle vx_rt_trace
  // writes back synchronously, so downstream WAIT(handle) ops can
  // look up "which slot is this ray in" without a round-trip. If the
  // pool is full OR the bus port is full, backpressure (caller
  // retries next cycle and the trace stays at SFU input head).
  if (req_out_.full()) {
    return nullptr;
  }
  int32_t slot = rtu_core_->allocate_slot();
  if (slot < 0) {
    return nullptr;  // pool full — try again next tick
  }

  auto& wregs = regfile_.at(trace->wid);
  auto& rs1 = trace->src_data[0];

  RtuReq req;
  req.kind     = RtuReqKind::TRACE_NEW;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);
  req.slot_idx = uint32_t(slot);
  req.trace    = trace;
  req.block_id = block_id;
  req.warp_id  = trace->wid;
  uint32_t bits = 0;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    bits |= (1u << t);
    auto& lregs = wregs.at(t);
    req.scene_root[t] = static_cast<uint32_t>(rs1.at(t).u);
    req.origin_x[t]   = bits_to_float(lregs[VX_RT_RAY_ORIGIN + 0]);
    req.origin_y[t]   = bits_to_float(lregs[VX_RT_RAY_ORIGIN + 1]);
    req.origin_z[t]   = bits_to_float(lregs[VX_RT_RAY_ORIGIN + 2]);
    req.dir_x[t]      = bits_to_float(lregs[VX_RT_RAY_DIRECTION + 0]);
    req.dir_y[t]      = bits_to_float(lregs[VX_RT_RAY_DIRECTION + 1]);
    req.dir_z[t]      = bits_to_float(lregs[VX_RT_RAY_DIRECTION + 2]);
    req.tmin[t]       = bits_to_float(lregs[VX_RT_T_MIN]);
    req.tmax[t]       = bits_to_float(lregs[VX_RT_T_MAX]);
    req.flags[t]      = lregs[VX_RT_RAY_FLAGS];
    req.cull_mask[t]  = lregs[VX_RT_CULL_MASK];
    // §8.6: every active lane gets the same handle (a TRACE op
    // allocates one slot covering all lanes in the warp).
    trace->dst_data[t].u = uint32_t(slot);
  }
  req.tmask_bits = bits;
  req_out_.send(req);
  DT(3, "rtu-trace submit: core=" << core_->id() << ", wid=" << trace->wid
       << ", slot=" << slot
       << ", tmask=0x" << std::hex << bits << std::dec);
  return trace;
}

uint32_t RtuUnit::wait_handle(const instr_trace_t* trace) {
  // §8.6: handle = TRACE's rd = WAIT's rs1. Phase-1 of §8.6 assumes
  // all active lanes carry the same handle (one TRACE allocates one
  // slot covering the whole warp). Read the first active lane's
  // rs1 as the canonical handle for the WAIT.
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (trace->tmask.test(t)) {
      return static_cast<uint32_t>(trace->src_data[0].at(t).u);
    }
  }
  return 0;
}

bool RtuUnit::wait_would_short_circuit(uint32_t wid, uint32_t slot) const {
  return pending_terminals_.at(wid).count(slot) != 0;
}

bool RtuUnit::terminal_would_writeback(const RtuRsp& rsp,
                                       uint32_t* out_block_id) const {
  const auto& parked = wait_parked_.at(rsp.warp_id);
  auto it = parked.find(rsp.slot_idx);
  if (it == parked.end()) return false;
  if (out_block_id) *out_block_id = it->second.block_id;
  return true;
}

instr_trace_t* RtuUnit::process_wait(instr_trace_t* trace, uint32_t block_id) {
  uint32_t slot = wait_handle(trace);
  auto& pending = pending_terminals_.at(trace->wid);
  auto it = pending.find(slot);
  if (it == pending.end()) {
    // TERMINAL hasn't landed yet — park the trace and bail. The
    // matching on_terminal_rsp() call will revive it. dst_data
    // stays uninitialised; SfuUnit won't output.send the parked
    // trace, so scoreboard keeps WAIT's rd reserved (which is
    // exactly the §10.3 ordering that gates vx_rt_get_after).
    wait_parked_.at(trace->wid)[slot] = ParkedWait{trace, block_id};
    DT(3, "rtu-wait park: core=" << core_->id() << ", wid=" << trace->wid
         << ", slot=" << slot);
    return nullptr;
  }
  // Fast path: TERMINAL was already cached. Apply it to the regfile
  // now (so vx_rt_get_after that follows reads coherent hit data)
  // and write the per-lane status word into trace's dst_data so
  // the SFU output.send delivers it.
  const RtuRsp& rsp = it->second;
  apply_response(rsp);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    trace->dst_data[t].u = trace->tmask.test(t) ? rsp.status[t] : 0;
  }
  pending.erase(it);
  rtu_core_->free_slot(slot);
  DT(3, "rtu-wait short-circuit: core=" << core_->id() << ", wid=" << trace->wid
       << ", slot=" << slot);
  return trace;
}

RtuUnit::PendingWriteback RtuUnit::on_terminal_rsp(const RtuRsp& rsp) {
  uint32_t wid  = rsp.warp_id;
  uint32_t slot = rsp.slot_idx;
  auto& parked = wait_parked_.at(wid);
  auto it = parked.find(slot);
  if (it == parked.end()) {
    // WAIT hasn't issued yet — latch the rsp. Slot stays live in
    // RtuCore (EMITTED state) until the eventual WAIT consumes the
    // pending_terminals_ entry and calls free_slot.
    pending_terminals_.at(wid)[slot] = rsp;
    DT(3, "rtu-terminal latch: core=" << core_->id() << ", wid=" << wid
         << ", slot=" << slot);
    return {nullptr, 0};
  }
  // Common path: WAIT was parked, now we can complete it. Apply
  // hit attrs to the regfile so post-WAIT vx_rt_get_after sees
  // coherent data; write status word into the parked trace's
  // dst_data; return it so SfuUnit can output.send.
  ParkedWait pw = it->second;
  parked.erase(it);
  apply_response(rsp);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    pw.trace->dst_data[t].u = pw.trace->tmask.test(t) ? rsp.status[t] : 0;
  }
  rtu_core_->free_slot(slot);
  DT(3, "rtu-terminal deliver: core=" << core_->id() << ", wid=" << wid
       << ", slot=" << slot << ", block=" << pw.block_id);
  return {pw.trace, pw.block_id};
}

instr_trace_t* RtuUnit::process_cb_ret(instr_trace_t* trace, uint32_t block_id) {
  // Phase 2 / 3-A2: vx_rt_cb_ret releases per-lane parked contexts. Per
  // lane it reports an action code (ACCEPT/IGNORE/TERMINATE) AND the
  // slot handle (from VX_RT_CB_HANDLE, staged by apply_callback_payload
  // at CB_YIELD time). The RtuCore CB_ACTION drain uses the per-lane
  // handle to route the action back to the originating slot — necessary
  // because Phase 3-A2 same-warp reformation may bundle lanes from
  // multiple slots into one CB_YIELD trap.
  if (req_out_.full()) {
    return nullptr;
  }
  auto& wregs = regfile_.at(trace->wid);
  RtuReq req;
  req.kind     = RtuReqKind::CB_ACTION;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);
  req.trace    = trace;
  req.block_id = block_id;
  req.warp_id  = trace->wid;
  uint32_t bits = 0;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    bits |= (1u << t);
    // rs1 holds the action (ACCEPT/IGNORE/TERMINATE).
    req.cb_action[t] = static_cast<uint32_t>(trace->src_data[0].at(t).u);
    req.cb_handle[t] = wregs.at(t)[VX_RT_CB_HANDLE];
    trace->dst_data[t].u = 0;  // no writeback
  }
  req.tmask_bits = bits;
  req_out_.send(req);
  DT(3, "rtu-cb_ret submit: core=" << core_->id() << ", wid=" << trace->wid
       << ", tmask=0x" << std::hex << bits << std::dec);
  return trace;
}

void RtuUnit::apply_response(const RtuRsp& rsp) {
  auto& wregs = regfile_.at(rsp.warp_id);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    auto& lregs = wregs.at(t);
    lregs[VX_RT_HIT_T]              = float_to_bits(rsp.hit_t[t]);
    lregs[VX_RT_HIT_BARY_U]         = float_to_bits(rsp.hit_bary_u[t]);
    lregs[VX_RT_HIT_BARY_V]         = float_to_bits(rsp.hit_bary_v[t]);
    lregs[VX_RT_HIT_PRIMITIVE_ID]   = rsp.hit_primitive_id[t];
    lregs[VX_RT_HIT_INSTANCE_ID]    = rsp.hit_instance_id[t];
    lregs[VX_RT_HIT_GEOMETRY_INDEX] = rsp.hit_geometry_index[t];
  }
}

void RtuUnit::apply_callback_payload(const RtuRsp& rsp) {
  // Stage candidate-hit attrs + cb_type + cb_handle into the RTU regs
  // for the lanes whose rays yielded, so the dispatcher's vx_rt_get
  // sees the right payload AND so vx_rt_cb_ret can route the action
  // back to the originating slot. Only the yielded lanes
  // (cb_active_mask) are touched. Phase 3-A2: with same-warp
  // reformation we may batch lanes from MULTIPLE slots into one
  // CB_YIELD, so VX_RT_CB_HANDLE is per-lane (not warp-scoped).
  auto& wregs = regfile_.at(rsp.warp_id);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (((rsp.cb_active_mask >> t) & 1u) == 0) continue;
    auto& lregs = wregs.at(t);
    lregs[VX_RT_HIT_T]              = float_to_bits(rsp.hit_t[t]);
    lregs[VX_RT_HIT_BARY_U]         = float_to_bits(rsp.hit_bary_u[t]);
    lregs[VX_RT_HIT_BARY_V]         = float_to_bits(rsp.hit_bary_v[t]);
    lregs[VX_RT_HIT_PRIMITIVE_ID]   = rsp.hit_primitive_id[t];
    lregs[VX_RT_HIT_INSTANCE_ID]    = rsp.hit_instance_id[t];
    lregs[VX_RT_HIT_GEOMETRY_INDEX] = rsp.hit_geometry_index[t];
    lregs[VX_RT_CB_TYPE]            = rsp.cb_type[t];
    lregs[VX_RT_CB_HANDLE]          = rsp.cb_handle[t];
    lregs[VX_RT_HIT_SBT_IDX]        = rsp.cb_sbt_idx[t];
  }
}
