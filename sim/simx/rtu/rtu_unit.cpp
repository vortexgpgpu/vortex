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
#include "rtu_core.h"  // async pool: allocate_slot / free_slot
#include <util.h>      // log2ceil (uop uuid derivation)
#include <cassert>
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

RtuUnit::RtuUnit(Core* core, SimChannel<RtuReq>& req_out, RtuWindow& window)
  : window_(window)
  , core_(core)
  , req_out_(req_out)
{
  trace_slot_.fill(-1);
  for (auto& s : trace_scene_) s.fill(0);
}

uint32_t RtuUnit::wait_handle(const instr_trace_t* trace) {
  // handle = TRACE's rd = WAIT's rs1. Assumes all active lanes carry
  // the same handle (one TRACE allocates one slot covering the whole
  // warp). Read the first active lane's rs1 as the canonical handle
  // for the WAIT.
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

namespace {
// Low 32 bits of a (NaN-boxed) FP source operand = the raw f32 bits.
inline uint32_t fp_src_bits(const instr_trace_t* trace, uint32_t src, uint32_t t) {
  return static_cast<uint32_t>(trace->src_data[src].at(t).u64 & 0xffffffffu);
}
} // namespace

instr_trace_t* RtuUnit::process_wait(instr_trace_t* trace, uint32_t block_id) {
  uint32_t slot = wait_handle(trace);
  auto& pending = pending_terminals_.at(trace->wid);
  auto it = pending.find(slot);
  if (it == pending.end()) {
    // No response has landed yet — park the trace and bail. The matching
    // on_terminal_rsp() / on_candidate_rsp() call revives it. dst_data stays
    // uninitialised; SfuUnit won't output.send the parked trace, so the
    // scoreboard keeps WAIT's rd reserved (which is exactly the ordering
    // that gates a post-WAIT windowed read).
    wait_parked_.at(trace->wid)[slot] = ParkedWait{trace, block_id};
    DT(3, "rtu-wait park: core=" << core_->id() << ", wid=" << trace->wid
         << ", slot=" << slot);
    return nullptr;
  }
  // Fast path: a response was already cached. A TERMINAL frees the slot; a
  // CANDIDATE keeps it live (traversal resumes after the warp's CONTINUE).
  // Either way, stage the payload into the regfile and write the per-lane
  // status word into trace's dst_data so the SFU output.send delivers it.
  const RtuRsp& rsp = it->second;
  const bool is_candidate = (rsp.kind == RtuRspKind::CB_YIELD);
  if (is_candidate) apply_callback_payload(rsp);
  else              apply_response(rsp);
  write_status(trace, rsp, is_candidate);
  if (is_candidate) last_cand_mask_.at(trace->wid) = rsp.cb_active_mask;
  pending.erase(it);
  if (!is_candidate) rtu_core_->free_slot(slot);
  DT(3, "rtu-wait short-circuit: core=" << core_->id() << ", wid=" << trace->wid
       << ", slot=" << slot << ", cand=" << is_candidate);
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
  // hit attrs to the regfile so post-WAIT windowed reads see
  // coherent data; write status word into the parked trace's
  // dst_data; return it so SfuUnit can output.send.
  ParkedWait pw = it->second;
  parked.erase(it);
  apply_response(rsp);
  write_status(pw.trace, rsp, false);
  rtu_core_->free_slot(slot);
  DT(3, "rtu-terminal deliver: core=" << core_->id() << ", wid=" << wid
       << ", slot=" << slot << ", block=" << pw.block_id);
  return {pw.trace, pw.block_id};
}

// Per-lane status writeback for a completing WAIT/CONTINUE. A terminal reports
// each lane's own DONE_* code. A candidate reports YIELD_* only for the lanes it
// actually yielded (cb_active_mask); every other active lane of the trace is
// still traversing, so it gets PENDING — it must stay in the warp's loop rather
// than exit on a stale status, and the RTU ignores any action it contributes
// (only lanes with a pending candidate are applied). This is what makes a
// partial candidate batch (e.g. divergent-SBT reformation) correct.
void RtuUnit::write_status(instr_trace_t* trace, const RtuRsp& rsp, bool is_candidate) {
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) {
      trace->dst_data[t].u = 0;
      continue;
    }
    if (is_candidate && (((rsp.cb_active_mask >> t) & 1u) == 0)) {
      trace->dst_data[t].u = VX_RT_STS_PENDING;
    } else {
      trace->dst_data[t].u = rsp.status[t];
    }
  }
}

// The candidate-return counterpart of on_terminal_rsp: a non-opaque candidate
// (AHS / procedural) is returned to the issuing warp. Complete the parked WAIT
// with the YIELD status (the warp then loops in software: read the candidate,
// decide, vx_rt_continue). The slot is NOT freed — the ray is still traversing
// and resumes when the warp's CONTINUE lands its CB_ACTION.
uint32_t RtuUnit::candidate_slot(const RtuRsp& rsp) {
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if ((rsp.cb_active_mask >> t) & 1u) return rsp.cb_handle[t];
  }
  return 0;
}

bool RtuUnit::candidate_would_writeback(const RtuRsp& rsp,
                                        uint32_t* out_block_id) const {
  const auto& parked = wait_parked_.at(rsp.warp_id);
  auto it = parked.find(candidate_slot(rsp));
  if (it == parked.end()) return false;
  if (out_block_id) *out_block_id = it->second.block_id;
  return true;
}

RtuUnit::PendingWriteback RtuUnit::on_candidate_rsp(const RtuRsp& rsp) {
  uint32_t wid  = rsp.warp_id;
  uint32_t slot = candidate_slot(rsp);
  auto& parked = wait_parked_.at(wid);
  auto it = parked.find(slot);
  if (it == parked.end()) {
    // WAIT hasn't issued yet — latch the candidate; the slot stays live.
    pending_terminals_.at(wid)[slot] = rsp;
    DT(3, "rtu-candidate latch: core=" << core_->id() << ", wid=" << wid
         << ", slot=" << slot);
    return {nullptr, 0};
  }
  ParkedWait pw = it->second;
  parked.erase(it);
  apply_callback_payload(rsp);
  write_status(pw.trace, rsp, true);
  last_cand_mask_.at(wid) = rsp.cb_active_mask;
  // Candidate: leave the slot live; the warp's CONTINUE resumes traversal.
  DT(3, "rtu-candidate deliver: core=" << core_->id() << ", wid=" << wid
       << ", slot=" << slot << ", block=" << pw.block_id);
  return {pw.trace, pw.block_id};
}

instr_trace_t* RtuUnit::process_cb_ret(instr_trace_t* trace, uint32_t block_id) {
  // vx_rt_continue resumes traversal for the returned candidate. Per lane it
  // reports an action code (ACCEPT/IGNORE/TERMINATE) AND the slot handle (from
  // VX_RT_CB_HANDLE, staged by apply_callback_payload when the candidate was
  // returned). The RtuCore CB_ACTION drain uses the per-lane handle to route
  // the action back to the originating slot — necessary because same-warp
  // reformation may bundle lanes from multiple slots into one candidate.
  //
  // The action mask is the mask of the candidate we actually returned, NOT the
  // warp's SIMT mask. A lane that is still traversing (PENDING) rides along in
  // the warp's loop and contributes a garbage action against a stale
  // VX_RT_CB_HANDLE; it may nonetheless have a candidate queued for a LATER
  // batch (divergent-SBT reformation), so a cb_pending check alone would let
  // that garbage action resolve the wrong batch. Masking to the returned
  // candidate is what keeps partial batches correct.
  if (req_out_.full()) {
    return nullptr;
  }
  RtuReq req;
  req.kind     = RtuReqKind::CB_ACTION;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);
  req.trace    = trace;
  req.block_id = block_id;
  req.warp_id  = trace->wid;
  uint32_t cand_mask = last_cand_mask_.at(trace->wid);
  uint32_t bits = 0;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    if (((cand_mask >> t) & 1u) == 0) {
      trace->dst_data[t].u = 0;  // still traversing: no action this iteration
      continue;
    }
    bits |= (1u << t);
    // rs1 holds the action (ACCEPT/IGNORE/TERMINATE).
    req.cb_action[t] = static_cast<uint32_t>(trace->src_data[0].at(t).u);
    req.cb_handle[t] = cb_handle_.at(trace->wid)[t];
    // The shader's verdict carries its OWN hit distance (rs2, FP) and
    // hitAttribute (rs3) — that is why the RTU never has to read the window back.
    // An intersection shader reports the real t, so RtuCore commits it (not the
    // pre-IS AABB-entry candidate) on ACCEPT of a procedural primitive.
    req.cb_hit_t[t] = bits_to_float(fp_src_bits(trace, 1, t));
    req.cb_attr[t]  = static_cast<uint32_t>(trace->src_data[2].at(t).u);
    trace->dst_data[t].u = 0;  // no writeback
  }
  req.tmask_bits = bits;
  req_out_.send(req);
  DT(3, "rtu-cb_ret submit: core=" << core_->id() << ", wid=" << trace->wid
       << ", tmask=0x" << std::hex << bits << std::dec);
  return trace;
}

void RtuUnit::apply_response(const RtuRsp& rsp) {
  auto& wregs = window_.warp(rsp.warp_id);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    auto& lregs = wregs.at(t);
    // Written by the RTU, not by any instruction: the payload pointer rode the
    // arm and the hitAttribute rode the CONTINUE that accepted this hit.
    lregs[VX_RT_PAYLOAD_PTR_LO]     = trace_payload_.at(rsp.warp_id);
    lregs[VX_RT_HIT_ATTR_0]         = rsp.hit_attr[t];
    lregs[VX_RT_HIT_T]              = float_to_bits(rsp.hit_t[t]);
    lregs[VX_RT_HIT_BARY_U]         = float_to_bits(rsp.hit_bary_u[t]);
    lregs[VX_RT_HIT_BARY_V]         = float_to_bits(rsp.hit_bary_v[t]);
    lregs[VX_RT_HIT_PRIMITIVE_ID]   = rsp.hit_primitive_id[t];
    lregs[VX_RT_HIT_INSTANCE_ID]    = rsp.hit_instance_id[t];
    lregs[VX_RT_HIT_INSTANCE_CUSTOM] = rsp.hit_instance_custom[t];
    lregs[VX_RT_HIT_GEOMETRY_INDEX] = rsp.hit_geometry_index[t];
    // Committed hit's object-space ray, for a CHS / post-wait read of
    // gl_ObjectRay{Origin,Direction}EXT.
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 0]    = float_to_bits(rsp.obj_o_x[t]);
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 1]    = float_to_bits(rsp.obj_o_y[t]);
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 2]    = float_to_bits(rsp.obj_o_z[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 0] = float_to_bits(rsp.obj_d_x[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 1] = float_to_bits(rsp.obj_d_y[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 2] = float_to_bits(rsp.obj_d_z[t]);
  }
}

void RtuUnit::apply_callback_payload(const RtuRsp& rsp) {
  // Stage candidate-hit attrs + cb_type + cb_handle into the RTU regs
  // for the lanes whose rays yielded, so the dispatcher's vx_rt_get
  // sees the right payload AND so vx_rt_cb_ret can route the action
  // back to the originating slot. Only the yielded lanes
  // (cb_active_mask) are touched. With same-warp reformation we may
  // batch lanes from MULTIPLE slots into one CB_YIELD, so
  // VX_RT_CB_HANDLE is per-lane (not warp-scoped).
  auto& wregs = window_.warp(rsp.warp_id);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (((rsp.cb_active_mask >> t) & 1u) == 0) continue;
    auto& lregs = wregs.at(t);
    // The callback shaders read the tracing shader's payload pointer, and they
    // are a different shader — it cannot reach them any other way. A closest-hit
    // shader likewise reads the accepted hit's attribute.
    lregs[VX_RT_PAYLOAD_PTR_LO]     = trace_payload_.at(rsp.warp_id);
    lregs[VX_RT_HIT_ATTR_0]         = rsp.hit_attr[t];
    lregs[VX_RT_HIT_T]              = float_to_bits(rsp.hit_t[t]);
    lregs[VX_RT_HIT_BARY_U]         = float_to_bits(rsp.hit_bary_u[t]);
    lregs[VX_RT_HIT_BARY_V]         = float_to_bits(rsp.hit_bary_v[t]);
    lregs[VX_RT_HIT_PRIMITIVE_ID]   = rsp.hit_primitive_id[t];
    lregs[VX_RT_HIT_INSTANCE_ID]    = rsp.hit_instance_id[t];
    lregs[VX_RT_HIT_INSTANCE_CUSTOM] = rsp.hit_instance_custom[t];
    lregs[VX_RT_HIT_GEOMETRY_INDEX] = rsp.hit_geometry_index[t];
    lregs[VX_RT_CB_TYPE]            = rsp.cb_type[t];
    lregs[VX_RT_CB_HANDLE]          = rsp.cb_handle[t];
    // ... and keep the RTU's own copy, so its CONTINUE never has to read a slot.
    cb_handle_.at(rsp.warp_id)[t]   = rsp.cb_handle[t];
    lregs[VX_RT_HIT_SBT_IDX]        = rsp.cb_sbt_idx[t];
    // Candidate's object-space ray, so the AHS/IS dispatcher can read
    // gl_ObjectRay{Origin,Direction}EXT before computing the
    // procedural intersection.
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 0]    = float_to_bits(rsp.obj_o_x[t]);
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 1]    = float_to_bits(rsp.obj_o_y[t]);
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 2]    = float_to_bits(rsp.obj_o_z[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 0] = float_to_bits(rsp.obj_d_x[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 1] = float_to_bits(rsp.obj_d_y[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 2] = float_to_bits(rsp.obj_d_z[t]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Macro-op micro-op generator + the per-uop TRACE / WAIT handlers.
///////////////////////////////////////////////////////////////////////////////


uint32_t RtuUopGen::uop_count(const Instr& instr) {
  if (instr.get_fu_type() != FUType::SFU)
    return 1;
  auto op = instr.get_op_type();
  if (auto rtu_p = std::get_if<GfxwType>(&op)) {
    if (*rtu_p == GfxwType::TRACE)  return 4;  // 1 GP config + 3 FP ray
    if (*rtu_p == GfxwType::GETWF || *rtu_p == GfxwType::GETW) {
      auto args = std::get<IntrGfxwArgs>(instr.get_args());  // one uop per slot
      return args.count ? args.count : 1;
    }
  }
  return 1;
}

Instr::Ptr RtuUopGen::get(const Instr& macro_instr, uint32_t uop_index) {
  auto rtu_type = std::get<GfxwType>(macro_instr.get_op_type());
  uint64_t parent_uuid = macro_instr.get_uuid();
  uint32_t total = uop_count(macro_instr);

  uint32_t uuid_hi = (parent_uuid >> 32) & 0xffffffff;
  uint32_t uuid_lo = parent_uuid & 0xffffffff;
  uint32_t steps_shift = (total > 1) ? (32 - log2ceil(total)) : 0;
  uint64_t uop_uuid = (uint64_t(uuid_hi) << 32) | ((uop_index << steps_shift) | uuid_lo);

  auto uop = std::allocate_shared<Instr>(pool_, uop_uuid, FUType::SFU);
  uop->set_parent_uuid(parent_uuid);
  uop->set_op_type(rtu_type);

  auto macro_args = std::get<IntrGfxwArgs>(macro_instr.get_args());
  IntrGfxwArgs args{};
  args.uop = uop_index;
  args.slot = macro_args.slot;
  args.count = macro_args.count;
  uop->set_args(args);

  uint32_t rd_idx  = macro_instr.get_dest_reg().idx;   // handle / status / window base
  uint32_t rs1_idx = macro_instr.get_src_reg(0).idx;   // config / handle

  if (rtu_type == GfxwType::GETWF || rtu_type == GfxwType::GETW) {
    // Windowed read: uop i writes window slot (start+i) into reg (rd_base + i).
    // No source operands — the data comes from the RTU regfile. GETWF -> FP
    // (NaN-boxed), GETW -> GP (raw).
    uop->set_dest_reg(rd_idx + uop_index,
                      rtu_type == GfxwType::GETWF ? RegType::Float : RegType::Integer);
  } else if (rtu_type == GfxwType::TRACE) {
    // f0..f7 ray window streamed three regs per uop.
    switch (uop_index) {
    case 0: // GP config: read rs1 lanes, alloc slot, write handle.
      uop->set_dest_reg(rd_idx, RegType::Integer);
      uop->set_src_reg(0, rs1_idx, RegType::Integer);
      break;
    case 1: // origin.xyz <- f0,f1,f2
      uop->set_src_reg(0, 0, RegType::Float);
      uop->set_src_reg(1, 1, RegType::Float);
      uop->set_src_reg(2, 2, RegType::Float);
      break;
    case 2: // dir.xyz <- f3,f4,f5
      uop->set_src_reg(0, 3, RegType::Float);
      uop->set_src_reg(1, 4, RegType::Float);
      uop->set_src_reg(2, 5, RegType::Float);
      break;
    case 3: // tmin,tmax <- f6,f7 (then arm)
      uop->set_src_reg(0, 6, RegType::Float);
      uop->set_src_reg(1, 7, RegType::Float);
      break;
    default:
      std::abort();
    }
  } else {
    std::abort();  // only TRACE / GETWF / GETW are SFU macro-ops
  }
  // Windowed reads carry an optional scoreboard-chain source on rs1 (x0 = none):
  // vx_rt_wait sets it to the WAIT status so the window issues only after the
  // block retired and apply_response staged the hit. In-trap callback reads
  // (vx_rt_get_objray) leave it x0 — the dispatcher already runs post-yield.
  if (rtu_type == GfxwType::GETWF || rtu_type == GfxwType::GETW) {
    uop->set_src_reg(0, rs1_idx, RegType::Integer);
  }
  return uop;
}

// Claim this warp's slot BEFORE the TRACE head uop is issued. A functional unit
// retries a uop it cannot complete by leaving it at the head of its input queue,
// so allocating inside the unit meant an empty pool jammed that queue -- and the
// WAIT stuck behind it was the only thing that could ever free a slot.
bool RtuUnit::trace2_reserve_slot(uint32_t wid) {
  if (rtu_core_ == nullptr) {
    return false;
  }
  if (trace_slot_.at(wid) >= 0) {
    return true; // this warp's TRACE already holds a slot
  }
  int32_t slot = rtu_core_->allocate_slot(core_->id());
  if (slot < 0) {
    return false;
  }
  trace_slot_.at(wid) = slot;
  return true;
}

instr_trace_t* RtuUnit::process_trace_uop(instr_trace_t* trace, uint32_t block_id, uint32_t uop) {
  uint32_t wid = trace->wid;
  // No window access anywhere in here: a TRACE writes no slot.
  switch (uop) {
  case 0: {
    // GP config uop: the pool slot was claimed at issue (trace2_reserve_slot),
    // so this uop has no backpressure source. Unpack the lane-packed config
    // (lane0=scene, lane1=payload, lane2=flags, lane3=cull — the implicit
    // vx_wgather layout) and stage it.
    int32_t slot = trace_slot_.at(wid);
    assert(slot >= 0 && "TRACE uop0 issued without a reserved pool slot");
    // Config rides the gathered wgather lanes (1..3), never the write-suppressed
    // self slot (lane 0), so every word survives a partial/lane-0-dead mask.
    // scene = wgather lane 1 (warp-uniform).
    auto& cfg = trace->src_data[0];
    uint32_t flagscull = static_cast<uint32_t>(cfg.at(3).u);
    // Warp-uniform: it rides the arm doorbell, so it is staged here, not written
    // into the window. The RTU writes the payload pointer back with the record.
    trace_payload_.at(wid) = static_cast<uint32_t>(cfg.at(2).u);
    trace_flags_.at(wid)   = flagscull & 0xffffu;
    trace_cull_.at(wid)    = flagscull >> 16;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      trace_scene_.at(wid)[t] = static_cast<uint32_t>(cfg.at(1).u);
      trace->dst_data[t].u    = uint32_t(slot);  // handle returns early
    }
    return trace;
  }
  case 1:  // origin.xyz <- f0,f1,f2
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      auto& ray = trace_ray_.at(wid)[t];
      ray.origin[0] = fp_src_bits(trace, 0, t);
      ray.origin[1] = fp_src_bits(trace, 1, t);
      ray.origin[2] = fp_src_bits(trace, 2, t);
    }
    return trace;
  case 2:  // dir.xyz <- f3,f4,f5
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      auto& ray = trace_ray_.at(wid)[t];
      ray.dir[0] = fp_src_bits(trace, 0, t);
      ray.dir[1] = fp_src_bits(trace, 1, t);
      ray.dir[2] = fp_src_bits(trace, 2, t);
    }
    return trace;
  case 3: {  // tmin,tmax <- f6,f7, then ARM the slot.
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      auto& ray = trace_ray_.at(wid)[t];
      ray.t_min = fp_src_bits(trace, 0, t);
      ray.t_max = fp_src_bits(trace, 1, t);
    }
    // ARM: hand the staged ray to RtuCore (bus full => retry uop 3 idempotently;
    // the slot was already latched at uop 0). This IS the ray's only destination
    // — it was never written to the window and is never read back from it.
    if (req_out_.full())
      return nullptr;
    int32_t slot = trace_slot_.at(wid);
    RtuReq req;
    req.kind     = RtuReqKind::TRACE_NEW;
    req.uuid     = trace->uuid;
    req.tag      = uint32_t(trace->uuid);
    req.slot_idx = uint32_t(slot);
    req.trace    = trace;
    req.block_id = block_id;
    req.warp_id  = wid;
    uint32_t bits = 0;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      bits |= (1u << t);
      const auto& ray = trace_ray_.at(wid)[t];
      req.scene_root[t] = trace_scene_.at(wid)[t];
      req.origin_x[t]   = bits_to_float(ray.origin[0]);
      req.origin_y[t]   = bits_to_float(ray.origin[1]);
      req.origin_z[t]   = bits_to_float(ray.origin[2]);
      req.dir_x[t]      = bits_to_float(ray.dir[0]);
      req.dir_y[t]      = bits_to_float(ray.dir[1]);
      req.dir_z[t]      = bits_to_float(ray.dir[2]);
      req.tmin[t]       = bits_to_float(ray.t_min);
      req.tmax[t]       = bits_to_float(ray.t_max);
      req.flags[t]      = trace_flags_.at(wid);
      req.cull_mask[t]  = trace_cull_.at(wid);
    }
    req.tmask_bits = bits;
    req_out_.send(req);
    trace_slot_.at(wid) = -1;
    DT(3, "rtu-trace arm: core=" << core_->id() << ", wid=" << wid
         << ", slot=" << slot << ", tmask=0x" << std::hex << bits << std::dec);
    return trace;
  }
  default:
    std::abort();
  }
}

