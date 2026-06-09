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
#include <util.h>      // log2ceil (uop uuid derivation)
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
// NaN-box a single-precision value into a 64-bit FP register write (the
// RV32F/RV64F convention SimX's FPU uses for all f32 writebacks).
inline uint64_t nan_box32(uint32_t bits) {
  return uint64_t(bits) | 0xffffffff00000000ull;
}
} // namespace

RtuUnit::RtuUnit(Core* core, SimChannel<RtuReq>& req_out)
  : regfile_(VX_CFG_NUM_WARPS)
  , core_(core)
  , req_out_(req_out)
{
  trace2_slot_.fill(-1);
  for (auto& s : trace2_scene_) s.fill(0);
  for (auto& w : regfile_) {
    for (auto& l : w) {
      l.fill(0);
      // §8.8 Vulkan instanceCullMask: a kernel that never touches
      // VX_RT_CULL_MASK should see the "no culling" default
      // (cull_mask = 0xff matches every instance mask). Zero would
      // mean "no rays hit any instance" per the spec — exactly the
      // opposite of what un-set state should imply.
      l[VX_RT_CULL_MASK] = 0xffu;
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
    trace->suspended = true;  // holds its rd reserved while parked; an async
                              // callback trap lifts only suspended reservations
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
  pw.trace->suspended = false;  // flowing again — releases its rd on commit
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
    // P1: the IS dispatcher may have written the real hit distance into
    // VX_RT_HIT_T; carry it back so the RtuCore commits the IS t (not the
    // pre-IS AABB-entry candidate) on ACCEPT of a procedural primitive.
    req.cb_hit_t[t]  = bits_to_float(wregs.at(t)[VX_RT_HIT_T]);
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
    // P1 §4.2 slots 8..13: committed hit's object-space ray, for a CHS /
    // post-wait read of gl_ObjectRay{Origin,Direction}EXT.
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
    // P1 §4.2 slots 8..13: candidate's object-space ray, so the AHS/IS
    // dispatcher can read gl_ObjectRay{Origin,Direction}EXT before
    // computing the procedural intersection.
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 0]    = float_to_bits(rsp.obj_o_x[t]);
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 1]    = float_to_bits(rsp.obj_o_y[t]);
    lregs[VX_RT_OBJECT_RAY_ORIGIN + 2]    = float_to_bits(rsp.obj_o_z[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 0] = float_to_bits(rsp.obj_d_x[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 1] = float_to_bits(rsp.obj_d_y[t]);
    lregs[VX_RT_OBJECT_RAY_DIRECTION + 2] = float_to_bits(rsp.obj_d_z[t]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// ISA v2 (rtu_isa_v2_proposal.md §5.6) — macro-op micro-op generator + the
// per-uop TRACE2 / WAIT2 handlers.
///////////////////////////////////////////////////////////////////////////////

namespace {
// Low 32 bits of a (NaN-boxed) FP source operand = the raw f32 bits.
inline uint32_t fp_src_bits(const instr_trace_t* trace, uint32_t src, uint32_t t) {
  return static_cast<uint32_t>(trace->src_data[src].at(t).u64 & 0xffffffffu);
}
} // namespace

uint32_t RtuUopGen::uop_count(const Instr& instr) {
  if (instr.get_fu_type() != FUType::SFU)
    return 1;
  auto op = instr.get_op_type();
  if (auto rtu_p = std::get_if<RtuType>(&op)) {
    if (*rtu_p == RtuType::TRACE2)  return 4;  // 1 GP config + 3 FP ray
    if (*rtu_p == RtuType::GETWF || *rtu_p == RtuType::GETW) {
      auto args = std::get<IntrRtuArgs>(instr.get_args());  // one uop per slot
      return args.count ? args.count : 1;
    }
  }
  return 1;
}

Instr::Ptr RtuUopGen::get(const Instr& macro_instr, uint32_t uop_index) {
  auto rtu_type = std::get<RtuType>(macro_instr.get_op_type());
  uint64_t parent_uuid = macro_instr.get_uuid();
  uint32_t total = uop_count(macro_instr);

  uint32_t uuid_hi = (parent_uuid >> 32) & 0xffffffff;
  uint32_t uuid_lo = parent_uuid & 0xffffffff;
  uint32_t steps_shift = (total > 1) ? (32 - log2ceil(total)) : 0;
  uint64_t uop_uuid = (uint64_t(uuid_hi) << 32) | ((uop_index << steps_shift) | uuid_lo);

  auto uop = std::allocate_shared<Instr>(pool_, uop_uuid, FUType::SFU);
  uop->set_parent_uuid(parent_uuid);
  uop->set_op_type(rtu_type);

  auto macro_args = std::get<IntrRtuArgs>(macro_instr.get_args());
  IntrRtuArgs args{};
  args.uop = uop_index;
  args.divergent = macro_args.divergent;
  args.slot = macro_args.slot;
  args.count = macro_args.count;
  uop->set_args(args);

  uint32_t rd_idx  = macro_instr.get_dest_reg().idx;   // handle / status / window base
  uint32_t rs1_idx = macro_instr.get_src_reg(0).idx;   // config / handle

  if (rtu_type == RtuType::GETWF || rtu_type == RtuType::GETW) {
    // Windowed read: uop i writes window slot (start+i) into reg (rd_base + i).
    // No source operands — the data comes from the RTU regfile. GETWF -> FP
    // (NaN-boxed), GETW -> GP (raw).
    uop->set_dest_reg(rd_idx + uop_index,
                      rtu_type == RtuType::GETWF ? RegType::Float : RegType::Integer);
  } else if (rtu_type == RtuType::TRACE2) {
    // f0..f7 ray window streamed three regs per uop.
    switch (uop_index) {
    case 0: // GP config: read rs1 lanes, alloc slot, write handle.
      uop->set_dest_reg(rd_idx, RegType::Integer);
      uop->set_src_reg(0, rs1_idx, RegType::Integer);
      // Multi-AS form: uop 0 also reads the per-lane scene from rs2.
      if (macro_args.divergent)
        uop->set_src_reg(1, macro_instr.get_src_reg(1).idx, RegType::Integer);
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
    std::abort();  // only TRACE2 / GETWF / GETW are SFU macro-ops
  }
  // Windowed reads carry an optional scoreboard-chain source on rs1 (x0 = none):
  // vx_rt_wait2 sets it to the WAIT2 status so the window issues only after the
  // block retired and apply_response staged the hit. In-trap callback reads
  // (vx_rt_get_objray) leave it x0 — the dispatcher already runs post-yield.
  if (rtu_type == RtuType::GETWF || rtu_type == RtuType::GETW) {
    uop->set_src_reg(0, rs1_idx, RegType::Integer);
  }
  return uop;
}

instr_trace_t* RtuUnit::process_trace2_uop(instr_trace_t* trace, uint32_t block_id, uint32_t uop) {
  uint32_t wid = trace->wid;
  auto& wregs = regfile_.at(wid);
  switch (uop) {
  case 0: {
    // GP config uop: allocate the pool slot first (only backpressure source
    // here), then unpack the lane-packed config (lane0=scene, lane1=payload,
    // lane2=flags, lane3=cull — the implicit vx_wgather layout) and stage it.
    int32_t slot = rtu_core_->allocate_slot();
    if (slot < 0)
      return nullptr;  // pool full — retry uop 0
    trace2_slot_.at(wid) = slot;
    // Config rides the gathered wgather lanes (1..3), never the write-suppressed
    // self slot (lane 0), so every word survives a partial/lane-0-dead mask.
    // Multi-AS (divergent) form: scene is per-lane in rs2; otherwise it is the
    // warp-uniform wgather lane 1.
    auto args = std::get<IntrRtuArgs>(trace->instr_ptr->get_args());
    auto& cfg = trace->src_data[0];
    uint32_t payload   = static_cast<uint32_t>(cfg.at(2).u);
    uint32_t flagscull = static_cast<uint32_t>(cfg.at(3).u);
    uint32_t flags     = flagscull & 0xffffu;
    uint32_t cull      = flagscull >> 16;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      trace2_scene_.at(wid)[t]    = args.divergent
                                      ? static_cast<uint32_t>(trace->src_data[1].at(t).u)
                                      : static_cast<uint32_t>(cfg.at(1).u);
      auto& lregs = wregs.at(t);
      lregs[VX_RT_PAYLOAD_PTR_LO] = payload;
      lregs[VX_RT_RAY_FLAGS]      = flags;
      lregs[VX_RT_CULL_MASK]      = cull;
      trace->dst_data[t].u        = uint32_t(slot);  // handle returns early
    }
    return trace;
  }
  case 1:  // origin.xyz <- f0,f1,f2
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      auto& lregs = wregs.at(t);
      lregs[VX_RT_RAY_ORIGIN + 0] = fp_src_bits(trace, 0, t);
      lregs[VX_RT_RAY_ORIGIN + 1] = fp_src_bits(trace, 1, t);
      lregs[VX_RT_RAY_ORIGIN + 2] = fp_src_bits(trace, 2, t);
    }
    return trace;
  case 2:  // dir.xyz <- f3,f4,f5
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      auto& lregs = wregs.at(t);
      lregs[VX_RT_RAY_DIRECTION + 0] = fp_src_bits(trace, 0, t);
      lregs[VX_RT_RAY_DIRECTION + 1] = fp_src_bits(trace, 1, t);
      lregs[VX_RT_RAY_DIRECTION + 2] = fp_src_bits(trace, 2, t);
    }
    return trace;
  case 3: {  // tmin,tmax <- f6,f7, then ARM the slot.
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!trace->tmask.test(t)) continue;
      auto& lregs = wregs.at(t);
      lregs[VX_RT_T_MIN] = fp_src_bits(trace, 0, t);
      lregs[VX_RT_T_MAX] = fp_src_bits(trace, 1, t);
    }
    // ARM: build + send the RtuReq (bus full => retry uop 3 idempotently;
    // the slot was already latched at uop 0). Mirrors Phase-1 process_trace's
    // body, but reads the scene from the staged config rather than rs1.
    if (req_out_.full())
      return nullptr;
    int32_t slot = trace2_slot_.at(wid);
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
      auto& lregs = wregs.at(t);
      req.scene_root[t] = trace2_scene_.at(wid)[t];
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
    }
    req.tmask_bits = bits;
    req_out_.send(req);
    trace2_slot_.at(wid) = -1;
    DT(3, "rtu-trace2 arm: core=" << core_->id() << ", wid=" << wid
         << ", slot=" << slot << ", tmask=0x" << std::hex << bits << std::dec);
    return trace;
  }
  default:
    std::abort();
  }
}

instr_trace_t* RtuUnit::process_getw_uop(instr_trace_t* trace, uint32_t uop, bool is_float) {
  // ISA v2.1 windowed read (§5.5): uop reads regfile slot (start + uop) for each
  // active lane into the uop's dst — FP (NaN-boxed) for GETWF, GP (raw) for
  // GETW. The window streams as one fetched macro-op. Synchronous: the regfile
  // is already staged (by a callback yield's apply_callback_payload, or by the
  // WAIT2 block's terminal when vx_rt_wait2 chains it on the status word).
  auto args = std::get<IntrRtuArgs>(trace->instr_ptr->get_args());
  uint32_t slot = args.slot + uop;
  if (slot >= VX_RT_SLOT_COUNT)
    return trace;  // out-of-range window — leave dst unwritten
  auto& wregs = regfile_.at(trace->wid);
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    uint32_t bits = wregs.at(t).at(slot);
    if (is_float)
      trace->dst_data[t].u64 = nan_box32(bits);
    else
      trace->dst_data[t].u = bits;
  }
  return trace;
}
