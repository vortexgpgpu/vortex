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
  // Snapshot all active lanes' ray descriptors into one RtuReq and submit
  // to RtuCore. Phase 1 returns handle=0 since one ray per lane at a time.
  if (req_out_.full()) {
    return nullptr;
  }
  auto& wregs = regfile_.at(trace->wid);
  auto& rs1 = trace->src_data[0];

  RtuReq req;
  req.kind     = RtuReqKind::TRACE_NEW;
  req.uuid     = trace->uuid;
  req.tag      = uint32_t(trace->uuid);
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
    // Phase 1: handle = 0 (one outstanding ray per (warp,lane)).
    trace->dst_data[t].u = 0;
  }
  req.tmask_bits = bits;
  req_out_.send(req);
  DT(3, "rtu-trace submit: core=" << core_->id() << ", wid=" << trace->wid
       << ", tmask=0x" << std::hex << bits << std::dec);
  return trace;
}

instr_trace_t* RtuUnit::process_wait(instr_trace_t* trace, uint32_t /*block_id*/) {
  // Phase 1/2 collapsed-trace+wait model: TRACE parks in RtuCore, and
  // when its TERMINAL rsp arrives SfuUnit writes the per-lane status
  // word into TRACE's dst_data — i.e. TRACE delivers the status via
  // its writeback. WAIT therefore needs to be a *pass-through*: copy
  // rs1 (= the TRACE's rd = status) into rd, so the kernel idiom
  //   uint32_t h   = vx_rt_trace(scene);  // delivers status
  //   uint32_t sts = vx_rt_wait(h);        // forwards it to sts
  // sees status, not zero. (A previous revision clobbered dst_data
  // with zeros and the ACCEPT path only "passed" because
  // VX_RT_STS_DONE_HIT happens to equal 0.)
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (trace->tmask.test(t)) {
      trace->dst_data[t].u = static_cast<uint32_t>(trace->src_data[0].at(t).u);
    } else {
      trace->dst_data[t].u = 0;
    }
  }
  return trace;
}

instr_trace_t* RtuUnit::process_cb_ret(instr_trace_t* trace, uint32_t block_id) {
  // Phase 2: vx_rt_cb_ret releases the warp's parked context in the
  // RtuCore with per-lane action codes. Emit a CB_ACTION packet over the
  // bus; the RtuCore matches it to the AWAIT_CALLBACK slot for this
  // (warp,uuid) tuple and resumes traversal.
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
  uint32_t bits = 0;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!trace->tmask.test(t)) continue;
    bits |= (1u << t);
    // rs1 holds the action (ACCEPT/IGNORE/TERMINATE).
    req.cb_action[t] = static_cast<uint32_t>(trace->src_data[0].at(t).u);
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
  // Stage candidate-hit attrs + cb_type into the RTU regs for the lanes
  // whose rays yielded, so the dispatcher's vx_rt_get sees the right
  // payload. Only the yielded lanes (cb_active_mask) are touched.
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
  }
}
