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

#include "types.h"

using namespace vortex;

LocalMemSwitch::LocalMemSwitch(
  const SimContext& ctx,
  const char* name,
  uint32_t delay
) : SimObject<LocalMemSwitch>(ctx, name)
  , ReqIn(this)
  , RspOut(this)
  , ReqOutLmem(this)
  , RspInLmem(this)
  , ReqOutDC(this)
  , RspInDC(this)
  , delay_(delay)
{}

void LocalMemSwitch::reset() {}

void LocalMemSwitch::tick() {
  // process outgoing responses
  if (!RspInLmem.empty()) {
    auto& out_rsp = RspInLmem.peek();
    if (RspOut.try_send(out_rsp, 1)) {
      DT(4, this->name() << " lmem-rsp: " << out_rsp);
      RspInLmem.pop();
    }
  }
  if (!RspInDC.empty()) {
    auto& out_rsp = RspInDC.peek();
    if (RspOut.try_send(out_rsp, 1)) {
      DT(4, this->name() << " dc-rsp: " << out_rsp);
      RspInDC.pop();
    }
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.peek();

    LsuReq out_dc_req(in_req.mask.size());
    out_dc_req.write = in_req.write;
    out_dc_req.tag   = in_req.tag;
    out_dc_req.cid   = in_req.cid;
    out_dc_req.uuid  = in_req.uuid;

    LsuReq out_lmem_req(out_dc_req);

    for (uint32_t i = 0; i < in_req.mask.size(); ++i) {
      if (in_req.mask.test(i)) {
        auto type = get_addr_type(in_req.addrs.at(i));
        if (type == AddrType::Shared) {
          out_lmem_req.mask.set(i);
          out_lmem_req.addrs.at(i) = in_req.addrs.at(i);
        } else {
          out_dc_req.mask.set(i);
          out_dc_req.addrs.at(i) = in_req.addrs.at(i);
        }
      }
    }

    bool send_to_dc = !out_dc_req.mask.none();
    bool send_to_lmem = !out_lmem_req.mask.none();

    // check DC backpressure
    if (send_to_dc && ReqOutDC.full())
      return; // stall

    // check LMem backpressure
    if (send_to_lmem && ReqOutLmem.full())
      return; // stall

    if (send_to_dc) {
      ReqOutDC.send(out_dc_req, delay_);
      DT(4, this->name() << " dc-req: " << out_dc_req);
    }

    if (send_to_lmem) {
      ReqOutLmem.send(out_lmem_req, delay_);
      DT(4, this->name() << " lmem-req: " << out_lmem_req);
    }
    ReqIn.pop();
  }
}

///////////////////////////////////////////////////////////////////////////////

LsuMemAdapter::LsuMemAdapter(
  const SimContext& ctx,
  const char* name,
  uint32_t num_inputs,
  uint32_t delay
) : SimObject<LsuMemAdapter>(ctx, name)
  , ReqIn(this)
  , RspOut(this)
  , ReqOut(num_inputs, this)
  , RspIn(num_inputs, this)
  , delay_(delay)
  , pending_mask_(num_inputs)
{
  assert(num_inputs > 0);
  if (num_inputs == 1) {
    // bypass mode
    ReqIn.bind(&ReqOut.at(0), [](const LsuReq& req) {
      return MemReq{ req.addrs.at(0), req.write, AddrType::Global, req.tag, req.cid, req.uuid };
    });
    RspIn.at(0).bind(&RspOut, [](const MemRsp& rsp) {
      LsuRsp lsuRsp(1);
      lsuRsp.mask.set(0);
      lsuRsp.tag = rsp.tag;
      lsuRsp.cid = rsp.cid;
      lsuRsp.uuid = rsp.uuid;
      return lsuRsp;
    });
  }
}

void LsuMemAdapter::reset() {}

void LsuMemAdapter::tick() {
  uint32_t input_size = ReqOut.size();
  if (input_size == 1)
    return;

  // process outgoing responses
  for (uint32_t i = 0; i < input_size; ++i) {
    if (RspIn.at(i).empty())
      continue;

    // check output backpressure
    if (RspOut.full())
      continue;

    auto& rsp_in = RspIn.at(i).peek();

    // build memory response
    LsuRsp out_rsp(input_size);
    out_rsp.mask.set(i);
    out_rsp.tag = rsp_in.tag;
    out_rsp.cid = rsp_in.cid;
    out_rsp.uuid = rsp_in.uuid;

    // merge other responses with the same tag
    for (uint32_t j = i + 1; j < input_size; ++j) {
      if (RspIn.at(j).empty())
        continue;
      auto& other_rsp = RspIn.at(j).peek();
      if (rsp_in.tag == other_rsp.tag) {
        out_rsp.mask.set(j);
        DT(4, this->name() << "-rsp" << j << ": " << other_rsp);
        RspIn.at(j).pop();
      }
    }

    // send memory response
    RspOut.send(out_rsp, 1);
    
    // remove input
    DT(4, this->name() << "-rsp" << i << ": " << rsp_in);
    RspIn.at(i).pop();
    break;
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.peek();
    assert(in_req.mask.size() == input_size);

    if (pending_mask_.none()) {
      pending_mask_ = in_req.mask;
      assert(!pending_mask_.none()); // should not be empty
    }

    for (uint32_t i = 0; i < input_size; ++i) {
      if (!pending_mask_.test(i))
        continue;

      MemReq out_req;
      out_req.write = in_req.write;
      out_req.addr  = in_req.addrs.at(i);
      out_req.type  = get_addr_type(in_req.addrs.at(i));
      out_req.tag   = in_req.tag;
      out_req.cid   = in_req.cid;
      out_req.uuid  = in_req.uuid;

      if (ReqOut.at(i).try_send(out_req, delay_)) {
        DT(4, this->name() << " req" << i << ": " << out_req);
        pending_mask_.reset(i); // mark lane done
      }
    }

    // Pop only when all required lanes have been sent
    if (pending_mask_.none()) {
      ReqIn.pop();
    }
  }
}
