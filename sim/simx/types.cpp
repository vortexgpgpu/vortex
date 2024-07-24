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

LocalMemDemux::LocalMemDemux(
  const SimContext& ctx,
  const char* name,
  uint32_t delay
) : SimObject<LocalMemDemux>(ctx, name)
  , ReqIn(this)
  , RspIn(this)
  , ReqLmem(this)
  , RspLmem(this)
  , ReqDC(this)
  , RspDC(this)
  , delay_(delay)
{}

void LocalMemDemux::reset() {}

void LocalMemDemux::tick() {
  // process incoming responses
  if (!RspLmem.empty()) {
    auto& out_rsp = RspLmem.front();
    DT(4, this->name() << "-" << out_rsp);
    RspIn.push(out_rsp, 1);
    RspLmem.pop();
  }
  if (!RspDC.empty()) {
    auto& out_rsp = RspDC.front();
    DT(4, this->name() << "-" << out_rsp);
    RspIn.push(out_rsp, 1);
    RspDC.pop();
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.front();

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

    if (!out_dc_req.mask.none()) {
      ReqDC.push(out_dc_req, delay_);
      DT(4, this->name() << "-" << out_dc_req);
    }

    if (!out_lmem_req.mask.none()) {
      ReqLmem.push(out_lmem_req, delay_);
      DT(4, this->name() << "-" << out_lmem_req);
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
  , RspIn(this)
  , ReqOut(num_inputs, this)
  , RspOut(num_inputs, this)
  , delay_(delay)
{}

void LsuMemAdapter::reset() {}

void LsuMemAdapter::tick() {
  uint32_t input_size = ReqOut.size();

  // process incoming responses
  for (uint32_t i = 0; i < input_size; ++i) {
    if (RspOut.at(i).empty())
      continue;
    auto& out_rsp = RspOut.at(i).front();
    DT(4, this->name() << "-" << out_rsp);

    // build memory response
    LsuRsp in_rsp(input_size);
    in_rsp.mask.set(i);
    in_rsp.tag = out_rsp.tag;
    in_rsp.cid = out_rsp.cid;
    in_rsp.uuid = out_rsp.uuid;

    // include other responses with the same tag
    for (uint32_t j = i + 1; j < input_size; ++j) {
      if (RspOut.at(j).empty())
        continue;
      auto& other_rsp = RspOut.at(j).front();
      if (out_rsp.tag == other_rsp.tag) {
        in_rsp.mask.set(j);
        RspOut.at(j).pop();
      }
    }

    // send memory response
    RspIn.push(in_rsp, 1);

    // remove input
    RspOut.at(i).pop();
    break;
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.front();
    assert(in_req.mask.size() == input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
      if (in_req.mask.test(i)) {
        // build memory request
        MemReq out_req;
        out_req.write = in_req.write;
        out_req.addr  = in_req.addrs.at(i);
        out_req.type  = get_addr_type(in_req.addrs.at(i));
        out_req.tag   = in_req.tag;
        out_req.cid   = in_req.cid;
        out_req.uuid  = in_req.uuid;

        // send memory request
        ReqOut.at(i).push(out_req, delay_);
        DT(4, this->name() << "-" << out_req);
      }
    }
    ReqIn.pop();
  }
}