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

#include "lsu_mem_adapter.h"

using namespace vortex;

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
      MemReq mr{ req.addrs.at(0), req.write, AddrType::Global, req.tag, req.cid, req.uuid };
      mr.data   = req.data.at(0);
      mr.byteen = req.byteen.at(0);
      return mr;
    });
    RspIn.at(0).bind(&RspOut, [](const MemRsp& rsp) {
      LsuRsp lsuRsp(1);
      lsuRsp.mask.set(0);
      lsuRsp.tag = rsp.tag;
      lsuRsp.cid = rsp.cid;
      lsuRsp.uuid = rsp.uuid;
      lsuRsp.data.at(0) = rsp.data;
      return lsuRsp;
    });
  }
}

void LsuMemAdapter::on_reset() {
  pending_mask_.reset();
}

void LsuMemAdapter::on_tick() {
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
    out_rsp.data.at(i) = rsp_in.data;

    // merge other responses with the same tag
    for (uint32_t j = i + 1; j < input_size; ++j) {
      if (RspIn.at(j).empty())
        continue;
      auto& other_rsp = RspIn.at(j).peek();
      if (rsp_in.tag == other_rsp.tag) {
        out_rsp.mask.set(j);
        out_rsp.data.at(j) = other_rsp.data;
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
      out_req.write  = in_req.write;
      out_req.addr   = in_req.addrs.at(i);
      out_req.type   = get_addr_type(in_req.addrs.at(i));
      out_req.tag    = in_req.tag;
      out_req.cid    = in_req.cid;
      out_req.uuid   = in_req.uuid;
      out_req.data   = in_req.data.at(i);
      out_req.byteen = in_req.byteen.at(i);

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
