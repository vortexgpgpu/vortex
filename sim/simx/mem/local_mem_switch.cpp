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

#include "local_mem_switch.h"

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

void LocalMemSwitch::on_tick() {
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

    [[maybe_unused]] const bool in_is_amo = in_req.is_amo();

    LsuReq out_dc_req(in_req.mask.size());
    out_dc_req.tag   = in_req.tag;
    out_dc_req.cid   = in_req.cid;
    out_dc_req.uuid  = in_req.uuid;
    out_dc_req.wid   = in_req.wid;
    // Op + per-lane tids carry through. Adapter recovers hart_id via
    // make_hart_id(cid, wid, tids[i]) at the dcache boundary.
    out_dc_req.op    = in_req.op;
    out_dc_req.flags = in_req.flags;
    out_dc_req.tids  = in_req.tids;

    LsuReq out_lmem_req(in_req.mask.size());
    out_lmem_req.tag   = in_req.tag;
    out_lmem_req.cid   = in_req.cid;
    out_lmem_req.uuid  = in_req.uuid;
    out_lmem_req.wid   = in_req.wid;
    // The lmem path never carries AMO traffic, but op still flows for
    // consistency (LD/ST distinction at the LMEM bank).
    out_lmem_req.op   = in_req.op;

    for (uint32_t i = 0; i < in_req.mask.size(); ++i) {
      if (in_req.mask.test(i)) {
        auto type = get_addr_type(in_req.addrs.at(i));
        // §3.13: AMO on Shared (LMEM) is unsupported; bail loudly so a
        // future LMEM-AMO mistake doesn't silently route through the
        // LMEM path (which has no AMO machinery).
        assert(!(in_is_amo && type == AddrType::Shared)
               && "AMO on Shared (LMEM) is unsupported in this build");
        if (type == AddrType::Shared) {
          out_lmem_req.mask.set(i);
          out_lmem_req.addrs.at(i)  = in_req.addrs.at(i);
          out_lmem_req.data.at(i)   = in_req.data.at(i);
          out_lmem_req.byteen.at(i) = in_req.byteen.at(i);
        } else {
          out_dc_req.mask.set(i);
          out_dc_req.addrs.at(i)  = in_req.addrs.at(i);
          out_dc_req.data.at(i)   = in_req.data.at(i);
          out_dc_req.byteen.at(i) = in_req.byteen.at(i);
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
