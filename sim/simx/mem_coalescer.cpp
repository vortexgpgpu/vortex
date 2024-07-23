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

#include "mem_coalescer.h"

using namespace vortex;

MemCoalescer::MemCoalescer(
  const SimContext& ctx,
  const char* name,
  uint32_t input_size,
  uint32_t output_size,
  uint32_t line_size,
  uint32_t queue_size,
  uint32_t delay
) : SimObject<MemCoalescer>(ctx, name)
  , ReqIn(this)
  , RspIn(this)
  , ReqOut(this)
  , RspOut(this)
  , input_size_(input_size)
  , output_size_(output_size)
  , output_ratio_(input_size / output_size)
  , pending_rd_reqs_(queue_size)
  , sent_mask_(input_size)
  , line_size_(line_size)
  , delay_(delay)
{}

void MemCoalescer::reset() {
  sent_mask_.reset();
}

void MemCoalescer::tick() {
  // process incoming responses
  if (!RspOut.empty()) {
    auto& out_rsp = RspOut.front();
    DT(4, this->name() << "-" << out_rsp);
    auto& entry = pending_rd_reqs_.at(out_rsp.tag);

    BitVector<> rsp_mask(input_size_);
    for (uint32_t o = 0; o < output_size_; ++o) {
      if (!out_rsp.mask.test(o))
        continue;
      for (uint32_t r = 0; r < output_ratio_; ++r) {
        uint32_t i = o * output_ratio_ + r;
        if (entry.mask.test(i))
          rsp_mask.set(i);
      }
    }

    // build memory response
    LsuRsp in_rsp(input_size_);
    in_rsp.mask = rsp_mask;
    in_rsp.tag = entry.tag;
    in_rsp.cid = out_rsp.cid;
    in_rsp.uuid = out_rsp.uuid;

    // send memory response
    RspIn.push(in_rsp, 1);

    // track remaining responses
    assert(!entry.mask.none());
		entry.mask &= ~rsp_mask;
		if (entry.mask.none()) {
      // whole response received, release tag
			pending_rd_reqs_.release(out_rsp.tag);
		}
    RspOut.pop();
  }

  // process incoming requests
  if (ReqIn.empty())
    return;

  auto& in_req = ReqIn.front();
  assert(in_req.mask.size() == input_size_);
  assert(!in_req.mask.none());

  // ensure we can allocate a response tag
  if (pending_rd_reqs_.full()) {
    DT(4, "*** " << this->name() << "-queue-full: " << in_req);
    return;
  }

  uint64_t addr_mask = ~uint64_t(line_size_-1);

  BitVector<> out_mask(output_size_);
  std::vector<uint64_t> out_addrs(output_size_);

  BitVector<> cur_mask(input_size_);

  for (uint32_t o = 0; o < output_size_; ++o) {
    for (uint32_t r = 0; r < output_ratio_; ++r) {
      uint32_t i = o * output_ratio_ + r;
      if (sent_mask_.test(i) || !in_req.mask.test(i))
        continue;

      uint64_t seed_addr = in_req.addrs.at(i) & addr_mask;
      cur_mask.set(i);

      // coalesce matching requests
      for (uint32_t s = r + 1; s < output_ratio_; ++s) {
        uint32_t j = o * output_ratio_ + s;
        if (sent_mask_.test(j) || !in_req.mask.test(j))
          continue;
        uint64_t match_addr = in_req.addrs.at(j) & addr_mask;
        if (match_addr == seed_addr) {
          cur_mask.set(j);
        }
      }

      out_mask.set(o);
      out_addrs.at(o) = seed_addr;
      break;
    }
  }

  assert(!out_mask.none());

  uint32_t tag = 0;
  if (!in_req.write) {
    // allocate a response tag for read requests
    tag = pending_rd_reqs_.allocate(pending_req_t{in_req.tag, cur_mask});
  }

  // build memory request
  LsuReq out_req{output_size_};
  out_req.mask = out_mask;
  out_req.tag = tag;
  out_req.write = in_req.write;
  out_req.addrs = out_addrs;
  out_req.cid = in_req.cid;
  out_req.uuid = in_req.uuid;

  // send memory request
  ReqOut.push(out_req, delay_);
  DT(4, this->name() << "-" << out_req << ", coalesced=" << cur_mask.count());

  // update sent mask
  sent_mask_ |= cur_mask;
  if (sent_mask_ == in_req.mask) {
    ReqIn.pop();
    sent_mask_.reset();
  }
}