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
#include "mem_block_pool.h"
#include <cstring>

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
  , RspOut(this)
  , ReqOut(this)
  , RspIn(this)
  , input_size_(input_size)
  , output_size_(output_size)
  , output_ratio_(input_size / output_size)
  , pending_rd_reqs_(queue_size)
  , sent_mask_(input_size)
  , line_size_(line_size)
  , delay_(delay)
{}

void MemCoalescer::on_reset() {
  sent_mask_.reset();
}

void MemCoalescer::on_tick() {
  // process outgoing responses
  if (!RspIn.empty()) {
    auto& rsp_in = RspIn.peek();
    auto& entry = pending_rd_reqs_.at(rsp_in.tag);

    BitVector<> rsp_mask(input_size_);
    for (uint32_t o = 0; o < output_size_; ++o) {
      if (!rsp_in.mask.test(o))
        continue;
      for (uint32_t r = 0; r < output_ratio_; ++r) {
        uint32_t i = o * output_ratio_ + r;
        if (entry.mask.test(i))
          rsp_mask.set(i);
      }
    }

    // build memory response — replicate each output-lane data block to all
    // coalesced input lanes (shared_ptr aliasing, no copy)
    LsuRsp out_rsp(input_size_);
    out_rsp.mask = rsp_mask;
    out_rsp.tag = entry.tag;
    out_rsp.cid = rsp_in.cid;
    out_rsp.uuid = rsp_in.uuid;
    for (uint32_t o = 0; o < output_size_; ++o) {
      if (!rsp_in.mask.test(o))
        continue;
      for (uint32_t r = 0; r < output_ratio_; ++r) {
        uint32_t i = o * output_ratio_ + r;
        if (entry.mask.test(i)) {
          out_rsp.data.at(i) = rsp_in.data.at(o);
        }
      }
    }

    // send memory response
    if (RspOut.try_send(out_rsp, 1)) {
      DT(4, this->name() << " mem-rsp: " << rsp_in);

      // track remaining responses
      assert(!entry.mask.none());
      entry.mask &= ~rsp_mask;
      if (entry.mask.none()) {
        // whole response received, release tag
        pending_rd_reqs_.release(rsp_in.tag);
      }
      RspIn.pop();
    }
  }

  // process incoming requests
  if (ReqIn.empty())
    return;

  // check request output backpressure
  if (ReqOut.full())
    return;

  auto& in_req = ReqIn.peek();
  assert(in_req.mask.size() == input_size_);
  assert(!in_req.mask.none());

  // ensure we can allocate a response tag
  if (pending_rd_reqs_.full()) {
    DT(4, this->name() << " queue-full: " << in_req);
    return;
  }

  uint64_t addr_mask = ~uint64_t(line_size_-1);

  const bool in_is_amo = in_req.is_amo();

  BitVector<> out_mask(output_size_);
  std::vector<uint64_t> out_addrs(output_size_);
  std::vector<std::shared_ptr<mem_block_t>> out_data(output_size_);
  std::vector<uint64_t> out_byteen(output_size_, 0);
  std::vector<amo_req_t> out_amo(output_size_);

  BitVector<> cur_mask(input_size_);

  for (uint32_t o = 0; o < output_size_; ++o) {
    for (uint32_t r = 0; r < output_ratio_; ++r) {
      uint32_t i = o * output_ratio_ + r;
      if (sent_mask_.test(i) || !in_req.mask.test(i))
        continue;

      uint64_t seed_addr = in_req.addrs.at(i) & addr_mask;
      cur_mask.set(i);

      // RVA gives no commutativity guarantee across AMO operands —
      // do not coalesce AMO lanes that share a line. Each AMO lane
      // emits its own bank request (proposal §D6). For non-AMO,
      // matching addresses still coalesce as before.
      if (!in_is_amo) {
        for (uint32_t s = r + 1; s < output_ratio_; ++s) {
          uint32_t j = o * output_ratio_ + s;
          if (sent_mask_.test(j) || !in_req.mask.test(j))
            continue;
          uint64_t match_addr = in_req.addrs.at(j) & addr_mask;
          if (match_addr == seed_addr) {
            cur_mask.set(j);
          }
        }
      }

      if (in_is_amo) {
        // Carry this lane's full AMO sideband (op, width, rhs, hart_id)
        // through the output slot. No coalescing across lanes.
        if (i < in_req.amo.size()) {
          out_amo.at(o) = in_req.amo.at(i);
        }
      }

      // For writes, merge per-lane data + byteen into the coalesced block.
      if (in_req.write) {
        std::shared_ptr<mem_block_t> merged;
        uint64_t merged_byteen = 0;
        for (uint32_t s = r; s < output_ratio_; ++s) {
          uint32_t j = o * output_ratio_ + s;
          if (!cur_mask.test(j) || !in_req.data.at(j))
            continue;
          if (!merged) {
            merged = make_mem_block();
            std::memset(merged->data(), 0, merged->size());
          }
          uint64_t lane_be = in_req.byteen.at(j);
          for (uint32_t b = 0; b < MEM_BLOCK_SIZE; ++b) {
            if (lane_be & (1ull << b)) {
              (*merged)[b] = (*in_req.data.at(j))[b];
            }
          }
          merged_byteen |= lane_be;
        }
        out_data.at(o) = merged;
        out_byteen.at(o) = merged_byteen;
      }

      out_mask.set(o);
      // AMOs need the byte-level address downstream so the bank can
      // place the RMW result at the correct offset within the line.
      // Non-AMO requests stay line-aligned (no semantic change).
      out_addrs.at(o) = in_is_amo ? in_req.addrs.at(i) : seed_addr;
      break;
    }
  }

  assert(!out_mask.none());

  uint32_t tag = 0;
  if (!in_req.write || in_is_amo) {
    // Allocate a response tag for read requests and AMOs (which always
    // return rd). Without the AMO branch the response would route through
    // the write path and the LSU MSHR replay would never fire.
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
  out_req.data = std::move(out_data);
  out_req.byteen = std::move(out_byteen);
  out_req.wid = in_req.wid;
  if (in_is_amo) {
    out_req.amo = std::move(out_amo);
  }

  // send memory request
  ReqOut.send(out_req, delay_);
  DT(4, this->name() << " mem-req: coalesced=" << cur_mask.count() << ", " << out_req);

  // track partial responses
  perf_stats_.misses += (cur_mask.count() != in_req.mask.count());

  // update sent mask
  sent_mask_ |= cur_mask;
  if (sent_mask_ == in_req.mask) {
    ReqIn.pop();
    sent_mask_.reset();
  }
}

const MemCoalescer::PerfStats& MemCoalescer::perf_stats() const {
  return perf_stats_;
}