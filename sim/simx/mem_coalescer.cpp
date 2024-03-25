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
  , ReqIn(input_size, this)
  , RspIn(input_size, this)
  , ReqOut(output_size, this)
  , RspOut(output_size, this)
  , pending_rd_reqs_(queue_size)
  , line_size_(line_size)
  , delay_(delay)
{}

void MemCoalescer::reset() {
  last_index_ =  0;
  sent_mask_.reset();
}

void MemCoalescer::tick() {    
  uint32_t I = ReqIn.size();
  uint32_t O = ReqOut.size();

  // process incoming responses
  for (uint32_t o = 0; o < O; ++o) {
    if (RspOut.at(o).empty())
      continue;
    auto& mem_rsp = RspOut.at(o).front();
    DT(3, this->name() << "-" << mem_rsp);
    auto& entry = pending_rd_reqs_.at(mem_rsp.tag);
    for (uint32_t i = 0; i < I; ++i) {
      if (entry.mask.test(i)) {
        MemRsp rsp(mem_rsp);
        rsp.tag = entry.tag;
        RspIn.at(i).push(rsp, 1);
      }
    }
    pending_rd_reqs_.release(mem_rsp.tag);
    RspOut.at(o).pop();
  }

  // process incoming requests
  uint64_t addr_mask = ~uint64_t(line_size_-1);
  bool completed = true;
  for (uint32_t i = last_index_; i < I; ++i) {
    if (sent_mask_.test(i) || ReqIn.at(i).empty())
      continue;

    auto& seed = ReqIn.at(i).front();

    // ensure we can allocate a response tag      
    if (!seed.write && pending_rd_reqs_.full()) {
      DT(4, "*** " << this->name() << "-queue-full: " << seed);
      last_index_ = i;
      completed = false;
      break;
    }

    std::bitset<64> mask(0);      
    mask.set(i);      

    // coalesce matching requests      
    uint64_t seed_addr = seed.addr & addr_mask;
    for (uint32_t j = i + 1; j < I; ++j) {
      if (sent_mask_.test(j) || ReqIn.at(j).empty())
        continue;
      auto& match = ReqIn.at(j).front();
      uint64_t match_addr = match.addr & addr_mask;
      if (match_addr == seed_addr) {
        mask.set(j);
        ReqIn.at(j).pop();   
      }
    }

    uint32_t tag = 0;
    if (!seed.write) {
      tag = pending_rd_reqs_.allocate(pending_req_t{seed.tag, mask});
    }

    MemReq mem_req{seed};
    mem_req.tag = tag;      
    DT(3, this->name() << "-" << mem_req << ", coalesced=" << mask.count());        
    uint32_t c = i % O;
    ReqOut.at(c).push(mem_req, delay_);
    ReqIn.at(i).pop();

    sent_mask_ |= mask;     
  }

  if (completed) {
    last_index_ = 0;
    sent_mask_.reset();
  }
}