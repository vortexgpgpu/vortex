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

#pragma once

#include <simobject.h>
#include "mem_sim.h"
#include "arch.h"
#include "dcrs.h"
#include "mem.h"
#include <vector>

namespace vortex {

class Cluster;

class DTensorCore : public SimObject<DTensorCore> {
public:
  struct Desc {
    uint64_t ptrA;
    uint64_t ptrB;
    uint64_t ptrC;
    uint64_t ptrD;
    // Leading dimensions in number of elements (not bytes) for different element size
    uint32_t ldmA;
    uint32_t ldmB;
    uint32_t ldmC;
    uint32_t ldmD;
    uint32_t fmt_s;   // input format (A,B)
    uint32_t fmt_d;   // output/acc format (C,D)
    uint32_t flags;   // bit0: C_is_zero (ignore ptrC)
  };

  SimChannel<MemReq> mem_req_out;
  SimChannel<MemRsp> mem_rsp_in;

  DTensorCore(const SimContext& ctx,
                   const char* name,
                   Cluster* cluster,
                   const Arch& arch,
                   const DCRS& dcrs);

  ~DTensorCore();

  void attach_ram(RAM* ram);

  void start(uint64_t desc_addr);

  uint32_t poll() const;

  void reset();

  void tick();

private:
  enum class State {
    IDLE,
    DESC_REQ,
    DESC_WAIT,
    OP_REQ,
    OP_WAIT,
    EXECUTE,
    OUT_REQ,
    OUT_WAIT,
    DONE
  };

  Cluster*  cluster_;
  const Arch& arch_;
  const DCRS& dcrs_;
  RAM*      ram_;

  State     state_;
  bool      busy_;
  bool      done_;

  uint64_t  desc_addr_;
  Desc      desc_;

  uint64_t  tag_alloc_;
  uint64_t  pending_tag_;

  std::vector<float> fragA_; // NUM_THREADS * cfg::NRA
  std::vector<float> fragB_; // NUM_THREADS * cfg::NRB
  std::vector<float> fragC_; // NUM_THREADS * cfg::NRC

  // Cacheline request lists 
  // For calculating # of cache line accesses during operand load and output store
  std::vector<uint64_t> op_req_lines_;
  uint32_t op_req_idx_ = 0;
  std::vector<uint64_t> out_req_lines_;
  uint32_t out_req_idx_ = 0;

  void build_req_lists_();
  bool issue_next_op_req_();
  bool issue_next_out_req_();

  void issue_mem_req(uint64_t addr, bool write);

  void load_desc();
  void load_operands();
  void execute_wmma();
  void store_output();
};

} // namespace vortex
