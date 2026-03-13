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
    uint16_t M;
    uint16_t N;
    uint16_t K;
    uint8_t  fmt_s;
    uint8_t  fmt_d;
    uint8_t  flags;
    uint8_t  reserved0;
    uint16_t reserved1;
    uint32_t reserved2;
  };

  static_assert(sizeof(Desc) == 64, "DTensorCore::Desc must be 64 bytes");

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

  // Internal state for iterating through tiles
  uint32_t tile_m_idx_ = 0;
  uint32_t tile_n_idx_ = 0;
  uint32_t tile_k_idx_ = 0;
  uint32_t tiles_m_ = 1;
  uint32_t tiles_n_ = 1;
  uint32_t tiles_k_ = 1;

  // Aggregate mem request counter for descriptor
  uint64_t total_op_reqs_ = 0;
  uint64_t total_out_reqs_ = 0;

  void init_tile_state_();
  bool advance_output_tile_();

  uint64_t tile_ptrA_() const;
  uint64_t tile_ptrB_() const;
  uint64_t tile_ptrC_() const;
  uint64_t tile_ptrD_() const;

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
