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
#include "types.h"

namespace vortex {

// Per-block A-tile line cache for WGMMA.
//
// Holds the active k-stripe's `M_STEPS × lines_per_row` lines for one warp.
// Lines are fetched via a dedicated LMEM port pair
// (`lmem_req_out` / `lmem_rsp_in`); responses carry the bank-row bytes that
// the consumer reads via `read_line()`.
//
// The cache is address-keyed: the consumer (TcuUnit) computes the byte
// address of each operand it needs, calls `plan()` with the set of line
// addresses for the current uop's k-stripe, waits for `ready()`, then
// extracts operand bytes from the resident `mem_block_t` payloads.
class TcuTbufA : public SimObject<TcuTbufA> {
public:
  using Ptr = std::shared_ptr<TcuTbufA>;

  SimChannel<MemReq> lmem_req_out;
  SimChannel<MemRsp> lmem_rsp_in;

  TcuTbufA(const SimContext& ctx, const char* name);
  virtual ~TcuTbufA();

  // Request the given line addresses; lines already resident or in-flight
  // are skipped. Lines outside `line_addrs` are evicted (k-stripe rollover).
  void plan(const std::vector<uint64_t>& line_addrs);

  // True when no MemReq is outstanding for any planned line.
  bool ready() const;

  // Resident line for line-aligned `line_addr`, or nullptr if not resident.
  std::shared_ptr<mem_block_t> read_line(uint64_t line_addr) const;

  // Drop all state (entry-on-WGMMA-end / unit reset).
  void invalidate();

  // Number of MemReqs issued since the last reset (perf counter).
  uint64_t reads() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<TcuTbufA>;
};

} // namespace vortex
