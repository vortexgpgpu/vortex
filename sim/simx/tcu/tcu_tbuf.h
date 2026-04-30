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

// TCU tile-buffer subsystem.
//
// Owns the A / B / metadata operand line caches for WGMMA and arbitrates
// their LMEM traffic onto a single LMEM port pair.
//
//   abuf × Q   per-block A-tile line cache (one per warp lane)
//   mbuf × Q   per-block sparse-metadata line cache (sparse SS only)
//   bbuf × 1   shared B-tile line cache (TB-shared across all Q blocks)
//
// All line caches are address-keyed. The consumer (TcuUnit) plans the set
// of line addresses each cache should fetch for the current WGMMA, waits
// for the corresponding `ready_*()` to clear, then extracts operand bytes
// from the resident `mem_block_t` payloads via `read_*()`.
class TcuTbuf : public SimObject<TcuTbuf> {
public:
  using Ptr = std::shared_ptr<TcuTbuf>;

  // Single external LMEM port pair. Internal arbitration fans 2Q+1 sources
  // (abuf×Q, mbuf×Q, bbuf×1) into one bank-row request channel.
  SimChannel<MemReq> lmem_req_out;
  SimChannel<MemRsp> lmem_rsp_in;

  TcuTbuf(const SimContext& ctx, const char* name);
  virtual ~TcuTbuf();

  // Plan/query/read API per role. `b` indexes the per-block lane.
  void plan_a(uint32_t b, const std::vector<uint64_t>& line_addrs);
  void plan_m(uint32_t b, const std::vector<uint64_t>& line_addrs);
  void plan_b(const std::vector<uint64_t>& line_addrs);

  bool ready_a(uint32_t b) const;
  bool ready_m(uint32_t b) const;
  bool ready_b() const;

  std::shared_ptr<mem_block_t> read_a(uint32_t b, uint64_t line_addr) const;
  std::shared_ptr<mem_block_t> read_m(uint32_t b, uint64_t line_addr) const;
  std::shared_ptr<mem_block_t> read_b(uint64_t line_addr) const;

  void invalidate_a(uint32_t b);
  void invalidate_m(uint32_t b);
  void invalidate_b();

  // Total LMEM port-cycles issued since last reset (perf counter).
  uint64_t reads() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<TcuTbuf>;
};

} // namespace vortex
