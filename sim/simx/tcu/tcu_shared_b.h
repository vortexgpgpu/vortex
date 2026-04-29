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

// Shared B-tile line cache for WGMMA, one per `TcuUnit`.
//
// All Q TCU blocks read B from this single buffer — the "M-direction warp
// tiling" production case where every warp shares the B descriptor. One
// LMEM port pair feeds the whole TB; the line cache de-duplicates requests
// across all blocks.
//
// Same address-keyed contract as `TcuTbufA`: caller plans the set of
// needed line addresses, waits for `ready()`, and reads from the resident
// `mem_block_t` payloads.
class TcuSharedB : public SimObject<TcuSharedB> {
public:
  using Ptr = std::shared_ptr<TcuSharedB>;

  SimChannel<MemReq> lmem_req_out;
  SimChannel<MemRsp> lmem_rsp_in;

  TcuSharedB(const SimContext& ctx, const char* name);
  virtual ~TcuSharedB();

  void plan(const std::vector<uint64_t>& line_addrs);
  bool ready() const;
  std::shared_ptr<mem_block_t> read_line(uint64_t line_addr) const;
  void invalidate();
  uint64_t reads() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<TcuSharedB>;
};

} // namespace vortex
