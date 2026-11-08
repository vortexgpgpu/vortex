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

#include "types.h"
#include "local_mem.h"

namespace vortex {

// TCU tile-buffer subsystem.
//
// Holds the WGMMA operand storage and fetches it from LMEM:
//
//   abuf × Q   per-block A k-stripe
//   bbuf × 1   B bank row, shared by all Q blocks
//   cbuf × 1   rows of a tensor-job TMEM copy (TMEM builds)
//
// The consumer (TcuUnit) owns the refill keys: it decides when a buffer
// refills and hands over the refill as a list of LMEM reads. Each read is
// one bank-row request on the hardware port and covers one or more
// mem_block lines here, sent together on parallel ports. A buffer keeps one
// read in flight (the B buffer may keep more); buffers share the port at fixed priority, abuf[0] first
// and cbuf last.
//
// Sparse metadata is preloaded into the TcuUnit's per-warp `sparse_meta_`
// SRAM via TCU_LD ahead of the MMA dispatch; it does not flow through here.
class TcuTbuf : public SimObject<TcuTbuf> {
public:
  using Ptr = std::shared_ptr<TcuTbuf>;

  // A bank row wider than a mem_block is read as same-cycle block reads on
  // adjacent LMEM DMA channels.
  static constexpr uint32_t LMEM_PORTS = LocalMem::DMA_PORTS;

  // Lines of one bank-row read.
  using LmemRead = std::vector<uint64_t>;

  std::vector<SimChannel<MemReq>> lmem_req_out;
  std::vector<SimChannel<MemRsp>> lmem_rsp_in;

  TcuTbuf(const SimContext& ctx, const char* name);
  virtual ~TcuTbuf();

  // Drop the buffer's contents and fetch `reads` in order. `b` indexes the
  // per-block A buffer.
  void fill_a(uint32_t b, std::vector<LmemRead> reads);
  // max_inflight: reads the B buffer keeps outstanding (a tensor op's dense
  // K-major block issues its row reads back to back).
  void fill_b(std::vector<LmemRead> reads, uint32_t max_inflight = 1);
  // Fetch `reads` after any in progress, keeping the buffer's contents.
  void prefetch_b(std::vector<LmemRead> reads, uint32_t max_inflight = 1);

  bool filling_a(uint32_t b) const;
  bool filling_b() const;

#ifdef VX_CFG_TCU_TMEM_ENABLE
  void fill_c(std::vector<LmemRead> reads);
  bool filling_c() const;
  std::shared_ptr<mem_block_t> read_c(uint64_t line_addr) const;
#endif

  std::shared_ptr<mem_block_t> read_a(uint32_t b, uint64_t line_addr) const;
  std::shared_ptr<mem_block_t> read_b(uint64_t line_addr) const;

  // Drop the buffer's contents and abandon a fill in progress.
  void invalidate_a(uint32_t b);
  void invalidate_b();

  // Bank-row reads issued since the last reset (perf counter).
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
