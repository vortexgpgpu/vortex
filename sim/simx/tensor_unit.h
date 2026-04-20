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
#include <mempool.h>
#include "instr_trace.h"
#include "instr.h"

namespace vortex {

class Core;

///////////////////////////////////////////////////////////////////////////////

// Micro-op generator for TCU instructions (WMMA/WGMMA).
// Owned by each per-warp Sequencer.
class TcuUopGen {
public:
  TcuUopGen(PoolAllocator<Instr, 64>& pool) : pool_(pool) {}

  // Returns total micro-op count for a macro instruction (>1 means macro-op).
  static uint32_t uop_count(const Instr& instr);

  // Generate micro-op Instr at uop_index for the given macro instruction.
  Instr::Ptr get(const Instr& macro_instr, uint32_t uop_index);

private:
  PoolAllocator<Instr, 64>& pool_;
};

///////////////////////////////////////////////////////////////////////////////

class TensorUnit : public SimObject<TensorUnit> {
public:

  static op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

  struct ExeTraceData : public ITraceData {
    using Ptr = std::shared_ptr<ExeTraceData>;
    bool is_last_k = true;        // false for non-last K-steps (suppress rd writeback)
    int  fetch_delay = 0;         // tile buffer fetch cycles (v2 timing model)
    bool tbuf_cache_hit = false;  // B tile was reused from tile buffer cache
  };

	struct PerfStats {
		uint64_t latency = 0;
		uint64_t tbuf_stalls = 0;      // cycles stalled waiting for tbuf data
		uint64_t tbuf_cache_hits = 0;  // B tile reuse from tile buffer cache
		uint64_t lmem_reads = 0;   // tile buffer local memory reads

		PerfStats& operator+=(const PerfStats& rhs) {
			this->latency          += rhs.latency;
			this->tbuf_stalls      += rhs.tbuf_stalls;
			this->tbuf_cache_hits  += rhs.tbuf_cache_hits;
			this->lmem_reads       += rhs.lmem_reads;
			return *this;
		}
	};

  std::vector<SimChannel<instr_trace_t*>> Inputs;
	std::vector<SimChannel<instr_trace_t*>> Outputs;

  TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core);
  virtual ~TensorUnit();

  virtual void reset();

  virtual void tick();

	void wmma(uint32_t wid,
	          uint32_t fmt_s,
	          uint32_t fmt_d,
	          uint32_t step_m,
	          uint32_t step_n,
	          uint32_t step_k,
	          const std::vector<reg_data_t>& rs1_data,
	          const std::vector<reg_data_t>& rs2_data,
	          const std::vector<reg_data_t>& rs3_data,
	          std::vector<reg_data_t>& rd_data,
	          ExeTraceData* trace_data,
	          bool is_sparse);

	void wgmma(uint32_t wid,
	           uint32_t fmt_s,
	           uint32_t fmt_d,
	           uint32_t step_m,
	           uint32_t step_n,
	           uint32_t step_k,
	           uint32_t a_desc,
	           uint32_t b_desc,
	           const std::vector<reg_data_t>& rs1_data,
	           const std::vector<reg_data_t>& rs3_data,
	           std::vector<reg_data_t>& rd_data,
	           ExeTraceData* trace_data,
	           bool is_sparse,
	           uint32_t cd_nregs,
	           uint32_t is_a_smem);

	void meta_store(uint32_t wid,
					uint32_t fmt_s,
					uint32_t col_idx,
					uint32_t meta_kind,
					const std::vector<reg_data_t>& rs1_data,
					ExeTraceData* trace_data);

	const PerfStats& perf_stats() const;

private:
	class Impl;
	Impl* impl_;
};

} // namespace vortex
