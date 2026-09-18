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
#include <mempool.h>
#include <array>
#include "instr_trace.h"
#include "instr.h"
#include "func_unit.h"
#include "tcu_tbuf.h"

// The TCU metadata SRAM is present when any metadata-consuming mode (MX or
// sparse) is enabled. Internal derived macro — not a VX_CFG_* knob; mirrors
// the hardware derivation.
#if defined(VX_CFG_TCU_MX_ENABLE) || defined(VX_CFG_TCU_SPARSE_ENABLE)
#define TCU_META_ENABLE
#endif

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

class TcuUnit : public FuncUnit<VX_CFG_NUM_TCU_BLOCKS> {
public:
  using Ptr = std::shared_ptr<TcuUnit>;

  static op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

	struct PerfStats {
		uint64_t latency = 0;
		uint64_t tbuf_stalls = 0;      // cycles stalled on TcuTbufA/TcuSharedB readiness
		uint64_t tbuf_cache_hits = 0;  // WGMMA entries with all lines already resident (cross-WGMMA reuse)
		uint64_t lmem_reads = 0;       // sum of TcuTbufA + TcuSharedB LmemReqs issued
#ifdef VX_CFG_TCU_TMEM_ENABLE
		uint64_t umma_instrs = 0;      // UMMA micro-ops executed
		uint64_t tmem_reads = 0;       // TMEM elements read (UMMA C + tmem_ld)
		uint64_t tmem_writes = 0;      // TMEM elements written (UMMA D + tmem_st)
		// The two below are microarchitectural and RTL-only: SimX doesn't
		// model bank arbitration or the RAW interlock, so it reports 0
		// for both.
		uint64_t tmem_bank_stalls = 0;
		uint64_t tmem_hazard_stalls = 0;
#endif

		PerfStats& operator+=(const PerfStats& rhs) {
			this->latency          += rhs.latency;
			this->tbuf_stalls      += rhs.tbuf_stalls;
			this->tbuf_cache_hits  += rhs.tbuf_cache_hits;
			this->lmem_reads       += rhs.lmem_reads;
#ifdef VX_CFG_TCU_TMEM_ENABLE
			this->umma_instrs      += rhs.umma_instrs;
			this->tmem_reads       += rhs.tmem_reads;
			this->tmem_writes      += rhs.tmem_writes;
			this->tmem_bank_stalls += rhs.tmem_bank_stalls;
			this->tmem_hazard_stalls += rhs.tmem_hazard_stalls;
#endif
			return *this;
		}
	};

  TcuUnit(const SimContext &ctx, const char* name, Core* core);
  virtual ~TcuUnit();

#ifdef TCU_META_ENABLE
  // Metadata AGU port: TCU_LD issues load requests through the LSU block-0
  // client port; response fragments return here and accumulate into the
  // metadata SRAM.
  SimChannel<LsuReq> agu_req_out;
  SimChannel<LsuRsp> agu_rsp_in;
#endif

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
	           const std::vector<reg_data_t>& rs2_data,
	           const std::vector<reg_data_t>& rs3_data,
	           std::vector<reg_data_t>& rd_data,
	           bool is_sparse,
	           uint32_t cd_nregs,
	           uint32_t is_a_smem,
	           uint32_t is_setup_uop);

#ifdef VX_CFG_TCU_TMEM_ENABLE
	// Allocate ncols columns of TMEM; returns a handle (base column).
	uint32_t tmem_alloc(uint32_t ncols, int32_t cta_id);

	// Free a TMEM allocation identified by its handle.
	void tmem_dealloc(uint32_t handle, int32_t cta_id, uint32_t wid);

	// Store one element per thread into TMEM. tmem_addr is software-computed
	// (vx_make_tmem_addr(lane_base, handle + local_col)) so the handle's base
	// column is already folded in. Inactive lanes are skipped.
	void tmem_st(uint32_t tmem_addr, const std::vector<reg_data_t>& value_data,
	             const ThreadMask& tmask);

	// Load one element per thread from TMEM. Inactive lanes are skipped.
	void tmem_ld(uint32_t tmem_addr, std::vector<reg_data_t>& rd_data,
	             const ThreadMask& tmask);

	// UMMA: A/B from SMEM via descriptors, C/D accumulator in TMEM addressed 
	// via `handle`.
	void umma(uint32_t wid,
	          uint32_t fmt_s,
	          uint32_t fmt_d,
	          uint32_t step_m,
	          uint32_t step_n,
	          uint32_t step_k,
	          uint32_t umma_nrc,
	          uint32_t a_desc,
	          uint32_t b_desc,
	          uint32_t handle,
	          const std::vector<reg_data_t>& rs1_data,
	          const std::vector<reg_data_t>& rs2_data);
#endif

	// Tile-buffer subsystem (owns abuf×Q + bbuf + LMEM arb).
	// Exposed so that `Core` can bind its single LMEM port pair.
	TcuTbuf::Ptr& tbuf();

	const PerfStats& perf_stats() const;

protected:
  void on_reset() override;
  void on_tick() override;

private:
	class Impl;
	Impl* impl_;

	TcuTbuf::Ptr tbuf_;
};

} // namespace vortex
