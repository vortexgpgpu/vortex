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
#include "instr.h"
#include "instr_trace.h"
#include <vector>
#include "sparse_cfg.h"

namespace vortex {

class Core;

// Register file type: 8 registers, each with 16 rows x 32 columns of fp32
using SparseRegFile_t = std::vector<std::vector<std::vector<typename sparse::fp32::dtype>>>;

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

class SparseUnit : public SimObject<SparseUnit> {
public:

  struct MemTraceData : public ITraceData {
    using Ptr = std::shared_ptr<MemTraceData>;
    std::vector<std::vector<mem_addr_size_t>> mem_addrs;
    MemTraceData(uint32_t num_threads = 0) : mem_addrs(num_threads) {}
  };

  struct ExeTraceData : public ITraceData {
    using Ptr = std::shared_ptr<ExeTraceData>;
  };

	struct PerfStats {
		uint64_t latency;

		PerfStats()
			: latency(0)
		{}

		PerfStats& operator+=(const PerfStats& rhs) {
			this->latency += rhs.latency;
			return *this;
		}
	};

  std::vector<SimPort<instr_trace_t*>> Inputs;
	std::vector<SimPort<instr_trace_t*>> Outputs;

  SparseUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core);
  virtual ~SparseUnit();

  virtual void reset();

  virtual void tick();

  void load(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, MemTraceData* trace_data);

  void store(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, MemTraceData* trace_data);

	void wmma(uint32_t wid,
			 	    uint32_t fmt_s,
						uint32_t fmt_d,
			 	    uint32_t step_m,
						uint32_t step_n,
	          const std::vector<reg_data_t>& rs1_data,
					  const std::vector<reg_data_t>& rs2_data,
					  const std::vector<reg_data_t>& rs3_data,
					  std::vector<reg_data_t>& rd_data,
					  ExeTraceData* trace_data,
					  const uint32_t* metadata = nullptr);

  void tile_gemm_t(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_treg);
  void tile_gemm_u(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_ureg, uint32_t meta_reg);
  void tile_gemm_v(uint32_t dst_treg, uint32_t src1_treg, uint32_t src2_vreg, uint32_t meta_reg);
  void tile_gemm_r(uint32_t dst_ureg, uint32_t src1_treg, uint32_t src2_ureg, uint32_t meta_reg);

	const PerfStats& perf_stats() const;

private:
	class Impl;
	Impl* impl_;
};

} // namespace vortex
