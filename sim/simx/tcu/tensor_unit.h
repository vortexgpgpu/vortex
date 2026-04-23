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
#include "instr_trace.h"

namespace vortex {

class Core;

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

class TensorUnit : public SimObject<TensorUnit> {
public:

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

  TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core);
  virtual ~TensorUnit();

  virtual void reset();

  virtual void tick();

	void wmma(uint32_t wid,
			 	    uint32_t fmt_s,
						uint32_t fmt_d,
			 	    uint32_t step_m,
						uint32_t step_n,
	          const std::vector<reg_data_t>& rs1_data,
					  const std::vector<reg_data_t>& rs2_data,
					  const std::vector<reg_data_t>& rs3_data,
					  std::vector<reg_data_t>& rd_data,
					  ExeTraceData* trace_data);

	const PerfStats& perf_stats() const;

private:
	class Impl;
	Impl* impl_;
};

} // namespace vortex
