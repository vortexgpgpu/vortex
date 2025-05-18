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

class TensorUnit : public SimObject<TensorUnit> {
public:
	struct Config {
		uint8_t num_ports;
		uint8_t mac_latency;

		Config()
			: num_ports(0)
			, mac_latency(0)
		{}
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

  TensorUnit(const SimContext &ctx, const char* name, const Config& config, Core* core);

  virtual ~TensorUnit();

  virtual void reset();

  virtual void tick();

	const PerfStats& perf_stats() const;

private:
	class Impl;
	Impl* impl_;
};

} // namespace vortex
