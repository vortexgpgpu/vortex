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

namespace vortex {

class CacheSim : public SimObject<CacheSim> {
public:
	struct Config {
		bool    bypass;         // cache bypass
		uint8_t C;              // log2 cache size
		uint8_t L;              // log2 line size
		uint8_t W;              // log2 word size
		uint8_t A;              // log2 associativity
		uint8_t B;              // log2 number of banks
		uint8_t addr_width;     // word address bits
		uint8_t ports_per_bank; // number of ports per bank
		uint8_t num_inputs;     // number of inputs
		bool    write_back;     // is write-back
		bool    write_reponse;  // enable write response
		uint16_t mshr_size;     // MSHR buffer size
		uint8_t latency;        // pipeline latency
	};

	struct PerfStats {
		uint64_t reads;
		uint64_t writes;
		uint64_t read_misses;
		uint64_t write_misses;
		uint64_t evictions;
		uint64_t pipeline_stalls;
		uint64_t bank_stalls;
		uint64_t mshr_stalls;
		uint64_t mem_latency;

		PerfStats()
			: reads(0)
			, writes(0)
			, read_misses(0)
			, write_misses(0)
			, evictions(0)
			, pipeline_stalls(0)
			, bank_stalls(0)
			, mshr_stalls(0)
			, mem_latency(0)
		{}

		PerfStats& operator+=(const PerfStats& rhs) {
			this->reads += rhs.reads;
			this->writes += rhs.writes;
			this->read_misses += rhs.read_misses;
			this->write_misses += rhs.write_misses;
			this->evictions += rhs.evictions;
			this->pipeline_stalls += rhs.pipeline_stalls;
			this->bank_stalls += rhs.bank_stalls;
			this->mshr_stalls += rhs.mshr_stalls;
			this->mem_latency += rhs.mem_latency;
			return *this;
		}
	};

	std::vector<SimPort<MemReq>> CoreReqPorts;
	std::vector<SimPort<MemRsp>> CoreRspPorts;
	SimPort<MemReq>              MemReqPort;
	SimPort<MemRsp>              MemRspPort;

	CacheSim(const SimContext& ctx, const char* name, const Config& config);
	~CacheSim();

	void reset();

	void tick();

	const PerfStats& perf_stats() const;

private:
	class Impl;
	Impl* impl_;
};

}