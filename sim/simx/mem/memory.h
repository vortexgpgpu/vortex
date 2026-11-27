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
#include <mem.h>
#include <dram_sim.h>
#include <functional>

namespace vortex {

class Memory : public SimObject<Memory>{
public:
	struct Config {
		uint32_t num_ports;
		DramSim::Config dram;  // one DRAM channel per memory bank
	};

	struct PerfStats {
		uint64_t bank_stalls = 0;
		uint64_t reads = 0;
		uint64_t writes = 0;
		uint64_t latency = 0;  // pending reads summed over cycles

		PerfStats& operator+=(const PerfStats& rhs) {
			this->bank_stalls += rhs.bank_stalls;
			this->reads += rhs.reads;
			this->writes += rhs.writes;
			this->latency += rhs.latency;
			return *this;
		}
	};

	// The crossbar ports that carry the memory interface.
	std::vector<SimPortIn<MemReq>*>  mem_req_in;
	std::vector<SimPortOut<MemRsp>*> mem_rsp_out;

	Memory(const SimContext& ctx, const char* name, const Config& config);
	~Memory();

	// Attach the backing RAM image so reads/writes can carry actual bytes
	// (TLM data path).
	void attach_ram(RAM* ram);

	// When a non-null hook is installed, Memory::tick() invokes it on
	// every accepted request just before enqueueing to the local DRAM
	// model. The hook receives the MemReq by const-ref — typically used
	// to forward to an SST::Interfaces::StandardMem link.
	//
	// The local data path (RAM read/write + dram_sim_) is unchanged;
	// this is timing-only telemetry, not a substitute backing store.
	// The std::function callback keeps this module SST-agnostic.
	using PreSendHook = std::function<void(const MemReq&)>;
	void set_pre_send_hook(PreSendHook hook);

	const PerfStats& perf_stats() const;

protected:
	void on_reset();
	void on_tick();
	// The DRAM model advances every cycle.

private:
	class Impl;
	Impl* impl_;

	friend class SimObject<Memory>;
};

};
