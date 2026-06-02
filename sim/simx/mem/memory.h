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
#include <mem.h>
#include <functional>
#include "types.h"

namespace vortex {

class Memory : public SimObject<Memory>{
public:
	struct Config {
		uint32_t num_banks;
		uint32_t num_ports;
		uint32_t block_size;
		float clock_ratio;
	};

	struct PerfStats {
		uint64_t bank_stalls = 0;

		PerfStats& operator+=(const PerfStats& rhs) {
			this->bank_stalls += rhs.bank_stalls;
			return *this;
		}
	};

	std::vector<SimChannel<MemReq>> mem_req_in;
	std::vector<SimChannel<MemRsp>> mem_rsp_out;

	Memory(const SimContext& ctx, const char* name, const Config& config);
	~Memory();

	// Attach the backing RAM image so reads/writes can carry actual bytes
	// (TLM data path).
	void attach_ram(RAM* ram);

	// Backdoor functional read from the attached RAM. Bypasses the
	// timing-modeled DRAM/cache hierarchy; used by TCU_LD's AGU mirror
	// to fetch metadata from global memory directly.
	uint32_t read_word(uint64_t byte_addr) const;

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

private:
	class Impl;
	Impl* impl_;

	friend class SimObject<Memory>;
};

};