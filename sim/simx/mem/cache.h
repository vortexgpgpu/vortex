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
#include "memory.h"

namespace vortex {

class Cache : public SimObject<Cache> {
public:
	enum ReplPolicy : uint8_t {
		RANDOM = 0,
		FIFO   = 1,
		PLRU   = 2
	};

	struct Config {
		bool    bypass;         // cache bypass
		uint8_t C;              // log2 cache size
		uint8_t L;              // log2 line size
		uint8_t S;              // log2 sector size (fill/eviction/mem granule); = L => 1 sector/line
		uint8_t W;              // log2 word size
		uint8_t A;              // log2 associativity
		uint8_t B;              // log2 number of banks
		uint8_t addr_width;     // word address bits
		uint16_t num_inputs;    // number of inputs
		uint8_t mem_ports;      // memory ports
		bool    write_back;     // is write-back
		bool    write_reponse;  // enable write response
		uint16_t mshr_size;     // MSHR buffer size
		uint16_t latency;       // pipeline latency
		uint8_t repl_policy;    // replacement policy (ReplPolicy)
		bool    is_llc;					// is last-level cache
		bool    nc_enable = false;   // non-cacheable bypass layer in front of the cache
		bool    bank_hash = false;	// hashed bank select
		bool    shared_llc = false;	// the LLC serves every memory client alone
		bool    lazy_fetch = false;	// write miss allocates without a fill (write-back LLC only)
		bool    core_out_buf = false; // registered core-response port
		uint8_t read_merge = 0;	// merge same-word reads within groups of this many adjacent ports (0: off)
		uint16_t mshr_lines = 0;	// MSHR distinct {line, sector} entries (0: one per request)
		bool    write_around = false; // a no-allocate write miss goes to memory (write-back only)
		bool    mem_rsp_queue = false; // memory responses enter through a response queue
		bool    mem_rsp_slice = false; // a registered boundary ahead of the memory-response port stands for that queue's register
		bool    merge_port0 = true;    // port 0's reads match other ports' tags for read merging
		bool    acq_forward = false;   // an acquire flush also invalidates the next level
	};

	struct PerfStats {
		uint64_t reads = 0;
		uint64_t writes = 0;
		uint64_t read_misses = 0;
		uint64_t write_misses = 0;
		uint64_t evictions = 0;
		uint64_t bank_stalls = 0;
		uint64_t mshr_stalls = 0;
		uint64_t mem_latency = 0;

		PerfStats& operator+=(const PerfStats& rhs) {
			this->reads += rhs.reads;
			this->writes += rhs.writes;
			this->read_misses += rhs.read_misses;
			this->write_misses += rhs.write_misses;
			this->evictions += rhs.evictions;
			this->bank_stalls += rhs.bank_stalls;
			this->mshr_stalls += rhs.mshr_stalls;
			this->mem_latency += rhs.mem_latency;
			return *this;
		}
	};

	std::vector<SimChannel<MemReq>> core_req_in;
	std::vector<SimChannel<MemRsp>> core_rsp_out;
	std::vector<SimChannel<MemReq>> mem_req_out;
	std::vector<SimChannel<MemRsp>> mem_rsp_in;

	Cache(const SimContext& ctx, const char* name, const Config& config);
	~Cache();

	PerfStats perf_stats() const;

	// Flush API: arm a write-back walk that emits dirty lines via mem_req_out.
	// flush_done() returns true once the walk completes; the caller is expected
	// to tick the simulator until then. Write-through caches no-op. On an
	// L1-only flush the write-back last-level cache skips the walk and keeps
	// its lines across the next launch reset.
	void flush_begin(bool l1_only = false);
	bool flush_done() const;

	// A shared last-level cache keeps its lines into the next launch.
	void keep_lines();

protected:
	void on_reset();
	void on_tick();

private:
	class Impl;
	Impl* impl_;

	friend class SimObject<Cache>;
};

}