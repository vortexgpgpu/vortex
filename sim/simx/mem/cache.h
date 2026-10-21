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
		uint8_t num_inputs;     // number of inputs
		uint8_t mem_ports;      // memory ports
		bool    write_back;     // is write-back
		bool    write_reponse;  // enable write response
		uint16_t mshr_size;     // MSHR buffer size
		uint8_t latency;        // pipeline latency
		uint8_t repl_policy;    // replacement policy (ReplPolicy)
		bool    is_llc;					// is last-level cache
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
		uint64_t write_nofetch = 0;   // full-sector write misses allocated without a DRAM fetch

		PerfStats& operator+=(const PerfStats& rhs) {
			this->reads += rhs.reads;
			this->writes += rhs.writes;
			this->read_misses += rhs.read_misses;
			this->write_misses += rhs.write_misses;
			this->evictions += rhs.evictions;
			this->bank_stalls += rhs.bank_stalls;
			this->write_nofetch += rhs.write_nofetch;
			this->mshr_stalls += rhs.mshr_stalls;
			this->mem_latency += rhs.mem_latency;
			return *this;
		}
	};

	// Per-requester counters (RFC 261019c), off unless the owner calls
	// enable_requester_stats(). Index = the requester row that the arbiter in front of
	// this cache ORed into the request tag; the owner maps rows to requester names.
	static constexpr uint32_t kMaxReqRows = 8;
	struct ReqStats {
		uint64_t reads[kMaxReqRows]   = {};
		uint64_t writes[kMaxReqRows]  = {};
		uint64_t miss_r[kMaxReqRows]  = {};
		uint64_t miss_w[kMaxReqRows]  = {};
		uint64_t xevicts[kMaxReqRows] = {}; // valid sectors of another row replaced by this row's fills
		uint64_t evicted[kMaxReqRows] = {}; // this row's valid sectors replaced by another row's fills
		uint64_t pipe_cap  = 0; // a core request waited: bank pipe full, no request yet at its head
		uint64_t pipe_head = 0; // a core request waited: bank pipe full, its head stalled downstream
		uint64_t pipe_slot = 0; // a core request waited: the input slot went to a replay or a fill
		std::vector<ArbPortStats> xbar_lanes; // bank crossbar outcome per request lane

		// Bank-side fields only; xbar_lanes is filled by the cache.
		ReqStats& operator+=(const ReqStats& rhs) {
			for (uint32_t r = 0; r < kMaxReqRows; ++r) {
				reads[r]   += rhs.reads[r];
				writes[r]  += rhs.writes[r];
				miss_r[r]  += rhs.miss_r[r];
				miss_w[r]  += rhs.miss_w[r];
				xevicts[r] += rhs.xevicts[r];
				evicted[r] += rhs.evicted[r];
			}
			pipe_cap  += rhs.pipe_cap;
			pipe_head += rhs.pipe_head;
			pipe_slot += rhs.pipe_slot;
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

	// Call once, before the first tick. row_bits = log2 of the arbiter row group in front
	// of this cache (0: every request is row 0). with_xbar also counts the bank crossbar
	// per request lane. No-op for a bypassed cache.
	void enable_requester_stats(uint32_t row_bits, bool with_xbar);
	ReqStats req_stats() const;

	// Flush API: arm a write-back walk that emits dirty lines via mem_req_out.
	// flush_done() returns true once the walk completes; the caller is expected
	// to tick the simulator until then. Write-through caches no-op.
	void flush_begin();
	bool flush_done() const;

protected:
	void on_reset();
	void on_tick();

private:
	class Impl;
	Impl* impl_;

	friend class SimObject<Cache>;
};

}