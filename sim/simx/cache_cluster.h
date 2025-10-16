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

#include "cache_sim.h"
#include "debug.h"
#ifdef VM_ENABLE
#include "tlb.h"
#endif

namespace vortex {

class CacheCluster : public SimObject<CacheCluster> {
public:
	struct PerfStats {
		CacheSim::PerfStats caches;
		#ifdef VM_ENABLE
		CacheSim::PerfStats tlb;
		#endif
	};

	std::vector<std::vector<SimPort<MemReq>>> CoreReqPorts;
	std::vector<std::vector<SimPort<MemRsp>>> CoreRspPorts;
	std::vector<SimPort<MemReq>> MemReqPorts;
	std::vector<SimPort<MemRsp>> MemRspPorts;

	CacheCluster(const SimContext& ctx,
							const char* name,
							uint32_t num_inputs,
							uint32_t num_units,
							const CacheSim::Config& cache_config)
		: SimObject(ctx, name)
		, CoreReqPorts(num_inputs, std::vector<SimPort<MemReq>>(cache_config.num_inputs, this))
		, CoreRspPorts(num_inputs, std::vector<SimPort<MemRsp>>(cache_config.num_inputs, this))
		, MemReqPorts(cache_config.mem_ports, this)
		, MemRspPorts(cache_config.mem_ports, this)
		, caches_(MAX(num_units, 0x1))
		#ifdef VM_ENABLE
        , tlb_(MAX(num_units, 0x1))       // Initialize tlb_ similarly when VM_ENABLE is defined
		, pending_tlb_requests_(MAX(num_units, 0x1), HashTable<MemReq>(TLB_MSHR_SIZE))  // One HashTable per unit
		, tlb_req_ports_(cache_config.num_inputs * MAX(num_units, 0x1), this)  // Internal ports: InputArbiter ReqOut → (here)
		, cache_rsp_ports_(cache_config.num_inputs * MAX(num_units, 0x1), this)  // Internal ports: Cache CoreRspPorts → (here)
		#endif
	{
		DT(1, "CacheCluster constructor: " << name << " num_inputs=" << num_inputs << " num_units=" << num_units);

		CacheSim::Config cache_config2(cache_config);
		if (0 == num_units) {
			num_units = 1;
			cache_config2.bypass = true;
		}

		char sname[100];

		// Arbitrate incoming core interfaces
		std::vector<MemArbiter::Ptr> input_arbs(cache_config.num_inputs);
		for (uint32_t i = 0; i < cache_config.num_inputs; ++i) {
			snprintf(sname, 100, "%s-input-arb%d", name, i);
			input_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_inputs, num_units, 1);
			for (uint32_t j = 0; j < num_inputs; ++j) {
				this->CoreReqPorts.at(j).at(i).bind(&input_arbs.at(i)->ReqIn.at(j));
				input_arbs.at(i)->RspIn.at(j).bind(&this->CoreRspPorts.at(j).at(i));
			}
		}

		// Arbitrate outgoing memory interfaces
		std::vector<MemArbiter::Ptr> mem_arbs(cache_config.mem_ports);
		for (uint32_t i = 0; i < cache_config.mem_ports; ++i) {
			snprintf(sname, 100, "%s-mem-arb%d", name, i);
			#ifdef VM_ENABLE
			// Memory arbiter needs ports for both TLB (page table walks) and Cache (misses)
			mem_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, 2 * num_units, 1);
			#else
			// Memory arbiter only needs ports for Cache misses
			mem_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_units, 1);
			#endif
			mem_arbs.at(i)->ReqOut.at(0).bind(&this->MemReqPorts.at(i));
			this->MemRspPorts.at(i).bind(&mem_arbs.at(i)->RspOut.at(0));
		}

		// Start looping for each cache
		for (uint32_t i = 0; i < num_units; ++i) {
			// Create caches for the unit
			snprintf(sname, 100, "%s-cache%d", name, i);

			// Create caches (TLB latency handled in request path)
			caches_.at(i) = CacheSim::Create(sname, cache_config2);

			#ifdef VM_ENABLE
			// VM_ENABLE: Create and connect TLB
			snprintf(sname, 100, "%s-tlb%d", name, i);
			tlb_.at(i) = TlbSim::Create(sname, 1, CacheSim::Config{
				false,  // Don't bypass TLB when VM is enabled
				log2ceil(TLB_SIZE),    // C
				log2ceil(TLB_LINE_SIZE), // L
				log2ceil(TLB_WORD_SIZE), // W
				log2ceil(TLB_NUM_WAYS), // A num ways
				log2ceil(TLB_NUM_BANKS),// B
				XLEN,                   // address bits
				1,                      // number of ports
				cache_config.num_inputs,// number of inputs
				cache_config.mem_ports, // memory ports
				false,                  // write-back
				false,                  // write response
				TLB_MSHR_SIZE,          // mshr size
				cache_config.latency,   // use same latency as cache
			});

			// Connect input arbiters to internal ports (processed in tick)
			for (uint32_t j = 0; j < cache_config.num_inputs; ++j) {
				uint32_t port_idx = j * num_units + i;
				input_arbs.at(j)->ReqOut.at(i).bind(&tlb_req_ports_.at(port_idx));
				caches_.at(i)->CoreRspPorts.at(j).bind(&cache_rsp_ports_.at(port_idx));
			}

			// Connect TLB and caches to memory arbiters
			for (uint32_t j = 0; j < cache_config.mem_ports; ++j) {
				// TLB MemReqPorts for page table walks
				tlb_.at(i)->MemReqPorts.at(j).bind(&mem_arbs.at(j)->ReqIn.at(2*i));
				mem_arbs.at(j)->RspIn.at(2*i).bind(&tlb_.at(i)->MemRspPorts.at(j));
				// Cache MemReqPorts for cache misses
				caches_.at(i)->MemReqPorts.at(j).bind(&mem_arbs.at(j)->ReqIn.at(2*i+1));
				mem_arbs.at(j)->RspIn.at(2*i+1).bind(&caches_.at(i)->MemRspPorts.at(j));
			}
			#else
			// Non-VM: Direct connection to caches
			for (uint32_t j = 0; j < cache_config.num_inputs; ++j) {
				input_arbs.at(j)->ReqOut.at(i).bind(&caches_.at(i)->CoreReqPorts.at(j));
				caches_.at(i)->CoreRspPorts.at(j).bind(&input_arbs.at(j)->RspOut.at(i));
			}

			// Connect caches to memory arbiters
			for (uint32_t j = 0; j < cache_config.mem_ports; ++j) {
				caches_.at(i)->MemReqPorts.at(j).bind(&mem_arbs.at(j)->ReqIn.at(i));
				mem_arbs.at(j)->RspIn.at(i).bind(&caches_.at(i)->MemRspPorts.at(j));
			}
			#endif
		}
		DT(1, "CacheCluster constructor completed: " << name);
	}

	~CacheCluster() {}

	void reset() {}

	void tick() {
		static bool first_call = true;
		if (first_call) {
			first_call = false;
			#ifdef VM_ENABLE
			std::cout << "=== CACHE_CLUSTER: VM_ENABLE IS DEFINED ===" << std::endl;
			#else
			std::cout << "=== CACHE_CLUSTER: VM_ENABLE IS NOT DEFINED ===" << std::endl;
			#endif
		}
		#ifdef VM_ENABLE
		// Process TLB and Cache in VM mode (similar to Core::fetch pattern)
		static uint64_t tick_count = 0;
		if (tick_count == 0) {
			std::cout << "=== VM CACHE_CLUSTER TICK() CALLED! ===" << std::endl;
		}
		if ((tick_count % 1000) == 0) {
			DT(2, this->name() << "-vm-tick: count=" << tick_count);
		}
		tick_count++;

		for (uint32_t i = 0; i < caches_.size(); ++i) {
			uint32_t num_inputs = tlb_.at(i) ? tlb_.at(i)->CoreRspPorts.size() : 0;
			if (num_inputs == 0) continue;

			// Step 1: Handle new requests from internal tlb_req_ports_ (fed by InputArbiter)
			// Do this FIRST to feed the TLB pipeline
			for (uint32_t j = 0; j < num_inputs; ++j) {
				uint32_t port_idx = j * caches_.size() + i;
				auto& tlb_req_port = tlb_req_ports_.at(port_idx);
				if (!tlb_req_port.empty()) {
					// Get request from internal port (came from InputArbiter)
					const auto& input_req = tlb_req_port.front();

					DT(1, ">>> MEM REQUEST: addr=0x" << std::hex << input_req.addr << std::dec
					   << " tag=" << input_req.tag << " type=" << input_req.type
					   << " (entering TLB pipeline)");

					// Check if we have space in pending queue
					bool is_full = pending_tlb_requests_.at(i).full();
					if (!is_full) {
						// Allocate a tag and store the request
						uint32_t tlb_tag = pending_tlb_requests_.at(i).allocate(input_req);

						// Create TLB request with allocated tag
						MemReq tlb_req;
						tlb_req.addr = input_req.addr;
						tlb_req.p_addr = input_req.p_addr;
						tlb_req.write = false;  // TLB lookups are always reads
						tlb_req.type = input_req.type;
						tlb_req.tag = tlb_tag;  // Use allocated tag
						tlb_req.cid = input_req.cid;
						tlb_req.uuid = input_req.uuid;

				// Send TLB request
				DT(2, this->name() << "-vm-push-to-tlb: unit=" << i << " input=" << j << " tag=" << tlb_tag << " tlb_null=" << (tlb_.at(i) ? 0 : 1));
				if (tlb_.at(i)) {
					tlb_.at(i)->CoreReqPorts.at(j).push(tlb_req, 1);  // delay=1, TLB has its own pipeline latency
					DT(2, this->name() << "-vm-pushed-to-tlb: unit=" << i << " input=" << j);
				}

					// Pop the input request from internal port
					tlb_req_port.pop();
					}
					// If pending queue is full, leave request in port (backpressure)
				}
			}

			// Step 2: Process TLB and Cache (they generate responses)
			if (tlb_.at(i)) {
				tlb_.at(i)->tick();
			}
			if (caches_.at(i)) {
				caches_.at(i)->tick();
			}

			// Step 3: Handle TLB responses and create cache requests
			// Do this AFTER tick() so TLB has generated responses
			for (uint32_t j = 0; j < num_inputs; ++j) {
				auto& tlb_rsp_port = tlb_.at(i)->CoreRspPorts.at(j);
				if (!tlb_rsp_port.empty()) {
					// Get TLB response (MemRsp with tag, cid, uuid)
					const auto& tlb_rsp = tlb_rsp_port.front();

					// Retrieve the original request using the tag
					const auto& orig_req = pending_tlb_requests_.at(i).at(tlb_rsp.tag);

					DT(1, "  TLB Response: addr=0x" << std::hex << orig_req.addr << std::dec
					   << " tag=" << tlb_rsp.tag << " (sending to Cache)");

					// Create cache request with physical address (from TLB translation)
					// For performance modeling, we assume TLB provides p_addr = addr (identity mapping)
					MemReq cache_req;
					cache_req.addr = orig_req.addr;  // Keep virtual address
					cache_req.p_addr = orig_req.addr; // Physical address (TLB translated - for now identity)
					cache_req.write = orig_req.write;
					cache_req.type = orig_req.type;
					cache_req.tag = orig_req.tag;  // Preserve original tag for final response
					cache_req.cid = orig_req.cid;
					cache_req.uuid = orig_req.uuid;

					// Send cache request
					caches_.at(i)->CoreReqPorts.at(j).push(cache_req, 1);

					// Release the pending TLB request
					pending_tlb_requests_.at(i).release(tlb_rsp.tag);

					// Pop the TLB response
					tlb_rsp_port.pop();
				}
			}

			// Step 4: Handle cache responses and forward back to core
			for (uint32_t j = 0; j < num_inputs; ++j) {
				uint32_t port_idx = j * caches_.size() + i;
				auto& cache_rsp_port = cache_rsp_ports_.at(port_idx);
				if (!cache_rsp_port.empty()) {
					// Get cache response
					const auto& cache_rsp = cache_rsp_port.front();

					DT(1, "<<< MEM RESPONSE COMPLETE: tag=" << cache_rsp.tag
					   << " (returning to Core) ===");

					// Forward response back through CacheCluster CoreRspPorts
					// The arbiter's RspIn is already bound to this port, so it will flow through
					this->CoreRspPorts.at(i).at(j).push(cache_rsp, 1);

					// Pop the cache response
					cache_rsp_port.pop();
				}
			}
		}
		#else
		// Process cache requests directly when VM is disabled
		static uint64_t tick_count_novm = 0;
		if (tick_count_novm == 0) {
			DT(1, "=== CacheCluster non-VM tick() called! ===");
		}
		if ((tick_count_novm % 1000) == 0) {
			DT(2, this->name() << "-novm-tick: count=" << tick_count_novm);
		}
		tick_count_novm++;

		// Add debug output for non-VM memory requests
		for (uint32_t i = 0; i < caches_.size(); ++i) {
			if (caches_.at(i)) {
				// Check for new requests
				for (uint32_t j = 0; j < caches_.at(i)->CoreReqPorts.size(); ++j) {
					if (!caches_.at(i)->CoreReqPorts.at(j).empty()) {
						const auto& req = caches_.at(i)->CoreReqPorts.at(j).front();
						DT(1, ">>> MEM REQUEST: addr=0x" << std::hex << req.addr << std::dec
						   << " tag=" << req.tag << " type=Global (direct cache access)");
					}
				}

				// Check for responses
				for (uint32_t j = 0; j < caches_.at(i)->CoreRspPorts.size(); ++j) {
					if (!caches_.at(i)->CoreRspPorts.at(j).empty()) {
						const auto& rsp = caches_.at(i)->CoreRspPorts.at(j).front();
						DT(1, "<<< MEM RESPONSE COMPLETE: tag=" << rsp.tag
						   << " (returning to Core) ===");
					}
				}

				caches_.at(i)->tick();
			}
		}
		#endif
	}

	PerfStats perf_stats() const {
		PerfStats perf;

		// Aggregate cache performance stats
		for (auto cache : caches_) {
			perf.caches += cache->perf_stats();
		}

		#ifdef VM_ENABLE
		// Aggregate TLB performance stats
		for (auto tlb : tlb_) {
			perf.tlb += tlb->perf_stats().tlb;
		}
		#endif

		return perf;
	}

	void print_perf_stats() const {
		auto perf_stats = this->perf_stats();

		std::cout << "=== Cache Cluster Performance Stats ===" << std::endl;

		// Cache stats
		const auto& cache_stats = perf_stats.caches;
		std::cout << "Cache Stats:" << std::endl;
		std::cout << "  Reads: " << cache_stats.reads << std::endl;
		std::cout << "  Writes: " << cache_stats.writes << std::endl;
		std::cout << "  Read Misses: " << cache_stats.read_misses << std::endl;
		std::cout << "  Write Misses: " << cache_stats.write_misses << std::endl;
		std::cout << "  Evictions: " << cache_stats.evictions << std::endl;
		std::cout << "  Memory Latency: " << cache_stats.mem_latency << std::endl;

		#ifdef VM_ENABLE
		// TLB stats
		const auto& tlb_stats = perf_stats.tlb;
		std::cout << "TLB Stats:" << std::endl;
		std::cout << "  Reads: " << tlb_stats.reads << std::endl;
		std::cout << "  Writes: " << tlb_stats.writes << std::endl;
		std::cout << "  Read Misses: " << tlb_stats.read_misses << std::endl;
		std::cout << "  Write Misses: " << tlb_stats.write_misses << std::endl;
		std::cout << "  Evictions: " << tlb_stats.evictions << std::endl;
		std::cout << "  Memory Latency: " << tlb_stats.mem_latency << std::endl;

		// Calculate TLB hit rate
		uint64_t total_tlb_accesses = tlb_stats.reads + tlb_stats.writes;
		if (total_tlb_accesses > 0) {
			uint64_t tlb_misses = tlb_stats.read_misses + tlb_stats.write_misses;
			uint64_t tlb_hits = total_tlb_accesses - tlb_misses;
			double tlb_hit_rate = (double)tlb_hits / total_tlb_accesses * 100.0;
			std::cout << "  TLB Hit Rate: " << tlb_hit_rate << "%" << std::endl;
		}
		#endif

		std::cout << "========================================" << std::endl;
	}

	#ifdef VM_ENABLE
	void set_satp(uint64_t satp) {
		// Propagate SATP to all TLBs
		for (auto tlb : tlb_) {
			if (tlb) {
				tlb->set_satp(satp);
			}
		}
	}
	#endif

private:
  std::vector<CacheSim::Ptr> caches_;
  #ifdef VM_ENABLE
    std::vector<TlbSim::Ptr> tlb_;  // TLBs are only defined if VM_ENABLE is defined
    std::vector<HashTable<MemReq>> pending_tlb_requests_;  // Store original requests while waiting for TLB translation
    std::vector<SimPort<MemReq>> tlb_req_ports_;  // Internal ports: InputArbiter ReqOut → (here) (processed in tick)
    std::vector<SimPort<MemRsp>> cache_rsp_ports_;  // Internal ports: Cache CoreRspPorts → (here) (processed in tick)
  #endif
};

}
