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
  #include "tlb.h"      // CacheSim-based TLB wrapper
  #include "ptw.h"
#endif

namespace vortex {

#ifdef VM_ENABLE
// Number of PTWs: configurable via NUM_PTWS (default: one per TLB/cache unit)
// When NUM_PTWS < num_units, multiple TLBs share PTWs via an arbiter
// This allows trading PTW count for resource efficiency
#endif

class CacheCluster : public SimObject<CacheCluster> {
public:
	struct PerfStats {
		CacheSim::PerfStats caches;
		#ifdef VM_ENABLE
		CacheSim::PerfStats tlb;
		PTW::PerfStats ptw;
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
		// Pending queue size = 1 (one request at a time per TLB)
		// With shared PTWs, multiple TLBs may share PTWs via arbiter
		, pending_tlb_requests_(MAX(num_units, 0x1), HashTable<MemReq>(1))  // One HashTable per unit, size=1 (one request at a time)
		, tlb_req_ports_(cache_config.num_inputs * MAX(num_units, 0x1), this)  // Internal ports: InputArbiter ReqOut → (here)
		, cache_rsp_ports_(cache_config.num_inputs * MAX(num_units, 0x1), this)  // Internal ports: Cache CoreRspPorts → (here)
		, shared_ptw_(MAX(num_units, 0x1))  // Configurable number of PTWs (will be resized to NUM_PTWS in constructor)
    	#endif
	{
		#ifdef VM_ENABLE
		// VM mode: TLB and PTW components will be created
		#endif
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

		#ifdef VM_ENABLE
		// Create configurable number of PTWs (NUM_PTWS, default: 1)
		// When NUM_PTWS < num_units, multiple TLBs share PTWs using round-robin distribution
		uint32_t num_ptws = MAX(NUM_PTWS, 0x1);  // Use NUM_PTWS from config
		uint32_t num_units_safe = MAX(num_units, 0x1);
		// Resize shared_ptw_ vector to match num_ptws (may be smaller than num_units)
		shared_ptw_.resize(num_ptws);
		
		// Count how many TLBs will connect to each PTW (for arbiter sizing)
		std::vector<uint32_t> tlbs_per_ptw(num_ptws, 0);
		for (uint32_t i = 0; i < num_units_safe; ++i) {
			uint32_t ptw_idx = i % num_ptws;
			tlbs_per_ptw.at(ptw_idx)++;
		}
		
		// Create PTW arbiters for shared PTWs (one arbiter per PTW)
		// These arbiters route requests from multiple TLBs to their assigned PTW
		std::vector<MemArbiter::Ptr> ptw_arbs(num_ptws);
		
		for (uint32_t i = 0; i < num_ptws; ++i) {
			snprintf(sname, 100, "%s-ptw%d", name, i);
			PTW::Config ptw_config;
			ptw_config.pt_levels = PT_LEVEL;
			ptw_config.pte_size = PTE_SIZE;
			ptw_config.base_ppn = 0x0;  // Default SATP base PPN - can be updated later
			// Use configurable PTW buffer size (max concurrent walks per PTW)
			ptw_config.buffer_size = PTW_BUFFER_SIZE;  // Configurable via PTW_BUFFER_SIZE parameter
			shared_ptw_.at(i) = PTW::Create(sname, ptw_config);
			
			// Create arbiter for this PTW (always create, even if only 1 TLB for simplicity)
			snprintf(sname, 100, "%s-ptw-arb%d", name, i);
			ptw_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, tlbs_per_ptw.at(i), 1);
			// Connect arbiter to PTW
			ptw_arbs.at(i)->ReqOut.at(0).bind(&shared_ptw_.at(i)->CoreReqPort);
			shared_ptw_.at(i)->CoreRspPort.bind(&ptw_arbs.at(i)->RspOut.at(0));
		}
		#endif

		// Arbitrate outgoing memory interfaces
		std::vector<MemArbiter::Ptr> mem_arbs(cache_config.mem_ports);
		for (uint32_t i = 0; i < cache_config.mem_ports; ++i) {
			snprintf(sname, 100, "%s-mem-arb%d", name, i);
			#ifdef VM_ENABLE
			// Memory arbiter needs ports for PTW (page table walks) and Cache (misses)
			// PTWs: num_ptws (one per TLB), Caches: num_units
			// num_ptws is already declared above in the VM_ENABLE block
			mem_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_ptws + num_units, 1);
			#else
			// Memory arbiter only needs ports for Cache misses
			mem_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_units, 1);
			#endif
			mem_arbs.at(i)->ReqOut.at(0).bind(&this->MemReqPorts.at(i));
			this->MemRspPorts.at(i).bind(&mem_arbs.at(i)->RspOut.at(0));
		}

		#ifdef VM_ENABLE
		// Connect PTW memory ports to memory arbiters
		// Each PTW has a single MemReqPort/MemRspPort
		// Each memory arbiter has inputs [0..num_ptws-1] for PTWs, then [num_ptws..] for caches
		// For simplicity, connect all PTWs to all memory arbiters
		// Each PTW uses the same input index across all memory arbiters
		// num_ptws is already declared above in the VM_ENABLE block
		for (uint32_t i = 0; i < num_ptws; ++i) {
			// Connect each PTW to all memory arbiters (for now, use first memory arbiter only)
			// TODO: Distribute PTWs across memory arbiters for better bandwidth
			shared_ptw_.at(i)->MemReqPort.bind(&mem_arbs.at(0)->ReqIn.at(i));
			mem_arbs.at(0)->RspIn.at(i).bind(&shared_ptw_.at(i)->MemRspPort);
		}
		#endif

		// Start looping for each cache
		for (uint32_t i = 0; i < num_units; ++i) {
			// Create caches for the unit
			snprintf(sname, 100, "%s-cache%d", name, i);

			// Create caches (TLB latency handled in request path)
			caches_.at(i) = CacheSim::Create(sname, cache_config2);

            #ifdef VM_ENABLE
		// VM_ENABLE: Create TLB using CacheSim structure
		// CRITICAL: TLB_LINE_SIZE must equal PAGE_SIZE (4096) for proper page-level caching!
            snprintf(sname, 100, "%s-tlb%d", name, i);
		tlb_.at(i) = TlbSim::Create(sname, 1, CacheSim::Config{
			false,                   // bypass: Enable TLB (don't bypass)
			log2ceil(TLB_SIZE),      // C: log2(total size) - determines entry count = 2^(C-L)
			log2ceil(TLB_LINE_SIZE), // L: log2(line size) - ⭐ MUST = log2(PAGE_SIZE) for VPN granularity!
			log2ceil(TLB_WORD_SIZE), // W: log2(word size) - for address decomposition
			log2ceil(TLB_NUM_WAYS),  // A: log2(associativity) - 8-way reduces conflict misses
			log2ceil(TLB_NUM_BANKS), // B: log2(num banks) - 2 banks for parallel lookups
			XLEN,                    // addr_width: 32 or 64-bit address space
			1,                       // ports_per_bank: 1 port per bank
			cache_config.num_inputs, // num_inputs: match cache input ports
			cache_config.mem_ports,  // mem_ports: ports to PTW for page table walks
			false,                   // write_back: false (TLB is read-only)
			false,                   // write_response: false (no write operations)
			8,                       // mshr_size: 8 concurrent PTW requests allowed
			cache_config.latency,    // latency: TLB lookup cycles (match cache)
		});

			// Connect input arbiters to internal ports (processed in tick)
            for (uint32_t j = 0; j < cache_config.num_inputs; ++j) {
				uint32_t port_idx = j * num_units + i;
				input_arbs.at(j)->ReqOut.at(i).bind(&tlb_req_ports_.at(port_idx));
				caches_.at(i)->CoreRspPorts.at(j).bind(&cache_rsp_ports_.at(port_idx));
			}

			// Connect TLB MemReqPorts to PTW using round-robin distribution via arbiter
			// When NUM_PTWS < num_units, multiple TLBs share PTWs via arbiters
			// Distribution: TLB[i] → PTW[i % num_ptws] via arbiter
			uint32_t ptw_idx = i % num_ptws;  // Round-robin assignment
			
			// Calculate which arbiter input this TLB should use
			// Count how many TLBs with lower index share the same PTW
			uint32_t arb_input_idx = 0;
			for (uint32_t j = 0; j < i; ++j) {
				if ((j % num_ptws) == ptw_idx) {
					arb_input_idx++;
				}
			}
			
			// Request path: TLB MemReqPorts → Arbiter → PTW CoreReqPort
			tlb_.at(i)->MemReqPorts.at(0).bind(&ptw_arbs.at(ptw_idx)->ReqIn.at(arb_input_idx));
			// Response path: PTW CoreRspPort → Arbiter → TLB MemRspPorts (forwarded to internal cache in TLB tick)
			// Note: PTW responses include tag/uuid to match back to originating TLB
			// Arbiter routes responses back to correct TLB based on request order
			ptw_arbs.at(ptw_idx)->RspIn.at(arb_input_idx).bind(&tlb_.at(i)->MemRspPorts.at(0));
			// If TLB has multiple mem_ports, connect them all to the same PTW (PTW buffer will handle serialization)
			for (uint32_t j = 1; j < cache_config.mem_ports; ++j) {
				// For now, only use the first mem_port - PTW buffer can handle queuing
				// TODO: Add per-TLB arbiter if multiple concurrent PTW requests are needed
			}

			// Connect Cache MemReqPorts to memory arbiters (for cache misses)
			// num_ptws is already declared above in the VM_ENABLE block
			for (uint32_t j = 0; j < cache_config.mem_ports; ++j) {
				uint32_t mem_arb_input = num_ptws + i;  // PTWs come first, then caches
				caches_.at(i)->MemReqPorts.at(j).bind(&mem_arbs.at(j)->ReqIn.at(mem_arb_input));
				mem_arbs.at(j)->RspIn.at(mem_arb_input).bind(&caches_.at(i)->MemRspPorts.at(j));
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
		#ifdef VM_ENABLE
		static uint64_t tick_count = 0;
		++tick_count;
		
		// Process TLB and Cache in VM mode
		// CRITICAL TICK ORDER:
		// 1. Tick PTW (sends responses to TLB MemRspPort)
		// 2. Tick TLB (forwards PTW responses, generates CoreRspPort responses)
		// 3. Process TLB responses (read from CoreRspPort)
		// 4. Handle new requests (push to TLB CoreReqPort)

		// Step 1: Tick PTW and TLB FIRST
		for (uint32_t ptw_idx = 0; ptw_idx < shared_ptw_.size(); ++ptw_idx) {
			auto ptw = shared_ptw_.at(ptw_idx);
			if (ptw) {
				ptw->tick();
			}
		}
		for (uint32_t i = 0; i < caches_.size(); ++i) {
			if (tlb_.at(i)) {
				tlb_.at(i)->tick();
			}
			if (caches_.at(i)) {
				caches_.at(i)->tick();
			}
		}

		// Step 2: NOW process TLB responses (after PTW and TLB have ticked)
		for (uint32_t i = 0; i < caches_.size(); ++i) {
			uint32_t num_inputs = tlb_.at(i) ? tlb_.at(i)->CoreRspPorts.size() : 0;
			if (num_inputs == 0) continue;

			// Process TLB responses to free up pending queue space
			for (uint32_t j = 0; j < num_inputs; ++j) {
				auto& tlb_rsp_port = tlb_.at(i)->CoreRspPorts.at(j);
				if (!tlb_rsp_port.empty()) {
					// Get TLB response (MemRsp with tag, cid, uuid)
					const auto& tlb_rsp = tlb_rsp_port.front();
					
					// Track TLB responses to measure latency
					static int tlb_rsp_trace_count = 0;
					if (tlb_rsp_trace_count < 50) {
						std::cout << "\n<<< [VM TLB RESPONSE TRACE #" << tlb_rsp_trace_count << "] TICK=" << tick_count << " <<< " << this->name() << std::endl;
						std::cout << "    TLB completed: tlb_tag=" << tlb_rsp.tag << " cid=" << tlb_rsp.cid << " uuid=" << tlb_rsp.uuid << std::endl;
						tlb_rsp_trace_count++;
					}

					// Validate that the tag exists in pending queue (prevent tag reuse/clash)
					if (!pending_tlb_requests_.at(i).contains(tlb_rsp.tag)) {
						std::cerr << "[CACHE_CLUSTER] ERROR: TLB response tag " << tlb_rsp.tag 
						   << " not found in pending queue! Possible tag reuse/clash or stale response. Ignoring response." << std::endl;
						tlb_rsp_port.pop();  // Drop invalid response
						continue;
					}

					// Retrieve the original request using the tag (guaranteed to exist due to validation above)
					const auto& orig_req = pending_tlb_requests_.at(i).at(tlb_rsp.tag);

					// Additional validation: Match UUID to ensure this response belongs to this request
					// This prevents tag reuse issues even if tags are somehow reused prematurely
					if (orig_req.uuid != tlb_rsp.uuid) {
						std::cerr << "[CACHE_CLUSTER] ERROR: TLB response UUID mismatch! tag=" << tlb_rsp.tag 
						   << " req_uuid=" << orig_req.uuid << " rsp_uuid=" << tlb_rsp.uuid 
						   << " addr=0x" << std::hex << orig_req.addr << std::dec 
						   << " Possible tag reuse/clash. Ignoring response." << std::endl;
						tlb_rsp_port.pop();  // Drop invalid response
						continue;
					}

					DT(3, "TLB Response: addr=0x" << std::hex << orig_req.addr << std::dec
					   << " tag=" << tlb_rsp.tag << " (sending to Cache)");

					// Create cache request with physical address (from TLB translation)
					// When TLB responds, it means the translation is cached (TLB hit) or just completed (PTW done)
					// For now, use identity mapping - the TLB cache stores VPN, and we use same for PPN
					// TODO: Store actual physical address in TLB cache or pass it through response
					MemReq cache_req;
					cache_req.addr = orig_req.addr;  // Keep virtual address  
					cache_req.p_addr = orig_req.addr; // Physical address (identity mapping for now)
					cache_req.write = orig_req.write;
					cache_req.type = orig_req.type;
					cache_req.tag = orig_req.tag;  // Preserve original tag for final response
					cache_req.cid = orig_req.cid;
					cache_req.uuid = orig_req.uuid;

					// Send cache request
					std::cout << "[CACHE_CLUSTER TICK=" << tick_count << "] " << this->name() 
					   << " SENDING TO CACHE: cache_tag=" << cache_req.tag << " addr=0x" << std::hex << cache_req.addr << std::dec 
					   << " (TLB translation complete)" << std::endl;
					caches_.at(i)->CoreReqPorts.at(j).push(cache_req, 1);

					// Release the pending TLB request
					uint32_t prev_pending = pending_tlb_requests_.at(i).size();
					pending_tlb_requests_.at(i).release(tlb_rsp.tag);
					uint32_t new_pending = pending_tlb_requests_.at(i).size();
					std::cout << "[CACHE_CLUSTER TICK=" << tick_count << "] " << this->name() 
					   << " RELEASED TLB_TAG=" << tlb_rsp.tag << " pending=" << new_pending << "/1 (was " << prev_pending << ")" << std::endl;

					// Pop the TLB response
					tlb_rsp_port.pop();
				}
			}

			// Step 2: Handle new requests from internal tlb_req_ports_ (fed by InputArbiter)
			// Do this AFTER processing responses to free up queue space first
			for (uint32_t j = 0; j < num_inputs; ++j) {
				uint32_t port_idx = j * caches_.size() + i;
				auto& tlb_req_port = tlb_req_ports_.at(port_idx);
				if (!tlb_req_port.empty()) {
					// Get request from internal port (came from InputArbiter)
					const auto& input_req = tlb_req_port.front();

					// Track requests to trace VM path and measure latency
					static int request_trace_count = 0;
					if (request_trace_count < 50) {
						std::cout << "\n>>> [VM REQUEST TRACE #" << request_trace_count << "] TICK=" << tick_count << " <<< " << this->name() << std::endl;
						std::cout << "    addr=0x" << std::hex << input_req.addr << std::dec
						   << " tag=" << input_req.tag << " cid=" << input_req.cid << " uuid=" << input_req.uuid 
						   << " write=" << input_req.write << " type=" << input_req.type << std::endl;
						std::cout << "    >> Entering TLB for address translation..." << std::endl;
						request_trace_count++;
					}

					DT(3, ">>> MEM REQUEST: addr=0x" << std::hex << input_req.addr << std::dec
					   << " tag=" << input_req.tag << " (entering TLB pipeline)");

					// Check if we have space in pending queue (size=1, since PTW processes one request at a time)
					// IMPORTANT: With one PTW per TLB, only one request can be in flight at a time
					// The TLB MSHR is set to 1, so it can only have 1 miss pending
					// The pending_tlb_requests_ queue is also set to 1, so only 1 request can be tracked
					bool is_full = pending_tlb_requests_.at(i).full();
					uint32_t pending_count = pending_tlb_requests_.at(i).size();
					
					// Debug: Verify queue capacity (when full, size == capacity)
					if (is_full && pending_count != 1) {
						std::cerr << "[CACHE_CLUSTER] ERROR: Queue is full but size=" << pending_count 
						   << " (expected 1)! This indicates the queue capacity is NOT 1!" << std::endl;
					}
					
					if (is_full) {
						// Queue full - PTW is processing a request, wait for it to complete
						// This is normal behavior since PTW processes one request at a time
						// Backpressure: don't pop the request, leave it in port
						DT(3, this->name() << " PENDING QUEUE FULL: waiting for PTW");
						continue;
					}
					// Allocate a tag and store the request
					uint32_t tlb_tag = pending_tlb_requests_.at(i).allocate(input_req);
					std::cout << "[CACHE_CLUSTER TICK=" << tick_count << "] " << this->name() << " ALLOCATED TLB_TAG=" << tlb_tag 
					   << " for addr=0x" << std::hex << input_req.addr << std::dec 
					   << " orig_tag=" << input_req.tag << " pending=" << pending_tlb_requests_.at(i).size() << "/1" << std::endl;

					// Create TLB request with allocated tag
					// CRITICAL FIX: TLB lookups must ALWAYS use write=false
					// Even though we want to track read vs write misses for stats, the CacheSim
					// response logic (line 565 in cache_sim.cpp) checks:
					//   if (!pipeline_req.write || config_.write_reponse)
					// Since TLB has write_response=false, if write=true, NO response will be sent!
					// This causes deadlock for store instructions that miss in TLB.
					// Solution: Always use write=false for TLB lookups (translation is a "read" operation)
					MemReq tlb_req;
					tlb_req.addr = input_req.addr;
					tlb_req.p_addr = input_req.p_addr;
					tlb_req.write = false;  // ALWAYS false for TLB lookups to ensure response is sent
					tlb_req.type = input_req.type;
					tlb_req.tag = tlb_tag;  // Use allocated tag
					tlb_req.cid = input_req.cid;
					tlb_req.uuid = input_req.uuid;

					// Send TLB request
					DT(3, this->name() << " SENDING TO TLB: tlb_tag=" << tlb_tag << " addr=0x" << std::hex << tlb_req.addr << std::dec);
					if (tlb_.at(i)) {
						tlb_.at(i)->CoreReqPorts.at(j).push(tlb_req, 1);  // delay=1, TLB has its own pipeline latency
					}

					// Pop the input request from internal port
					tlb_req_port.pop();
				}
			}

		}

		// Step 4: Handle cache responses and forward back to core
		for (uint32_t i = 0; i < caches_.size(); ++i) {
			uint32_t num_inputs = tlb_.at(i) ? tlb_.at(i)->CoreRspPorts.size() : 0;
			if (num_inputs == 0) continue;

			// Handle cache responses and forward back to core
			for (uint32_t j = 0; j < num_inputs; ++j) {
				uint32_t port_idx = j * caches_.size() + i;
				auto& cache_rsp_port = cache_rsp_ports_.at(port_idx);
				if (!cache_rsp_port.empty()) {
					// Get cache response
					const auto& cache_rsp = cache_rsp_port.front();

					// Track cache responses to measure total end-to-end latency
					static int cache_rsp_trace_count = 0;
					if (cache_rsp_trace_count < 50) {
						std::cout << "\n*** [CACHE RESPONSE #" << cache_rsp_trace_count << "] TICK=" << tick_count << " *** " << this->name() << std::endl;
						std::cout << "    Cache completed: tag=" << cache_rsp.tag << " cid=" << cache_rsp.cid << " uuid=" << cache_rsp.uuid << std::endl;
						std::cout << "    >> FINAL RESPONSE TO CORE" << std::endl;
						cache_rsp_trace_count++;
					}

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

		// Process cache requests directly when VM is disabled
		for (uint32_t i = 0; i < caches_.size(); ++i) {
			if (caches_.at(i)) {
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
			if (tlb) {
				auto tlb_stats = tlb->perf_stats().tlb;
				perf.tlb += tlb_stats;
			}
		}
		// Aggregate PTW performance stats
		for (auto ptw : shared_ptw_) {
			if (ptw) perf.ptw += ptw->perf_stats();
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

		// PTW stats
		const auto& ptw_stats = perf_stats.ptw;
		std::cout << "PTW Stats:" << std::endl;
		std::cout << "  Walks: " << ptw_stats.walks << std::endl;
		std::cout << "  Memory Accesses: " << ptw_stats.mem_accesses << std::endl;
		std::cout << "  Max Concurrent Walks: " << ptw_stats.max_concurrent_walks << std::endl;
		if (ptw_stats.walks > 0) {
			double avg_accesses_per_walk = (double)ptw_stats.mem_accesses / ptw_stats.walks;
			std::cout << "  Avg Accesses Per Walk: " << avg_accesses_per_walk << std::endl;
		}
		#endif

		std::cout << "========================================" << std::endl;
	}

	#ifdef VM_ENABLE
	void set_satp(uint64_t satp) {
		// Extract base PPN from SATP (bits [21:0] for SV32)
		uint64_t base_ppn = satp & 0x3FFFFF;
		std::cout << "[CACHE_CLUSTER] " << this->name() << " set_satp CALLED: satp=0x" << std::hex << satp 
		   << " base_ppn=0x" << base_ppn << std::dec << " num_ptws=" << shared_ptw_.size() << std::endl;
		DT(1, "CacheCluster::set_satp: satp=0x" << std::hex << satp << " base_ppn=0x" << base_ppn << std::dec);
		// Propagate SATP to shared PTW pool
		for (uint32_t i = 0; i < shared_ptw_.size(); ++i) {
			if (shared_ptw_.at(i)) {
				std::cout << "[CACHE_CLUSTER] " << this->name() << " Setting base_ppn for PTW[" << i << "] = 0x" 
				   << std::hex << base_ppn << std::dec << std::endl;
				shared_ptw_.at(i)->set_base_ppn(base_ppn);
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
    std::vector<PTW::Ptr> shared_ptw_;  // Configurable number of PTWs (NUM_PTWS, default: one per TLB)
    // When NUM_PTWS < num_units, multiple TLBs share PTWs via arbiters (created in constructor)
  #endif
};

}
