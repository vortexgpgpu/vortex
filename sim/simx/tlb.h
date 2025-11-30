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
#include <iostream>
#include <iomanip>
#include <map>

namespace vortex {

class TlbSim : public SimObject<TlbSim> {
public:
	struct PerfStats {
    	CacheSim::PerfStats tlb;

		PerfStats& operator+=(const PerfStats& rhs) {
			this->tlb += rhs.tlb;
			return *this;
		}
  	};
	std::vector<SimPort<MemReq>> CoreReqPorts;
	std::vector<SimPort<MemRsp>> CoreRspPorts;
	std::vector<SimPort<MemReq>> MemReqPorts;  // TLB miss requests (for shared PTW)
	std::vector<SimPort<MemRsp>> MemRspPorts;  // PTW responses (from shared PTW)

	TlbSim(const SimContext& ctx,
			const char* name,
			uint32_t num_units,
			const CacheSim::Config& tlb_config)
		: SimObject(ctx, name)
		, CoreReqPorts(tlb_config.num_inputs, this)
		, CoreRspPorts(tlb_config.num_inputs, this)
		, MemReqPorts(tlb_config.mem_ports, this)
		, MemRspPorts(tlb_config.mem_ports, this)
		, tlb_(MAX(num_units, 0x1)) {

		CacheSim::Config tlb_config2(tlb_config);
		if (0 == num_units) {
			num_units = 1;
			tlb_config2.bypass = true;
		}

		char sname[100];

		// Create TLB cache
		snprintf(sname, 100, "%s-cache%d", name, 0);
		tlb_.at(0) = CacheSim::Create(sname, tlb_config2);

		// Connect input to TLB(Cache)
		for (uint32_t j = 0; j < tlb_config.num_inputs; ++j) {
			this->CoreReqPorts.at(j).bind(&tlb_.at(0)->CoreReqPorts.at(j));
			tlb_.at(0)->CoreRspPorts.at(j).bind(&this->CoreRspPorts.at(j));
		}

		// Connect TLB cache's memory ports to TlbSim's memory ports (for PTW)
		// TLB cache misses go to TlbSim's MemReqPorts (connected to PTW in CacheCluster)
		// PTW responses come back through TlbSim's MemRspPorts to TLB cache's MemRspPorts
		// NOTE: MemReqPorts binding is done here, but MemRspPorts binding is done externally
		// in CacheCluster to allow PTW -> TlbSim -> Internal Cache chain
		for (uint32_t j = 0; j < tlb_config.mem_ports; ++j) {
			tlb_.at(0)->MemReqPorts.at(j).bind(&this->MemReqPorts.at(j));
			// REMOVED: this->MemRspPorts.at(j).bind(&tlb_.at(0)->MemRspPorts.at(j));
			// This binding is now done by the external component (PTW -> TlbSim MemRspPorts)
			// and TlbSim MemRspPorts must forward to internal cache manually in tick()
		}

		DT(1, "TlbSim created (PTW shared): " << name);
	}

	~TlbSim() {}

	void reset() {}

	void tick() {
		static uint64_t tick_count = 0;
		static int tlb_trace_count = 0;
		
		// Trace first few TLB requests
		if (tlb_.at(0) && tlb_trace_count < 5) {
			for (uint32_t j = 0; j < CoreReqPorts.size(); ++j) {
				if (!CoreReqPorts.at(j).empty()) {
					auto& req = CoreReqPorts.at(j).front();
					std::cout << "\n=== [TLB LOOKUP TRACE #" << tlb_trace_count << "] === " << this->name() 
					   << " at tick=" << tick_count << std::endl;
					std::cout << "    Received request: addr=0x" << std::hex << req.addr << std::dec 
					   << " tag=" << req.tag << " (checking TLB cache...)" << std::endl;
					tlb_trace_count++;
				}
			}
		}
		
		if ((tick_count % 1000) == 0) {
			DT(2, this->name() << "-tick: count=" << tick_count
				<< " tlb_req_empty=" << (tlb_.at(0) ? tlb_.at(0)->CoreReqPorts.at(0).empty() : 1)
				<< " tlb_rsp_empty=" << (tlb_.at(0) ? tlb_.at(0)->CoreRspPorts.at(0).empty() : 1));
		}
		tick_count++;

			// Forward PTW responses to internal TLB cache
			// Since we removed the automatic binding, we need to manually forward
			if (tlb_.at(0)) {
				static int ptw_rsp_trace_count = 0;
				static std::map<std::string, int> received_count;
				// Debug: Always check and log for dcaches
				if (this->name() == std::string("socket0-dcaches-tlb0")) {
					for (uint32_t j = 0; j < MemRspPorts.size(); ++j) {
						if (!MemRspPorts.at(j).empty()) {
							received_count[this->name()]++;
							std::cout << "[TLB_DEBUG] " << this->name() << " tick=" << tick_count 
							   << " MemRspPorts[" << j << "] HAS RESPONSE #" << received_count[this->name()]
							   << " (before forwarding)" << std::endl;
						}
					}
				}
				for (uint32_t j = 0; j < MemRspPorts.size(); ++j) {
					if (!MemRspPorts.at(j).empty()) {
						auto rsp = MemRspPorts.at(j).front();
						// Always trace for dcaches to debug missing responses
						if (this->name() == std::string("socket0-dcaches-tlb0")) {
							static int dcache_forward_count = 0;
							dcache_forward_count++;
							std::cout << "\n=== [DCACHE TLB FORWARD #" << dcache_forward_count << "] === " << this->name() 
							   << " at tick=" << tick_count << std::endl;
							std::cout << "    Forwarding PTW response: tag=" << rsp.tag << " cid=" << rsp.cid 
							   << " uuid=" << rsp.uuid << " to internal cache" << std::endl;
						}
						if (ptw_rsp_trace_count < 10) {
							std::cout << "\n=== [PTW FILL TRACE #" << ptw_rsp_trace_count << "] === " << this->name() 
							   << " at tick=" << tick_count << std::endl;
							std::cout << "    Received PTW response: tag=" << rsp.tag << " cid=" << rsp.cid 
							   << " (filling TLB cache)" << std::endl;
							ptw_rsp_trace_count++;
						}
						// Forward to internal cache's MemRspPorts (delay=1 for proper pipeline timing)
						// CRITICAL FIX: Use delay=1 instead of delay=0 to allow cache to process in next tick
						// With delay=0, the response arrives in the same tick as the forward, which may cause
						// timing issues if the cache has already processed its mem_rsp queue this tick
						tlb_.at(0)->MemRspPorts.at(j).push(rsp, 1);
						MemRspPorts.at(j).pop();
						
						if (this->name() == std::string("socket0-dcaches-tlb0")) {
							std::cout << "    >> Using delay=1 to ensure cache processes response in next tick" << std::endl;
						}
					}
				}
			}

		// Tick TLB only (PTW is now shared in CacheCluster)
		for (auto tlb : tlb_) {
			if (tlb) tlb->tick();
		}
		
		// Trace TLB responses (hits or filled from PTW)
		static int tlb_output_trace_count = 0;
		if (tlb_.at(0) && tlb_output_trace_count < 5) {
			for (uint32_t j = 0; j < CoreRspPorts.size(); ++j) {
				if (!CoreRspPorts.at(j).empty()) {
					auto& rsp = CoreRspPorts.at(j).front();
					std::cout << "\n=== [TLB OUTPUT TRACE #" << tlb_output_trace_count << "] === " << this->name() 
					   << " at tick=" << tick_count << std::endl;
					std::cout << "    TLB sending response: tag=" << rsp.tag << " cid=" << rsp.cid 
					   << " (translation available)" << std::endl;
					tlb_output_trace_count++;
				}
			}
		}
	}

	TlbSim::PerfStats perf_stats() const {
		TlbSim::PerfStats perf;
		for (auto tlb : tlb_) {
			if (tlb) perf.tlb += tlb->perf_stats();
		}
		return perf;
	}

private:
	std::vector<CacheSim::Ptr> tlb_;
};

}
