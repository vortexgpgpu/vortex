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

namespace vortex {

class TlbSim : public SimObject<TlbSim> {
public:
	struct PerfStats {
    	CacheSim::PerfStats tlb;
  	};
	std::vector<SimPort<MemReq>> CoreReqPorts;
	std::vector<SimPort<MemRsp>> CoreRspPorts;
	std::vector<SimPort<MemReq>> MemReqPorts;
	std::vector<SimPort<MemRsp>> MemRspPorts;

	TlbSim(const SimContext& ctx,
			const char* name,
			uint32_t /* num_inputs */,
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

		// Page table Walker Implementation
		// std::vector<MemArbiter::Ptr> ptw();

		// Connect tlb
		snprintf(sname, 100, "%s-cache%d", name, 0);
		tlb_.at(0) = CacheSim::Create(sname, tlb_config2);

		//Connect input to TLB(Cache)
		for (uint32_t j = 0; j < tlb_config.num_inputs; ++j) {			
			this->CoreReqPorts.at(j).bind(&tlb_.at(0)->CoreReqPorts.at(j));
			tlb_.at(0)->CoreRspPorts.at(j).bind(&this->CoreRspPorts.at(j));
		}

		// TLB memory ports for page table walks
		for (uint32_t i = 0; i < tlb_config.mem_ports; ++i) {
			tlb_.at(0)->MemReqPorts.at(i).bind(&this->MemReqPorts.at(i));
			this->MemRspPorts.at(i).bind(&tlb_.at(0)->MemRspPorts.at(i));
		}
	}

	~TlbSim() {}

	void reset() {}

	void tick() {
		static uint64_t tick_count = 0;
		if ((tick_count % 1000) == 0) {
			DT(2, this->name() << "-tick: count=" << tick_count 
				<< " tlb_req_empty=" << (tlb_.at(0) ? tlb_.at(0)->CoreReqPorts.at(0).empty() : 1)
				<< " tlb_rsp_empty=" << (tlb_.at(0) ? tlb_.at(0)->CoreRspPorts.at(0).empty() : 1));
		}
		tick_count++;
		
		for (auto tlb : tlb_) {
			tlb->tick();
		}
	}

	TlbSim::PerfStats perf_stats() const {
		TlbSim::PerfStats perf;
		for (auto tlb : tlb_) {
			perf.tlb += tlb->perf_stats();
		}
		return perf;
	}

private:
  std::vector<CacheSim::Ptr> tlb_;
};

}
