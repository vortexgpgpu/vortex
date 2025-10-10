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
#include "ptw.h"

namespace vortex {

class TlbSim : public SimObject<TlbSim> {
public:
	struct PerfStats {
    	CacheSim::PerfStats tlb;
		PTW::PerfStats ptw;
		
		PerfStats& operator+=(const PerfStats& rhs) {
			this->tlb += rhs.tlb;
			this->ptw += rhs.ptw;
			return *this;
		}
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
		, tlb_(MAX(num_units, 0x1))
		, ptw_(MAX(num_units, 0x1)) {

		CacheSim::Config tlb_config2(tlb_config);
		if (0 == num_units) {
			num_units = 1;
			tlb_config2.bypass = true;
		}

		char sname[100];

		// Create TLB cache
		snprintf(sname, 100, "%s-cache%d", name, 0);
		tlb_.at(0) = CacheSim::Create(sname, tlb_config2);

		// Create Page Table Walker (simplified - single port)
		snprintf(sname, 100, "%s-ptw%d", name, 0);
		PTW::Config ptw_config;
		ptw_config.pt_levels = PT_LEVEL;
		ptw_config.pte_size = PTE_SIZE;
		ptw_.at(0) = PTW::Create(sname, ptw_config);

		// Connect input to TLB(Cache)
		for (uint32_t j = 0; j < tlb_config.num_inputs; ++j) {			
			this->CoreReqPorts.at(j).bind(&tlb_.at(0)->CoreReqPorts.at(j));
			tlb_.at(0)->CoreRspPorts.at(j).bind(&this->CoreRspPorts.at(j));
		}

		// Connect TLB miss port (first MemReqPort) to PTW
		// PTW handles one walk at a time sequentially
		tlb_.at(0)->MemReqPorts.at(0).bind(&ptw_.at(0)->CoreReqPort);
		ptw_.at(0)->CoreRspPort.bind(&tlb_.at(0)->MemRspPorts.at(0));

		// Connect PTW to external memory port for PTE reads
		ptw_.at(0)->MemReqPort.bind(&this->MemReqPorts.at(0));
		this->MemRspPorts.at(0).bind(&ptw_.at(0)->MemRspPort);
		
		DT(1, "TlbSim created with PTW: " << name);
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
		
		// Tick TLB and PTW
		for (auto tlb : tlb_) {
			if (tlb) tlb->tick();
		}
		for (auto ptw : ptw_) {
			if (ptw) ptw->tick();
		}
	}

	TlbSim::PerfStats perf_stats() const {
		TlbSim::PerfStats perf;
		for (auto tlb : tlb_) {
			if (tlb) perf.tlb += tlb->perf_stats();
		}
		for (auto ptw : ptw_) {
			if (ptw) perf.ptw += ptw->perf_stats();
		}
		return perf;
	}

private:
	std::vector<CacheSim::Ptr> tlb_;
	std::vector<PTW::Ptr> ptw_;
};

}
