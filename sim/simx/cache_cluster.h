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

namespace vortex {

class CacheCluster : public SimObject<CacheCluster> {
public:
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
		, caches_(MAX(num_units, 0x1)) {

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
			input_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_inputs, num_units);
			for (uint32_t j = 0; j < num_inputs; ++j) {
				this->CoreReqPorts.at(j).at(i).bind(&input_arbs.at(i)->ReqIn.at(j));
				input_arbs.at(i)->RspIn.at(j).bind(&this->CoreRspPorts.at(j).at(i));
			}
		}

		// Arbitrate outgoing memory interfaces
		std::vector<MemArbiter::Ptr> mem_arbs(cache_config.mem_ports);
		for (uint32_t i = 0; i < cache_config.mem_ports; ++i) {
			snprintf(sname, 100, "%s-mem-arb%d", name, i);
			mem_arbs.at(i) = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_units, 1);
			mem_arbs.at(i)->ReqOut.at(0).bind(&this->MemReqPorts.at(i));
			this->MemRspPorts.at(i).bind(&mem_arbs.at(i)->RspOut.at(0));
		}

		// Connect caches
		for (uint32_t i = 0; i < num_units; ++i) {
			snprintf(sname, 100, "%s-cache%d", name, i);
			caches_.at(i) = CacheSim::Create(sname, cache_config2);

			for (uint32_t j = 0; j < cache_config.num_inputs; ++j) {
				input_arbs.at(j)->ReqOut.at(i).bind(&caches_.at(i)->CoreReqPorts.at(j));
				caches_.at(i)->CoreRspPorts.at(j).bind(&input_arbs.at(j)->RspOut.at(i));
			}

			for (uint32_t j = 0; j < cache_config.mem_ports; ++j) {
				caches_.at(i)->MemReqPorts.at(j).bind(&mem_arbs.at(j)->ReqIn.at(i));
				mem_arbs.at(j)->RspIn.at(i).bind(&caches_.at(i)->MemRspPorts.at(j));
			}
		}
	}

	~CacheCluster() {}

	void reset() {}

	void tick() {}

	CacheSim::PerfStats perf_stats() const {
		CacheSim::PerfStats perf;
		for (auto cache : caches_) {
			perf += cache->perf_stats();
		}
		return perf;
	}

private:
  std::vector<CacheSim::Ptr> caches_;
};

}
