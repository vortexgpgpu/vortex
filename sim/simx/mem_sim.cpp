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

#include "mem_sim.h"
#include <vector>
#include <queue>
#include <stdlib.h>
#include <dram_sim.h>

#include "constants.h"
#include "types.h"
#include "debug.h"

using namespace vortex;

class MemSim::Impl {
private:
	MemSim*   simobject_;
	Config    config_;
	MemCrossBar::Ptr mem_xbar_;
	DramSim   dram_sim_;
	mutable PerfStats perf_stats_;
	struct DramCallbackArgs {
		MemSim::Impl* memsim;
		MemReq request;
		uint32_t bank_id;
	};

public:
	Impl(MemSim* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, dram_sim_(config.num_banks, config.block_size, config.clock_ratio)
	{
		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, config.num_ports, config.num_banks, 1,
			[lg2_block_size = log2ceil(config.block_size), num_banks = config.num_banks](const MemCrossBar::ReqType& req) {
    	// Custom logic to calculate the output index using bank interleaving
			return (uint32_t)((req.addr >> lg2_block_size) & (num_banks-1));
		});
		for (uint32_t i = 0; i < config.num_ports; ++i) {
			simobject->MemReqPorts.at(i).bind(&mem_xbar_->ReqIn.at(i));
			mem_xbar_->RspIn.at(i).bind(&simobject->MemRspPorts.at(i));
		}
	}

	~Impl() {
		//--
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->req_collisions();
		return perf_stats_;
	}

	void reset() {
		dram_sim_.reset();
	}

	void tick() {
		dram_sim_.tick();

		for (uint32_t i = 0; i < config_.num_banks; ++i) {
			if (mem_xbar_->ReqOut.at(i).empty())
				continue;

			auto& mem_req = mem_xbar_->ReqOut.at(i).front();

			// enqueue the request to the memory system
			auto req_args = new DramCallbackArgs{this, mem_req, i};
			dram_sim_.send_request(
				mem_req.addr,
				mem_req.write,
				[](void* arg) {
					auto rsp_args = reinterpret_cast<const DramCallbackArgs*>(arg);
					if (!rsp_args->request.write) {
						// only send a response for read requests
						MemRsp mem_rsp{rsp_args->request.tag, rsp_args->request.cid, rsp_args->request.uuid};
						rsp_args->memsim->mem_xbar_->RspOut.at(rsp_args->bank_id).push(mem_rsp, 1);
						DT(3, rsp_args->memsim->simobject_->name() << "-mem-rsp[" << rsp_args->bank_id << "]: " << mem_rsp);
					}
					delete rsp_args;
				},
				req_args
			);

			DT(3, simobject_->name() << "-mem-req[" << i << "]: " << mem_req);
			mem_xbar_->ReqOut.at(i).pop();
		}
	}
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<MemSim>(ctx, name)
	, MemReqPorts(config.num_ports, this)
	, MemRspPorts(config.num_ports, this)
	, impl_(new Impl(this, config))
{}

MemSim::~MemSim() {
  delete impl_;
}

void MemSim::reset() {
  impl_->reset();
}

void MemSim::tick() {
  impl_->tick();
}

const MemSim::PerfStats &MemSim::perf_stats() const {
	return impl_->perf_stats();
}