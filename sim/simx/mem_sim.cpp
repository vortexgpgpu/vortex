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
	DramSim   dram_sim_;
	PerfStats perf_stats_;

	struct DramCallbackArgs {
		MemSim* simobject;
		MemReq  request;
	};

public:
	Impl(MemSim* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, dram_sim_(MEM_CLOCK_RATIO)
	{}

	~Impl() {
		//--
	}

	const PerfStats& perf_stats() const {
		return perf_stats_;
	}

	void reset() {
		dram_sim_.reset();
	}

	void tick() {
		dram_sim_.tick();

		if (simobject_->MemReqPort.empty())
			return;

		auto& mem_req = simobject_->MemReqPort.front();

		// try to enqueue the request to the memory system
		auto req_args = new DramCallbackArgs{simobject_, mem_req};
		auto enqueue_success = dram_sim_.send_request(
			mem_req.write,
			mem_req.addr,
			0,
			[](void* arg) {
				auto rsp_args = reinterpret_cast<const DramCallbackArgs*>(arg);
				// only send a response for read requests
				if (!rsp_args->request.write) {
					MemRsp mem_rsp{rsp_args->request.tag, rsp_args->request.cid, rsp_args->request.uuid};
					rsp_args->simobject->MemRspPort.push(mem_rsp, 1);
					DT(3, rsp_args->simobject->name() << " mem-rsp: " << mem_rsp);
				}
				delete rsp_args;
			},
			req_args
		);

		// check if the request was enqueued successfully
		if (!enqueue_success) {
			delete req_args;
			return;
		}

		if (mem_req.write) {
			++perf_stats_.writes;
		} else {
			++perf_stats_.reads;
		}

		DT(3, simobject_->name() << " mem-req: " << mem_req);

		simobject_->MemReqPort.pop();
	}
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<MemSim>(ctx, name)
	, MemReqPort(this)
	, MemRspPort(this)
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