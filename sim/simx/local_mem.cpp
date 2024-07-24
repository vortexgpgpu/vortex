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

#include "local_mem.h"
#include "core.h"
#include <bitmanip.h>
#include <vector>
#include "types.h"

using namespace vortex;

class LocalMem::Impl {
protected:
	LocalMem* simobject_;
	Config    config_;
	RAM       ram_;
	int32_t   bank_sel_addr_start_;
  int32_t   bank_sel_addr_end_;
	PerfStats perf_stats_;

	uint64_t to_local_addr(uint64_t addr) {
		uint32_t total_lines = config_.capacity / config_.line_size;
		uint32_t line_bits = log2ceil(total_lines);
		uint32_t offset = bit_getw(addr, 0, line_bits-1);
		return offset;
	}

public:
	Impl(LocalMem* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, ram_(config.capacity)
		, bank_sel_addr_start_(0)
		, bank_sel_addr_end_(config.B-1)
	{}

	virtual ~Impl() {}

	void reset() {
		perf_stats_ = PerfStats();
	}

	void read(void* data, uint64_t addr, uint32_t size) {
		auto s_addr = to_local_addr(addr);
		DPH(3, "Local Mem addr=0x" << std::hex << s_addr << std::endl);
		ram_.read(data, s_addr, size);
	}

	void write(const void* data, uint64_t addr, uint32_t size) {
		auto s_addr = to_local_addr(addr);
		DPH(3, "Local Mem addr=0x" << std::hex << s_addr << std::endl);
		ram_.write(data, s_addr, size);
	}

	void tick() {
		std::vector<bool> in_used_banks(1 << config_.B);
		for (uint32_t req_id = 0; req_id < config_.num_reqs; ++req_id) {
			auto& core_req_port = simobject_->Inputs.at(req_id);
			if (core_req_port.empty())
				continue;

			auto& core_req = core_req_port.front();

			uint32_t bank_id = 0;
			if (bank_sel_addr_end_ >= bank_sel_addr_start_) {
				bank_id = (uint32_t)bit_getw(core_req.addr, bank_sel_addr_start_, bank_sel_addr_end_);
			}

			// bank conflict check
			if (in_used_banks.at(bank_id)) {
				++perf_stats_.bank_stalls;
				continue;
			}

			DT(4, simobject_->name() << "-" << core_req);

			in_used_banks.at(bank_id) = true;

			if (!core_req.write || config_.write_reponse) {
				// send response
				MemRsp core_rsp{core_req.tag, core_req.cid, core_req.uuid};
				simobject_->Outputs.at(req_id).push(core_rsp, 1);
			}

			// update perf counters
			perf_stats_.reads += !core_req.write;
			perf_stats_.writes += core_req.write;

			// remove input
			core_req_port.pop();
		}
	}

	const PerfStats& perf_stats() const {
		return perf_stats_;
	}
};

///////////////////////////////////////////////////////////////////////////////

LocalMem::LocalMem(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<LocalMem>(ctx, name)
	, Inputs(config.num_reqs, this)
	, Outputs(config.num_reqs, this)
	, impl_(new Impl(this, config))
{}

LocalMem::~LocalMem() {
  delete impl_;
}

void LocalMem::reset() {
  impl_->reset();
}

void LocalMem::read(void* data, uint64_t addr, uint32_t size) {
  impl_->read(data, addr, size);
}

void LocalMem::write(const void* data, uint64_t addr, uint32_t size) {
  impl_->write(data, addr, size);
}

void LocalMem::tick() {
  impl_->tick();
}

const LocalMem::PerfStats& LocalMem::perf_stats() const {
  return impl_->perf_stats();
}