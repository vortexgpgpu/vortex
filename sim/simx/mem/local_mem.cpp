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
#include "mem_block_pool.h"
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
	uint32_t 	addr_bits_;
	MemCrossBar::Ptr mem_xbar_;
	mutable PerfStats perf_stats_;

	uint64_t to_local_addr(uint64_t addr) {
		return bit_getw(addr, 0, addr_bits_-1);
	}

public:
	Impl(LocalMem* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, ram_(config.capacity)
	{
		line_bits_ = log2ceil(config.capacity);

		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		uint32_t lg2_line_size = log2ceil(config_.line_size);
		uint32_t num_banks = 1 << config.B;
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::Priority, config.num_reqs, num_banks,
		 [lg2_line_size, num_banks](const MemCrossBar::ReqType& req) {
    	// Custom logic to calculate the output index using bank interleaving
			return (uint32_t)((req.addr >> lg2_line_size) & (num_banks-1));
		});
		for (uint32_t i = 0; i < config.num_reqs; ++i) {
			simobject->Inputs.at(i).bind(&mem_xbar_->ReqIn.at(i));
			mem_xbar_->RspOut.at(i).bind(&simobject->Outputs.at(i));
		}
	}

	virtual ~Impl() {}

	void reset() {
		perf_stats_ = PerfStats();
	}

	void tick() {
#ifdef EXT_DXA_ENABLE
		// DXA writes take priority and stall normal Inputs for this cycle.
		if (!simobject_->dxa_req_in.empty()) {
			auto& req = simobject_->dxa_req_in.peek();
			DT(4, simobject_->name() << "-dxa smem write addr=0x"
			   << std::hex << (req.addr) << std::dec << ", is_last=" << req.is_last);
			perf_stats_.writes++;
			if (req.is_last) {
				req.core->barrier_event_release(req.bar_id);
			}
			simobject_->dxa_req_in.pop();
			return; // Inputs stalled this cycle
		}
#endif
		// process bank requets from xbar
		uint32_t num_banks = (1 << config_.B);
		for (uint32_t i = 0; i < num_banks; ++i) {
			auto& xbar_req_out = mem_xbar_->ReqOut.at(i);
			if (xbar_req_out.empty())
				continue;

			auto& bank_req = xbar_req_out.peek();

			// Apply byte-enabled writes from TLM payload to local RAM.
			if (bank_req.write && bank_req.data) {
				uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(MEM_BLOCK_SIZE - 1);
				for (uint32_t b = 0; b < MEM_BLOCK_SIZE; ++b) {
					if (bank_req.byteen & (1ull << b)) {
						uint8_t value = (*bank_req.data)[b];
						ram_.write(&value, line_addr + b, 1);
					}
				}
			}

			if (!bank_req.write || config_.write_reponse) {
				// send xbar response — for reads, capture the line payload.
				MemRsp bank_rsp{bank_req.tag, bank_req.cid, bank_req.uuid};
				if (!bank_req.write) {
					auto rsp_data = make_mem_block();
					uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(MEM_BLOCK_SIZE - 1);
					ram_.read(rsp_data->data(), line_addr, MEM_BLOCK_SIZE);
					bank_rsp.data = rsp_data;
				}
				if (!mem_xbar_->RspIn.at(i).try_send(bank_rsp))
					continue; // stall
			}

			DT(4, simobject_->name() << "-bank" << i << " req : " << bank_req);

			// update perf counters
			perf_stats_.reads += !bank_req.write;
			perf_stats_.writes += bank_req.write;

			// remove input
			xbar_req_out.pop();
		}
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
		return perf_stats_;
	}
};

///////////////////////////////////////////////////////////////////////////////

LocalMem::LocalMem(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<LocalMem>(ctx, name)
	, Inputs(config.num_reqs, this)
	, Outputs(config.num_reqs, this)
#ifdef EXT_DXA_ENABLE
	, dxa_req_in(this)
#endif
	, impl_(new Impl(this, config))
{}

LocalMem::~LocalMem() {
  delete impl_;
}

void LocalMem::on_reset() {
  impl_->reset();
}

void LocalMem::on_tick() {
  impl_->tick();
}

const LocalMem::PerfStats& LocalMem::perf_stats() const {
  return impl_->perf_stats();
}