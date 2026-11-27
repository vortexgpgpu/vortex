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

#include <VX_types.h>
#include "memory.h"
#include <vector>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <stdlib.h>
#include <dram_sim.h>

#include "mem_block_pool.h"
#include "constants.h"
#include "types.h"
#include "debug.h"
#include "VX_config.h"

using namespace vortex;

class Memory::Impl {
private:
	// A completed read is not visible to the device the same cycle it leaves
	// the DRAM model: the platform stages the response and completes a
	// registered valid/ready handshake at the device boundary, adding two
	// cycles beyond the response register.
	static constexpr uint64_t RSP_BOUNDARY_DELAY = 3;

	Memory*   simobject_;
	Config    config_;
	MemCrossBar::Ptr mem_xbar_;
	// Per DRAM channel: the crossbar's requests, and the responses back to it.
	SimPortInArray<MemReq>  bank_req_in_;
	SimPortOutArray<MemRsp> bank_rsp_out_;
	DramSim   dram_sim_;
	RAM*      ram_;
	Memory::PreSendHook pre_send_hook_;
	mutable PerfStats perf_stats_;
	Occupancy pending_reads_;
	struct DramCallbackArgs {
		Memory::Impl* memsim;
		MemReq request;
		uint32_t bank_id;
		std::shared_ptr<mem_block_t> rsp_data;  // captured at request time for reads
	};

public:
	Impl(Memory* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, bank_req_in_(simobject, config.dram.num_channels, 2)
		, bank_rsp_out_(simobject, config.dram.num_channels)
		, dram_sim_(config.dram)
		, ram_(nullptr)
	{
		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, config.num_ports, config.dram.num_channels,
			[lg2_block_size = log2ceil(config.dram.block_size), num_banks = config.dram.num_channels](const MemCrossBar::ReqType& req) {
    	// Custom logic to calculate the output index using bank interleaving
			return (uint32_t)((req.addr >> lg2_block_size) & (num_banks-1));
		});
		for (uint32_t i = 0; i < config.num_ports; ++i) {
			simobject->mem_req_in.at(i) = &mem_xbar_->ReqIn.at(i);
			simobject->mem_rsp_out.at(i) = &mem_xbar_->RspOut.at(i);
		}
		for (uint32_t b = 0; b < config.dram.num_channels; ++b) {
			mem_xbar_->ReqOut.at(b).bind(&bank_req_in_.at(b));
			bank_rsp_out_.at(b).bind(&mem_xbar_->RspIn.at(b));
		}
	}

	~Impl() {}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
		perf_stats_.latency = pending_reads_.total();
		return perf_stats_;
	}

	// The counters are never cleared; a device reset drops the reads in
	// flight.
	void reset() {
		dram_sim_.reset();
		if (!SimPlatform::instance().launch_resetting()) {
			pending_reads_.add(-int64_t(pending_reads_.value()));
		}
	}

	void tick() {
		dram_sim_.tick();

		uint32_t num_banks = config_.dram.num_channels;
		uint32_t lg2_block_size = log2ceil(config_.dram.block_size);
		for (uint32_t i = 0; i < num_banks; ++i) {
			if (bank_req_in_.at(i).empty() || !dram_sim_.can_accept(i))
				continue;

			auto& mem_req = bank_req_in_.at(i).peek();

			std::shared_ptr<mem_block_t> rsp_data;
			if (ram_) {
				uint64_t line_addr = mem_req.addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
				// Cache fills/writebacks are simulator-internal traffic and
				// don't carry the kernel's intent (e.g. a write-back cache
				// reads a write-only buffer to fill the line on write-miss,
				// since memory buses lack per-region read/write permissions).
				// Suppress ACL for the duration; ACL still guards upload/download.
				ram_->enable_acl(false);
				if (mem_req.is_write()) {
					// Apply byte-enabled write to RAM at request arrival.
					// IO_COUT-range bytes are tapped to the print buffer and
					// not stored in RAM.
					if (mem_req.data) {
						for (uint32_t b = 0; b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
							if (mem_req.byteen & (1ull << b)) {
								uint8_t value = (*mem_req.data)[b];
								ram_->write(&value, line_addr + b, 1);
							}
						}
					}
				} else {
					// Capture the line at request time; response carries it back.
					rsp_data = make_mem_block();
					ram_->read(rsp_data->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
				}
				ram_->enable_acl(true);
			}

			if (pre_send_hook_) {
				pre_send_hook_(mem_req);
			}

			// enqueue the request to the memory system
			auto req_args = new DramCallbackArgs{this, mem_req, i, rsp_data};
			uint64_t bank_addr = ((mem_req.addr >> lg2_block_size) / num_banks) << lg2_block_size;
			dram_sim_.send_request(
				i,
				bank_addr,
				mem_req.is_write(),
				[](void* arg)->bool {
					auto rsp_args = reinterpret_cast<const DramCallbackArgs*>(arg);
					if (rsp_args->request.is_write()) {
						delete rsp_args;
						return true;
					} else {
								MemRsp mem_rsp{rsp_args->request.tag, rsp_args->request.hart_id, rsp_args->request.uuid};
						mem_rsp.data = rsp_args->rsp_data;
						if (rsp_args->memsim->bank_rsp_out_.at(rsp_args->bank_id).try_send(mem_rsp, RSP_BOUNDARY_DELAY)) {
							rsp_args->memsim->pending_reads_.add(-1);
							DT(3, rsp_args->memsim->simobject_->name() << " mem-rsp" << rsp_args->bank_id << ": " << mem_rsp);
							delete rsp_args;
							return true;
						}
					}
					return false; // stall
				},
				req_args
			);

			DT(3, simobject_->name() << " mem-req" << i << ": " << mem_req);
			if (mem_req.is_write()) {
				++perf_stats_.writes;
			} else {
				++perf_stats_.reads;
				pending_reads_.add(+1);
			}
			bank_req_in_.at(i).pop();
		}
	}

	void attach_ram(RAM* ram) {
		ram_ = ram;
	}

	void set_pre_send_hook(Memory::PreSendHook hook) {
		pre_send_hook_ = std::move(hook);
	}
};

///////////////////////////////////////////////////////////////////////////////

Memory::Memory(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<Memory>(ctx, name)
	, mem_req_in(config.num_ports)
	, mem_rsp_out(config.num_ports)
	, impl_(new Impl(this, config))
{}

Memory::~Memory() {
  delete impl_;
}

void Memory::on_reset() {
  impl_->reset();
}

void Memory::on_tick() {
  impl_->tick();
}

void Memory::attach_ram(RAM* ram) {
  impl_->attach_ram(ram);
}

void Memory::set_pre_send_hook(PreSendHook hook) {
  impl_->set_pre_send_hook(std::move(hook));
}

const Memory::PerfStats &Memory::perf_stats() const {
	return impl_->perf_stats();
}