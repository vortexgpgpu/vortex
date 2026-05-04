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
	Memory*   simobject_;
	Config    config_;
	MemCrossBar::Ptr mem_xbar_;
	DramSim   dram_sim_;
	RAM*      ram_;
	mutable PerfStats perf_stats_;
	std::unordered_map<int, std::stringstream> print_bufs_;

public:
	// Phase 3 SST: hook called for every accepted MemReq just before
	// enqueueing to dram_sim_. See Memory::set_pre_send_hook().
	Memory::PreSendHook pre_send_hook_;
private:

	// Tap byte writes that fall in the IO_COUT range and route them to the
	// per-thread print buffer. Returns true if the byte was consumed (no RAM
	// write needed).
	bool io_cout_tap(uint64_t addr, uint8_t byte) {
		if (addr < uint64_t(IO_COUT_ADDR)
		 || addr >= (uint64_t(IO_COUT_ADDR) + IO_COUT_SIZE))
			return false;
		uint32_t tid = (addr - IO_COUT_ADDR) & (IO_COUT_SIZE - 1);
		auto& ss_buf = print_bufs_[tid];
		char c = (char)byte;
		ss_buf << c;
		if (c == '\n') {
			std::cout << "#" << tid << ": " << ss_buf.str() << std::flush;
			ss_buf.str("");
		}
		return true;
	}

	void cout_flush() {
		for (auto& buf : print_bufs_) {
			auto str = buf.second.str();
			if (!str.empty()) {
				std::cout << "#" << buf.first << ": " << str << std::endl;
			}
		}
		print_bufs_.clear();
	}
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
		, dram_sim_(config.num_banks, config.block_size, config.clock_ratio)
		, ram_(nullptr)
	{
		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, config.num_ports, config.num_banks,
			[lg2_block_size = log2ceil(config.block_size), num_banks = config.num_banks](const MemCrossBar::ReqType& req) {
    	// Custom logic to calculate the output index using bank interleaving
			return (uint32_t)((req.addr >> lg2_block_size) & (num_banks-1));
		});
		for (uint32_t i = 0; i < config.num_ports; ++i) {
			simobject->mem_req_in.at(i).bind(&mem_xbar_->ReqIn.at(i));
			mem_xbar_->RspOut.at(i).bind(&simobject->mem_rsp_out.at(i));
		}
	}

	~Impl() {
		this->cout_flush();
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
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

			auto& mem_req = mem_xbar_->ReqOut.at(i).peek();

			std::shared_ptr<mem_block_t> rsp_data;
			if (ram_) {
				uint64_t line_addr = mem_req.addr & ~uint64_t(MEM_BLOCK_SIZE - 1);
				// Cache fills/writebacks are simulator-internal traffic and
				// don't carry the kernel's intent (e.g. a write-back cache
				// reads a write-only buffer to fill the line on write-miss,
				// matching real-hardware behavior since memory buses lack
				// per-region read/write permissions). Suppress ACL for the
				// duration of the access; ACL still guards upload/download.
				ram_->enable_acl(false);
				if (mem_req.write) {
					// Apply byte-enabled write to RAM at request arrival.
					// IO_COUT-range bytes are tapped to the print buffer and
					// not stored in RAM.
					if (mem_req.data) {
						for (uint32_t b = 0; b < MEM_BLOCK_SIZE; ++b) {
							if (mem_req.byteen & (1ull << b)) {
								uint8_t value = (*mem_req.data)[b];
								uint64_t byte_addr = line_addr + b;
								if (this->io_cout_tap(byte_addr, value))
									continue;
								ram_->write(&value, byte_addr, 1);
							}
						}
					}
				} else {
					// Capture the line at request time; response carries it back.
					rsp_data = make_mem_block();
					ram_->read(rsp_data->data(), line_addr, MEM_BLOCK_SIZE);
				}
				ram_->enable_acl(true);
			}

			// Phase 3 SST: notify external observer (e.g. SST memHierarchy
			// link) before enqueueing locally. Hook is no-op when unset.
			if (pre_send_hook_) {
				pre_send_hook_(mem_req);
			}

			// enqueue the request to the memory system
			auto req_args = new DramCallbackArgs{this, mem_req, i, rsp_data};
			dram_sim_.send_request(
				mem_req.addr,
				mem_req.write,
				[](void* arg)->bool {
					auto rsp_args = reinterpret_cast<const DramCallbackArgs*>(arg);
					if (rsp_args->request.write) {
						delete rsp_args;
						return true;
					} else {
						// only send a response for read requests
						MemRsp mem_rsp{rsp_args->request.tag, rsp_args->request.cid, rsp_args->request.uuid};
						mem_rsp.data = rsp_args->rsp_data;
						if (rsp_args->memsim->mem_xbar_->RspIn.at(rsp_args->bank_id).try_send(mem_rsp)) {
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
			mem_xbar_->ReqOut.at(i).pop();
		}
	}

	void attach_ram(RAM* ram) {
		ram_ = ram;
	}
};

///////////////////////////////////////////////////////////////////////////////

Memory::Memory(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<Memory>(ctx, name)
	, mem_req_in(config.num_ports, this)
	, mem_rsp_out(config.num_ports, this)
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

void Memory::set_pre_send_hook(Memory::PreSendHook hook) {
  impl_->pre_send_hook_ = std::move(hook);
}

const Memory::PerfStats &Memory::perf_stats() const {
	return impl_->perf_stats();
}