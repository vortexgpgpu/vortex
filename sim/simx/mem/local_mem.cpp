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
#include <mem.h>
#include <bitmanip.h>
#include <cstring>
#include <vector>
#include "types.h"
#if VX_CFG_EXT_A_ENABLED
#include "amo_unit.h"
#endif

using namespace vortex;

class LocalMem::Impl {
protected:
	LocalMem* simobject_;
	Config    config_;
	RAM       ram_;
	uint32_t 	addr_bits_;
	MemCrossBar::Ptr mem_xbar_;
	mutable PerfStats perf_stats_;
#if VX_CFG_EXT_A_ENABLED
	AmoUnit amo_unit_;
#endif

	uint64_t to_local_addr(uint64_t addr) {
		return bit_getw(addr, 0, addr_bits_-1);
	}

	uint32_t byte_offset(uint64_t addr) {
		return (uint32_t)(to_local_addr(addr) & (VX_CFG_MEM_BLOCK_SIZE - 1));
	}

#if VX_CFG_EXT_A_ENABLED
	bool commit_amo(uint32_t bank_id, const MemReq& bank_req) {
		auto& rsp_in = mem_xbar_->RspIn.at(bank_id);
		if (rsp_in.full())
			return false;

		const uint32_t byte_off = byte_offset(bank_req.addr);
		const uint8_t width = (__builtin_popcountll(bank_req.byteen) >= 8) ? 3 : 2;
		const uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
		const uint64_t rhs = bank_req.data
		                   ? amo_load_word(bank_req.data->data(), byte_off, width)
		                   : 0ull;

		auto line = make_mem_block();
		ram_.read(line->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);

		const uint64_t old_word = amo_load_word(line->data(), byte_off, width);
		const uint32_t hid = bank_req.hart_id;
		const bool sc_fail = (bank_req.op == MemOp::AMO_SC) && !amo_unit_.check(hid, line_addr);
		const bool do_store = (bank_req.op != MemOp::AMO_LR) && !sc_fail;
		auto rmw = amo_unit_.compute(bank_req.op, width, old_word, rhs, bank_req.flags.amo_unsigned);

		auto rsp_data = make_mem_block();
		std::memset(rsp_data->data(), 0, rsp_data->size());
		const uint64_t rsp_word = (bank_req.op == MemOp::AMO_SC)
		                        ? (sc_fail ? 1ull : 0ull)
		                        : rmw.ret_word;
		amo_store_word(rsp_data->data(), byte_off, width, rsp_word);

		if (bank_req.op == MemOp::AMO_LR) {
			amo_unit_.reserve(hid, line_addr);
		} else if (bank_req.op == MemOp::AMO_SC) {
			amo_unit_.clear(hid, line_addr);
		}

		if (do_store) {
			amo_store_word(line->data(), byte_off, width, rmw.new_word);
			ram_.write(line->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
			amo_unit_.invalidate(line_addr, hid);
		}

		MemRsp bank_rsp{bank_req.tag, bank_req.hart_id, bank_req.uuid, rsp_data};
		rsp_in.send(bank_rsp);
		return true;
	}
#endif

public:
	Impl(LocalMem* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, ram_(config.capacity)
		, addr_bits_(log2ceil(config.capacity))
#if VX_CFG_EXT_A_ENABLED
		, amo_unit_(__MAX(2u, (uint32_t)VX_CFG_AMO_RS_SIZE))
#endif
	{
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
#if VX_CFG_EXT_A_ENABLED
		amo_unit_.reset();
#endif
	}

	void tick() {
		// process bank requets from xbar
		uint32_t num_banks = (1 << config_.B);
		for (uint32_t i = 0; i < num_banks; ++i) {
			auto& xbar_req_out = mem_xbar_->ReqOut.at(i);
			if (xbar_req_out.empty())
				continue;

			auto& bank_req = xbar_req_out.peek();

			if (memop_is_atomic(bank_req.op)) {
#if VX_CFG_EXT_A_ENABLED
				if (!commit_amo(i, bank_req))
					continue; // stall
				DT(4, simobject_->name() << "-bank" << i << " amo-req : " << bank_req);
				perf_stats_.reads += 1;
				if (bank_req.op != MemOp::AMO_LR)
					perf_stats_.writes += 1;
				xbar_req_out.pop();
				continue;
#else
				assert(false && "AMO on Shared (LMEM) requires VX_CFG_EXT_A_ENABLE");
#endif
			}

			// Apply byte-enabled writes from TLM payload to local RAM.
			if (bank_req.is_write() && bank_req.data) {
				uint32_t replay_count = 1;
				uint32_t replay_stride = 0;
			#ifdef VX_CFG_EXT_DXA_ENABLE
				replay_count = bank_req.dxa_mcast_count ? bank_req.dxa_mcast_count : 1;
				replay_stride = bank_req.dxa_mcast_stride;
			#endif
				for (uint32_t r = 0; r < replay_count; ++r) {
					uint64_t replay_addr = bank_req.addr + uint64_t(r) * replay_stride;
					uint64_t line_addr = to_local_addr(replay_addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
					for (uint32_t b = 0; b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
						if (bank_req.byteen & (1ull << b)) {
							uint8_t value = (*bank_req.data)[b];
							ram_.write(&value, line_addr + b, 1);
						}
					}
#if VX_CFG_EXT_A_ENABLED
					amo_unit_.invalidate(line_addr, bank_req.hart_id);
#endif
				}
			}

			// Loads always respond. Stores respond when configured globally OR
			// the request opts in via MEM_FLAG_STRSP.
			if (!bank_req.is_write() || config_.write_reponse || bank_req.flags.strsp) {
				// send xbar response — for reads, capture the line payload.
				MemRsp bank_rsp{bank_req.tag, bank_req.hart_id, bank_req.uuid};
				if (!bank_req.is_write()) {
					auto rsp_data = make_mem_block();
					uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
					ram_.read(rsp_data->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
					bank_rsp.data = rsp_data;
				}
				if (!mem_xbar_->RspIn.at(i).try_send(bank_rsp))
					continue; // stall
			}

			DT(4, simobject_->name() << "-bank" << i << " req : " << bank_req);

			// update perf counters
			perf_stats_.reads += !bank_req.is_write();
			if (bank_req.is_write()) {
				uint32_t replay_count = 1;
			#ifdef VX_CFG_EXT_DXA_ENABLE
				replay_count = bank_req.dxa_mcast_count ? bank_req.dxa_mcast_count : 1;
			#endif
				perf_stats_.writes += replay_count;
			}

			// remove input
			xbar_req_out.pop();
		}
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
		return perf_stats_;
	}

	uint32_t read_word(uint64_t local_addr) {
		uint32_t word = 0;
		uint64_t off = bit_getw(local_addr, 0, addr_bits_-1);
		ram_.read(&word, off, 4);
		return word;
	}

	int32_t atomic_add_word(uint64_t local_addr, int32_t value) {
		uint64_t off = bit_getw(local_addr, 0, addr_bits_-1);
#if VX_CFG_EXT_A_ENABLED
		uint64_t line_addr = off & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
#endif
		int32_t old_word = 0;
		ram_.read(&old_word, off, 4);
		int32_t new_word = old_word + value;
		ram_.write(&new_word, off, 4);
#if VX_CFG_EXT_A_ENABLED
		amo_unit_.invalidate(line_addr, ~uint32_t(0));
#endif
		return old_word;
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

void LocalMem::on_reset() {
  impl_->reset();
}

void LocalMem::on_tick() {
  impl_->tick();
}

const LocalMem::PerfStats& LocalMem::perf_stats() const {
  return impl_->perf_stats();
}

uint32_t LocalMem::read_word(uint64_t local_addr) {
  return impl_->read_word(local_addr);
}

int32_t LocalMem::atomic_add_word(uint64_t local_addr, int32_t value) {
  return impl_->atomic_add_word(local_addr, value);
}
