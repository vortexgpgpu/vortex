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
#include <vector>
#include <cstring>
#include "types.h"
#if VX_CFG_EXT_A_ENABLED
#include "amo_ops.h"
#include "amo_unit.h"
#endif

using namespace vortex;

class LocalMem::Impl {
protected:
	LocalMem* simobject_;
	Config    config_;
	RAM       ram_;
	uint32_t 	addr_bits_;
#if VX_CFG_EXT_A_ENABLED
	AmoUnit   amo_unit_;
#endif
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
		, addr_bits_(log2ceil(config.capacity))
#if VX_CFG_EXT_A_ENABLED
		, amo_unit_(VX_CFG_AMO_RS_SIZE < 2 ? 2u : (uint32_t)VX_CFG_AMO_RS_SIZE)
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

#if VX_CFG_EXT_A_ENABLED
			// Shared-memory atomics: read-modify-write in place and ALWAYS
			// return the old word. An AMO has a destination register, so the
			// LSU reserved a response slot; a missing response hangs the warp
			// (this was the prior behaviour — LMEM had no AMO path). LR/SC
			// reservations are tracked per hart for correct contended
			// compare-swap, mirroring the dcache AMO path used for global/SSBO.
			if (memop_is_atomic(bank_req.op)) {
				const uint32_t bmask   = VX_CFG_MEM_BLOCK_SIZE - 1;
				const uint64_t la      = to_local_addr(bank_req.addr);
				const uint32_t byte_off = (uint32_t)(la & bmask);
				const uint32_t wbytes  = bank_req.byteen
					? (uint32_t)__builtin_popcountll(bank_req.byteen) : 4u;
				const uint8_t  width   = (wbytes >= 8) ? 3 : 2;
				const MemOp    op      = bank_req.op;
				const bool     is_lr   = (op == MemOp::AMO_LR);
				const bool     is_sc   = (op == MemOp::AMO_SC);
				const bool     sc_fail = is_sc && !amo_unit_.check(bank_req.hart_id, la);
				const bool     do_store = !is_lr && !sc_fail;

				uint8_t obuf[8] = {0};
				ram_.read(obuf, la, wbytes);
				const uint64_t old_word = amo_load_word(obuf, 0, width);
				const uint64_t rhs = bank_req.data
					? amo_load_word(bank_req.data->data(), byte_off, width) : 0;
				auto rmw = amo_compute(op, width, old_word, rhs,
				                       bank_req.flags.amo_unsigned);

				// Build the response before mutating any state so a full RspIn
				// can stall-retry cleanly.
				MemRsp bank_rsp{bank_req.tag, bank_req.hart_id, bank_req.uuid};
				auto rsp_data = make_mem_block();
				std::memset(rsp_data->data(), 0, rsp_data->size());
				const uint64_t ret_word = is_sc ? (sc_fail ? 1ull : 0ull) : rmw.ret_word;
				amo_store_word(rsp_data->data(), byte_off, width, ret_word);
				bank_rsp.data = rsp_data;
				if (!mem_xbar_->RspIn.at(i).try_send(bank_rsp))
					continue; // stall; no state mutated yet

				if (is_lr)      amo_unit_.reserve(bank_req.hart_id, la);
				else if (is_sc) amo_unit_.clear(bank_req.hart_id, la);
				else            amo_unit_.invalidate(la, bank_req.hart_id);
				if (do_store) {
					uint8_t sbuf[8];
					amo_store_word(sbuf, 0, width, rmw.new_word);
					ram_.write(sbuf, la, wbytes);
				}
				perf_stats_.reads  += !do_store;
				perf_stats_.writes += do_store;
				DT(4, simobject_->name() << "-bank" << i << " amo : " << bank_req);
				xbar_req_out.pop();
				continue;
			}
#endif

			// Apply byte-enabled writes from TLM payload to local RAM.
			if (bank_req.is_write() && bank_req.data) {
				uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
				for (uint32_t b = 0; b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
					if (bank_req.byteen & (1ull << b)) {
						uint8_t value = (*bank_req.data)[b];
						ram_.write(&value, line_addr + b, 1);
					}
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
			perf_stats_.writes += bank_req.is_write();

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
