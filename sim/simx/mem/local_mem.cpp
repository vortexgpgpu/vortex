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
#include <algorithm>
#include <cassert>
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
	// Per-bank atomic occupancy. A bank has a single port and the value read out
	// is registered before the update is computed, so an atomic owns its bank
	// for three cycles -- read, capture, write back -- and it accepts nothing
	// else meanwhile.
	static constexpr uint32_t AMO_BANK_STALL_CYCLES = 2;
	std::vector<uint32_t> amo_busy_;
	// A read issued the cycle after a write to the same word returns the word's
	// pre-update value, so the bank holds such a read off for one cycle. The
	// write-back is the third cycle of the occupancy, so this shadow covers the
	// fourth -- the cycle the next contending atomic arrives, which is why
	// back-to-back atomics on one word cost four cycles and not three.
	std::vector<uint64_t> amo_rdw_addr_;
	std::vector<bool>     amo_rdw_valid_;
	// Whether the atomic now occupying the bank ends in a write at all. A
	// load-reserved never writes and a failed store-conditional must not, so
	// neither leaves a shadow behind.
	std::vector<bool>     amo_rdw_pending_;
#endif
	MemCrossBar::Ptr lsu_xbar_;
	MemCrossBar::Ptr dma_xbar_;
	mutable PerfStats perf_stats_;

	uint64_t to_local_addr(uint64_t addr) {
		return bit_getw(addr, 0, addr_bits_-1);
	}

#if VX_CFG_EXT_A_ENABLED
	// The word a bank is addressed by. Two byte addresses falling in one word
	// are the same access as far as the bank is concerned.
	uint64_t bank_word(uint64_t addr) {
		return to_local_addr(addr) >> log2ceil(config_.line_size);
	}
#endif

public:
	Impl(LocalMem* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, ram_(config.capacity)
		, addr_bits_(log2ceil(config.capacity))
#if VX_CFG_EXT_A_ENABLED
		, amo_unit_(VX_CFG_AMO_RS_SIZE < 2 ? 2u : (uint32_t)VX_CFG_AMO_RS_SIZE)
		, amo_busy_(1 << config.B, 0)
		, amo_rdw_addr_(1 << config.B, 0)
		, amo_rdw_valid_(1 << config.B, false)
		, amo_rdw_pending_(1 << config.B, false)
#endif
	{
		char sname[100];
		uint32_t lg2_line_size = log2ceil(config_.line_size);
		uint32_t num_banks = 1 << config.B;
		auto bank_sel = [lg2_line_size, num_banks](const MemCrossBar::ReqType& req) {
			// bank interleaving
			return (uint32_t)((req.addr >> lg2_line_size) & (num_banks-1));
		};
		// Inputs [0, num_lsu_reqs) are LSU ports; the rest are DMA ports.
		assert(config.num_lsu_reqs != 0 && config.num_lsu_reqs <= config.num_reqs);
		uint32_t num_dma_reqs = config.num_reqs - config.num_lsu_reqs;
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		lsu_xbar_ = MemCrossBar::Create(sname, ArbiterType::Priority, config.num_lsu_reqs, num_banks, bank_sel);
		if (num_dma_reqs != 0) {
			snprintf(sname, 100, "%s-dma_xbar", simobject->name().c_str());
			dma_xbar_ = MemCrossBar::Create(sname, ArbiterType::Priority, num_dma_reqs, num_banks, bank_sel);
		}
		for (uint32_t i = 0; i < config.num_reqs; ++i) {
			bool is_lsu = (i < config.num_lsu_reqs);
			auto& xbar = is_lsu ? lsu_xbar_ : dma_xbar_;
			uint32_t port = is_lsu ? i : (i - config.num_lsu_reqs);
			simobject->Inputs.at(i).bind(&xbar->ReqIn.at(port));
			xbar->RspOut.at(port).bind(&simobject->Outputs.at(i));
		}
	}

	virtual ~Impl() {}

	void reset() {
		perf_stats_ = PerfStats();
#if VX_CFG_EXT_A_ENABLED
		amo_unit_.reset();
		std::fill(amo_busy_.begin(), amo_busy_.end(), 0);
		std::fill(amo_rdw_valid_.begin(), amo_rdw_valid_.end(), false);
		std::fill(amo_rdw_pending_.begin(), amo_rdw_pending_.end(), false);
#endif
	}

	enum class Served { None, Read, Write };

	// Serve the head request this crossbar routed to bank i.
	Served serve(MemCrossBar& xbar, uint32_t i, bool rdw_shadow, uint64_t rdw_addr) {
		auto& xbar_req_out = xbar.ReqOut.at(i);
		if (xbar_req_out.empty())
			return Served::None;
#if !VX_CFG_EXT_A_ENABLED
		(void)rdw_shadow; (void)rdw_addr;
#endif
		auto& bank_req = xbar_req_out.peek();
		const bool bank_req_is_write = bank_req.is_write();

#if VX_CFG_EXT_A_ENABLED
		// A read of the word the write-back just stored waits a cycle. An
		// atomic reads the bank for its old value, so it counts as a reader
		// here even though the request carries a write flag; that is what
		// makes contending atomics on one word cost four cycles.
		const bool reads_bank = !bank_req.is_write() || memop_is_atomic(bank_req.op);
		if (rdw_shadow
		 && reads_bank
		 && this->bank_word(bank_req.addr) == rdw_addr) {
			return Served::None;
		}

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
			                       bank_req.flags.amo_unsigned,
			                       bank_req.amo_cmp);

			// Build the response before mutating any state so a full RspIn
			// can stall-retry cleanly.
			MemRsp bank_rsp{bank_req.tag, bank_req.hart_id, bank_req.uuid};
			auto rsp_data = make_mem_block();
			std::memset(rsp_data->data(), 0, rsp_data->size());
			const uint64_t ret_word = is_sc ? (sc_fail ? 1ull : 0ull) : rmw.ret_word;
			amo_store_word(rsp_data->data(), byte_off, width, ret_word);
			bank_rsp.data = rsp_data;
			if (!xbar.RspIn.at(i).try_send(bank_rsp))
				return Served::None; // stall; no state mutated yet

			// An SC gives up its own reservation whether it succeeds or
			// fails; an LR claims one. Everything else is a plain RMW,
			// whose commit breaks the other harts' reservations below.
			if (is_lr)      amo_unit_.reserve(bank_req.hart_id, la);
			else if (is_sc) amo_unit_.clear(bank_req.hart_id, la);
			if (do_store) {
				uint8_t sbuf[8];
				amo_store_word(sbuf, 0, width, rmw.new_word);
				ram_.write(sbuf, la, wbytes);
				// Any committed store breaks every other hart's reservation
				// on the word. Without this a second hart's stale
				// reservation survives the first hart's successful SC, so
				// both report success and one update is lost.
				amo_unit_.invalidate(la, bank_req.hart_id);
			}
			perf_stats_.reads  += !do_store;
			perf_stats_.writes += do_store;
			DT(4, simobject_->name() << "-bank" << i << " amo : " << bank_req);
			amo_rdw_addr_[i]    = this->bank_word(bank_req.addr);
			amo_rdw_pending_[i] = do_store;
			xbar_req_out.pop();
			amo_busy_[i] = AMO_BANK_STALL_CYCLES;   // capture, then write back
			return Served::Write;
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
#if VX_CFG_EXT_A_ENABLED
			// A plain store breaks a reservation on the word it writes, the
			// same as an atomic's commit does; otherwise a store between a
			// hart's load-reserved and its store-conditional would go
			// unnoticed and the conditional would wrongly succeed.
			amo_unit_.invalidate(to_local_addr(bank_req.addr), bank_req.hart_id);
#endif
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
			if (!xbar.RspIn.at(i).try_send(bank_rsp))
				return Served::None; // stall
		}

		DT(4, simobject_->name() << "-bank" << i << " req : " << bank_req);

		// update perf counters
		perf_stats_.reads += !bank_req.is_write();
		perf_stats_.writes += bank_req.is_write();

		// remove input
		xbar_req_out.pop();
		return bank_req_is_write ? Served::Write : Served::Read;
	}

	void tick() {
		// The bank SRAM is 1R1W. The DMA side (TCU reads, DXA writes) takes the
		// port in its direction first; an LSU request uses the other port in
		// the same cycle unless it needs the one the DMA side just took.
		uint32_t num_banks = (1 << config_.B);
		for (uint32_t i = 0; i < num_banks; ++i) {
			bool rdw_shadow = false;
			uint64_t rdw_addr = 0;
#if VX_CFG_EXT_A_ENABLED
			// Cycles an in-flight atomic owns; the bank serves nothing else.
			if (amo_busy_[i] != 0) {
				if (0 == --amo_busy_[i]) {
					amo_rdw_valid_[i] = amo_rdw_pending_[i]; // write-back just issued
				}
				continue;
			}
			// The shadow is one cycle wide: consume it here whether or not
			// anything is waiting on the bank.
			rdw_shadow = amo_rdw_valid_[i];
			rdw_addr   = amo_rdw_addr_[i];
			amo_rdw_valid_[i] = false;
#endif
			Served dma = dma_xbar_ ? this->serve(*dma_xbar_, i, rdw_shadow, rdw_addr) : Served::None;
			auto& lsu_out = lsu_xbar_->ReqOut.at(i);
			if (lsu_out.empty())
				continue;
			auto& lsu_req = lsu_out.peek();
			bool lsu_atomic = memop_is_atomic(lsu_req.op);
			bool lsu_write  = lsu_req.is_write();
			if (dma != Served::None
			 && (lsu_atomic || (lsu_write == (dma == Served::Write))))
				continue;
			this->serve(*lsu_xbar_, i, rdw_shadow, rdw_addr);
		}
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = lsu_xbar_->collisions()
		                        + (dma_xbar_ ? dma_xbar_->collisions() : 0);
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
