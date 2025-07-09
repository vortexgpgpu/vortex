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

#include "cache_sim.h"
#include "debug.h"
#include "types.h"
#include <util.h>
#include <unordered_map>
#include <vector>
#include <list>
#include <queue>

using namespace vortex;

struct params_t {
	uint32_t sets_per_bank;
	uint32_t lines_per_set;
	uint32_t words_per_line;
	uint32_t log2_num_inputs;

	int32_t word_select_addr_start;
	int32_t word_select_addr_end;

	int32_t bank_select_addr_start;
	int32_t bank_select_addr_end;

	int32_t set_select_addr_start;
	int32_t set_select_addr_end;

	int32_t tag_select_addr_start;
	int32_t tag_select_addr_end;

	params_t(const CacheSim::Config& config) {
		int32_t offset_bits = config.L - config.W;
		int32_t index_bits = config.C - (config.L + config.A + config.B);
		assert(offset_bits >= 0);
		assert(index_bits >= 0);

		this->log2_num_inputs = log2ceil(config.num_inputs);

		this->sets_per_bank  = 1 << index_bits;
		this->lines_per_set  = 1 << config.A;
		this->words_per_line = 1 << offset_bits;

		// Word select
		this->word_select_addr_start = config.W;
		this->word_select_addr_end = (this->word_select_addr_start+offset_bits-1);

		// Bank select
		this->bank_select_addr_start = (1+this->word_select_addr_end);
		this->bank_select_addr_end = (this->bank_select_addr_start+config.B-1);

		// Set select
		this->set_select_addr_start = (1+this->bank_select_addr_end);
		this->set_select_addr_end = (this->set_select_addr_start+index_bits-1);

		// Tag select
		this->tag_select_addr_start = (1+this->set_select_addr_end);
		this->tag_select_addr_end = (config.addr_width-1);
	}

	uint32_t addr_bank_id(uint64_t addr) const {
		if (bank_select_addr_end >= bank_select_addr_start)
			return (uint32_t)bit_getw(addr, bank_select_addr_start, bank_select_addr_end);
		else
			return 0;
	}

	uint32_t addr_set_id(uint64_t addr) const {
		if (set_select_addr_end >= set_select_addr_start)
			return (uint32_t)bit_getw(addr, set_select_addr_start, set_select_addr_end);
		else
			return 0;
	}

	uint64_t addr_tag(uint64_t addr) const {
		if (tag_select_addr_end >= tag_select_addr_start)
			return bit_getw(addr, tag_select_addr_start, tag_select_addr_end);
		else
			return 0;
	}

	uint64_t mem_addr(uint32_t bank_id, uint32_t set_id, uint64_t tag) const {
		uint64_t addr(0);
		if (bank_select_addr_end >= bank_select_addr_start)
			addr = bit_setw(addr, bank_select_addr_start, bank_select_addr_end, bank_id);
		if (set_select_addr_end >= set_select_addr_start)
			addr = bit_setw(addr, set_select_addr_start, set_select_addr_end, set_id);
		if (tag_select_addr_end >= tag_select_addr_start)
			addr = bit_setw(addr, tag_select_addr_start, tag_select_addr_end, tag);
		return addr;
	}
};

struct line_t {
	uint64_t tag;
	uint32_t lru_ctr;
	bool     valid;
	bool     dirty;

	void reset() {
		valid = false;
		dirty = false;
	}
};

struct set_t {
	std::vector<line_t> lines;

	set_t(uint32_t num_ways)
		: lines(num_ways)
	{}

	void reset() {
		for (auto& line : lines) {
			line.reset();
		}
	}

	int tag_lookup(uint64_t tag, int* free_line_id, int* repl_line_id) {
		uint32_t max_cnt = 0;
		int hit_line_id = -1;
		*free_line_id = -1;
		*repl_line_id = -1;
		for (uint32_t i = 0, n = lines.size(); i < n; ++i) {
			auto& line = lines.at(i);
			if (max_cnt < line.lru_ctr) {
				max_cnt = line.lru_ctr;
				*repl_line_id = i;
			}
			if (line.valid) {
				if (line.tag == tag) {
					hit_line_id = i;
					line.lru_ctr = 0;
				} else {
					++line.lru_ctr;
				}
			} else {
				*free_line_id = i;
			}
		}
		return hit_line_id;
	}
};

struct bank_req_t {

  using Ptr = std::shared_ptr<bank_req_t>;

	enum ReqType {
		None   = 0,
		Replay = 2,
		Core   = 3
	};

	uint64_t addr_tag;
	uint32_t set_id;
	uint32_t cid;
	uint64_t req_tag;
	uint64_t uuid;
	ReqType  type;
	bool     write;

	bank_req_t() {
		this->reset();
	}

	void reset() {
		type = ReqType::None;
	}

	friend std::ostream &operator<<(std::ostream &os, const bank_req_t& req) {
		os << "set=" << req.set_id << ", rw=" << req.write;
		os << ", type=" << req.type;
		os << ", addr_tag=0x" << std::hex << req.addr_tag;
		os << ", req_tag=" << req.req_tag;
		os << ", cid=" << std::dec << req.cid;
		os << " (#" << req.uuid << ")";
		return os;
	}
};

struct mshr_entry_t {
	bank_req_t bank_req;
	uint32_t line_id;

	mshr_entry_t() {}

	void reset() {
		bank_req.reset();
	}
};

class MSHR {
public:
	MSHR(uint32_t size)
		: entries_(size)
		, ready_reqs_(0)
		, size_(0)
	{}

	uint32_t capacity() const {
		return entries_.size();
	}

	uint32_t size() const {
		return size_;
	}

	bool empty() const {
		return (0 == size_);
	}

	bool full() const {
		assert(size_ <= entries_.size());
		return (size_ == entries_.size());
	}

	bool has_ready_reqs() const {
		return (ready_reqs_ != 0);
	}

	bool lookup(const bank_req_t& bank_req) {
		for (auto& entry : entries_) {;
			if (entry.bank_req.type != bank_req_t::None
		 	 && entry.bank_req.set_id == bank_req.set_id
		   && entry.bank_req.addr_tag == bank_req.addr_tag) {
				return true;
			}
		}
		return false;
	}

	int enqueue(const bank_req_t& bank_req, uint32_t line_id) {
		assert(bank_req.type == bank_req_t::Core);
		for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
			auto& entry = entries_.at(i);
			if (entry.bank_req.type == bank_req_t::None) {
				entry.bank_req = bank_req;
				entry.line_id = line_id;
				++size_;
				return i;
			}
		}
		std::abort();
		return -1;
	}

	mshr_entry_t& replay(uint32_t id) {
		auto& root_entry = entries_.at(id);
		assert(root_entry.bank_req.type == bank_req_t::Core);
		assert(ready_reqs_ == 0);
		// mark all related mshr entries for replay
		for (auto& entry : entries_) {
			if (entry.bank_req.type == bank_req_t::Core
			 && entry.bank_req.set_id == root_entry.bank_req.set_id
			 && entry.bank_req.addr_tag == root_entry.bank_req.addr_tag) {
				entry.bank_req.type = bank_req_t::Replay;
				++ready_reqs_;
			}
		}
		return root_entry;
	}

	void dequeue(bank_req_t* out) {
		assert(ready_reqs_ > 0);
		for (auto& entry : entries_) {
			if (entry.bank_req.type == bank_req_t::Replay) {
				*out = entry.bank_req;
				entry.bank_req.type = bank_req_t::None;
				--ready_reqs_;
				--size_;
				break;
			}
		}
	}

	void reset() {
		for (auto& entry : entries_) {
			entry.reset();
		}
		size_ = 0;
	}

private:
	std::vector<mshr_entry_t> entries_;
	uint32_t ready_reqs_;
	uint32_t size_;
};

class CacheBank : public SimObject<CacheBank> {
public:
	SimPort<MemReq> core_req_port;
  SimPort<MemRsp> core_rsp_port;

  SimPort<MemReq> mem_req_port;
  SimPort<MemRsp> mem_rsp_port;

  CacheBank(const SimContext& ctx,
	          const char* name,
	          const CacheSim::Config& config,
				    const params_t& params,
						uint32_t bank_id)
    : SimObject<CacheBank>(ctx, name)
		, core_req_port(this)
		, core_rsp_port(this)
		, mem_req_port(this)
		, mem_rsp_port(this)
		, config_(config)
	  , params_(params)
		, bank_id_(bank_id)
		, sets_(params.sets_per_bank, params.lines_per_set)
		, mshr_(config.mshr_size)
		, pipe_req_(TFifo<bank_req_t>::Create("", config.latency-1))
	{
		this->reset();
	}

  void reset() {
		perf_stats_ = CacheSim::PerfStats();
		pending_mshr_size_ = 0;
    pending_read_reqs_ = 0;
		pending_write_reqs_ = 0;
		pending_fill_reqs_ = 0;
  }

  void tick() {
		// process input requests
		this->processInputs();

		// process pipeline requests
		this->processRequests();

		// calculate memory latency
		perf_stats_.mem_latency += pending_fill_reqs_;
	}

	const CacheSim::PerfStats& perf_stats() const {
		return perf_stats_;
	}

private:

	void processInputs() {
		// proces inputs in prioroty order
		do {
			bank_req_t bank_req;

			// first: schedule MSHR replay
			if (mshr_.has_ready_reqs()) {
				mshr_.dequeue(&bank_req);
				--pending_mshr_size_;
				pipe_req_->push(bank_req);
				break;
			}

			// second: schedule memory fill
			if (!this->mem_rsp_port.empty()) {
				auto& mem_rsp = mem_rsp_port.front();
				DT(3, this->name() << "-fill-rsp: " << mem_rsp);
				// update MSHR
				auto& entry = mshr_.replay(mem_rsp.tag);
				auto& set   = sets_.at(entry.bank_req.set_id);
				auto& line  = set.lines.at(entry.line_id);
				line.valid  = true;
				line.tag    = entry.bank_req.addr_tag;
				mshr_.dequeue(&bank_req);
				--pending_mshr_size_;
				pipe_req_->push(bank_req);
				mem_rsp_port.pop();
				--pending_fill_reqs_;
				break;
			}

			// third: schedule core request
			if (!this->core_req_port.empty()) {
				auto& core_req = core_req_port.front();
				// check MSHR capacity
				if ((!core_req.write || config_.write_back)
				 && (pending_mshr_size_ >= mshr_.capacity())) {
					++perf_stats_.mshr_stalls;
					break;
				}
				++pending_mshr_size_;
				DT(3, this->name() << "-core-req: " << core_req);
				bank_req.type = bank_req_t::Core;
				bank_req.cid = core_req.cid;
				bank_req.uuid = core_req.uuid;
				bank_req.set_id = params_.addr_set_id(core_req.addr);
				bank_req.addr_tag = params_.addr_tag(core_req.addr);
				bank_req.req_tag = core_req.tag;
				bank_req.write = core_req.write;
				pipe_req_->push(bank_req);
				if (core_req.write)
					++perf_stats_.writes;
				else
					++perf_stats_.reads;
				core_req_port.pop();
				break;
			}
		} while (false);
	}

	void processRequests() {
		if (pipe_req_->empty())
			return;
		auto bank_req = pipe_req_->front();

		switch (bank_req.type) {
		case bank_req_t::None:
			break;
		case bank_req_t::Replay: {
			// send core response
			if (!bank_req.write || config_.write_reponse) {
				MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
				this->core_rsp_port.push(core_rsp);
				DT(3, this->name() << "-replay: " << core_rsp);
			}
		} break;
		case bank_req_t::Core: {
			int32_t free_line_id = -1;
			int32_t repl_line_id = 0;
			auto& set = sets_.at(bank_req.set_id);
			// tag lookup
			int hit_line_id = set.tag_lookup(bank_req.addr_tag, &free_line_id, &repl_line_id);
			if (hit_line_id != -1) {
				// Hit handling
				if (bank_req.write) {
					// handle write has_hit
					auto& hit_line = set.lines.at(hit_line_id);
					if (!config_.write_back) {
						// forward write request to memory
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = true;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-writethrough: " << mem_req);
					} else {
						// mark line as dirty
						hit_line.dirty = true;
					}
				}
				// send core response
				if (!bank_req.write || config_.write_reponse) {
					MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
					this->core_rsp_port.push(core_rsp);
					DT(3, this->name() << "-core-rsp: " << core_rsp);
				}
				--pending_mshr_size_;
			} else {
				// Miss handling
				if (bank_req.write)
					++perf_stats_.write_misses;
				else
					++perf_stats_.read_misses;

				if (free_line_id == -1 && config_.write_back) {
					// write back dirty line
					auto& repl_line = set.lines.at(repl_line_id);
					if (repl_line.dirty) {
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, repl_line.tag);
						mem_req.write = true;
						mem_req.cid   = bank_req.cid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-writeback: " << mem_req);
						++perf_stats_.evictions;
					}
				}

				if (bank_req.write && !config_.write_back) {
					// forward write request to memory
					{
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = true;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-writethrough: " << mem_req);
					}
					// send core response
					if (config_.write_reponse) {
						MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
						this->core_rsp_port.push(core_rsp);
						DT(3, this->name() << "-core-rsp: " << core_rsp);
					}
					--pending_mshr_size_;
				} else {
					// MSHR lookup
					auto mshr_pending = mshr_.lookup(bank_req);

					// allocate MSHR
					auto mshr_id = mshr_.enqueue(bank_req, (free_line_id != -1) ? free_line_id : repl_line_id);
					DT(3, this->name() << "-mshr-enqueue: " << bank_req);

					// send fill request
					if (!mshr_pending) {
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = false;
						mem_req.tag   = mshr_id;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-fill-req: " << mem_req);
						++pending_fill_reqs_;
					}
				}
			}
		} break;
		default:
			std::abort();
		}

		pipe_req_->pop();
	}

	CacheSim::Config config_;
	params_t params_;
	uint32_t bank_id_;

  std::vector<set_t> sets_;
	MSHR mshr_;
	uint32_t pending_mshr_size_;
	TFifo<bank_req_t>::Ptr pipe_req_;

	CacheSim::PerfStats perf_stats_;

	uint64_t pending_read_reqs_;
	uint64_t pending_write_reqs_;
	uint64_t pending_fill_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

class CacheSim::Impl {
public:
	Impl(CacheSim* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, params_(config)
		, banks_(1 << config.B)
		, nc_mem_arbs_(config.mem_ports)
	{
		char sname[100];

		uint32_t num_banks = (1 << config.B);

		if (config_.bypass) {
			snprintf(sname, 100, "%s-bypass_arb", simobject->name().c_str());
			auto bypass_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, config_.num_inputs, config_.mem_ports);
			for (uint32_t i = 0; i < config_.num_inputs; ++i) {
				simobject->CoreReqPorts.at(i).bind(&bypass_arb->ReqIn.at(i));
				bypass_arb->RspIn.at(i).bind(&simobject->CoreRspPorts.at(i));
			}
			for (uint32_t i = 0; i < config_.mem_ports; ++i) {
				bypass_arb->ReqOut.at(i).bind(&simobject->MemReqPorts.at(i));
				simobject->MemRspPorts.at(i).bind(&bypass_arb->RspOut.at(i));
			}
			return;
		}

		// create non-cacheable arbiter
		for (uint32_t i = 0; i < config_.mem_ports; ++i) {
			snprintf(sname, 100, "%s-nc_arb%d", simobject->name().c_str(), i);
			nc_mem_arbs_.at(i) = MemArbiter::Create(sname, ArbiterType::Priority, 2, 1);
		}

		// Connect non-cacheable arbiter output port 0 to outgoing memory ports
		for (uint32_t i = 0; i < config_.mem_ports; ++i) {
			nc_mem_arbs_.at(i)->ReqOut.at(0).bind(&simobject->MemReqPorts.at(i));
			simobject->MemRspPorts.at(i).bind(&nc_mem_arbs_.at(i)->RspOut.at(0));
		}

		// Create bank's memory arbiter
		snprintf(sname, 100, "%s-mem_arb", simobject->name().c_str());
		auto bank_mem_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_banks, config_.mem_ports);

		// Connect bank's memory arbiter to non-cacheable arbiter's input 0
		for (uint32_t i = 0; i < config_.mem_ports; ++i) {
			bank_mem_arb->ReqOut.at(i).bind(&nc_mem_arbs_.at(i)->ReqIn.at(0));
			nc_mem_arbs_.at(i)->RspIn.at(0).bind(&bank_mem_arb->RspOut.at(i));
		}

		// Create bank's core crossbar
		snprintf(sname, 100, "%s-core_xbar", simobject->name().c_str());
		bank_core_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, config_.num_inputs, num_banks,
			[&](const MemCrossBar::ReqType& req) {
				return params_.addr_bank_id(req.addr);
			});

		// Create cache banks
		for (uint32_t i = 0, n = num_banks; i < n; ++i) {
			snprintf(sname, 100, "%s-bank%d", simobject->name().c_str(), i);
			banks_.at(i) = CacheBank::Create(sname, config, params_, i);

			// bind core ports
			bank_core_xbar_->ReqOut.at(i).bind(&banks_.at(i)->core_req_port);
			banks_.at(i)->core_rsp_port.bind(&bank_core_xbar_->RspOut.at(i));

			// bind memory ports
			banks_.at(i)->mem_req_port.bind(&bank_mem_arb->ReqIn.at(i));
			bank_mem_arb->RspIn.at(i).bind(&banks_.at(i)->mem_rsp_port);
		}
	}

  void reset() {
		if (config_.bypass)
			return;

		// calculate cache initialization cycles
		init_cycles_ = params_.sets_per_bank;
	}

  void tick() {
		if (config_.bypass)
			return;

		// wait on cache initialization cycles
		if (init_cycles_ != 0) {
			--init_cycles_;
			DT(3, simobject_->name() << "-init: line=" << init_cycles_);
			return;
		}

		// handle cache bypasss responses
		for (uint32_t i = 0, n = config_.mem_ports; i < n; ++i) {
			// Forward non-cacheable arbiter's output 1 to core response ports
			auto& bypass_port = nc_mem_arbs_.at(i)->RspIn.at(1);
			if (!bypass_port.empty()) {
				auto& mem_rsp = bypass_port.front();
				this->processBypassResponse(mem_rsp);
				bypass_port.pop();
			}
		}

		// schedule core responses
		for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
			auto& bank_rsp_port = bank_core_xbar_->RspIn.at(req_id);
			if (bank_rsp_port.empty())
				continue;
			auto& core_rsp = bank_rsp_port.front();
			simobject_->CoreRspPorts.at(req_id).push(core_rsp, 0);
			DT(3, simobject_->name() << "-core-rsp: " << core_rsp);
			bank_rsp_port.pop();
		}

		// schedule core requests
		for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
			auto& core_req_port = simobject_->CoreReqPorts.at(req_id);
			if (core_req_port.empty())
				continue;
			auto& core_req = core_req_port.front();
			if (core_req.type == AddrType::IO) {
				this->processBypassRequest(core_req, req_id);
			} else {
				bank_core_xbar_->ReqIn.at(req_id).push(core_req, 0);
			}
			core_req_port.pop();
		}
	}

	PerfStats perf_stats() const {
		PerfStats perf_stats;
		if (!config_.bypass) {
			for (const auto& bank : banks_) {
				perf_stats += bank->perf_stats();
			}
			perf_stats.bank_stalls = bank_core_xbar_->collisions();
		}
		return perf_stats;
	}

private:

	void processBypassResponse(const MemRsp& mem_rsp) {
		uint32_t req_id = mem_rsp.tag & ((1 << params_.log2_num_inputs)-1);
		uint64_t tag = mem_rsp.tag >> params_.log2_num_inputs;
		MemRsp core_rsp{tag, mem_rsp.cid, mem_rsp.uuid};
		simobject_->CoreRspPorts.at(req_id).push(core_rsp, 0);
		DT(3, simobject_->name() << "-bypass-core-rsp: " << core_rsp);
	}

	void processBypassRequest(const MemReq& core_req, uint32_t req_id) {
		{
			// Push core request to non-cacheable arbiter's input 1
			MemReq mem_req(core_req);
			mem_req.tag = (core_req.tag << params_.log2_num_inputs) + req_id;
			uint32_t mem_port = req_id % config_.mem_ports;
			nc_mem_arbs_.at(mem_port)->ReqIn.at(1).push(mem_req, 0);
			DT(3, simobject_->name() << "-bypass-dram-req: " << mem_req);
		}

		if (core_req.write && config_.write_reponse) {
			MemRsp core_rsp{core_req.tag, core_req.cid, core_req.uuid};
			simobject_->CoreRspPorts.at(req_id).push(core_rsp, 0);
			DT(3, simobject_->name() << "-bypass-core-rsp: " << core_rsp);
		}
	}

	CacheSim* const simobject_;
	Config config_;
	params_t params_;
	std::vector<CacheBank::Ptr> banks_;
	MemArbiter::Ptr bank_arb_;
	std::vector<MemArbiter::Ptr> nc_mem_arbs_;
	MemCrossBar::Ptr bank_core_xbar_;
	uint32_t init_cycles_;
};

///////////////////////////////////////////////////////////////////////////////

CacheSim::CacheSim(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<CacheSim>(ctx, name)
	, CoreReqPorts(config.num_inputs, this)
	, CoreRspPorts(config.num_inputs, this)
	, MemReqPorts(config.mem_ports, this)
	, MemRspPorts(config.mem_ports, this)
	, impl_(new Impl(this, config))
{}

CacheSim::~CacheSim() {
  delete impl_;
}

void CacheSim::reset() {
  impl_->reset();
}

void CacheSim::tick() {
  impl_->tick();
}

CacheSim::PerfStats CacheSim::perf_stats() const {
  return impl_->perf_stats();
}