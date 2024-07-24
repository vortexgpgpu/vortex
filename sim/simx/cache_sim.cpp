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

		assert(config.ports_per_bank <= this->words_per_line);

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

	void clear() {
		valid = false;
		dirty = false;
	}
};

struct set_t {
	std::vector<line_t> lines;

	set_t(uint32_t num_ways)
		: lines(num_ways)
	{}

	void clear() {
		for (auto& line : lines) {
			line.clear();
		}
	}
};

struct bank_req_port_t {
	uint32_t req_id;
	uint64_t req_tag;
	bool     valid;

	void clear() {
		valid = false;
	}
};

struct bank_req_t {

	enum ReqType {
		None   = 0,
		Fill   = 1,
		Replay = 2,
		Core   = 3
	};

	std::vector<bank_req_port_t> ports;
	uint64_t tag;
	uint32_t set_id;
	uint32_t cid;
	uint64_t uuid;
	ReqType  type;
	bool     write;

	bank_req_t(uint32_t num_ports)
		: ports(num_ports)
	{}

	void clear() {
		for (auto& port : ports) {
			port.clear();
		}
		type = ReqType::None;
	}
};

struct mshr_entry_t {
	bank_req_t bank_req;
	uint32_t   line_id;

	mshr_entry_t(uint32_t num_ports)
		: bank_req(num_ports)
	{}

	void clear() {
		bank_req.clear();
	}
};

class MSHR {
private:
	std::vector<mshr_entry_t> entries_;
	uint32_t size_;

public:
	MSHR(uint32_t size, uint32_t num_ports)
		: entries_(size, num_ports)
		, size_(0)
	{}

	bool empty() const {
		return (0 == size_);
	}

	bool full() const {
		return (size_ == entries_.size());
	}

	bool lookup(const bank_req_t& bank_req) {
		for (auto& entry : entries_) {;
			if (entry.bank_req.type != bank_req_t::None
		 	 && entry.bank_req.set_id == bank_req.set_id
		   && entry.bank_req.tag == bank_req.tag) {
				return true;
			}
		}
		return false;
	}

	int allocate(const bank_req_t& bank_req, uint32_t line_id) {
		for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
			auto& entry = entries_.at(i);
			if (entry.bank_req.type == bank_req_t::None) {
				entry.bank_req = bank_req;
				entry.line_id = line_id;
				++size_;
				return i;
			}
		}
		return -1;
	}

	mshr_entry_t& replay(uint32_t id) {
		auto& root_entry = entries_.at(id);
		assert(root_entry.bank_req.type == bank_req_t::Core);
		// mark all related mshr entries for replay
		for (auto& entry : entries_) {
			if (entry.bank_req.type == bank_req_t::Core
			 && entry.bank_req.set_id == root_entry.bank_req.set_id
			 && entry.bank_req.tag == root_entry.bank_req.tag) {
				entry.bank_req.type = bank_req_t::Replay;
			}
		}
		return root_entry;
	}

	bool pop(bank_req_t* out) {
		for (auto& entry : entries_) {
			if (entry.bank_req.type == bank_req_t::Replay) {
				*out = entry.bank_req;
				entry.bank_req.type = bank_req_t::None;
				--size_;
				return true;
			}
		}
		return false;
	}

	void clear() {
		for (auto& entry : entries_) {
			entry.clear();
		}
		size_ = 0;
	}
};

struct bank_t {
	std::vector<set_t> sets;
	MSHR               mshr;

	bank_t(const CacheSim::Config& config,
				 const params_t& params)
		: sets(params.sets_per_bank, params.lines_per_set)
		, mshr(config.mshr_size, config.ports_per_bank)
	{}

	void clear() {
		for (auto& set : sets) {
			set.clear();
		}
		mshr.clear();
	}
};

///////////////////////////////////////////////////////////////////////////////

class CacheSim::Impl {
private:
	CacheSim* const simobject_;
	Config config_;
	params_t params_;
	std::vector<bank_t> banks_;
	MemSwitch::Ptr bank_switch_;
	MemSwitch::Ptr bypass_switch_;
	std::vector<SimPort<MemReq>> mem_req_ports_;
	std::vector<SimPort<MemRsp>> mem_rsp_ports_;
	std::vector<bank_req_t> pipeline_reqs_;
	uint32_t init_cycles_;
	PerfStats perf_stats_;
	uint64_t pending_read_reqs_;
	uint64_t pending_write_reqs_;
	uint64_t pending_fill_reqs_;

public:
	Impl(CacheSim* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, params_(config)
		, banks_((1 << config.B), {config, params_})
		, mem_req_ports_((1 << config.B), simobject)
		, mem_rsp_ports_((1 << config.B), simobject)
		, pipeline_reqs_((1 << config.B), config.ports_per_bank)
	{
		char sname[100];
		snprintf(sname, 100, "%s-bypass-arb", simobject->name().c_str());

		if (config_.bypass) {
			bypass_switch_ = MemSwitch::Create(sname, ArbiterType::RoundRobin, config_.num_inputs);
			for (uint32_t i = 0; i < config_.num_inputs; ++i) {
				simobject->CoreReqPorts.at(i).bind(&bypass_switch_->ReqIn.at(i));
				bypass_switch_->RspIn.at(i).bind(&simobject->CoreRspPorts.at(i));
			}
			bypass_switch_->ReqOut.at(0).bind(&simobject->MemReqPort);
			simobject->MemRspPort.bind(&bypass_switch_->RspOut.at(0));
			return;
		}

		bypass_switch_ = MemSwitch::Create(sname, ArbiterType::Priority, 2);
		bypass_switch_->ReqOut.at(0).bind(&simobject->MemReqPort);
		simobject->MemRspPort.bind(&bypass_switch_->RspOut.at(0));

		if (config.B != 0) {
			snprintf(sname, 100, "%s-bank-arb", simobject->name().c_str());
			bank_switch_ = MemSwitch::Create(sname, ArbiterType::RoundRobin, (1 << config.B));
			for (uint32_t i = 0, n = (1 << config.B); i < n; ++i) {
				mem_req_ports_.at(i).bind(&bank_switch_->ReqIn.at(i));
				bank_switch_->RspIn.at(i).bind(&mem_rsp_ports_.at(i));
			}
			bank_switch_->ReqOut.at(0).bind(&bypass_switch_->ReqIn.at(0));
			bypass_switch_->RspIn.at(0).bind(&bank_switch_->RspOut.at(0));
		} else {
			mem_req_ports_.at(0).bind(&bypass_switch_->ReqIn.at(0));
			bypass_switch_->RspIn.at(0).bind(&mem_rsp_ports_.at(0));
		}

		// calculate cache initialization cycles
		init_cycles_ = params_.sets_per_bank * params_.lines_per_set;
	}

  void reset() {
		if (config_.bypass)
			return;

		for (auto& bank : banks_) {
			bank.clear();
		}
		perf_stats_ = PerfStats();
		pending_read_reqs_  = 0;
		pending_write_reqs_ = 0;
		pending_fill_reqs_  = 0;
	}

  void tick() {
		if (config_.bypass)
			return;

		// wait on cache initialization cycles
		if (init_cycles_ != 0) {
			--init_cycles_;
			return;
		}

		// handle cache bypasss responses
		{
			auto& bypass_port = bypass_switch_->RspIn.at(1);
			if (!bypass_port.empty()) {
				auto& mem_rsp = bypass_port.front();
				this->processBypassResponse(mem_rsp);
				bypass_port.pop();
			}
		}

		// initialize pipeline request
		for (auto& pipeline_req : pipeline_reqs_) {
			pipeline_req.clear();
		}

		// first: schedule MSHR replay (flush MSHR queue)
		for (uint32_t bank_id = 0, n = (1 << config_.B); bank_id < n; ++bank_id) {
			auto& bank = banks_.at(bank_id);
			auto& pipeline_req = pipeline_reqs_.at(bank_id);
			bank.mshr.pop(&pipeline_req);
		}

		// second: schedule memory fill (flush memory queue)
		for (uint32_t bank_id = 0, n = (1 << config_.B); bank_id < n; ++bank_id) {
			auto& mem_rsp_port = mem_rsp_ports_.at(bank_id);
			if (mem_rsp_port.empty())
				continue;

			auto& pipeline_req = pipeline_reqs_.at(bank_id);

			// skip if bank already busy
			if (pipeline_req.type != bank_req_t::None)
				continue;

			auto& mem_rsp = mem_rsp_port.front();
			DT(3, simobject_->name() << "-dram-" << mem_rsp);
			pipeline_req.type = bank_req_t::Fill;
			pipeline_req.tag = mem_rsp.tag;
			mem_rsp_port.pop();
		}

		// last: schedule core requests (flush core queue)
		for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
			auto& core_req_port = simobject_->CoreReqPorts.at(req_id);
			if (core_req_port.empty())
				continue;

			auto& core_req = core_req_port.front();

			// check cache bypassing
			if (core_req.type == AddrType::IO) {
				// send bypass request
				this->processBypassRequest(core_req, req_id);
				// remove request
				core_req_port.pop();
				continue;
			}

			auto bank_id = params_.addr_bank_id(core_req.addr);
			auto& bank = banks_.at(bank_id);
			auto& pipeline_req = pipeline_reqs_.at(bank_id);

			// skip if bank already busy
			if (pipeline_req.type != bank_req_t::None)
				continue;

			auto set_id  = params_.addr_set_id(core_req.addr);
			auto tag     = params_.addr_tag(core_req.addr);
			auto port_id = req_id % config_.ports_per_bank;

			// check MSHR capacity
			if ((!core_req.write || !config_.write_through)
		   && bank.mshr.full()) {
				++perf_stats_.mshr_stalls;
				continue;
			}

			// check bank conflicts
			if (pipeline_req.type == bank_req_t::Core) {
				// check port conflict
				if (pipeline_req.write != core_req.write
				 || pipeline_req.set_id != set_id
				 || pipeline_req.tag != tag
				 || pipeline_req.ports.at(port_id).valid) {
					++perf_stats_.bank_stalls;
					continue;
				}
				// extend request ports
				pipeline_req.ports.at(port_id) = bank_req_port_t{req_id, core_req.tag, true};
			} else {
				// schedule new request
				bank_req_t bank_req(config_.ports_per_bank);
				bank_req.ports.at(port_id) = bank_req_port_t{req_id, core_req.tag, true};
				bank_req.tag   = tag;
				bank_req.set_id = set_id;
				bank_req.cid   = core_req.cid;
				bank_req.uuid  = core_req.uuid;
				bank_req.type  = bank_req_t::Core;
				bank_req.write = core_req.write;
				pipeline_req   = bank_req;
			}

			if (core_req.write)
				++perf_stats_.writes;
			else
				++perf_stats_.reads;

			// remove request
			DT(3, simobject_->name() << "-core-" << core_req);
			auto time = core_req_port.pop();
			perf_stats_.pipeline_stalls += (SimPlatform::instance().cycles() - time);
		}

		// process active request
		this->processBankRequests();
	}

	const PerfStats& perf_stats() const {
		return perf_stats_;
	}

private:

	void processBypassResponse(const MemRsp& mem_rsp) {
		uint32_t req_id = mem_rsp.tag & ((1 << params_.log2_num_inputs)-1);
		uint64_t tag = mem_rsp.tag >> params_.log2_num_inputs;
		MemRsp core_rsp{tag, mem_rsp.cid, mem_rsp.uuid};
		simobject_->CoreRspPorts.at(req_id).push(core_rsp, config_.latency);
		DT(3, simobject_->name() << "-core-" << core_rsp);
	}

	void processBypassRequest(const MemReq& core_req, uint32_t req_id) {
		DT(3, simobject_->name() << "-core-" << core_req);

		{
			MemReq mem_req(core_req);
			mem_req.tag = (core_req.tag << params_.log2_num_inputs) + req_id;
			bypass_switch_->ReqIn.at(1).push(mem_req, 1);
			DT(3, simobject_->name() << "-dram-" << mem_req);
		}

		if (core_req.write && config_.write_reponse) {
			MemRsp core_rsp{core_req.tag, core_req.cid, core_req.uuid};
			simobject_->CoreRspPorts.at(req_id).push(core_rsp, 1);
			DT(3, simobject_->name() << "-core-" << core_rsp);
		}
	}

	void processBankRequests() {
		for (uint32_t bank_id = 0, n = (1 << config_.B); bank_id < n; ++bank_id) {
			auto& bank = banks_.at(bank_id);
			auto pipeline_req = pipeline_reqs_.at(bank_id);

			switch (pipeline_req.type) {
			case bank_req_t::None:
				break;
			case bank_req_t::Fill: {
				// update cache line
				auto& bank  = banks_.at(bank_id);
				auto& entry = bank.mshr.replay(pipeline_req.tag);
				auto& set   = bank.sets.at(entry.bank_req.set_id);
				auto& line  = set.lines.at(entry.line_id);
				line.valid  = true;
				line.tag    = entry.bank_req.tag;
				--pending_fill_reqs_;
			} break;
			case bank_req_t::Replay: {
				// send core response
				if (!pipeline_req.write || config_.write_reponse) {
					for (auto& info : pipeline_req.ports) {
						if (!info.valid)
							continue;
						MemRsp core_rsp{info.req_tag, pipeline_req.cid, pipeline_req.uuid};
						simobject_->CoreRspPorts.at(info.req_id).push(core_rsp, config_.latency);
						DT(3, simobject_->name() << "-replay-" << core_rsp);
					}
				}
			} break;
			case bank_req_t::Core: {
				int32_t hit_line_id  = -1;
				int32_t free_line_id = -1;
				int32_t repl_line_id = 0;
				uint32_t max_cnt = 0;

				auto& set = bank.sets.at(pipeline_req.set_id);

				// tag lookup
				for (uint32_t i = 0, n = set.lines.size(); i < n; ++i) {
					auto& line = set.lines.at(i);
					if (max_cnt < line.lru_ctr) {
						max_cnt = line.lru_ctr;
						repl_line_id = i;
					}
					if (line.valid) {
						if (line.tag == pipeline_req.tag) {
							hit_line_id = i;
							line.lru_ctr = 0;
						} else {
							++line.lru_ctr;
						}
					} else {
						free_line_id = i;
					}
				}

				if (hit_line_id != -1) {
					// Hit handling
					if (pipeline_req.write) {
						// handle write has_hit
						auto& hit_line = set.lines.at(hit_line_id);
						if (config_.write_through) {
							// forward write request to memory
							MemReq mem_req;
							mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, pipeline_req.tag);
							mem_req.write = true;
							mem_req.cid   = pipeline_req.cid;
							mem_req.uuid  = pipeline_req.uuid;
							mem_req_ports_.at(bank_id).push(mem_req, 1);
							DT(3, simobject_->name() << "-writethrough-" << mem_req);
						} else {
							// mark line as dirty
							hit_line.dirty = true;
						}
					}
					// send core response
					if (!pipeline_req.write || config_.write_reponse) {
						for (auto& info : pipeline_req.ports) {
							if (!info.valid)
								continue;
							MemRsp core_rsp{info.req_tag, pipeline_req.cid, pipeline_req.uuid};
							simobject_->CoreRspPorts.at(info.req_id).push(core_rsp, config_.latency);
							DT(3, simobject_->name() << "-core-" << core_rsp);
						}
					}
				} else {
					// Miss handling
					if (pipeline_req.write)
						++perf_stats_.write_misses;
					else
						++perf_stats_.read_misses;

					if (free_line_id == -1 && !config_.write_through) {
						// write back dirty line
						auto& repl_line = set.lines.at(repl_line_id);
						if (repl_line.dirty) {
							MemReq mem_req;
							mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, repl_line.tag);
							mem_req.write = true;
							mem_req.cid   = pipeline_req.cid;
							mem_req_ports_.at(bank_id).push(mem_req, 1);
							DT(3, simobject_->name() << "-writeback-" << mem_req);
							++perf_stats_.evictions;
						}
					}

					if (pipeline_req.write && config_.write_through) {
						// forward write request to memory
						{
							MemReq mem_req;
							mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, pipeline_req.tag);
							mem_req.write = true;
							mem_req.cid   = pipeline_req.cid;
							mem_req.uuid  = pipeline_req.uuid;
							mem_req_ports_.at(bank_id).push(mem_req, 1);
							DT(3, simobject_->name() << "-writethrough-" << mem_req);
						}
						// send core response
						if (config_.write_reponse) {
							for (auto& info : pipeline_req.ports) {
								if (!info.valid)
									continue;
								MemRsp core_rsp{info.req_tag, pipeline_req.cid, pipeline_req.uuid};
								simobject_->CoreRspPorts.at(info.req_id).push(core_rsp, config_.latency);
								DT(3, simobject_->name() << "-core-" << core_rsp);
							}
						}
					} else {
						// MSHR lookup
						auto mshr_pending = bank.mshr.lookup(pipeline_req);

						// allocate MSHR
						auto mshr_id = bank.mshr.allocate(pipeline_req, (free_line_id != -1) ? free_line_id : repl_line_id);

						// send fill request
						if (!mshr_pending) {
							MemReq mem_req;
							mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, pipeline_req.tag);
							mem_req.write = false;
							mem_req.tag   = mshr_id;
							mem_req.cid   = pipeline_req.cid;
							mem_req.uuid  = pipeline_req.uuid;
							mem_req_ports_.at(bank_id).push(mem_req, 1);
							DT(3, simobject_->name() << "-dram-" << mem_req);
							++pending_fill_reqs_;
						}
					}
				}
			} break;
			}
		}
		// calculate memory latency
		perf_stats_.mem_latency += pending_fill_reqs_;
	}
};

///////////////////////////////////////////////////////////////////////////////

CacheSim::CacheSim(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<CacheSim>(ctx, name)
	, CoreReqPorts(config.num_inputs, this)
	, CoreRspPorts(config.num_inputs, this)
	, MemReqPort(this)
	, MemRspPort(this)
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

const CacheSim::PerfStats& CacheSim::perf_stats() const {
  return impl_->perf_stats();
}