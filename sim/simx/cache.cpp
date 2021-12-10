#include "cache.h"
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
    uint32_t blocks_per_set;    
    uint32_t words_per_block;
    uint32_t log2_num_inputs;

    uint32_t word_select_addr_start;
    uint32_t word_select_addr_end;

    uint32_t bank_select_addr_start;
    uint32_t bank_select_addr_end;

    uint32_t set_select_addr_start;
    uint32_t set_select_addr_end;

    uint32_t tag_select_addr_start;
    uint32_t tag_select_addr_end;

    params_t(const Cache::Config& config) {
        uint32_t bank_bits   = log2ceil(config.num_banks);
        uint32_t offset_bits = config.B - config.W;
        uint32_t log2_bank_size  = config.C - bank_bits;
        uint32_t index_bits  = log2_bank_size - (config.B << config.A);        
        assert(log2_bank_size >= config.B);   

        this->log2_num_inputs = log2ceil(config.num_inputs);

        this->words_per_block = 1 << offset_bits;
        this->blocks_per_set  = 1 << config.A;
        this->sets_per_bank   = 1 << index_bits;

        assert(config.ports_per_bank <= this->words_per_block);
                
        // Word select
        this->word_select_addr_start = config.W;
        this->word_select_addr_end = (this->word_select_addr_start+offset_bits-1);

        // Bank select
        this->bank_select_addr_start = (1+this->word_select_addr_end);
        this->bank_select_addr_end = (this->bank_select_addr_start+bank_bits-1);

        // Set select
        this->set_select_addr_start = (1+this->bank_select_addr_end);
        this->set_select_addr_end = (this->set_select_addr_start+index_bits-1);

        // Tag select
        this->tag_select_addr_start = (1+this->set_select_addr_end);
        this->tag_select_addr_end = (config.addr_width-1);
    }

    uint32_t addr_bank_id(uint64_t word_addr) const {
        if (bank_select_addr_end >= bank_select_addr_start)
            return (uint32_t)bit_getw(word_addr, bank_select_addr_start, bank_select_addr_end);
        else    
            return 0;
    }

    uint32_t addr_set_id(uint64_t word_addr) const {
        if (set_select_addr_end >= set_select_addr_start)
            return (uint32_t)bit_getw(word_addr, set_select_addr_start, set_select_addr_end);
        else
            return 0;
    }

    uint64_t addr_tag(uint64_t word_addr) const {
        if (tag_select_addr_end >= tag_select_addr_start)
            return bit_getw(word_addr, tag_select_addr_start, tag_select_addr_end);
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

struct block_t {
    bool     valid;
    bool     dirty;        
    uint64_t tag;
    uint32_t lru_ctr;
};

struct set_t {
    std::vector<block_t> blocks;    
    set_t(uint32_t size) : blocks(size) {}

    void clear() {
        for (auto& block : blocks) {
            block.valid = false;
        }
    }
};

struct bank_req_info_t {
    bool     valid;    
    uint32_t req_id;
    uint64_t req_tag;
};

struct bank_req_t {
    bool valid;
    bool write;
    bool mshr_replay;
    uint64_t tag;
    uint32_t set_id;
    uint32_t core_id;
    uint64_t uuid;
    std::vector<bank_req_info_t> infos;

    bank_req_t(uint32_t size) 
        : valid(false)
        , write(false)
        , mshr_replay(false)
        , tag(0)
        , set_id(0)
        , core_id(0)
        , uuid(0)
        , infos(size)
    {}
};

struct mshr_entry_t : public bank_req_t {
    uint32_t block_id;

    mshr_entry_t(uint32_t size = 0) 
        : bank_req_t(size) 
        , block_id(0)
    {}
};

class MSHR {
private:
    std::vector<mshr_entry_t> entries_;
    uint32_t size_;

public:    
    MSHR(uint32_t size)
        : entries_(size)
        , size_(0) 
    {}

    bool empty() const {
        return (0 == size_);
    }
    
    bool full() const {
        return (size_ == entries_.size());
    }

    int lookup(const bank_req_t& bank_req) {
         for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
            auto& entry = entries_.at(i);
            if (entry.valid 
             && entry.set_id == bank_req.set_id 
             && entry.tag == bank_req.tag) {
                return i;
            }
        }
        return -1;
    }

    int allocate(const bank_req_t& bank_req, uint32_t block_id) {
        for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
            auto& entry = entries_.at(i);
            if (!entry.valid) {
                *(bank_req_t*)&entry = bank_req;
                entry.valid = true;
                entry.mshr_replay = false;
                entry.block_id = block_id;  
                ++size_;              
                return i;
            }
        }
        return -1;
    }

    mshr_entry_t& replay(uint32_t id) {
        auto& root_entry = entries_.at(id);
        assert(root_entry.valid);
        // make all related mshr entries for replay
        for (auto& entry : entries_) {
            if (entry.valid 
             && entry.set_id == root_entry.set_id 
             && entry.tag == root_entry.tag) {
                entry.mshr_replay = true;
            }
        }
        return root_entry;
    }

    bool pop(bank_req_t* out) {
        for (auto& entry : entries_) {
            if (entry.valid && entry.mshr_replay) {
                *out = entry;
                entry.valid = false;
                --size_;
                return true;
            }
        }
        return false;
    }

    void clear() {
        for (auto& entry : entries_) {
            if (entry.valid && entry.mshr_replay) {
                entry.valid = false;
            }
        }
        size_ = 0;
    }
};

struct bank_t {
    std::vector<set_t>  sets;    
    MSHR                mshr;

    bank_t(const Cache::Config& config, 
           const params_t& params) 
        : sets(params.sets_per_bank, params.blocks_per_set)
        , mshr(config.mshr_size)
    {}

    void clear() {
        mshr.clear();
        for (auto& set : sets) {
            set.clear();
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

class Cache::Impl {
private:
    Cache* const simobject_;
    Config config_;
    params_t params_;
    std::vector<bank_t> banks_;
    Switch<MemReq, MemRsp>::Ptr mem_switch_;    
    Switch<MemReq, MemRsp>::Ptr bypass_switch_;
    std::vector<SimPort<MemReq>> mem_req_ports_;
    std::vector<SimPort<MemRsp>>  mem_rsp_ports_;
    uint32_t flush_cycles_;
    PerfStats perf_stats_;
    uint64_t pending_read_reqs_;
    uint64_t pending_write_reqs_;
    uint64_t pending_fill_reqs_;    

public:
    Impl(Cache* simobject, const Config& config) 
        : simobject_(simobject)
        , config_(config)
        , params_(config)
        , banks_(config.num_banks, {config, params_})
        , mem_req_ports_(config.num_banks, simobject)
        , mem_rsp_ports_(config.num_banks, simobject)
    {
        bypass_switch_ = Switch<MemReq, MemRsp>::Create("bypass_arb", ArbiterType::Priority, 2);
        bypass_switch_->ReqOut.bind(&simobject->MemReqPort);
        simobject->MemRspPort.bind(&bypass_switch_->RspIn);

        if (config.num_banks > 1) {
            mem_switch_ = Switch<MemReq, MemRsp>::Create("mem_arb", ArbiterType::RoundRobin, config.num_banks);
            for (uint32_t i = 0, n = config.num_banks; i < n; ++i) {
                mem_req_ports_.at(i).bind(&mem_switch_->ReqIn.at(i));
                mem_switch_->RspOut.at(i).bind(&mem_rsp_ports_.at(i));
            }    
            mem_switch_->ReqOut.bind(&bypass_switch_->ReqIn.at(0));
            bypass_switch_->RspOut.at(0).bind(&mem_switch_->RspIn);
        } else {
            mem_req_ports_.at(0).bind(&bypass_switch_->ReqIn.at(0));
            bypass_switch_->RspOut.at(0).bind(&mem_rsp_ports_.at(0));
        }

        // calculate tag flush cycles
        flush_cycles_ = params_.sets_per_bank * params_.blocks_per_set;
    }

    void reset() {
        for (auto& bank : banks_) {
            bank.clear();
        }
        perf_stats_ = PerfStats();
        pending_read_reqs_ = 0;
        pending_write_reqs_ = 0;
        pending_fill_reqs_ = 0;
    }

    void tick() {
        // wait on flush cycles
        if (flush_cycles_ != 0) {
            --flush_cycles_;
            return;
        }

        // per-bank pipeline request
        std::vector<bank_req_t> pipeline_reqs(config_.num_banks, config_.ports_per_bank);

        // calculate memory latency
        perf_stats_.mem_latency += pending_fill_reqs_;

        // handle bypasss responses
        auto& bypass_port = bypass_switch_->RspOut.at(1);            
        if (!bypass_port.empty()) {
            auto& mem_rsp = bypass_port.front();
            uint32_t req_id = mem_rsp.tag & ((1 << params_.log2_num_inputs)-1);                
            uint64_t tag = mem_rsp.tag >> params_.log2_num_inputs;
            MemRsp core_rsp{tag, mem_rsp.core_id, mem_rsp.uuid};
            simobject_->CoreRspPorts.at(req_id).send(core_rsp, config_.latency);
            DT(3, simobject_->name() << "-" << core_rsp);
            bypass_port.pop();
        }        

        // handle MSHR replay
        for (uint32_t bank_id = 0, n = config_.num_banks; bank_id < n; ++bank_id) {
            auto& bank = banks_.at(bank_id);
            auto& pipeline_req = pipeline_reqs.at(bank_id);
            bank.mshr.pop(&pipeline_req);
        }       

        // handle memory fills
        std::vector<bool> pending_fill_req(config_.num_banks, false);
        for (uint32_t bank_id = 0, n = config_.num_banks; bank_id < n; ++bank_id) {
            auto& mem_rsp_port = mem_rsp_ports_.at(bank_id);
            if (!mem_rsp_port.empty()) {
                auto& mem_rsp = mem_rsp_port.front();
                this->processMemoryFill(bank_id, mem_rsp.tag);                
                pending_fill_req.at(bank_id) = true;
                mem_rsp_port.pop();
            }
        }
        
        // handle incoming core requests
        for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
            auto& core_req_port = simobject_->CoreReqPorts.at(req_id);            
            if (core_req_port.empty())
                continue;

            auto& core_req = core_req_port.front();

            // check cache bypassing
            if (core_req.non_cacheable) {
                // send IO request
                this->processIORequest(core_req, req_id);

                // remove request
                core_req_port.pop();
                continue;
            }

            auto bank_id = params_.addr_bank_id(core_req.addr);
            auto set_id  = params_.addr_set_id(core_req.addr);
            auto tag     = params_.addr_tag(core_req.addr);
            auto port_id = req_id % config_.ports_per_bank;
            
            // create bank request
            bank_req_t bank_req(config_.ports_per_bank);
            bank_req.valid = true;
            bank_req.write = core_req.write;
            bank_req.mshr_replay = false;
            bank_req.tag = tag;            
            bank_req.set_id = set_id;       
            bank_req.core_id = core_req.core_id;
            bank_req.uuid = core_req.uuid;
            bank_req.infos.at(port_id) = {true, req_id, core_req.tag};

            auto& bank = banks_.at(bank_id);            
            auto& pipeline_req = pipeline_reqs.at(bank_id);

            // check pending MSHR replay
            if (pipeline_req.valid 
             && pipeline_req.mshr_replay) {
                 // stall
                continue;
            }    

            // check pending fill request
            if (pending_fill_req.at(bank_id)) {
                // stall
                continue;
            }
            
            // check MSHR capacity if read or writeback
            if ((!core_req.write || !config_.write_through)
             && bank.mshr.full()) {
                ++perf_stats_.mshr_stalls;
                continue;
            }    

            // check bank conflicts
            if (pipeline_req.valid) {
                // check port conflict
                if (pipeline_req.write != core_req.write
                 || pipeline_req.set_id != set_id
                 || pipeline_req.tag != tag
                 || pipeline_req.infos[port_id].valid) {
                    ++perf_stats_.bank_stalls;
                    continue;
                }
                // update pending request infos
                pipeline_req.infos[port_id] = bank_req.infos[port_id];
            } else {
                // schedule new request
                pipeline_req = bank_req;
            }

            if (core_req.write)
                ++perf_stats_.writes;
            else
                ++perf_stats_.reads;

            // remove request
            auto time = core_req_port.pop();
            perf_stats_.pipeline_stalls += (SimPlatform::instance().cycles() - time);
        }
    
        // process active request        
        this->processBankRequest(pipeline_reqs);
    } 

    const PerfStats& perf_stats() const {
        return perf_stats_;
    }

private:
    
    void processIORequest(const MemReq& core_req, uint32_t req_id) {
        {
            MemReq mem_req(core_req);
            mem_req.tag = (core_req.tag << params_.log2_num_inputs) + req_id;
            bypass_switch_->ReqIn.at(1).send(mem_req, 1);
            DT(3, simobject_->name() << "-" << mem_req);
        }

        if (core_req.write && config_.write_reponse) {
            MemRsp core_rsp{core_req.tag, core_req.core_id, core_req.uuid};
            simobject_->CoreRspPorts.at(req_id).send(core_rsp, 1);            
            DT(3, simobject_->name() << "-" << core_rsp);
        }
    }

    void processMemoryFill(uint32_t bank_id, uint32_t mshr_id) {
        // update block
        auto& bank  = banks_.at(bank_id);
        auto& entry = bank.mshr.replay(mshr_id);
        auto& set   = bank.sets.at(entry.set_id);
        auto& block = set.blocks.at(entry.block_id);
        block.valid = true;
        block.tag   = entry.tag;
        --pending_fill_reqs_;
    }

    void processBankRequest(const std::vector<bank_req_t>& pipeline_reqs) {
        for (uint32_t bank_id = 0, n = config_.num_banks; bank_id < n; ++bank_id) {
            auto& pipeline_req = pipeline_reqs.at(bank_id);
            if (!pipeline_req.valid)
                continue;

            auto& bank = banks_.at(bank_id);
            auto& set = bank.sets.at(pipeline_req.set_id);

            if (pipeline_req.mshr_replay) {
                // send core response
                for (auto& info : pipeline_req.infos) {
                    MemRsp core_rsp{info.req_tag, pipeline_req.core_id, pipeline_req.uuid};
                    simobject_->CoreRspPorts.at(info.req_id).send(core_rsp, config_.latency);  
                    DT(3, simobject_->name() << "-" << core_rsp);         
                }
            } else {        
                bool hit = false;
                bool found_free_block = false;            
                int hit_block_id = 0;
                int repl_block_id = 0;            
                uint32_t max_cnt = 0;
                
                for (int i = 0, n = set.blocks.size(); i < n; ++i) {
                    auto& block = set.blocks.at(i);
                    if (block.valid) {
                        if (block.tag == pipeline_req.tag) {
                            block.lru_ctr = 0;                        
                            hit_block_id = i;
                            hit = true;
                        } else {
                            ++block.lru_ctr;
                        }
                        if (max_cnt < block.lru_ctr) {
                            max_cnt = block.lru_ctr;
                            repl_block_id = i;
                        }
                    } else {                    
                        found_free_block = true;
                        repl_block_id = i;
                    }
                }

                if (hit) {     
                    //
                    // Hit handling   
                    //                
                    if (pipeline_req.write) {
                        // handle write hit
                        auto& hit_block = set.blocks.at(hit_block_id);
                        if (config_.write_through) {
                            // forward write request to memory
                            MemReq mem_req;
                            mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, hit_block.tag);
                            mem_req.write = true;
                            mem_req.core_id = pipeline_req.core_id;
                            mem_req.uuid = pipeline_req.uuid;
                            mem_req_ports_.at(bank_id).send(mem_req, 1);
                            DT(3, simobject_->name() << "-" << mem_req);
                        } else {
                            // mark block as dirty
                            hit_block.dirty = true;
                        }
                    }
                    // send core response
                    if (!pipeline_req.write || config_.write_reponse) {
                        for (auto& info : pipeline_req.infos) {     
                            MemRsp core_rsp{info.req_tag, pipeline_req.core_id, pipeline_req.uuid};
                            simobject_->CoreRspPorts.at(info.req_id).send(core_rsp, config_.latency);
                            DT(3, simobject_->name() << "-" << core_rsp);
                        }
                    }
                } else {     
                    //
                    // Miss handling   
                    //
                    if (pipeline_req.write)
                        ++perf_stats_.write_misses;
                    else
                        ++perf_stats_.read_misses;

                    if (!found_free_block && !config_.write_through) {
                        // write back dirty block
                        auto& repl_block = set.blocks.at(repl_block_id);
                        if (repl_block.dirty) {                       
                            MemReq mem_req;
                            mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, repl_block.tag);
                            mem_req.write = true;
                            mem_req.core_id = pipeline_req.core_id;
                            mem_req_ports_.at(bank_id).send(mem_req, 1);
                            DT(3, simobject_->name() << "-" << mem_req);
                            ++perf_stats_.evictions;
                        }
                    }

                    if (pipeline_req.write && config_.write_through) {
                        // forward write request to memory
                        {
                            MemReq mem_req;
                            mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, pipeline_req.tag);
                            mem_req.write = true;
                            mem_req.core_id = pipeline_req.core_id;
                            mem_req.uuid = pipeline_req.uuid;
                            mem_req_ports_.at(bank_id).send(mem_req, 1);
                            DT(3, simobject_->name() << "-" << mem_req);
                        }
                        // send core response
                        if (config_.write_reponse) {
                            for (auto& info : pipeline_req.infos) {         
                                MemRsp core_rsp{info.req_tag, pipeline_req.core_id, pipeline_req.uuid};
                                simobject_->CoreRspPorts.at(info.req_id).send(core_rsp, config_.latency);
                                DT(3, simobject_->name() << "-" << core_rsp);
                            }
                        }
                    } else {
                        // MSHR lookup
                        int pending = bank.mshr.lookup(pipeline_req);

                        // allocate MSHR
                        int mshr_id = bank.mshr.allocate(pipeline_req, repl_block_id);
                        
                        // send fill request
                        if (pending == -1) {
                            MemReq mem_req;
                            mem_req.addr  = params_.mem_addr(bank_id, pipeline_req.set_id, pipeline_req.tag);
                            mem_req.write = false;
                            mem_req.tag   = mshr_id;
                            mem_req.core_id = pipeline_req.core_id;
                            mem_req.uuid = pipeline_req.uuid;
                            mem_req_ports_.at(bank_id).send(mem_req, 1);
                            DT(3, simobject_->name() << "-" << mem_req);
                            ++pending_fill_reqs_;
                        }
                    }
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

Cache::Cache(const SimContext& ctx, const char* name, const Config& config) 
    : SimObject<Cache>(ctx, name)    
    , CoreReqPorts(config.num_inputs, this)
    , CoreRspPorts(config.num_inputs, this)
    , MemReqPort(this)
    , MemRspPort(this)
    , impl_(new Impl(this, config))
{}

Cache::~Cache() {
    delete impl_;
}

void Cache::reset() {
    impl_->reset();
}

void Cache::tick() {
    impl_->tick();
}

const Cache::PerfStats& Cache::perf_stats() const {
    return impl_->perf_stats();
}