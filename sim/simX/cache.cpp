#include "cache.h"
#include "debug.h"
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

    uint32_t word_select_addr_start;
    uint32_t word_select_addr_end;

    uint32_t bank_select_addr_start;
    uint32_t bank_select_addr_end;

    uint32_t set_select_addr_start;
    uint32_t set_select_addr_end;

    uint32_t tag_select_addr_start;
    uint32_t tag_select_addr_end;

    params_t(const CacheConfig& config) {
        uint32_t bank_bits   = log2ceil(config.num_banks);
        uint32_t offset_bits = config.B - config.W;
        uint32_t log2_bank_size  = config.C - bank_bits;
        uint32_t index_bits  = log2_bank_size - (config.B << config.A);        
        assert(log2_bank_size >= config.B);
        
        
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
};

struct bank_req_info_t {
    bool     valid;    
    uint32_t req_id;
    uint32_t req_tag;
};

struct bank_req_t {
    bool valid;
    bool write;
    bool mshr_replay;
    uint64_t tag;
    uint32_t set_id;
    std::vector<bank_req_info_t> infos;

    bank_req_t(uint32_t size) 
        : valid(false)
        , write(false)
        , mshr_replay(false)
        , tag(0)
        , set_id(0)
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
    uint32_t capacity_;

public:    
    MSHR(uint32_t size)
        : entries_(size)
        , capacity_(0) 
    {}

    bool empty() const {
        return (0 == capacity_);
    }
    
    bool full() const {
        return (capacity_ == entries_.size());
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
                ++capacity_;              
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

    bool try_pop(bank_req_t* out) {
        for (auto& entry : entries_) {
            if (entry.valid && entry.mshr_replay) {
                *out = entry;
                entry.valid = false;
                --capacity_;
                return true;
            }
        }
        return false;
    }
};

struct bank_t {
    std::vector<set_t>      sets;    
    MSHR                    mshr;
    std::queue<bank_req_t>  stall_buffer;
    bank_req_t              active_req;

    bank_t(const CacheConfig& config, 
           const params_t& params) 
        : sets(params.sets_per_bank, params.blocks_per_set)
        , mshr(config.mshr_size)
        , active_req(config.ports_per_bank) 
    {}
};

///////////////////////////////////////////////////////////////////////////////

class Cache::Impl {
private:
    Cache* const simobject_;
    CacheConfig config_;
    params_t params_;
    std::vector<bank_t> banks_;
    std::vector<std::pair<bool, MemReq>> core_reqs_;
    std::pair<bool, MemRsp> mem_rsp_;
    std::vector<std::queue<uint32_t>> core_rsps_;

public:
    Impl(Cache* simobject, const CacheConfig& config) 
        : simobject_(simobject)
        , config_(config)
        , params_(config)
        , banks_(config.num_banks, {config, params_})
        , core_reqs_(config.num_inputs)
        , core_rsps_(config.num_inputs)
    {}    

    void handleMemResponse(const MemRsp& response, uint32_t) {        
        mem_rsp_ = {true, response};
    }

    void handleCoreRequest(const MemReq& request, uint32_t port_id) {
        core_reqs_.at(port_id) = {true, request};
    }

    void step(uint64_t /*cycle*/) {
        // process core response
        for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
            auto& core_rsp = core_rsps_.at(req_id);
            if (!core_rsp.empty()) {
                simobject_->CoreRspPorts.at(req_id).send(MemRsp{core_rsp.front()}, config_.latency);
                core_rsp.pop();
            }
        }

        for (auto& bank : banks_) {
            auto& active_req = bank.active_req;

            // try chedule mshr replay
            if (!active_req.valid) {
                bank.mshr.try_pop(&active_req);
            }

            // try schedule stall replay
            if (!active_req.valid 
             && !bank.stall_buffer.empty()) {            
                active_req = bank.stall_buffer.front();
                bank.stall_buffer.pop();
            }
        }

        // handle memory fills
        if (mem_rsp_.first) {
            mem_rsp_.first = false;
            auto bank_id = bit_getw(mem_rsp_.second.tag, 0, 15);
            auto mshr_id = bit_getw(mem_rsp_.second.tag, 16, 31);
            this->processMemoryFill(bank_id, mshr_id);        
        }
        
        // handle incoming core requests
        for (uint32_t i = 0, n = core_reqs_.size(); i < n; ++i) {
            auto& entry = core_reqs_.at(i);
            if (!entry.first)
                continue;
                
            entry.first = false;

            auto& core_req = entry.second;
            auto bank_id   = params_.addr_bank_id(core_req.addr);
            auto set_id    = params_.addr_set_id(core_req.addr);
            auto tag       = params_.addr_tag(core_req.addr);
            auto port_id   = i % config_.ports_per_bank;
            
            // create abnk request
            bank_req_t bank_req(config_.ports_per_bank);
            bank_req.valid = true;
            bank_req.write = core_req.write;
            bank_req.mshr_replay = false;
            bank_req.tag = tag;            
            bank_req.set_id = set_id;       
            bank_req.infos.at(port_id) = {true, i, core_req.tag};

            auto& bank = banks_.at(bank_id);
            
            // check MSHR capacity
            if (bank.mshr.full()) {
                // add to stall buffer
                bank.stall_buffer.emplace(bank_req);
                continue;
            }

            auto& active_req = bank.active_req;

            // check pending MSHR request
            if (active_req.valid 
             && active_req.mshr_replay) {
                // add to stall buffer
                bank.stall_buffer.emplace(bank_req);
                continue;
            }        

            // check bank conflicts
            if (active_req.valid) {
                // check port conflict
                if (active_req.write != core_req.write
                 || active_req.set_id != set_id
                 || active_req.tag != tag
                 || active_req.infos[port_id].valid) {
                    // add to stall buffer
                    bank.stall_buffer.emplace(bank_req);
                    continue;
                }
                // update pending request infos
                active_req.infos[port_id] = bank_req.infos[port_id];
            } else {
                // schedule new request
                active_req = bank_req;
            }
        }
    
        // process active request
        for (uint32_t bank_id = 0, n = config_.num_banks; bank_id < n; ++bank_id) {
            this->processBankRequest(bank_id);
        }
    }

    void processMemoryFill(uint32_t bank_id, uint32_t mshr_id) {
        // update block
        auto& bank = banks_.at(bank_id);
        auto& root_entry = bank.mshr.replay(mshr_id);
        auto& set   = bank.sets.at(root_entry.set_id);
        auto& block = set.blocks.at(root_entry.block_id);
        block.valid = true;
        block.tag   = root_entry.tag;
    }

    void processBankRequest(uint32_t bank_id) {
        auto& bank = banks_.at(bank_id);
        auto& active_req = bank.active_req;
        if (!active_req.valid)
            return;

        active_req.valid = false;

        auto& set = bank.sets.at(active_req.set_id);

        if (active_req.mshr_replay) {
            // send core response
            for (auto& info : active_req.infos) {
                core_rsps_.at(info.req_id).emplace(info.req_tag);            
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
                    if (block.tag == active_req.tag) {
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
                // MISS handling   
                //                
                if (active_req.write) {
                    // handle write hit
                    auto& hit_block = set.blocks.at(hit_block_id);
                    if (config_.write_through) {
                        // forward write request to memory
                        MemReq mem_req;
                        mem_req.addr  = params_.mem_addr(bank_id, active_req.set_id, hit_block.tag);
                        mem_req.write = true;
                        mem_req.tag   = 0;
                        simobject_->MemReqPort.send(mem_req, 1);
                    } else {
                        // mark block as dirty
                        hit_block.dirty = true;
                    }
                }
                // send core response
                for (auto& info : active_req.infos) {
                    core_rsps_.at(info.req_id).emplace(info.req_tag);            
                }
            } else {     
                //
                // MISS handling   
                //                 
                if (!found_free_block && !config_.write_through) {
                     // write back dirty block
                    auto& repl_block = set.blocks.at(repl_block_id);
                    if (repl_block.dirty) {                       
                        MemReq mem_req;
                        mem_req.addr  = params_.mem_addr(bank_id, active_req.set_id, repl_block.tag);
                        mem_req.write = true;
                        simobject_->MemReqPort.send(mem_req, 1);
                    }
                }

                if (active_req.write && config_.write_through) {
                    // forward write request to memory
                    {
                        MemReq mem_req;
                        mem_req.addr  = params_.mem_addr(bank_id, active_req.set_id, active_req.tag);
                        mem_req.write = true;
                        mem_req.tag   = 0;
                        simobject_->MemReqPort.send(mem_req, 1);
                    }
                    // send core response
                    for (auto& info : active_req.infos) {
                        core_rsps_.at(info.req_id).emplace(info.req_tag);            
                    }
                } else {
                    // lookup
                    int pending = bank.mshr.lookup(active_req);

                    // allocate MSHR
                    int mshr_id = bank.mshr.allocate(active_req, repl_block_id);
                    
                    // send fill request
                    if (pending == -1) {
                        MemReq mem_req;
                        mem_req.addr  = params_.mem_addr(bank_id, active_req.set_id, active_req.tag);
                        mem_req.write = active_req.write;
                        mem_req.tag = bit_setw(0,            0, 15, bank_id);
                        mem_req.tag = bit_setw(mem_req.tag, 16, 31, mshr_id);
                        simobject_->MemReqPort.send(mem_req, 1);
                    }
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

Cache::Cache(const SimContext& ctx, const char* name, const CacheConfig& config) 
    : SimObject<Cache>(ctx, name)
    , impl_(new Impl(this, config))
    , CoreReqPorts(config.num_inputs, {this, impl_, &Cache::Impl::handleCoreRequest})
    , CoreRspPorts(config.num_inputs, this)
    , MemReqPort(this)
    , MemRspPort(this, impl_, &Impl::handleMemResponse)
{}

Cache::~Cache() {
    delete impl_;
}

void Cache::step(uint64_t cycle) {
    impl_->step(cycle);
}