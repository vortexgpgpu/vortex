#include "shared_mem.h"
#include "core.h"
#include <bitmanip.h>
#include <vector>
#include "types.h"

using namespace vortex;

class SharedMem::Impl {
protected:
    SharedMem* simobject_;
    Config    config_;
    RAM       ram_;
    uint32_t  bank_sel_addr_start_;
    uint32_t  bank_sel_addr_end_;
    PerfStats perf_stats_;

    uint64_t to_local_addr(uint64_t addr) {
        uint32_t smem_local_size_w = log2ceil(SMEM_LOCAL_SIZE);
        uint32_t stack_size_w = log2ceil(STACK_SIZE);
        uint32_t wid_tid_w = log2ceil(NUM_WARPS * NUM_THREADS);    
        uint32_t wid_tid = bit_getw(addr, stack_size_w, stack_size_w + wid_tid_w-1);
        uint32_t offset = bit_getw(addr, 0, smem_local_size_w-1);
        uint32_t s_addr = bit_setw(offset, smem_local_size_w, smem_local_size_w + wid_tid_w-1, wid_tid);
        return s_addr;
    }

public:
    Impl(SharedMem* simobject, const Config& config) 
        : simobject_(simobject)
        , config_(config)
        , ram_(config.capacity, config.capacity)
        , bank_sel_addr_start_(config.bank_offset)
        , bank_sel_addr_end_(config.bank_offset + log2up(config.num_banks)-1)
    {}    
    
    virtual ~Impl() {}

    void reset() {
        perf_stats_ = PerfStats();
    }

    void read(void* data, uint64_t addr, uint32_t size) {
        auto s_addr = to_local_addr(addr);
        //std::cout << "*** s_addr=" << std::dec << s_addr << std::endl;
        ram_.read(data, s_addr, size);
    }

    void write(const void* data, uint64_t addr, uint32_t size) {
        auto s_addr = to_local_addr(addr);
        //std::cout << "*** s_addr=" << std::dec << s_addr << std::endl;
        ram_.write(data, s_addr, size);
    }

    void tick() {
        std::vector<bool> in_used_banks(config_.num_banks);
        for (uint32_t req_id = 0; req_id < config_.num_reqs; ++req_id) {
            auto& core_req_port = simobject_->Inputs.at(req_id);            
            if (core_req_port.empty())
                continue;

            auto& core_req = core_req_port.front();

            uint32_t bank_id = (uint32_t)bit_getw(
                core_req.addr, bank_sel_addr_start_, bank_sel_addr_end_);

            // bank conflict check
            if (in_used_banks.at(bank_id)) {
                ++perf_stats_.bank_stalls;
                continue;
            }

            in_used_banks.at(bank_id) = true;

            if (!core_req.write || config_.write_reponse) {
                // send response
                MemRsp core_rsp{core_req.tag, core_req.core_id};
                simobject_->Outputs.at(req_id).send(core_rsp, 1);
            }

            // update perf counters
            perf_stats_.reads += !core_req.write;            
            perf_stats_.writes += core_req.write;

            // remove input
            core_req_port.pop();
        }
    }

    const PerfStats& perf_stats() const { 
        return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

SharedMem::SharedMem(const SimContext& ctx, const char* name, const Config& config) 
    : SimObject<SharedMem>(ctx, name)   
    , Inputs(config.num_reqs, this)
    , Outputs(config.num_reqs, this)
    , impl_(new Impl(this, config))
{}

SharedMem::~SharedMem() {
    delete impl_;
}

void SharedMem::reset() {
    impl_->reset();
}

void SharedMem::read(void* data, uint64_t addr, uint32_t size) {
    impl_->read(data, addr, size);
}

void SharedMem::write(const void* data, uint64_t addr, uint32_t size) {
    impl_->write(data, addr, size);
}

void SharedMem::tick() {
    impl_->tick();
}

const SharedMem::PerfStats& SharedMem::perf_stats() const {
    return impl_->perf_stats();
}