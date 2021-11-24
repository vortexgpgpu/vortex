#include "exeunit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "constants.h"

using namespace vortex;

NopUnit::NopUnit(Core*) : ExeUnit("NOP") {}
    
void NopUnit::step(uint64_t /*cycle*/) {
    if (inputs_.empty()) 
        return;
    auto trace = inputs_.top();
    this->schedule_output(trace, 1);
    inputs_.pop();
}

///////////////////////////////////////////////////////////////////////////////

LsuUnit::LsuUnit(Core* core) 
    : ExeUnit("LSU")
    , core_(core)
    , num_threads_(core->arch().num_threads()) 
    , pending_dcache_(LSUQ_SIZE)
    , fence_lock_(false)
{}

void LsuUnit::step(uint64_t cycle) {
    __unused (cycle);

    // handle dcache response
    for (uint32_t t = 0; t < num_threads_; ++t) {
        auto& dcache_rsp_port = core_->dcache_switch_.at(t)->RspOut.at(0);
        if (dcache_rsp_port.empty())
            continue;
        auto& mem_rsp = dcache_rsp_port.top();
        auto& entry = pending_dcache_.at(mem_rsp.tag);          
        auto trace = entry.first;
        DT(3, cycle, "dcache-rsp: tag=" << mem_rsp.tag << ", type=" << trace->lsu.type 
            << ", tid=" << t << ", " << *trace);  
        assert(entry.second);
        --entry.second; // track remaining blocks 
        if (0 == entry.second) {        
            auto latency = (SimPlatform::instance().cycles() - trace->dcache_latency);
            trace->dcache_latency = latency;
            this->schedule_output(trace, 1);
            pending_dcache_.release(mem_rsp.tag);
        } 
        dcache_rsp_port.pop();  
    }

    if (fence_lock_) {
        // wait for all pending memory operations to complete
        if (!pending_dcache_.empty())
            return;
        this->schedule_output(fence_state_, 1);
        fence_lock_ = false;
        DT(3, cycle, "fence-unlock: " << fence_state_);
    }

    // check input queue
    if (inputs_.empty())
        return;

    auto trace = inputs_.top();

    if (trace->lsu.type == LsuType::FENCE) {
        // schedule fence lock
        fence_state_ = trace;
        fence_lock_ = true;        
        DT(3, cycle, "fence-lock: " << *trace);
        // remove input
        inputs_.pop(); 
        return;
    }

    // check pending queue capacity
    if (!trace->check_stalled(pending_dcache_.full())) {
        DT(3, cycle, "*** lsu-queue-stall: " << *trace);
    }
    if (pending_dcache_.full())
        return;

    // send memory request

    bool has_shared_memory = false;
    bool mem_rsp_pending = false;    
    bool is_write = (trace->lsu.type == LsuType::STORE);

    uint32_t valid_addrs = 0;
    for (auto& mem_addr : trace->mem_addrs) {
        valid_addrs += mem_addr.size();
    }    

    trace->dcache_latency = SimPlatform::instance().cycles();
    auto tag = pending_dcache_.allocate({trace, valid_addrs});

    for (uint32_t t = 0; t < num_threads_; ++t) {
        if (!trace->tmask.test(t))
            continue;

        auto& dcache_req_port = core_->dcache_switch_.at(t)->ReqIn.at(0);
        for (auto mem_addr : trace->mem_addrs.at(t)) {
            // check shared memory address
            if (SM_ENABLE) {
                if ((mem_addr >= (SMEM_BASE_ADDR-SMEM_SIZE))
                && (mem_addr < SMEM_BASE_ADDR)) {
                    DT(3, cycle, "smem-access: addr=" << std::hex << mem_addr << ", tag=" << tag 
                        << ", type=" << trace->lsu.type << ", tid=" << t << ", " << *trace);
                    has_shared_memory = true;
                    continue;
                }
            }

            bool is_io = (mem_addr >= IO_BASE_ADDR);

            MemReq mem_req;
            mem_req.addr  = mem_addr;
            mem_req.write = is_write;
            mem_req.tag   = tag;
            mem_req.is_io = is_io; 
            dcache_req_port.send(mem_req, 1);
            DT(3, cycle, "dcache-req: addr=" << std::hex << mem_addr << ", tag=" << tag 
                << ", type=" << trace->lsu.type << ", tid=" << t << ", io=" << is_io << ", "<< trace);            
            // do not wait on writes
            mem_rsp_pending = !is_write;
        }
    }

    // do not wait 
    if (!mem_rsp_pending) {        
        pending_dcache_.release(tag);
        uint32_t delay = 1;
        if (has_shared_memory) {
            // all threads accessed shared memory
            delay += Constants::SMEM_DELAY;
        }
        this->schedule_output(trace, delay);
    }

    // remove input
    inputs_.pop();
}

///////////////////////////////////////////////////////////////////////////////

AluUnit::AluUnit(Core*) : ExeUnit("ALU") {}
    
void AluUnit::step(uint64_t /*cycle*/) {    
    if (inputs_.empty())
        return;
    auto trace = inputs_.top();    
    switch (trace->alu.type) {
    case AluType::ARITH:        
    case AluType::BRANCH:
    case AluType::CMOV:
        this->schedule_output(trace, 1);
        inputs_.pop();
        break;
    case AluType::IMUL:
        this->schedule_output(trace, LATENCY_IMUL);
        inputs_.pop();
        break;
    case AluType::IDIV:
        this->schedule_output(trace, XLEN);
        inputs_.pop();
        break;
    default:
        std::abort();
    }
}

///////////////////////////////////////////////////////////////////////////////

CsrUnit::CsrUnit(Core*) : ExeUnit("CSR") {}
    
void CsrUnit::step(uint64_t /*cycle*/) {
    if (inputs_.empty()) 
        return;
    auto trace = inputs_.top();
    this->schedule_output(trace, 1);
    inputs_.pop();
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(Core*) : ExeUnit("FPU") {}
    
void FpuUnit::step(uint64_t /*cycle*/) {
    if (inputs_.empty()) 
        return;
    auto trace = inputs_.top();
    switch (trace->fpu.type) {
    case FpuType::FNCP:
        this->schedule_output(trace, 1);
        inputs_.pop();
        break;
    case FpuType::FMA:
        this->schedule_output(trace, LATENCY_FMA);
        inputs_.pop();
        break;
    case FpuType::FDIV:
        this->schedule_output(trace, LATENCY_FDIV);
        inputs_.pop();
        break;
    case FpuType::FSQRT:
        this->schedule_output(trace, LATENCY_FSQRT);
        inputs_.pop();
        break;
    case FpuType::FCVT:
        this->schedule_output(trace, LATENCY_FCVT);
        inputs_.pop();
        break;
    default:
        std::abort();
    }
}

///////////////////////////////////////////////////////////////////////////////

GpuUnit::GpuUnit(Core* core) 
    : ExeUnit("GPU") 
    , core_(core)
    , num_threads_(core->arch().num_threads()) 
    , pending_tex_reqs_(TEXQ_SIZE)
{}
    
void GpuUnit::step(uint64_t cycle) {
    __unused (cycle);
#ifdef EXT_TEX_ENABLE
    // handle memory response
    for (uint32_t t = 0; t < num_threads_; ++t) {
        auto& dcache_rsp_port = core_->dcache_switch_.at(t)->RspOut.at(1);
        if (dcache_rsp_port.empty())
            continue;
        auto& mem_rsp = dcache_rsp_port.top();
        auto& entry = pending_tex_reqs_.at(mem_rsp.tag);  
        auto trace = entry.first;
        DT(3, cycle, "tex-rsp: tag=" << mem_rsp.tag << ", tid=" << t << ", " << *trace);  
        assert(entry.second);
        --entry.second; // track remaining blocks 
        if (0 == entry.second) {             
            auto latency = (SimPlatform::instance().cycles() - trace->dcache_latency);
            trace->dcache_latency = latency;
            this->schedule_output(trace, 1);
            pending_tex_reqs_.release(mem_rsp.tag);
        }   
        dcache_rsp_port.pop();
    }
#endif

    // check input queue
    if (inputs_.empty())
        return;

    auto trace = inputs_.top();

    switch  (trace->gpu.type) {
    case GpuType::TMC:
    case GpuType::WSPAWN:
    case GpuType::SPLIT:
    case GpuType::JOIN:
    case GpuType::BAR:
        this->schedule_output(trace, 1);
        inputs_.pop();
        break;
    case GpuType::TEX: {
        if (this->processTexRequest(cycle, trace))
            inputs_.pop();
    }   break;
    default:
        std::abort();
    }
}

bool GpuUnit::processTexRequest(uint64_t cycle, pipeline_trace_t* trace) {
    __unused (cycle);
    
    // check pending queue capacity
    if (!trace->check_stalled(pending_tex_reqs_.full())) {
        DT(3, cycle, "*** tex-queue-stall: " << *trace);
    }
    if (pending_tex_reqs_.full())
        return false;

    // send memory request

    uint32_t valid_addrs = 0;
    for (auto& mem_addr : trace->mem_addrs) {
        valid_addrs += mem_addr.size();
    }

    trace->tex_latency = SimPlatform::instance().cycles();
    auto tag = pending_tex_reqs_.allocate({trace, valid_addrs});

    for (uint32_t t = 0; t < num_threads_; ++t) {
        if (!trace->tmask.test(t))
            continue;

        auto& dcache_req_port = core_->dcache_switch_.at(t)->ReqIn.at(1);
        for (auto mem_addr : trace->mem_addrs.at(t)) {
            MemReq mem_req;
            mem_req.addr  = mem_addr;
            mem_req.write = (trace->lsu.type == LsuType::STORE);
            mem_req.tag   = tag;
            dcache_req_port.send(mem_req, 1);
            DT(3, cycle, "tex-req: addr=" << std::hex << mem_addr << ", tag=" << tag 
                << ", tid=" << t << ", "<< trace);
        }
    }

    return true;
}