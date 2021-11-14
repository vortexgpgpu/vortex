#include "exeunit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"

using namespace vortex;

NopUnit::NopUnit(Core*) : ExeUnit("NOP") {}
    
void NopUnit::step(uint64_t /*cycle*/) {
    pipeline_state_t state;
    if (!inputs_.try_pop(&state))
        return;
    this->schedule_output(state, 1);
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
        MemRsp mem_rsp;
        if (!core_->dcache_->CoreRspPorts.at(t).read(&mem_rsp))
            continue;
        auto& entry = pending_dcache_.at(mem_rsp.tag);  
        DT(3, cycle, "dcache-rsp: addr=" << std::hex << entry.first.mem_addrs.at(t) << ", tag=" << mem_rsp.tag << ", type=" << entry.first.lsu.type << ", tid=" << t << ", " << entry.first);  
        assert(entry.second.test(t));
        entry.second.reset(t); // track remaining blocks        
        if (!entry.second.any()) {        
            auto latency = (SimPlatform::instance().cycles() - entry.first.dcache_latency);
            entry.first.dcache_latency = latency;
            this->schedule_output(entry.first, 1);
            pending_dcache_.release(mem_rsp.tag);
        }   
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

    auto state = inputs_.top();

    if (state.lsu.type == LsuType::FENCE) {
        // schedule fence lock
        fence_state_ = state;
        fence_lock_ = true;
        inputs_.pop();
        DT(3, cycle, "fence-lock: " << state);
        return;
    }

    // check pending queue capacity
    if (pending_dcache_.full()) {
        DT(3, cycle, "*** lsu-queue-stall: " << state);
        return;
    }

    // send dcache request 
    state.dcache_latency = SimPlatform::instance().cycles();
    auto tag = pending_dcache_.allocate({state, state.tmask});         
    for (uint32_t t = 0; t < num_threads_; ++t) {
        if (!state.tmask.test(t))
            continue;
        MemReq mem_req;
        mem_req.addr  = state.mem_addrs.at(t);
        mem_req.write = (state.lsu.type == LsuType::STORE);
        mem_req.tag   = tag;
        core_->dcache_->CoreReqPorts.at(t).send(mem_req, 1);
        DT(3, cycle, "dcache-req: addr=" << std::hex << mem_req.addr << ", tag=" << mem_req.tag << ", type=" << state.lsu.type << ", tid=" << t << ", " << state);
    }            
    inputs_.pop();
}

///////////////////////////////////////////////////////////////////////////////

AluUnit::AluUnit(Core*) : ExeUnit("ALU") {}
    
void AluUnit::step(uint64_t /*cycle*/) {
    pipeline_state_t state;
    if (!inputs_.try_pop(&state))
        return;
    switch  (state.alu.type) {
    case AluType::ARITH:
        this->schedule_output(state, 1);
        break;
    case AluType::BRANCH:
        this->schedule_output(state, 1);
        break;
    case AluType::IMUL:
        this->schedule_output(state, LATENCY_IMUL);
        break;
    case AluType::IDIV:
        this->schedule_output(state, XLEN);
        break;
    }
}

///////////////////////////////////////////////////////////////////////////////

CsrUnit::CsrUnit(Core*) : ExeUnit("CSR") {}
    
void CsrUnit::step(uint64_t /*cycle*/) {
    pipeline_state_t state;
    if (!inputs_.try_pop(&state))
        return;
    this->schedule_output(state, 1);
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(Core*) : ExeUnit("FPU") {}
    
void FpuUnit::step(uint64_t /*cycle*/) {
    pipeline_state_t state;
    if (!inputs_.try_pop(&state))
        return;
    switch  (state.fpu.type) {
    case FpuType::FNCP:
        this->schedule_output(state, 1);
        break;
    case FpuType::FMA:
        this->schedule_output(state, LATENCY_FMA);
        break;
    case FpuType::FDIV:
        this->schedule_output(state, LATENCY_FDIV);
        break;
    case FpuType::FSQRT:
        this->schedule_output(state, LATENCY_FSQRT);
        break;
    case FpuType::FCVT:
        this->schedule_output(state, LATENCY_FCVT);
        break;
    }
}

///////////////////////////////////////////////////////////////////////////////

GpuUnit::GpuUnit(Core*) : ExeUnit("GPU") {}
    
void GpuUnit::step(uint64_t /*cycle*/) {
    pipeline_state_t state;
    if (!inputs_.try_pop(&state))
        return;
    switch  (state.gpu.type) {
    case GpuType::TMC:
    case GpuType::WSPAWN:
    case GpuType::SPLIT:
    case GpuType::JOIN:
    case GpuType::BAR:
        this->schedule_output(state, 1);
        break;
    case GpuType::TEX:
        /* TODO */
        break;
    }
}