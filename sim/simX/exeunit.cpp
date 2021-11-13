#include "exeunit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"

using namespace vortex;

LsuUnit::LsuUnit(Core* core) 
    : ExeUnit("LSU")
    , core_(core)
    , num_threads_(core->arch().num_threads()) 
    , pending_dcache_(LSUQ_SIZE)
    , fence_lock_(false)
{}

void LsuUnit::handleCacheReponse(const MemRsp& response, uint32_t port_id) {
    auto entry = pending_dcache_.at(response.tag);    
    entry.second.reset(port_id); // track remaining blocks
    if (!entry.second.any()) {        
        auto latency = (SimPlatform::instance().cycles() - entry.first.dcache_latency);
        entry.first.dcache_latency = latency;
        this->schedule_output(entry.first, 1);
        pending_dcache_.release(response.tag);
    }
}

void LsuUnit::step() {
    if (fence_lock_) {
        // wait for all pending memory operations to complete
        if (!pending_dcache_.empty())
            return;
        this->schedule_output(fence_state_, 1);
        fence_lock_ = false;
    }

    if (inputs_.empty())
        return;

    auto state = inputs_.top();

    if (state.lsu.fence) {
        // schedule fence lock
        fence_state_ = state;
        fence_lock_ = true;
        inputs_.pop();
        return;
    }

    // send dcache requests
    if (!pending_dcache_.full()) {   
        state.dcache_latency = SimPlatform::instance().cycles();
        auto tag = pending_dcache_.allocate({state, state.tmask});         
        for (uint32_t t = 0; t < num_threads_; ++t) {
            if (!state.tmask.test(t))
                continue;
            MemReq mem_req;
            mem_req.addr  = state.mem_addrs.at(t);
            mem_req.write = state.lsu.store;
            mem_req.tag   = tag;
            core_->dcache_->CoreReqPorts.at(t).send(mem_req, 1);
        }            
        inputs_.pop();
    }
}

///////////////////////////////////////////////////////////////////////////////

AluUnit::AluUnit(Core*) : ExeUnit("ALU") {}
    
void AluUnit::step() {
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
    
void CsrUnit::step() {
    pipeline_state_t state;
    if (!inputs_.try_pop(&state))
        return;
    this->schedule_output(state, 1);
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(Core*) : ExeUnit("FPU") {}
    
void FpuUnit::step() {
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
    
void GpuUnit::step() {
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