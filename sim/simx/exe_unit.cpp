#include "exe_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "constants.h"
#include "cache_sim.h"

using namespace vortex;

LsuUnit::LsuUnit(const SimContext& ctx, Core* core) 
    : ExeUnit(ctx, core, "LSU")
    , pending_rd_reqs_(LSUQ_SIZE)
    , num_threads_(core->arch().num_threads())     
    , pending_loads_(0)
    , fence_lock_(false)
{}

void LsuUnit::reset() {
    pending_rd_reqs_.clear();
    pending_loads_ = 0;
    fence_lock_ = false;
}

void LsuUnit::tick() {
    core_->perf_stats_.load_latency += pending_loads_;

    // handle dcache response
    for (uint32_t t = 0; t < num_threads_; ++t) {
        auto& dcache_rsp_port = core_->dcache_rsp_ports.at(t);
        if (dcache_rsp_port.empty())
            continue;
        auto& mem_rsp = dcache_rsp_port.front();
        auto& entry = pending_rd_reqs_.at(mem_rsp.tag);          
        auto trace = entry.trace;
        DT(3, "dcache-rsp: tag=" << mem_rsp.tag << ", type=" << trace->lsu_type 
            << ", tid=" << t << ", " << *trace);  
        assert(entry.count);
        --entry.count; // track remaining addresses 
        if (0 == entry.count) {
            Output.send(trace, 1);
            pending_rd_reqs_.release(mem_rsp.tag);
        } 
        dcache_rsp_port.pop();
        --pending_loads_;
    }

    // handle shared memory response
    for (uint32_t t = 0; t < num_threads_; ++t) {
        auto& smem_rsp_port = core_->sharedmem_->Outputs.at(t);
        if (smem_rsp_port.empty())
            continue;
        auto& mem_rsp = smem_rsp_port.front();
        auto& entry = pending_rd_reqs_.at(mem_rsp.tag);          
        auto trace = entry.trace;
        DT(3, "smem-rsp: tag=" << mem_rsp.tag << ", type=" << trace->lsu_type 
            << ", tid=" << t << ", " << *trace);  
        assert(entry.count);
        --entry.count; // track remaining addresses 
        if (0 == entry.count) {
            Output.send(trace, 1);
            pending_rd_reqs_.release(mem_rsp.tag);
        } 
        smem_rsp_port.pop();  
        --pending_loads_;
    }

    if (fence_lock_) {
        // wait for all pending memory operations to complete
        if (!pending_rd_reqs_.empty())
            return;
        Output.send(fence_state_, 1);
        fence_lock_ = false;
        DT(3, "fence-unlock: " << fence_state_);
    }    

    // check input queue
    if (Input.empty())
        return;

    auto trace = Input.front();
    auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);

    if (trace->lsu_type == LsuType::FENCE) {
        // schedule fence lock
        fence_state_ = trace;
        fence_lock_ = true;        
        DT(3, "fence-lock: " << *trace);
        // remove input
        auto time = Input.pop(); 
        core_->perf_stats_.lsu_stalls += (SimPlatform::instance().cycles() - time);
        return;
    }

    // check pending queue capacity    
    if (pending_rd_reqs_.full()) {
        if (!trace->log_once(true)) {
            DT(3, "*** " << this->name() << "-lsu-queue-stall: " << *trace);
        }
        return;
    } else {
        trace->log_once(false);
    }
    
    bool is_write = (trace->lsu_type == LsuType::STORE);

    // duplicates detection
    bool is_dup = false;
    if (trace->tmask.test(0)) {
        uint64_t addr_mask = sizeof(uint32_t)-1;
        uint32_t addr0 = trace_data->mem_addrs.at(0).addr & ~addr_mask;
        uint32_t matches = 1;
        for (uint32_t t = 1; t < num_threads_; ++t) {
            if (!trace->tmask.test(t))
                continue;
            auto mem_addr = trace_data->mem_addrs.at(t).addr & ~addr_mask;
            matches += (addr0 == mem_addr);
        }
        is_dup = (matches == trace->tmask.count());
    }

    uint32_t addr_count;
    if (is_dup) {
        addr_count = 1;
    } else {
        addr_count = trace->tmask.count();
    }

    auto tag = pending_rd_reqs_.allocate({trace, addr_count});

    for (uint32_t t = 0; t < num_threads_; ++t) {
        if (!trace->tmask.test(t))
            continue;
        
        auto& dcache_req_port = core_->dcache_req_ports.at(t);
        auto mem_addr = trace_data->mem_addrs.at(t);
        auto type = core_->get_addr_type(mem_addr.addr);

        MemReq mem_req;
        mem_req.addr  = mem_addr.addr;
        mem_req.write = is_write;
        mem_req.type  = type; 
        mem_req.tag   = tag;
        mem_req.cid   = trace->cid;
        mem_req.uuid  = trace->uuid;        
             
        dcache_req_port.send(mem_req, 2);
        DT(3, "dcache-req: addr=" << std::hex << mem_req.addr << ", tag=" << tag 
            << ", lsu_type=" << trace->lsu_type << ", tid=" << t << ", addr_type=" << mem_req.type << ", " << *trace);

        ++pending_loads_;
        ++core_->perf_stats_.loads;        
        if (is_dup)
            break;
    }

    // do not wait on writes
    if (is_write) {
        pending_rd_reqs_.release(tag);
        Output.send(trace, 1);
        ++core_->perf_stats_.stores;
    }

    // remove input
    auto time = Input.pop();
    core_->perf_stats_.lsu_stalls += (SimPlatform::instance().cycles() - time);
}

///////////////////////////////////////////////////////////////////////////////

AluUnit::AluUnit(const SimContext& ctx, Core* core) : ExeUnit(ctx, core, "ALU") {}
    
void AluUnit::tick() {    
    if (Input.empty())
        return;
    auto trace = Input.front();    
    bool release_warp = trace->fetch_stall;
    switch (trace->alu_type) {
    case AluType::ARITH:        
    case AluType::BRANCH:
    case AluType::SYSCALL:
    case AluType::IMUL:
        Output.send(trace, LATENCY_IMUL+1);
        break;
    case AluType::IDIV:
        Output.send(trace, XLEN+1);
        break;
    default:
        std::abort();
    }
    DT(3, "pipeline-execute: op=" << trace->alu_type << ", " << *trace);
    if (release_warp) {
        core_->stalled_warps_.reset(trace->wid);
    }
    auto time = Input.pop();
    core_->perf_stats_.alu_stalls += (SimPlatform::instance().cycles() - time);
}

///////////////////////////////////////////////////////////////////////////////

CsrUnit::CsrUnit(const SimContext& ctx, Core* core) : ExeUnit(ctx, core, "CSR") {}
    
void CsrUnit::tick() {
    if (Input.empty()) 
        return;
    auto trace = Input.front();
    Output.send(trace, 1);
    auto time = Input.pop();
    core_->perf_stats_.csr_stalls += (SimPlatform::instance().cycles() - time);
    DT(3, "pipeline-execute: op=CSR, " << *trace);
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(const SimContext& ctx, Core* core) : ExeUnit(ctx, core, "FPU") {}
    
void FpuUnit::tick() {
    if (Input.empty()) 
        return;
    auto trace = Input.front();
    switch (trace->fpu_type) {
    case FpuType::FNCP:
        Output.send(trace, 2);
        break;
    case FpuType::FMA:
        Output.send(trace, LATENCY_FMA+1);
        break;
    case FpuType::FDIV:
        Output.send(trace, LATENCY_FDIV+1);
        break;
    case FpuType::FSQRT:
        Output.send(trace, LATENCY_FSQRT+1);
        break;
    case FpuType::FCVT:
        Output.send(trace, LATENCY_FCVT+1);
        break;
    default:
        std::abort();
    }    
    DT(3, "pipeline-execute: op=" << trace->fpu_type << ", " << *trace);
    auto time = Input.pop();
    core_->perf_stats_.fpu_stalls += (SimPlatform::instance().cycles() - time);
}

///////////////////////////////////////////////////////////////////////////////

GpuUnit::GpuUnit(const SimContext& ctx, Core* core) 
    : ExeUnit(ctx, core, "GPU")   
    , raster_units_(core->raster_units_)    
    , rop_units_(core->rop_units_)
    , tex_units_(core->tex_units_)
{
    for (auto& raster_unit : raster_units_) {
        pending_rsps_.push_back(&raster_unit->Output);
    }
    for (auto& rop_unit : rop_units_) {
        pending_rsps_.push_back(&rop_unit->Output);
    }
    for (auto& tex_unit : tex_units_) {
        pending_rsps_.push_back(&tex_unit->Output);
    }
}
    
void GpuUnit::tick() {
    // handle pending reponses
    for (auto pending_rsp : pending_rsps_) {
        if (pending_rsp->empty())
            continue;
        auto trace = pending_rsp->front();
        if (trace->cid != core_->id())
            continue;
        Output.send(trace, 1);
        pending_rsp->pop();
    }

    // check input queue
    if (Input.empty())
        return;

    auto trace = Input.front();

    auto gpu_type = trace->gpu_type;
    bool release_warp = trace->fetch_stall;

    switch  (gpu_type) {
    case GpuType::TMC: 
    case GpuType::WSPAWN:
    case GpuType::SPLIT:
    case GpuType::JOIN:
        Output.send(trace, 1);
        break;
    case GpuType::BAR: {
        Output.send(trace, 1);
        auto trace_data = std::dynamic_pointer_cast<GPUTraceData>(trace->data);
        core_->barrier(trace_data->bar.id, trace_data->bar.count, trace->wid);
        release_warp = false;
    }   break;
    case GpuType::RASTER: {
        auto trace_data = std::dynamic_pointer_cast<RasterUnit::TraceData>(trace->data);
        raster_units_.at(trace_data->raster_idx)->Input.send(trace, 1);
    }   break;
    case GpuType::ROP: {
        auto trace_data = std::dynamic_pointer_cast<RopUnit::TraceData>(trace->data);
        rop_units_.at(trace_data->rop_idx)->Input.send(trace, 1);
    }   break;    
    case GpuType::TEX: {
        auto trace_data = std::dynamic_pointer_cast<TexUnit::TraceData>(trace->data);
        tex_units_.at(trace_data->tex_idx)->Input.send(trace, 1);
    }   break;
    case GpuType::CMOV:
        Output.send(trace, 3);
        break;
    case GpuType::IMADD:
        Output.send(trace, 3);
        break;
    default:
        std::abort();
    }

    DT(3, "pipeline-execute: op=" << trace->gpu_type << ", " << *trace);
    if (release_warp)  {
        core_->stalled_warps_.reset(trace->wid);
    }

    auto time = Input.pop();
    auto stalls = (SimPlatform::instance().cycles() - time);

    core_->perf_stats_.gpu_stalls += stalls;

    switch (gpu_type) {
    case GpuType::TEX:
        core_->perf_stats_.tex_issue_stalls += stalls;
        break;
    case GpuType::ROP:
        core_->perf_stats_.rop_issue_stalls += stalls;
        break;
    case GpuType::RASTER:
        core_->perf_stats_.raster_issue_stalls += stalls;
        break;
    default:        
        break;
    }    
}