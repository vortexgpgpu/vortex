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

AluUnit::AluUnit(const SimContext& ctx, Core* core) : ExeUnit(ctx, core, "ALU") {}
    
void AluUnit::tick() {    
    for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
        auto& input = Inputs.at(i);
        if (input.empty()) 
            continue;
        auto& output = Outputs.at(i);
        auto trace = input.front();
        switch (trace->alu_type) {
        case AluType::ARITH:        
        case AluType::BRANCH:
        case AluType::SYSCALL:
        case AluType::IMUL:
            output.send(trace, LATENCY_IMUL+1);
            break;
        case AluType::IDIV:
            output.send(trace, XLEN+1);
            break;
        default:
            std::abort();
        }
        DT(3, "pipeline-execute: op=" << trace->alu_type << ", " << *trace);
        if (trace->eop && trace->fetch_stall) {
            assert(core_->stalled_warps_.test(trace->wid));
            core_->stalled_warps_.reset(trace->wid);
        }
        input.pop();
    }
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(const SimContext& ctx, Core* core) : ExeUnit(ctx, core, "FPU") {}
    
void FpuUnit::tick() {
    for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
        auto& input = Inputs.at(i);
        if (input.empty()) 
            continue;
        auto& output = Outputs.at(i);
        auto trace = input.front();
        switch (trace->fpu_type) {
        case FpuType::FNCP:
            output.send(trace, 2);
            break;
        case FpuType::FMA:
            output.send(trace, LATENCY_FMA+1);
            break;
        case FpuType::FDIV:
            output.send(trace, LATENCY_FDIV+1);
            break;
        case FpuType::FSQRT:
            output.send(trace, LATENCY_FSQRT+1);
            break;
        case FpuType::FCVT:
            output.send(trace, LATENCY_FCVT+1);
            break;
        default:
            std::abort();
        }    
        DT(3, "pipeline-execute: op=" << trace->fpu_type << ", " << *trace);
        input.pop();
    }
}

///////////////////////////////////////////////////////////////////////////////

LsuUnit::LsuUnit(const SimContext& ctx, Core* core) 
    : ExeUnit(ctx, core, "LSU")
    , pending_rd_reqs_(LSUQ_SIZE)
    , num_lanes_(NUM_LSU_LANES)     
    , pending_loads_(0)
    , fence_lock_(false)
    , input_idx_(0)
{}

void LsuUnit::reset() {
    pending_rd_reqs_.clear();
    pending_loads_ = 0;
    fence_lock_ = false;
}

void LsuUnit::tick() {    
    core_->perf_stats_.load_latency += pending_loads_;

    // handle dcache response    
    for (uint32_t t = 0; t < num_lanes_; ++t) {
        auto& dcache_rsp_port = core_->smem_demuxs_.at(t)->RspIn;
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
            int iw = trace->wid % ISSUE_WIDTH;
            auto& output = Outputs.at(iw);
            output.send(trace, 1);
            pending_rd_reqs_.release(mem_rsp.tag);
        } 
        dcache_rsp_port.pop();
        --pending_loads_;
    }

    // handle shared memory response
    for (uint32_t t = 0; t < num_lanes_; ++t) {
        auto& smem_rsp_port = core_->shared_mem_->Outputs.at(t);
        if (smem_rsp_port.empty())
            continue;
        auto& mem_rsp = smem_rsp_port.front();
        auto& entry = pending_rd_reqs_.at(mem_rsp.tag);          
        auto trace = entry.trace;
        DT(3, "smem-rsp: tag=" << mem_rsp.tag << ", type=" << trace->lsu_type << ", tid=" << t << ", " << *trace);
        assert(entry.count);
        --entry.count; // track remaining addresses 
        if (0 == entry.count) {
            int iw = trace->wid % ISSUE_WIDTH;
            auto& output = Outputs.at(iw);
            output.send(trace, 1);
            pending_rd_reqs_.release(mem_rsp.tag);
        } 
        smem_rsp_port.pop();  
        --pending_loads_;
    }

    if (fence_lock_) {
        // wait for all pending memory operations to complete
        if (!pending_rd_reqs_.empty())
            return;
        int iw = fence_state_->wid % ISSUE_WIDTH;
        auto& output = Outputs.at(iw);
        output.send(fence_state_, 1);
        fence_lock_ = false;
        DT(3, "fence-unlock: " << fence_state_);
    }    

    // check input queue
    for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
        int iw = (input_idx_ + i) % ISSUE_WIDTH;
        auto& input = Inputs.at(iw);
        if (input.empty())
            continue;
        auto& output = Outputs.at(iw);
        auto trace = input.front();
        auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);

        auto t0 = trace->pid * num_lanes_;

        if (trace->lsu_type == LsuType::FENCE) {
            // schedule fence lock
            fence_state_ = trace;
            fence_lock_ = true;        
            DT(3, "fence-lock: " << *trace);
            // remove input
            input.pop(); 
            break;
        }

        // check pending queue capacity    
        if (pending_rd_reqs_.full()) {
            if (!trace->log_once(true)) {
                DT(3, "*** " << this->name() << "-lsu-queue-stall: " << *trace);
            }
            break;
        } else {
            trace->log_once(false);
        }
        
        bool is_write = (trace->lsu_type == LsuType::STORE);

        // duplicates detection
        bool is_dup = false;
        if (trace->tmask.test(t0)) {
            uint64_t addr_mask = sizeof(uint32_t)-1;
            uint32_t addr0 = trace_data->mem_addrs.at(0).addr & ~addr_mask;
            uint32_t matches = 1;
            for (uint32_t t = 1; t < num_lanes_; ++t) {
                if (!trace->tmask.test(t0 + t))
                    continue;
                auto mem_addr = trace_data->mem_addrs.at(t + t0).addr & ~addr_mask;
                matches += (addr0 == mem_addr);
            }
        #ifdef LSU_DUP_ENABLE
            is_dup = (matches == trace->tmask.count());
        #endif
        }

        uint32_t addr_count;
        if (is_dup) {
            addr_count = 1;
        } else {
            addr_count = trace->tmask.count();
        }

        auto tag = pending_rd_reqs_.allocate({trace, addr_count});

        for (uint32_t t = 0; t < num_lanes_; ++t) {
            if (!trace->tmask.test(t0 + t))
                continue;
            
            auto& dcache_req_port = core_->smem_demuxs_.at(t)->ReqIn;
            auto mem_addr = trace_data->mem_addrs.at(t + t0);
            auto type = core_->get_addr_type(mem_addr.addr);

            MemReq mem_req;
            mem_req.addr  = mem_addr.addr;
            mem_req.write = is_write;
            mem_req.type  = type; 
            mem_req.tag   = tag;
            mem_req.cid   = trace->cid;
            mem_req.uuid  = trace->uuid;        
                
            dcache_req_port.send(mem_req, 1);
            DT(3, "dcache-req: addr=0x" << std::hex << mem_req.addr << ", tag=" << tag 
                << ", lsu_type=" << trace->lsu_type << ", tid=" << t << ", addr_type=" << mem_req.type << ", " << *trace);

            if (is_write) {
                ++core_->perf_stats_.stores;
            } else {                
                ++core_->perf_stats_.loads;
                ++pending_loads_;
            }
            if (is_dup)
                break;
        }

        // do not wait on writes
        if (is_write) {
            pending_rd_reqs_.release(tag);
            output.send(trace, 1);            
        }

        // remove input
        input.pop();

        break; // single block
    }
    ++input_idx_;
}

///////////////////////////////////////////////////////////////////////////////

SfuUnit::SfuUnit(const SimContext& ctx, Core* core) 
    : ExeUnit(ctx, core, "SFU")
    , input_idx_(0)
{}
    
void SfuUnit::tick() {
    // check input queue
    for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
        int iw = (input_idx_ + i) % ISSUE_WIDTH;        
        auto& input = Inputs.at(iw);
        if (input.empty())
            continue;
        auto& output = Outputs.at(iw);
        auto trace = input.front();
        auto sfu_type = trace->sfu_type;
        bool release_warp = trace->fetch_stall;

        switch  (sfu_type) {
        case SfuType::TMC: 
        case SfuType::WSPAWN:
        case SfuType::SPLIT:
        case SfuType::JOIN:
        case SfuType::PRED:
        case SfuType::CSRRW:
        case SfuType::CSRRS:
        case SfuType::CSRRC:
            output.send(trace, 1);
            break;
        case SfuType::BAR: {
            output.send(trace, 1);
            auto trace_data = std::dynamic_pointer_cast<SFUTraceData>(trace->data);
            if (trace->eop) {
                core_->barrier(trace_data->bar.id, trace_data->bar.count, trace->wid);
            }
            release_warp = false;
        }   break;
        case SfuType::CMOV:
            output.send(trace, 3);
            break;
        default:
            std::abort();
        }

        DT(3, "pipeline-execute: op=" << trace->sfu_type << ", " << *trace);
        if (trace->eop && release_warp)  {
            assert(core_->stalled_warps_.test(trace->wid));
            core_->stalled_warps_.reset(trace->wid);
        }

        input.pop();

        break; // single block
    }
    ++input_idx_;
}
