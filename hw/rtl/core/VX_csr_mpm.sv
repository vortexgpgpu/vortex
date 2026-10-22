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

`include "VX_define.vh"

`ifdef VX_CFG_XLEN_64
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = `VX_CFG_XLEN'(src)
`else
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = src[31:0]; \
        addr+12'h80 : dst = 32'(src[$bits(src)-1:32])
`endif

// Host-side machine performance-monitor reads (DCR path). The counters are not
// reachable from kernel CSR instructions, so one instance serves the whole core.
module VX_csr_mpm import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
`ifdef PERF_ENABLE
    input sysmem_perf_t                 sysmem_perf,
    input pipeline_perf_t               pipeline_perf,
`endif
    input wire [PERF_CTR_BITS-1:0]      cycles,
    input wire [PERF_CTR_BITS-1:0]      instret,

    VX_dcr_csr_if.slave                 dcr_csr_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    wire [7:0]                   dcr_mpm_class = dcr_csr_if.mpm_class;
    wire [`VX_CSR_ADDR_BITS-1:0] dcr_addr      = dcr_csr_if.addr;
    wire [`VX_CFG_XLEN-1:0]      dcr_data;

    // Host counter reads decode their own address and never wait on a request.
    assign dcr_csr_if.ready = 1'b1;
    assign dcr_csr_if.value = VX_DCR_DATA_WIDTH'(dcr_data);
    `UNUSED_VAR (dcr_data)
    `UNUSED_VAR (dcr_csr_if.valid)

    // Host performance-counter reads ////////////////////////////////////////

    reg [`VX_CFG_XLEN-1:0] dcr_data_w;

    always @(*) begin
        dcr_data_w = '0;
        case (dcr_addr)
            `CSR_READ_64(`VX_CSR_MCYCLE, dcr_data_w, cycles);
            `CSR_READ_64(`VX_CSR_MINSTRET, dcr_data_w, instret);
            default: begin
            `ifdef PERF_ENABLE
                if ((dcr_addr >= `VX_CSR_MPM_USER   && dcr_addr < (`VX_CSR_MPM_USER + 32))
                 || (dcr_addr >= `VX_CSR_MPM_USER_H && dcr_addr < (`VX_CSR_MPM_USER_H + 32))) begin
                    case (dcr_mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (dcr_addr)
                        // PERF: pipeline
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_IDLE, dcr_data_w, pipeline_perf.sched.idles);
                        `CSR_READ_64(`VX_CSR_MPM_ACTIVE_WARPS, dcr_data_w, pipeline_perf.sched.active_warps);
                        `CSR_READ_64(`VX_CSR_MPM_STALLED_WARPS, dcr_data_w, pipeline_perf.sched.stalled_warps);
                        `CSR_READ_64(`VX_CSR_MPM_ISSUED_WARPS, dcr_data_w, pipeline_perf.sched.issued_warps);
                        `CSR_READ_64(`VX_CSR_MPM_ISSUED_THREADS, dcr_data_w, pipeline_perf.sched.issued_threads);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_FETCH, dcr_data_w, pipeline_perf.fetch.stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_IBUF, dcr_data_w, pipeline_perf.issue.ibf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_SCRB, dcr_data_w, pipeline_perf.issue.scb_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_OPDS, dcr_data_w, pipeline_perf.issue.opd_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_ALU, dcr_data_w, pipeline_perf.issue.dispatch_stalls[EX_ALU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_ALU, dcr_data_w, pipeline_perf.issue.dispatch_instrs[EX_ALU]);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_LSU, dcr_data_w, pipeline_perf.issue.dispatch_stalls[EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_LSU, dcr_data_w, pipeline_perf.issue.dispatch_instrs[EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_SFU, dcr_data_w, pipeline_perf.issue.dispatch_stalls[EX_SFU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_SFU, dcr_data_w, pipeline_perf.issue.dispatch_instrs[EX_SFU]);
                    `ifdef VX_CFG_EXT_F_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_STALL_FPU, dcr_data_w, pipeline_perf.issue.dispatch_stalls[EX_FPU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_FPU, dcr_data_w, pipeline_perf.issue.dispatch_instrs[EX_FPU]);
                    `endif
                    `ifdef VX_CFG_EXT_TCU_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_STALL_TCU, dcr_data_w, pipeline_perf.issue.dispatch_stalls[EX_TCU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_TCU, dcr_data_w, pipeline_perf.issue.dispatch_instrs[EX_TCU]);
                    `endif
                        // PERF: branches
                        `CSR_READ_64(`VX_CSR_MPM_BRANCHES, dcr_data_w, pipeline_perf.sched.branches);
                        `CSR_READ_64(`VX_CSR_MPM_DIVERGENCE, dcr_data_w, pipeline_perf.sched.divergence);
                        // PERF: memory (core-issued requests; DRAM traffic is in the MEM class)
                        `CSR_READ_64(`VX_CSR_MPM_IFETCHES, dcr_data_w, pipeline_perf.ifetches);
                        `CSR_READ_64(`VX_CSR_MPM_LOADS, dcr_data_w, pipeline_perf.loads);
                        `CSR_READ_64(`VX_CSR_MPM_STORES, dcr_data_w, pipeline_perf.stores);
                        `CSR_READ_64(`VX_CSR_MPM_IFETCH_LT, dcr_data_w, pipeline_perf.ifetch_latency);
                        `CSR_READ_64(`VX_CSR_MPM_LOAD_LT, dcr_data_w, pipeline_perf.load_latency);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_ICACHE: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_READS, dcr_data_w, sysmem_perf.icache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MISS_R, dcr_data_w, sysmem_perf.icache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MSHR_ST, dcr_data_w, sysmem_perf.icache.mshr_stalls);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_DCACHE: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_READS, dcr_data_w, sysmem_perf.dcache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_WRITES, dcr_data_w, sysmem_perf.dcache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_R, dcr_data_w, sysmem_perf.dcache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_W, dcr_data_w, sysmem_perf.dcache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_EVICTS, dcr_data_w, sysmem_perf.dcache.evictions);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_BANK_ST, dcr_data_w, sysmem_perf.dcache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MSHR_ST, dcr_data_w, sysmem_perf.dcache.mshr_stalls);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_L2CACHE: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_READS, dcr_data_w, sysmem_perf.l2cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_WRITES, dcr_data_w, sysmem_perf.l2cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_R, dcr_data_w, sysmem_perf.l2cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_W, dcr_data_w, sysmem_perf.l2cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_EVICTS, dcr_data_w, sysmem_perf.l2cache.evictions);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_BANK_ST, dcr_data_w, sysmem_perf.l2cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MSHR_ST, dcr_data_w, sysmem_perf.l2cache.mshr_stalls);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_L3CACHE: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_READS, dcr_data_w, sysmem_perf.l3cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_WRITES, dcr_data_w, sysmem_perf.l3cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_R, dcr_data_w, sysmem_perf.l3cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_W, dcr_data_w, sysmem_perf.l3cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_EVICTS, dcr_data_w, sysmem_perf.l3cache.evictions);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_BANK_ST, dcr_data_w, sysmem_perf.l3cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MSHR_ST, dcr_data_w, sysmem_perf.l3cache.mshr_stalls);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_MEM: begin
                        case (dcr_addr)
                        // PERF: off-chip memory
                        `CSR_READ_64(`VX_CSR_MPM_MEM_READS, dcr_data_w, sysmem_perf.mem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_WRITES, dcr_data_w, sysmem_perf.mem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_LT, dcr_data_w, sysmem_perf.mem.latency);
                        // PERF: lmem
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_READS, dcr_data_w, sysmem_perf.lmem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_WRITES, dcr_data_w, sysmem_perf.lmem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_BANK_ST, dcr_data_w, sysmem_perf.lmem.bank_stalls);
                        // PERF: coalescer
                        `CSR_READ_64(`VX_CSR_MPM_COALESCER_MISS, dcr_data_w, sysmem_perf.coalescer.misses);
                    `ifdef VX_CFG_VM_ENABLE
                        // PERF: VM/MMU (icache + dcache MMU summed)
                        `CSR_READ_64(`VX_CSR_MPM_TLB_READS,   dcr_data_w, pipeline_perf.mmu.tlb_reads);
                        `CSR_READ_64(`VX_CSR_MPM_TLB_HITS,    dcr_data_w, pipeline_perf.mmu.tlb_hits);
                        `CSR_READ_64(`VX_CSR_MPM_TLB_MISSES,  dcr_data_w, pipeline_perf.mmu.tlb_misses);
                        `CSR_READ_64(`VX_CSR_MPM_TLB_EVICTS,  dcr_data_w, pipeline_perf.mmu.tlb_evictions);
                        `CSR_READ_64(`VX_CSR_MPM_PTW_WALKS,   dcr_data_w, pipeline_perf.mmu.ptw_walks);
                        `CSR_READ_64(`VX_CSR_MPM_PTW_LATENCY, dcr_data_w, pipeline_perf.mmu.ptw_latency);
                    `endif
                        default:;
                        endcase
                    end
                `ifdef VX_CFG_EXT_DXA_ENABLE
                    `VX_DCR_MPM_CLASS_DXA: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_DXA_TRANSFERS,  dcr_data_w, sysmem_perf.dxa.transfers);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_GMEM_READS, dcr_data_w, sysmem_perf.dxa.gmem_reads);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_GMEM_DEDUP, dcr_data_w, sysmem_perf.dxa.gmem_dedup);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_LMEM_WRITES,dcr_data_w, sysmem_perf.dxa.lmem_writes);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_GMEM_LT,    dcr_data_w, sysmem_perf.dxa.gmem_latency);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_NOSLOT_STALLS, dcr_data_w, sysmem_perf.dxa.noslot_stalls);
                        default:;
                        endcase
                    end
                `endif
                `ifdef VX_CFG_EXT_TCU_ENABLE
                    `VX_DCR_MPM_CLASS_TCU: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_TCU_TBUF_STALLS,      dcr_data_w, pipeline_perf.tcu.tbuf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_TCU_TBUF_CACHE_HITS, dcr_data_w, pipeline_perf.tcu.tbuf_cache_hits);
                        `CSR_READ_64(`VX_CSR_MPM_TCU_LMEM_READS,     dcr_data_w, pipeline_perf.tcu.lmem_reads);
                        default:;
                        endcase
                    end
                `endif
                `ifdef VX_CFG_EXT_TEX_ENABLE
                    `VX_DCR_MPM_CLASS_TEX: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_TEX_READS,      dcr_data_w, sysmem_perf.tex.mem_reads);
                        `CSR_READ_64(`VX_CSR_MPM_TEX_LAT,        dcr_data_w, sysmem_perf.tex.mem_latency);
                        `CSR_READ_64(`VX_CSR_MPM_TEX_ST,         dcr_data_w, sysmem_perf.tex.stall_cycles);
                        `CSR_READ_64(`VX_CSR_MPM_TCACHE_READS,   dcr_data_w, sysmem_perf.tcache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_TCACHE_MISS_R,  dcr_data_w, sysmem_perf.tcache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_TCACHE_BANK_ST, dcr_data_w, sysmem_perf.tcache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_TCACHE_MSHR_ST, dcr_data_w, sysmem_perf.tcache.mshr_stalls);
                        default:;
                        endcase
                    end
                `endif
                `ifdef VX_CFG_EXT_RASTER_ENABLE
                    `VX_DCR_MPM_CLASS_RASTER: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_RASTER_READS,   dcr_data_w, sysmem_perf.raster.mem_reads);
                        `CSR_READ_64(`VX_CSR_MPM_RASTER_LAT,     dcr_data_w, sysmem_perf.raster.mem_latency);
                        `CSR_READ_64(`VX_CSR_MPM_RASTER_ST,      dcr_data_w, sysmem_perf.raster.stall_cycles);
                        `CSR_READ_64(`VX_CSR_MPM_RCACHE_READS,   dcr_data_w, sysmem_perf.rcache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_RCACHE_MISS_R,  dcr_data_w, sysmem_perf.rcache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_RCACHE_BANK_ST, dcr_data_w, sysmem_perf.rcache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_RCACHE_MSHR_ST, dcr_data_w, sysmem_perf.rcache.mshr_stalls);
                        default:;
                        endcase
                    end
                `endif
                `ifdef VX_CFG_EXT_OM_ENABLE
                    `VX_DCR_MPM_CLASS_OM: begin
                        case (dcr_addr)
                        `CSR_READ_64(`VX_CSR_MPM_OM_READS,       dcr_data_w, sysmem_perf.om.mem_reads);
                        `CSR_READ_64(`VX_CSR_MPM_OM_WRITES,      dcr_data_w, sysmem_perf.om.mem_writes);
                        `CSR_READ_64(`VX_CSR_MPM_OM_LAT,         dcr_data_w, sysmem_perf.om.mem_latency);
                        `CSR_READ_64(`VX_CSR_MPM_OM_ST,          dcr_data_w, sysmem_perf.om.stall_cycles);
                        `CSR_READ_64(`VX_CSR_MPM_OCACHE_READS,   dcr_data_w, sysmem_perf.ocache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_OCACHE_WRITES,  dcr_data_w, sysmem_perf.ocache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_OCACHE_MISS_R,  dcr_data_w, sysmem_perf.ocache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_OCACHE_MISS_W,  dcr_data_w, sysmem_perf.ocache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_OCACHE_BANK_ST, dcr_data_w, sysmem_perf.ocache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_OCACHE_MSHR_ST, dcr_data_w, sysmem_perf.ocache.mshr_stalls);
                        default:;
                        endcase
                    end
                `endif
                    default:;
                    endcase
                end
            `endif
            end
        endcase
    end

    assign dcr_data = dcr_data_w;
`ifndef PERF_ENABLE
    `UNUSED_VAR (dcr_mpm_class)
`endif

`ifdef PERF_ENABLE
    `UNUSED_VAR (sysmem_perf.icache);
    `UNUSED_VAR (sysmem_perf.lmem);
`endif

endmodule
