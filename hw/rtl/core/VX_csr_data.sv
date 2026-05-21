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

`ifdef VX_CFG_EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

`ifdef VX_CFG_XLEN_64
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = `VX_CFG_XLEN'(src)
`else
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = src[31:0]; \
        addr+12'h80 : dst = 32'(src[$bits(src)-1:32])
`endif

module VX_csr_data
import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_F_ENABLE
import VX_fpu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire                          clk,
    input wire                          reset,

    input wire [7:0]                    mpm_class,

`ifdef PERF_ENABLE
    input sysmem_perf_t                 sysmem_perf,
    input pipeline_perf_t               pipeline_perf,
`endif

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if.slave                 fpu_csr_if [`VX_CFG_NUM_FPU_BLOCKS],
`endif

    VX_sched_csr_if.slave               sched_csr_if,

    input wire                          read_enable,
    input wire [UUID_WIDTH-1:0]         read_uuid,
    input wire [NW_WIDTH-1:0]           read_wid,
    input wire [NCTA_WIDTH-1:0]         read_cta_id,
    input wire [`VX_CSR_ADDR_BITS-1:0]  read_addr,
    output wire [`VX_CFG_XLEN-1:0]             read_data_ro,
    output wire [`VX_CFG_XLEN-1:0]             read_data_rw,

    input wire                          write_enable,
    input wire [UUID_WIDTH-1:0]         write_uuid,
    input wire [NW_WIDTH-1:0]           write_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  write_addr,
    input wire [`VX_CFG_XLEN-1:0]              write_data
);

    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR ({mpm_class, read_data_rw, read_enable, read_uuid});
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] __cta_param = sched_csr_if.cta_csrs.param;
    `UNUSED_VAR (__cta_param)
    `UNUSED_VAR ({write_data, write_uuid})

    // CSRs Write /////////////////////////////////////////////////////////////

    // Scheduler CSRs write interface
    assign sched_csr_if.csr_wr_valid = write_enable && (write_addr == `VX_CSR_MSCRATCH);
    assign sched_csr_if.csr_wr_wid   = write_wid;
    assign sched_csr_if.csr_wr_data  = `VX_CFG_MEM_ADDR_WIDTH'(write_data);

    // Machine-mode trap CSRs are stored in the scheduler; forward csrw
    // writes to them (csr_wr_wid above carries the warp id).
    wire is_trap_csr = (write_addr == `VX_CSR_MSTATUS)
                    || (write_addr == `VX_CSR_MTVEC)
                    || (write_addr == `VX_CSR_MEPC)
                    || (write_addr == `VX_CSR_MCAUSE)
                    || (write_addr == `VX_CSR_MTVAL);
    assign sched_csr_if.trap_csr_wr_valid = write_enable && is_trap_csr;
    assign sched_csr_if.trap_csr_wr_addr  = write_addr;
    assign sched_csr_if.trap_csr_wr_data  = write_data;

`ifdef VX_CFG_EXT_F_ENABLE
    reg [`VX_CFG_NUM_WARPS-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr, fcsr_n;
    wire [`VX_CFG_NUM_FPU_BLOCKS-1:0]              fpu_write_enable;
    wire [`VX_CFG_NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0] fpu_write_wid;
    fflags_t [`VX_CFG_NUM_FPU_BLOCKS-1:0]          fpu_write_fflags;

    for (genvar i = 0; i < `VX_CFG_NUM_FPU_BLOCKS; ++i) begin : g_fpu_write
        assign fpu_write_enable[i] = fpu_csr_if[i].write_enable;
        assign fpu_write_wid[i]    = fpu_csr_if[i].write_wid;
        assign fpu_write_fflags[i] = fpu_csr_if[i].write_fflags;
    end

    always @(*) begin
        fcsr_n = fcsr;
        for (integer i = 0; i < `VX_CFG_NUM_FPU_BLOCKS; ++i) begin
            if (fpu_write_enable[i]) begin
                fcsr_n[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0] = fcsr[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0]
                                                             | fpu_write_fflags[i];
            end
        end
        if (write_enable) begin
            case (write_addr)
                `VX_CSR_FFLAGS: fcsr_n[write_wid][`FP_FLAGS_BITS-1:0] = write_data[`FP_FLAGS_BITS-1:0];
                `VX_CSR_FRM:    fcsr_n[write_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] = write_data[INST_FRM_BITS-1:0];
                `VX_CSR_FCSR:   fcsr_n[write_wid] = write_data[`FP_FLAGS_BITS+INST_FRM_BITS-1:0];
            default:;
            endcase
        end
    end

    for (genvar i = 0; i < `VX_CFG_NUM_FPU_BLOCKS; ++i) begin : g_fpu_csr_read_frm
        assign fpu_csr_if[i].read_frm = fcsr[fpu_csr_if[i].read_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS];
    end

    always @(posedge clk) begin
        if (reset) begin
            fcsr <= '0;
        end else begin
            fcsr <= fcsr_n;
        end
    end
`endif

`ifdef VX_CFG_VM_ENABLE
    // Per-core SATP CSR. Initialized to 0 (BARE mode); kernel writes
    // it from vx_start.S after the runtime has installed the page table.
    // Surfaced on sched_csr_if.satp so VX_core can pick it up directly
    // off the shared interface instead of routing through SFU/execute.
    reg [`VX_CFG_XLEN-1:0] satp;
    always @(posedge clk) begin
        if (reset) begin
            satp <= '0;
        end else if (write_enable && write_addr == `VX_CSR_SATP) begin
            satp <= write_data;
        end
    end
    assign sched_csr_if.csr_satp = satp;
`endif

    always @(posedge clk) begin
        if (write_enable) begin
            case (write_addr)
            `ifdef VX_CFG_EXT_F_ENABLE
                `VX_CSR_FFLAGS,
                `VX_CSR_FRM,
                `VX_CSR_FCSR,
            `endif
                `VX_CSR_SATP,
                `VX_CSR_MSTATUS,
                `VX_CSR_MNSTATUS,
                `VX_CSR_MEDELEG,
                `VX_CSR_MIDELEG,
                `VX_CSR_MIE,
                `VX_CSR_MTVEC,
                `VX_CSR_MEPC,
                `VX_CSR_MCAUSE,
                `VX_CSR_MTVAL,
                `VX_CSR_PMPCFG0,
                `VX_CSR_PMPADDR0,
                `VX_CSR_MSCRATCH: begin
                    // do nothing — mscratch and the trap CSRs are stored
                    // in the scheduler and written via sched_csr_if above.
                end
                default: begin
                    `ASSERT(0, ("invalid CSR write address: %0h (#%0d)", write_addr, write_uuid));
                end
            endcase
        end
    end

    // CSRs read //////////////////////////////////////////////////////////////

    // Scheduler CSRs read interface
    assign sched_csr_if.csr_rd_wid = read_wid;
    assign sched_csr_if.csr_rd_cta_id = read_cta_id;

    reg [`VX_CFG_XLEN-1:0] read_data_ro_w;
    reg [`VX_CFG_XLEN-1:0] read_data_rw_w;
    reg read_addr_valid_w;

    always @(*) begin
        read_data_ro_w    = '0;
        read_data_rw_w    = '0;
        read_addr_valid_w = 1;
        case (read_addr)
            `VX_CSR_MVENDORID  : read_data_ro_w = `VX_CFG_XLEN'(`VX_ISA_VENDOR_ID);
            `VX_CSR_MARCHID    : read_data_ro_w = `VX_CFG_XLEN'(`VX_ISA_ARCH_ID);
            `VX_CSR_MIMPID     : read_data_ro_w = `VX_CFG_XLEN'(`VX_ISA_IMPL_ID);
            `VX_CSR_MISA       : read_data_ro_w = `VX_CFG_XLEN'({2'(`CLOG2(`VX_CFG_XLEN/16)), 30'(`VX_CFG_MISA_STD)});
        `ifdef VX_CFG_EXT_F_ENABLE
            `VX_CSR_FFLAGS     : read_data_rw_w = `VX_CFG_XLEN'(fcsr[read_wid][`FP_FLAGS_BITS-1:0]);
            `VX_CSR_FRM        : read_data_rw_w = `VX_CFG_XLEN'(fcsr[read_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
            `VX_CSR_FCSR       : read_data_rw_w = `VX_CFG_XLEN'(fcsr[read_wid]);
        `endif
            `VX_CSR_MSCRATCH   : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.mscratch);

            `VX_CSR_CTA_ID          : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.cta_id);
            `VX_CSR_CTA_RANK        : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.cta_rank);
            `VX_CSR_CTA_SIZE        : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.cta_size);
            `VX_CSR_CTA_BLOCK_ID_X  : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.block_idx[0]);
            `VX_CSR_CTA_BLOCK_ID_Y  : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.block_idx[1]);
            `VX_CSR_CTA_BLOCK_ID_Z  : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.block_idx[2]);
            `VX_CSR_CTA_BLOCK_DIM_X : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.block_dim[0]);
            `VX_CSR_CTA_BLOCK_DIM_Y : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.block_dim[1]);
            `VX_CSR_CTA_BLOCK_DIM_Z : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.block_dim[2]);
            `VX_CSR_CTA_GRID_DIM_X  : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.grid_dim[0]);
            `VX_CSR_CTA_GRID_DIM_Y  : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.grid_dim[1]);
            `VX_CSR_CTA_GRID_DIM_Z  : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.grid_dim[2]);
            `VX_CSR_CTA_LMEM_ADDR   : read_data_rw_w = `VX_CFG_XLEN'(sched_csr_if.cta_csrs.lmem_addr);

            `VX_CSR_WARP_ID    : read_data_ro_w = `VX_CFG_XLEN'(read_wid);
            `VX_CSR_CORE_ID    : read_data_ro_w = `VX_CFG_XLEN'(CORE_ID);
            `VX_CSR_ACTIVE_THREADS: read_data_ro_w = `VX_CFG_XLEN'(sched_csr_if.thread_masks[read_wid]);
            `VX_CSR_ACTIVE_WARPS: read_data_ro_w = `VX_CFG_XLEN'(sched_csr_if.active_warps);
            `VX_CSR_NUM_THREADS: read_data_ro_w = `VX_CFG_XLEN'(`VX_CFG_NUM_THREADS);
            `VX_CSR_NUM_WARPS  : read_data_ro_w = `VX_CFG_XLEN'(`VX_CFG_NUM_WARPS);
            `VX_CSR_NUM_CORES  : read_data_ro_w = `VX_CFG_XLEN'(`VX_CFG_NUM_CORES * `VX_CFG_NUM_CLUSTERS);
            `VX_CSR_LOCAL_MEM_BASE: read_data_ro_w = `VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR);
            `VX_CSR_NUM_BARRIERS: read_data_ro_w = `VX_CFG_XLEN'(`VX_CFG_NUM_BARRIERS);

            `CSR_READ_64(`VX_CSR_MCYCLE, read_data_ro_w, sched_csr_if.cycles);
            `CSR_READ_64(`VX_CSR_MINSTRET, read_data_ro_w, sched_csr_if.instret);
            `VX_CSR_MPM_RESERVED : read_data_ro_w = 'x;
            `VX_CSR_MPM_RESERVED_H : read_data_ro_w = 'x;

        `ifdef VX_CFG_VM_ENABLE
            `VX_CSR_SATP       : read_data_rw_w = satp;
        `else
            `VX_CSR_SATP,
        `endif
            `VX_CSR_MNSTATUS,
            `VX_CSR_MEDELEG,
            `VX_CSR_MIDELEG,
            `VX_CSR_MIE,
            `VX_CSR_PMPCFG0,
            `VX_CSR_PMPADDR0 : read_data_ro_w = `VX_CFG_XLEN'(0);

            // Machine-mode trap CSRs (stored in the scheduler).
            `VX_CSR_MSTATUS : read_data_rw_w = sched_csr_if.csr_mstatus;
            `VX_CSR_MTVEC   : read_data_rw_w = sched_csr_if.csr_mtvec;
            `VX_CSR_MEPC    : read_data_rw_w = sched_csr_if.csr_mepc;
            `VX_CSR_MCAUSE  : read_data_rw_w = sched_csr_if.csr_mcause;
            `VX_CSR_MTVAL   : read_data_rw_w = sched_csr_if.csr_mtval;

            default: begin
                read_addr_valid_w = 0;
                if ((read_addr >= `VX_CSR_MPM_USER   && read_addr < (`VX_CSR_MPM_USER + 32))
                 || (read_addr >= `VX_CSR_MPM_USER_H && read_addr < (`VX_CSR_MPM_USER_H + 32))) begin
                    read_addr_valid_w = 1;
                `ifdef PERF_ENABLE
                    case (mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_IDLE, read_data_ro_w, pipeline_perf.sched.idles);
                        `CSR_READ_64(`VX_CSR_MPM_ACTIVE_WARPS, read_data_ro_w, pipeline_perf.sched.active_warps);
                        `CSR_READ_64(`VX_CSR_MPM_STALLED_WARPS, read_data_ro_w, pipeline_perf.sched.stalled_warps);
                        `CSR_READ_64(`VX_CSR_MPM_ISSUED_WARPS, read_data_ro_w, pipeline_perf.sched.issued_warps);
                        `CSR_READ_64(`VX_CSR_MPM_ISSUED_THREADS, read_data_ro_w, pipeline_perf.sched.issued_threads);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_FETCH, read_data_ro_w, pipeline_perf.fetch.stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_IBUF, read_data_ro_w, pipeline_perf.issue.ibf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_SCRB, read_data_ro_w, pipeline_perf.issue.scb_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_OPDS, read_data_ro_w, pipeline_perf.issue.opd_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_ALU, read_data_ro_w, pipeline_perf.issue.dispatch_stalls[EX_ALU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_ALU, read_data_ro_w, pipeline_perf.issue.dispatch_instrs[EX_ALU]);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_LSU, read_data_ro_w, pipeline_perf.issue.dispatch_stalls[EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_LSU, read_data_ro_w, pipeline_perf.issue.dispatch_instrs[EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_STALL_SFU, read_data_ro_w, pipeline_perf.issue.dispatch_stalls[EX_SFU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_SFU, read_data_ro_w, pipeline_perf.issue.dispatch_instrs[EX_SFU]);
                    `ifdef VX_CFG_EXT_F_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_STALL_FPU, read_data_ro_w, pipeline_perf.issue.dispatch_stalls[EX_FPU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_FPU, read_data_ro_w, pipeline_perf.issue.dispatch_instrs[EX_FPU]);
                    `endif
                    `ifdef VX_CFG_EXT_TCU_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_STALL_TCU, read_data_ro_w, pipeline_perf.issue.dispatch_stalls[EX_TCU]);
                        `CSR_READ_64(`VX_CSR_MPM_INSTR_TCU, read_data_ro_w, pipeline_perf.issue.dispatch_instrs[EX_TCU]);
                    `endif
                        // PERF: branches
                        `CSR_READ_64(`VX_CSR_MPM_BRANCHES, read_data_ro_w, pipeline_perf.sched.branches);
                        `CSR_READ_64(`VX_CSR_MPM_DIVERGENCE, read_data_ro_w, pipeline_perf.sched.divergence);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_MEM_READS, read_data_ro_w, sysmem_perf.mem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_WRITES, read_data_ro_w, sysmem_perf.mem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_IFETCHES, read_data_ro_w, pipeline_perf.ifetches);
                        `CSR_READ_64(`VX_CSR_MPM_LOADS, read_data_ro_w, pipeline_perf.loads);
                        `CSR_READ_64(`VX_CSR_MPM_STORES, read_data_ro_w, pipeline_perf.stores);
                        `CSR_READ_64(`VX_CSR_MPM_IFETCH_LT, read_data_ro_w, pipeline_perf.ifetch_latency);
                        `CSR_READ_64(`VX_CSR_MPM_LOAD_LT, read_data_ro_w, pipeline_perf.load_latency);
                        default:;
                        endcase
                    end
                `ifdef VX_CFG_VM_ENABLE
                    `VX_DCR_MPM_CLASS_VM: begin
                        case (read_addr)
                        // PERF: VM/MMU (icache + dcache MMU summed)
                        `CSR_READ_64(`VX_CSR_MPM_TLB_READS,   read_data_ro_w, pipeline_perf.mmu.tlb_reads);
                        `CSR_READ_64(`VX_CSR_MPM_TLB_HITS,    read_data_ro_w, pipeline_perf.mmu.tlb_hits);
                        `CSR_READ_64(`VX_CSR_MPM_TLB_MISSES,  read_data_ro_w, pipeline_perf.mmu.tlb_misses);
                        `CSR_READ_64(`VX_CSR_MPM_TLB_EVICTS,  read_data_ro_w, pipeline_perf.mmu.tlb_evictions);
                        `CSR_READ_64(`VX_CSR_MPM_PTW_WALKS,   read_data_ro_w, pipeline_perf.mmu.ptw_walks);
                        `CSR_READ_64(`VX_CSR_MPM_PTW_LATENCY, read_data_ro_w, pipeline_perf.mmu.ptw_latency);
                        default:;
                        endcase
                    end
                `endif
                    `VX_DCR_MPM_CLASS_MEM: begin
                        case (read_addr)
                        // PERF: icache
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_READS, read_data_ro_w, sysmem_perf.icache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MISS_R, read_data_ro_w, sysmem_perf.icache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MSHR_ST, read_data_ro_w, sysmem_perf.icache.mshr_stalls);
                        // PERF: dcache
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_READS, read_data_ro_w, sysmem_perf.dcache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_WRITES, read_data_ro_w, sysmem_perf.dcache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_R, read_data_ro_w, sysmem_perf.dcache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_W, read_data_ro_w, sysmem_perf.dcache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_BANK_ST, read_data_ro_w, sysmem_perf.dcache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MSHR_ST, read_data_ro_w, sysmem_perf.dcache.mshr_stalls);
                        // PERF: lmem
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_READS, read_data_ro_w, sysmem_perf.lmem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_WRITES, read_data_ro_w, sysmem_perf.lmem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_BANK_ST, read_data_ro_w, sysmem_perf.lmem.bank_stalls);
                        // PERF: l2cache
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_READS, read_data_ro_w, sysmem_perf.l2cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_WRITES, read_data_ro_w, sysmem_perf.l2cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_R, read_data_ro_w, sysmem_perf.l2cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_W, read_data_ro_w, sysmem_perf.l2cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_BANK_ST, read_data_ro_w, sysmem_perf.l2cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MSHR_ST, read_data_ro_w, sysmem_perf.l2cache.mshr_stalls);
                        // PERF: l3cache
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_READS, read_data_ro_w, sysmem_perf.l3cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_WRITES, read_data_ro_w, sysmem_perf.l3cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_R, read_data_ro_w, sysmem_perf.l3cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_W, read_data_ro_w, sysmem_perf.l3cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_BANK_ST, read_data_ro_w, sysmem_perf.l3cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MSHR_ST, read_data_ro_w, sysmem_perf.l3cache.mshr_stalls);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_MEM_READS, read_data_ro_w, sysmem_perf.mem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_WRITES, read_data_ro_w, sysmem_perf.mem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_LT, read_data_ro_w, sysmem_perf.mem.latency);
                        // PERF: coalescer
                        `CSR_READ_64(`VX_CSR_MPM_COALESCER_MISS, read_data_ro_w, sysmem_perf.coalescer.misses);
                        default:;
                        endcase
                    end
                `ifdef VX_CFG_EXT_DXA_ENABLE
                    `VX_DCR_MPM_CLASS_DXA: begin
                        case (read_addr)
                        `CSR_READ_64(`VX_CSR_MPM_DXA_TRANSFERS,  read_data_ro_w, sysmem_perf.dxa.transfers);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_GMEM_READS, read_data_ro_w, sysmem_perf.dxa.gmem_reads);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_GMEM_DEDUP, read_data_ro_w, sysmem_perf.dxa.gmem_dedup);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_LMEM_WRITES,read_data_ro_w, sysmem_perf.dxa.lmem_writes);
                        `CSR_READ_64(`VX_CSR_MPM_DXA_GMEM_LT,    read_data_ro_w, sysmem_perf.dxa.gmem_latency);
                        default:;
                        endcase
                    end
                `endif
                `ifdef VX_CFG_EXT_TCU_ENABLE
                    `VX_DCR_MPM_CLASS_TCU: begin
                        case (read_addr)
                        `CSR_READ_64(`VX_CSR_MPM_TCU_TBUF_STALLS,      read_data_ro_w, pipeline_perf.tcu.tbuf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_TCU_TBUF_CACHE_HITS, read_data_ro_w, pipeline_perf.tcu.tbuf_cache_hits);
                        `CSR_READ_64(`VX_CSR_MPM_TCU_LMEM_READS,     read_data_ro_w, pipeline_perf.tcu.lmem_reads);
                        default:;
                        endcase
                    end
                `endif
                    default:;
                    endcase
                `endif
                end
            end
        endcase
        // If still invalid after decode, return zero instead of halting.
        if (!read_addr_valid_w) begin
            read_data_ro_w = '0;
            read_data_rw_w = '0;
        end
    end

    assign read_data_ro = read_data_ro_w;
    assign read_data_rw = read_data_rw_w;

`ifdef PERF_ENABLE
    `UNUSED_VAR (sysmem_perf.icache);
    `UNUSED_VAR (sysmem_perf.lmem);
`endif

endmodule
