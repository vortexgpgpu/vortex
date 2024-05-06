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

`ifdef EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

`ifdef XLEN_64
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = `XLEN'(src)
`else
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = src[31:0]; \
        addr+12'h80 : dst = 32'(src[$bits(src)-1:32])
`endif


module VX_csr_data
import VX_gpu_pkg::*;
`ifdef EXT_F_ENABLE
import VX_fpu_pkg::*;
`endif
#(
    parameter CORE_ID = 0
) (
    input wire                          clk,
    input wire                          reset,

    input base_dcrs_t                   base_dcrs,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave                mem_perf_if,
    VX_pipeline_perf_if.slave           pipeline_perf_if,
`endif

    VX_commit_csr_if.slave              commit_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_csr_if.slave                 fpu_csr_if [`NUM_FPU_BLOCKS],
`endif

    input wire [`PERF_CTR_BITS-1:0]     cycles,
    input wire [`NUM_WARPS-1:0]         active_warps,
    input wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks,

    input wire                          read_enable,
    input wire [`UUID_WIDTH-1:0]        read_uuid,
    input wire [`NW_WIDTH-1:0]          read_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  read_addr,
    output wire [`XLEN-1:0]             read_data_ro,
    output wire [`XLEN-1:0]             read_data_rw,

    input wire                          write_enable,
    input wire [`UUID_WIDTH-1:0]        write_uuid,
    input wire [`NW_WIDTH-1:0]          write_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  write_addr,
    input wire [`XLEN-1:0]              write_data
);

    `UNUSED_VAR (reset)
    `UNUSED_VAR (write_wid)
    `UNUSED_VAR (write_data)

    // CSRs Write /////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] mscratch;

`ifdef EXT_F_ENABLE
    reg [`NUM_WARPS-1:0][`INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr, fcsr_n;
    wire [`NUM_FPU_BLOCKS-1:0]              fpu_write_enable;
    wire [`NUM_FPU_BLOCKS-1:0][`NW_WIDTH-1:0] fpu_write_wid;
    fflags_t [`NUM_FPU_BLOCKS-1:0]          fpu_write_fflags;

    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
        assign fpu_write_enable[i] = fpu_csr_if[i].write_enable;
        assign fpu_write_wid[i]    = fpu_csr_if[i].write_wid;
        assign fpu_write_fflags[i] = fpu_csr_if[i].write_fflags;
    end

    always @(*) begin
        fcsr_n = fcsr;
        for (integer i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
            if (fpu_write_enable[i]) begin
                fcsr_n[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0] = fcsr[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0]
                                                             | fpu_write_fflags[i];
            end
        end
        if (write_enable) begin
            case (write_addr)
                `VX_CSR_FFLAGS: fcsr_n[write_wid][`FP_FLAGS_BITS-1:0] = write_data[`FP_FLAGS_BITS-1:0];
                `VX_CSR_FRM:    fcsr_n[write_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] = write_data[`INST_FRM_BITS-1:0];
                `VX_CSR_FCSR:   fcsr_n[write_wid] = write_data[`FP_FLAGS_BITS+`INST_FRM_BITS-1:0];
            default:;
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
        assign fpu_csr_if[i].read_frm = fcsr[fpu_csr_if[i].read_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS];
    end

    always @(posedge clk) begin
        if (reset) begin
            fcsr <= '0;
        end else begin
            fcsr <= fcsr_n;
        end
    end
`endif

    always @(posedge clk) begin
        if (reset) begin
            mscratch <= base_dcrs.startup_arg;
        end
        if (write_enable) begin
            case (write_addr)
            `ifdef EXT_F_ENABLE
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
                `VX_CSR_PMPCFG0,
                `VX_CSR_PMPADDR0: begin
                    // do nothing!
                end
                `VX_CSR_MSCRATCH: begin
                    mscratch <= write_data;
                end
                default: begin
                    `ASSERT(0, ("%t: *** invalid CSR write address: %0h (#%0d)", $time, write_addr, write_uuid));
                end
            endcase
        end
    end

    // CSRs read //////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] read_data_ro_r;
    reg [`XLEN-1:0] read_data_rw_r;
    reg read_addr_valid_r;

    always @(*) begin
        read_data_ro_r    = '0;
        read_data_rw_r    = '0;
        read_addr_valid_r = 1;
        case (read_addr)
            `VX_CSR_MVENDORID  : read_data_ro_r = `XLEN'(`VENDOR_ID);
            `VX_CSR_MARCHID    : read_data_ro_r = `XLEN'(`ARCHITECTURE_ID);
            `VX_CSR_MIMPID     : read_data_ro_r = `XLEN'(`IMPLEMENTATION_ID);
            `VX_CSR_MISA       : read_data_ro_r = `XLEN'({2'(`CLOG2(`XLEN/16)), 30'(`MISA_STD)});
        `ifdef EXT_F_ENABLE
            `VX_CSR_FFLAGS     : read_data_rw_r = `XLEN'(fcsr[read_wid][`FP_FLAGS_BITS-1:0]);
            `VX_CSR_FRM        : read_data_rw_r = `XLEN'(fcsr[read_wid][`INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
            `VX_CSR_FCSR       : read_data_rw_r = `XLEN'(fcsr[read_wid]);
        `endif
            `VX_CSR_MSCRATCH   : read_data_rw_r = mscratch;

            `VX_CSR_WARP_ID    : read_data_ro_r = `XLEN'(read_wid);
            `VX_CSR_CORE_ID    : read_data_ro_r = `XLEN'(CORE_ID);
            `VX_CSR_ACTIVE_THREADS: read_data_ro_r = `XLEN'(thread_masks[read_wid]);
            `VX_CSR_ACTIVE_WARPS: read_data_ro_r = `XLEN'(active_warps);
            `VX_CSR_NUM_THREADS: read_data_ro_r = `XLEN'(`NUM_THREADS);
            `VX_CSR_NUM_WARPS  : read_data_ro_r = `XLEN'(`NUM_WARPS);
            `VX_CSR_NUM_CORES  : read_data_ro_r = `XLEN'(`NUM_CORES * `NUM_CLUSTERS);
            `VX_CSR_NUM_BARRIERS: read_data_ro_r = `XLEN'(`NUM_BARRIERS);

            `CSR_READ_64(`VX_CSR_MCYCLE, read_data_ro_r, cycles);

            `VX_CSR_MPM_RESERVED : read_data_ro_r = 'x;
            `VX_CSR_MPM_RESERVED_H : read_data_ro_r = 'x;

            `CSR_READ_64(`VX_CSR_MINSTRET, read_data_ro_r, commit_csr_if.instret);

            `VX_CSR_SATP,
            `VX_CSR_MSTATUS,
            `VX_CSR_MNSTATUS,
            `VX_CSR_MEDELEG,
            `VX_CSR_MIDELEG,
            `VX_CSR_MIE,
            `VX_CSR_MTVEC,
            `VX_CSR_MEPC,
            `VX_CSR_PMPCFG0,
            `VX_CSR_PMPADDR0 : read_data_ro_r = `XLEN'(0);

            default: begin
                read_addr_valid_r = 0;
                if ((read_addr >= `VX_CSR_MPM_USER   && read_addr < (`VX_CSR_MPM_USER + 32))
                 || (read_addr >= `VX_CSR_MPM_USER_H && read_addr < (`VX_CSR_MPM_USER_H + 32))) begin
                    read_addr_valid_r = 1;
                `ifdef PERF_ENABLE
                    case (base_dcrs.mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_ID, read_data_ro_r, pipeline_perf_if.sched_idles);
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_ST, read_data_ro_r, pipeline_perf_if.sched_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_IBUF_ST, read_data_ro_r, pipeline_perf_if.ibf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_ST, read_data_ro_r, pipeline_perf_if.scb_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_ALU, read_data_ro_r, pipeline_perf_if.units_uses[`EX_ALU]);
                    `ifdef EXT_F_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_FPU, read_data_ro_r, pipeline_perf_if.units_uses[`EX_FPU]);
                    `else
                        `VX_CSR_MPM_SCRB_FPU   : read_data_ro_r = '0;
                        `VX_CSR_MPM_SCRB_FPU_H : read_data_ro_r = '0;
                    `endif
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_LSU, read_data_ro_r, pipeline_perf_if.units_uses[`EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_SFU, read_data_ro_r, pipeline_perf_if.units_uses[`EX_SFU]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_CSRS, read_data_ro_r, pipeline_perf_if.sfu_uses[`SFU_CSRS]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_WCTL, read_data_ro_r, pipeline_perf_if.sfu_uses[`SFU_WCTL]);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_IFETCHES, read_data_ro_r, pipeline_perf_if.ifetches);
                        `CSR_READ_64(`VX_CSR_MPM_LOADS, read_data_ro_r, pipeline_perf_if.loads);
                        `CSR_READ_64(`VX_CSR_MPM_STORES, read_data_ro_r, pipeline_perf_if.stores);
                        `CSR_READ_64(`VX_CSR_MPM_IFETCH_LT, read_data_ro_r, pipeline_perf_if.ifetch_latency);
                        `CSR_READ_64(`VX_CSR_MPM_LOAD_LT, read_data_ro_r, pipeline_perf_if.load_latency);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_MEM: begin
                        case (read_addr)
                        // PERF: icache
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_READS, read_data_ro_r, mem_perf_if.icache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MISS_R, read_data_ro_r, mem_perf_if.icache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MSHR_ST, read_data_ro_r, mem_perf_if.icache.mshr_stalls);
                        // PERF: dcache
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_READS, read_data_ro_r, mem_perf_if.dcache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_WRITES, read_data_ro_r, mem_perf_if.dcache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_R, read_data_ro_r, mem_perf_if.dcache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_W, read_data_ro_r, mem_perf_if.dcache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_BANK_ST, read_data_ro_r, mem_perf_if.dcache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MSHR_ST, read_data_ro_r, mem_perf_if.dcache.mshr_stalls);
                        // PERF: lmem
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_READS, read_data_ro_r, mem_perf_if.lmem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_WRITES, read_data_ro_r, mem_perf_if.lmem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_BANK_ST, read_data_ro_r, mem_perf_if.lmem.bank_stalls);
                        // PERF: l2cache
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_READS, read_data_ro_r, mem_perf_if.l2cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_WRITES, read_data_ro_r, mem_perf_if.l2cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_R, read_data_ro_r, mem_perf_if.l2cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_W, read_data_ro_r, mem_perf_if.l2cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_BANK_ST, read_data_ro_r, mem_perf_if.l2cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MSHR_ST, read_data_ro_r, mem_perf_if.l2cache.mshr_stalls);
                        // PERF: l3cache
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_READS, read_data_ro_r, mem_perf_if.l3cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_WRITES, read_data_ro_r, mem_perf_if.l3cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_R, read_data_ro_r, mem_perf_if.l3cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_W, read_data_ro_r, mem_perf_if.l3cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_BANK_ST, read_data_ro_r, mem_perf_if.l3cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MSHR_ST, read_data_ro_r, mem_perf_if.l3cache.mshr_stalls);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_MEM_READS, read_data_ro_r, mem_perf_if.mem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_WRITES, read_data_ro_r, mem_perf_if.mem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_LT, read_data_ro_r, mem_perf_if.mem.latency);
                        default:;
                        endcase
                    end
                    default:;
                    endcase
                `endif
                end
            end
        endcase
    end

    assign read_data_ro = read_data_ro_r;
    assign read_data_rw = read_data_rw_r;

    `UNUSED_VAR (base_dcrs)

    `RUNTIME_ASSERT(~read_enable || read_addr_valid_r, ("%t: *** invalid CSR read address: 0x%0h (#%0d)", $time, read_addr, read_uuid))

`ifdef PERF_ENABLE
    `UNUSED_VAR (mem_perf_if.icache);
    `UNUSED_VAR (mem_perf_if.lmem);
`endif

endmodule
