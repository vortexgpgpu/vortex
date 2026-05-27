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
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire                          clk,
    input wire                          reset,

    input base_dcrs_t                   base_dcrs,

`ifdef PERF_ENABLE
    input sysmem_perf_t                 sysmem_perf,
    input pipeline_perf_t               pipeline_perf,
`endif

    VX_commit_csr_if.slave              commit_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_csr_if.slave                 fpu_csr_if [`NUM_FPU_BLOCKS],
`endif

    input wire [PERF_CTR_BITS-1:0]      cycles,
    input wire [`NUM_WARPS-1:0]         active_warps,
    input wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks,

    input wire                          read_enable,
    input wire [UUID_WIDTH-1:0]         read_uuid,
    input wire [NW_WIDTH-1:0]           read_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  read_addr,
    output wire [`XLEN-1:0]             read_data_ro,
    output wire [`XLEN-1:0]             read_data_rw,

    input wire                          write_enable,
    input wire [UUID_WIDTH-1:0]         write_uuid,
    input wire [NW_WIDTH-1:0]           write_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  write_addr,
    input wire [`XLEN-1:0]              write_data,

    //==========================================================================
    // DFV Controllability: CSR-driven control signals
    //==========================================================================
    output wire                         dfv_enable,
    output wire                         dfv_stall_icache_req,
    output wire                         dfv_stall_dcache_req,
    output wire                         dfv_stall_writeback,
    output wire                         dfv_stall_fill,
    output wire [15:0]                  dfv_throttle_threshold
);

    `UNUSED_VAR (reset)
    `UNUSED_VAR (write_wid)
    `UNUSED_VAR (write_data)

    // CSRs Write /////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] mscratch;

    //==========================================================================
    // DFV Control Registers
    //==========================================================================
    reg dfv_ctrl_enable = 1'b0;          // Initialize to 0 for simulation
    reg dfv_ctrl_icache_stall = 1'b0;    // Initialize to 0 for simulation (1=enable LFSR-based stall)
    reg dfv_ctrl_dcache_stall = 1'b0;    // Initialize to 0 for simulation (1=enable LFSR-based dcache req stall)
    reg dfv_ctrl_writeback_stall = 1'b0; // Initialize to 0 for simulation (1=enable LFSR-based writeback stall)
    reg dfv_ctrl_fill_stall = 1'b0;      // Initialize to 0 for simulation (1=enable LFSR-based cache fill stall)
    reg [31:0] dfv_random_seed = 32'h12345678;     // LFSR1 seed (set timing)
    reg [31:0] dfv_release_seed = 32'h87654321;   // LFSR2 seed (release timing)
    reg [7:0] dfv_set_threshold = 8'd128;          // SET probability: activate stall when lfsr1 < threshold
    reg [15:0] dfv_release_threshold = 16'd16;     // RELEASE probability: release when lfsr2[15:0] >= threshold
    reg dfv_release_forever = 1'b0;               // When 1: once released, stalls stay off permanently
    reg [15:0] dfv_throttle_thresh_r = 16'h1800;   // Throttle counter threshold

    //==========================================================================
    // DFV Dual-LFSR Architecture
    //==========================================================================
    // LFSR1 (SET):     Controls when each stall point activates (independent per point)
    // LFSR2 (RELEASE): Controls when ALL active stalls release (synchronized)
    //
    // Each stall point uses different LFSR1 bits for SET (stalls activate at different times)
    // All stall points use the same LFSR2 comparison for RELEASE (synchronized deactivation)
    //==========================================================================

    // LFSR1: 32-bit Galois LFSR for stall SET decisions (per-point bits)
    reg [31:0] dfv_lfsr1 = 32'h12345678;

    always @(posedge clk) begin
        if (reset) begin
            dfv_lfsr1 <= 32'h12345678;
        end else if (write_enable && write_addr == 12'h7C2) begin
            dfv_lfsr1 <= write_data[31:0];
        end else if (dfv_ctrl_enable) begin
            dfv_lfsr1 <= {1'b0, dfv_lfsr1[31:1]} ^ (dfv_lfsr1[0] ? 32'hD0000001 : 32'h0);
        end
    end

    // LFSR2: 32-bit Galois LFSR for stall RELEASE decisions (shared across all points)
    // Uses different polynomial (0xB4000001) for decorrelated sequence
    reg [31:0] dfv_lfsr2 = 32'h87654321;

    always @(posedge clk) begin
        if (reset) begin
            dfv_lfsr2 <= 32'h87654321;
        end else if (write_enable && write_addr == 12'h7C8) begin
            dfv_lfsr2 <= write_data[31:0];
        end else if (dfv_ctrl_enable) begin
            dfv_lfsr2 <= {1'b0, dfv_lfsr2[31:1]} ^ (dfv_lfsr2[0] ? 32'hB4000001 : 32'h0);
        end
    end

`ifdef EXT_F_ENABLE
    reg [`NUM_WARPS-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr, fcsr_n;
    wire [`NUM_FPU_BLOCKS-1:0]              fpu_write_enable;
    wire [`NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0] fpu_write_wid;
    fflags_t [`NUM_FPU_BLOCKS-1:0]          fpu_write_fflags;

    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin : g_fpu_write
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
                `VX_CSR_FRM:    fcsr_n[write_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] = write_data[INST_FRM_BITS-1:0];
                `VX_CSR_FCSR:   fcsr_n[write_wid] = write_data[`FP_FLAGS_BITS+INST_FRM_BITS-1:0];
            default:;
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin : g_fpu_csr_read_frm
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

    always @(posedge clk) begin
        if (reset) begin
            mscratch <= base_dcrs.startup_arg;
            dfv_ctrl_enable <= 1'b0;
            dfv_ctrl_icache_stall <= 1'b0;
            dfv_ctrl_dcache_stall <= 1'b0;
            dfv_ctrl_writeback_stall <= 1'b0;
            dfv_ctrl_fill_stall <= 1'b0;
            dfv_random_seed <= 32'h12345678;
            dfv_release_seed <= 32'h87654321;
            dfv_set_threshold <= 8'd128;
            dfv_release_threshold <= 16'd16;
            dfv_release_forever <= 1'b0;
            dfv_throttle_thresh_r <= 16'h1800;
            dfv_release_delay_icache <= 4'd0;
            dfv_release_delay_dcache <= 4'd0;
            dfv_release_delay_wb     <= 4'd0;
            dfv_release_delay_fill   <= 4'd0;
        end else if (write_enable) begin
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
                12'h7C0: begin  // VX_CSR_DFV_CTRL
                    dfv_ctrl_enable <= write_data[0];
                end
                12'h7C1: begin  // VX_CSR_DFV_ICACHE_STALL (enable LFSR-based stall)
                    dfv_ctrl_icache_stall <= write_data[0];
                end
                12'h7C2: begin  // VX_CSR_DFV_RANDOM_SEED (handled in LFSR always block)
                    dfv_random_seed <= write_data[31:0];
                end
                12'h7C3: begin  // VX_CSR_DFV_SET_THRESHOLD
                    dfv_set_threshold <= write_data[7:0];
                end
                12'h7C4: begin  // VX_CSR_DFV_DCACHE_STALL (enable LFSR-based dcache req stall)
                    dfv_ctrl_dcache_stall <= write_data[0];
                end
                12'h7C5: begin  // VX_CSR_DFV_WRITEBACK_STALL (enable LFSR-based writeback stall)
                    dfv_ctrl_writeback_stall <= write_data[0];
                end
                12'h7C6: begin  // VX_CSR_DFV_FILL_STALL (enable LFSR-based cache fill stall)
                    dfv_ctrl_fill_stall <= write_data[0];
                end
                12'h7C7: begin  // VX_CSR_DFV_RELEASE_THRESHOLD
                    dfv_release_threshold <= write_data[15:0];
                end
                12'h7C8: begin  // VX_CSR_DFV_RELEASE_SEED (handled in LFSR2 always block)
                    dfv_release_seed <= write_data[31:0];
                end
                12'h7C9: begin  // VX_CSR_DFV_RELEASE_DELAY (per-point release delay, 0-8 cycles)
                    dfv_release_delay_icache <= write_data[3:0];
                    dfv_release_delay_dcache <= write_data[7:4];
                    dfv_release_delay_wb     <= write_data[11:8];
                    dfv_release_delay_fill   <= write_data[15:12];
                end
                12'h7CA: begin  // VX_CSR_DFV_RELEASE_FOREVER
                    dfv_release_forever <= write_data[0];
                end
                12'h7CB: begin  // VX_CSR_DFV_THROTTLE_THRESHOLD
                    dfv_throttle_thresh_r <= write_data[15:0];
                end
                default: begin
                    `ASSERT(0, ("%t: *** %s invalid CSR write address: %0h (#%0d)", $time, INSTANCE_ID, write_addr, write_uuid));
                end
            endcase
        end
    end

    // CSRs read //////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] read_data_ro_w;
    reg [`XLEN-1:0] read_data_rw_w;
    reg read_addr_valid_w;

    always @(*) begin
        read_data_ro_w    = '0;
        read_data_rw_w    = '0;
        read_addr_valid_w = 1;
        case (read_addr)
            `VX_CSR_MVENDORID  : read_data_ro_w = `XLEN'(`VENDOR_ID);
            `VX_CSR_MARCHID    : read_data_ro_w = `XLEN'(`ARCHITECTURE_ID);
            `VX_CSR_MIMPID     : read_data_ro_w = `XLEN'(`IMPLEMENTATION_ID);
            `VX_CSR_MISA       : read_data_ro_w = `XLEN'({2'(`CLOG2(`XLEN/16)), 30'(`MISA_STD)});
        `ifdef EXT_F_ENABLE
            `VX_CSR_FFLAGS     : read_data_rw_w = `XLEN'(fcsr[read_wid][`FP_FLAGS_BITS-1:0]);
            `VX_CSR_FRM        : read_data_rw_w = `XLEN'(fcsr[read_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
            `VX_CSR_FCSR       : read_data_rw_w = `XLEN'(fcsr[read_wid]);
        `endif
            `VX_CSR_MSCRATCH   : read_data_rw_w = mscratch;

            // DFV CSRs
            12'h7C0            : read_data_rw_w = `XLEN'(dfv_ctrl_enable);       // VX_CSR_DFV_CTRL
            12'h7C1            : read_data_rw_w = `XLEN'(dfv_ctrl_icache_stall); // VX_CSR_DFV_ICACHE_STALL
            12'h7C2            : read_data_rw_w = `XLEN'(dfv_random_seed);       // VX_CSR_DFV_RANDOM_SEED
            12'h7C3            : read_data_rw_w = `XLEN'(dfv_set_threshold);     // VX_CSR_DFV_SET_THRESHOLD
            12'h7C4            : read_data_rw_w = `XLEN'(dfv_ctrl_dcache_stall); // VX_CSR_DFV_DCACHE_STALL
            12'h7C5            : read_data_rw_w = `XLEN'(dfv_ctrl_writeback_stall); // VX_CSR_DFV_WRITEBACK_STALL
            12'h7C6            : read_data_rw_w = `XLEN'(dfv_ctrl_fill_stall);      // VX_CSR_DFV_FILL_STALL
            12'h7C7            : read_data_rw_w = `XLEN'(dfv_release_threshold);  // VX_CSR_DFV_RELEASE_THRESHOLD
            12'h7C8            : read_data_rw_w = `XLEN'(dfv_release_seed);       // VX_CSR_DFV_RELEASE_SEED
            12'h7C9            : read_data_rw_w = `XLEN'({dfv_release_delay_fill, dfv_release_delay_wb, dfv_release_delay_dcache, dfv_release_delay_icache}); // VX_CSR_DFV_RELEASE_DELAY
            12'h7CA            : read_data_rw_w = `XLEN'(dfv_release_forever); // VX_CSR_DFV_RELEASE_FOREVER
            12'h7CB            : read_data_rw_w = `XLEN'(dfv_throttle_thresh_r); // VX_CSR_DFV_THROTTLE_THRESHOLD

            `VX_CSR_WARP_ID    : read_data_ro_w = `XLEN'(read_wid);
            `VX_CSR_CORE_ID    : read_data_ro_w = `XLEN'(CORE_ID);
            `VX_CSR_ACTIVE_THREADS: read_data_ro_w = `XLEN'(thread_masks[read_wid]);
            `VX_CSR_ACTIVE_WARPS: read_data_ro_w = `XLEN'(active_warps);
            `VX_CSR_NUM_THREADS: read_data_ro_w = `XLEN'(`NUM_THREADS);
            `VX_CSR_NUM_WARPS  : read_data_ro_w = `XLEN'(`NUM_WARPS);
            `VX_CSR_NUM_CORES  : read_data_ro_w = `XLEN'(`NUM_CORES * `NUM_CLUSTERS);
            `VX_CSR_LOCAL_MEM_BASE: read_data_ro_w = `XLEN'(`LMEM_BASE_ADDR);

            `CSR_READ_64(`VX_CSR_MCYCLE, read_data_ro_w, cycles);

            `VX_CSR_MPM_RESERVED : read_data_ro_w = 'x;
            `VX_CSR_MPM_RESERVED_H : read_data_ro_w = 'x;

            `CSR_READ_64(`VX_CSR_MINSTRET, read_data_ro_w, commit_csr_if.instret);

            `VX_CSR_SATP,
            `VX_CSR_MSTATUS,
            `VX_CSR_MNSTATUS,
            `VX_CSR_MEDELEG,
            `VX_CSR_MIDELEG,
            `VX_CSR_MIE,
            `VX_CSR_MTVEC,
            `VX_CSR_MEPC,
            `VX_CSR_PMPCFG0,
            `VX_CSR_PMPADDR0 : read_data_ro_w = `XLEN'(0);

            default: begin
                read_addr_valid_w = 0;
                if ((read_addr >= `VX_CSR_MPM_USER   && read_addr < (`VX_CSR_MPM_USER + 32))
                 || (read_addr >= `VX_CSR_MPM_USER_H && read_addr < (`VX_CSR_MPM_USER_H + 32))) begin
                    read_addr_valid_w = 1;
                `ifdef PERF_ENABLE
                    case (base_dcrs.mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_ID, read_data_ro_w, pipeline_perf.sched.idles);
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_ST, read_data_ro_w, pipeline_perf.sched.stalls);
                        `CSR_READ_64(`VX_CSR_MPM_IBUF_ST, read_data_ro_w, pipeline_perf.issue.ibf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_ST, read_data_ro_w, pipeline_perf.issue.scb_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_OPDS_ST, read_data_ro_w, pipeline_perf.issue.opd_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_ALU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_ALU]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_LSU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_SFU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_SFU]);
                    `ifdef EXT_F_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_FPU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_FPU]);
                    `endif
                    `ifdef EXT_TCU_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_TCU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_TCU]);
                    `endif
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_CSRS, read_data_ro_w, pipeline_perf.issue.sfu_uses[SFU_CSRS]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_WCTL, read_data_ro_w, pipeline_perf.issue.sfu_uses[SFU_WCTL]);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_IFETCHES, read_data_ro_w, pipeline_perf.ifetches);
                        `CSR_READ_64(`VX_CSR_MPM_LOADS, read_data_ro_w, pipeline_perf.loads);
                        `CSR_READ_64(`VX_CSR_MPM_STORES, read_data_ro_w, pipeline_perf.stores);
                        `CSR_READ_64(`VX_CSR_MPM_IFETCH_LT, read_data_ro_w, pipeline_perf.ifetch_latency);
                        `CSR_READ_64(`VX_CSR_MPM_LOAD_LT, read_data_ro_w, pipeline_perf.load_latency);
                        default:;
                        endcase
                    end
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
                    default:;
                    endcase
                `endif
                end
            end
        endcase
    end

    assign read_data_ro = read_data_ro_w;
    assign read_data_rw = read_data_rw_w;

    //==========================================================================
    // DFV Control Signal Outputs — Dual-LFSR Set/Release with Delay Line
    //==========================================================================
    // Each stall point has:
    //   - Independent SET (LFSR1, different bits per point)
    //   - Shared RELEASE (LFSR2) with per-point programmable delay (0-8 cycles)
    //
    // The delay line compensates for pipeline depth differences between DFV
    // gates and the collision target. Example: dcache req gate is 1 cycle
    // farther from the cache bank than the fill gate. Setting dcache delay=0
    // and fill delay=1 makes the dcache request release 1 cycle earlier,
    // so both events arrive at the cache bank in the same cycle.
    //
    // CSR 0x7C9 (DFV_RELEASE_DELAY) packs per-point delays:
    //   [3:0]   = icache delay (0-8)
    //   [7:4]   = dcache delay (0-8)
    //   [11:8]  = writeback delay (0-8)
    //   [15:12] = fill delay (0-8)
    //==========================================================================
    assign dfv_enable = dfv_ctrl_enable;

    // Raw release signal from LFSR2
    wire dfv_release_raw = (dfv_lfsr2[15:0] >= dfv_release_threshold);

    // 8-stage delay line on the release signal
    reg [7:0] dfv_release_pipe;
    always @(posedge clk) begin
        if (reset) begin
            dfv_release_pipe <= '0;
        end else begin
            dfv_release_pipe <= {dfv_release_pipe[6:0], dfv_release_raw};
        end
    end

    // Tap array: index 0 = no delay (immediate), index 8 = 8-cycle delay
    wire [8:0] dfv_release_taps = {dfv_release_pipe, dfv_release_raw};

    // Per-point delay configuration (from CSR 0x7C9, 4 bits each)
    reg [3:0] dfv_release_delay_icache;
    reg [3:0] dfv_release_delay_dcache;
    reg [3:0] dfv_release_delay_wb;
    reg [3:0] dfv_release_delay_fill;

    // Per-point delayed release signals
    wire dfv_release_icache = dfv_release_taps[dfv_release_delay_icache];
    wire dfv_release_dcache = dfv_release_taps[dfv_release_delay_dcache];
    wire dfv_release_wb     = dfv_release_taps[dfv_release_delay_wb];
    wire dfv_release_fill   = dfv_release_taps[dfv_release_delay_fill];

    // Expose undelayed release for waveform debugging
    wire dfv_release = dfv_release_raw;
    `UNUSED_VAR(dfv_release)

    // Per-point SET conditions (independent, using different LFSR1 bit slices)
    wire dfv_set_icache = dfv_ctrl_icache_stall    && (dfv_lfsr1[7:0]   < dfv_set_threshold);
    wire dfv_set_dcache = dfv_ctrl_dcache_stall    && (dfv_lfsr1[15:8]  < dfv_set_threshold);
    wire dfv_set_wb     = dfv_ctrl_writeback_stall && (dfv_lfsr1[23:16] < dfv_set_threshold);
    wire dfv_set_fill   = dfv_ctrl_fill_stall      && (dfv_lfsr1[31:24] < dfv_set_threshold);

    // Set/release latches: once SET, stay active until per-point delayed RELEASE
    // When dfv_release_forever=1: once released, permanently stays off (cannot re-set)
    reg dfv_stall_active_icache;
    reg dfv_stall_active_dcache;
    reg dfv_stall_active_wb;
    reg dfv_stall_active_fill;
    reg dfv_released_permanently;  // latches high after first release when release_forever=1

    always @(posedge clk) begin
        if (reset || !dfv_ctrl_enable) begin
            dfv_stall_active_icache <= 1'b0;
            dfv_stall_active_dcache <= 1'b0;
            dfv_stall_active_wb     <= 1'b0;
            dfv_stall_active_fill   <= 1'b0;
            dfv_released_permanently <= 1'b0;
        end else if (dfv_released_permanently) begin
            // Permanently released: all stalls forced off, no re-setting
            dfv_stall_active_icache <= 1'b0;
            dfv_stall_active_dcache <= 1'b0;
            dfv_stall_active_wb     <= 1'b0;
            dfv_stall_active_fill   <= 1'b0;
        end else begin
            // Check if any release fires while release_forever is enabled
            if (dfv_release_forever && (dfv_release_icache || dfv_release_dcache || dfv_release_wb || dfv_release_fill)) begin
                dfv_released_permanently <= 1'b1;
            end

            // icache: delayed release has priority over set
            if (dfv_release_icache)
                dfv_stall_active_icache <= 1'b0;
            else if (!dfv_stall_active_icache && dfv_set_icache)
                dfv_stall_active_icache <= 1'b1;

            // dcache
            if (dfv_release_dcache)
                dfv_stall_active_dcache <= 1'b0;
            else if (!dfv_stall_active_dcache && dfv_set_dcache)
                dfv_stall_active_dcache <= 1'b1;

            // writeback
            if (dfv_release_wb)
                dfv_stall_active_wb <= 1'b0;
            else if (!dfv_stall_active_wb && dfv_set_wb)
                dfv_stall_active_wb <= 1'b1;

            // fill
            if (dfv_release_fill)
                dfv_stall_active_fill <= 1'b0;
            else if (!dfv_stall_active_fill && dfv_set_fill)
                dfv_stall_active_fill <= 1'b1;
        end
    end

    assign dfv_stall_icache_req   = dfv_stall_active_icache;
    assign dfv_stall_dcache_req   = dfv_stall_active_dcache;
    assign dfv_stall_writeback    = dfv_stall_active_wb;
    assign dfv_stall_fill         = dfv_stall_active_fill;
    assign dfv_throttle_threshold = dfv_throttle_thresh_r;


    `UNUSED_VAR (base_dcrs)

    `RUNTIME_ASSERT(~read_enable || read_addr_valid_w, ("%t: *** invalid CSR read address: 0x%0h (#%0d)", $time, read_addr, read_uuid))

`ifdef PERF_ENABLE
    `UNUSED_VAR (sysmem_perf.icache);
    `UNUSED_VAR (sysmem_perf.lmem);
`endif

endmodule
