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

module VX_csr_data
import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_F_ENABLE
import VX_fpu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire                          clk,
    input wire                          reset,

    input wire [CORE_ID_WIDTH-1:0]      core_id,

`ifdef VX_CFG_EXT_F_ENABLE
    // FP CSR of the requesting warp: read live, written at fire
    input wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0]  fcsr_rd_data,
    output wire [2:0]                              fcsr_wr_sel,  // {fcsr, frm, fflags}
    output wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr_wr_data,
`endif

    VX_sched_csr_if.slave               sched_csr_if,

    // context of the request's CTA, read by req_cta_id
    input cta_ctx_t                     cta_ctx,

    // The request is held for at least one cycle before it fires; that cycle
    // decodes it, and the fire cycle only merges the values that can still move.
    input wire                          req_fire,
    input wire [UUID_WIDTH-1:0]         req_uuid,
    input wire [NW_WIDTH-1:0]           req_wid,
    input wire [NCTA_WIDTH-1:0]         req_cta_id,
    input wire [`VX_CSR_ADDR_BITS-1:0]  req_addr,
    input wire [INST_SFU_BITS-1:0]      req_op,
    input wire [`VX_CFG_XLEN-1:0]       req_src,
    output wire [`VX_CFG_XLEN-1:0]      read_data
);

    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (req_uuid)
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] __cta_param = cta_ctx.param;
    `UNUSED_VAR (__cta_param)

    // Values that can change between the decode cycle and the fire cycle are
    // read live; each is selected by a registered one-hot.
    localparam LIVE_CYCLE       = 0;
    localparam LIVE_CYCLE_H     = 1;
    localparam LIVE_INSTRET     = 2;
    localparam LIVE_INSTRET_H   = 3;
    localparam LIVE_THREADS     = 4;
    localparam LIVE_WARPS       = 5;
    localparam LIVE_FFLAGS      = 6;
    localparam LIVE_FRM         = 7;
    localparam LIVE_FCSR        = 8;
    localparam LIVE_CTA_RANK    = 9;
    localparam LIVE_CTA_SIZE    = 10;
    localparam LIVE_BLOCK_ID    = 11; // x, y, z
    localparam LIVE_BLOCK_DIM   = 14; // x, y, z
    localparam LIVE_GRID_DIM    = 17; // x, y, z
    localparam LIVE_LMEM_ADDR   = 20;
    localparam LIVE_CLUSTER     = 21;
    localparam LIVE_ENTRY       = 22;
    localparam LIVE_SATP        = 23;
    localparam NUM_LIVE         = 24;

    // Writable CSRs
    localparam WR_MSCRATCH      = 0;
    localparam WR_SATP          = 1;
    localparam WR_FFLAGS        = 2;
    localparam WR_FRM           = 3;
    localparam WR_FCSR          = 4;
    localparam WR_TRAP          = 5;
    localparam NUM_WR           = WR_TRAP + NUM_TRAP_CSRS;

    function automatic logic [`VX_CFG_XLEN-1:0] csr_rmw(
        input logic [INST_SFU_BITS-1:0] op,
        input logic [`VX_CFG_XLEN-1:0]  value,
        input logic [`VX_CFG_XLEN-1:0]  src
    );
        case (op)
            INST_SFU_CSRRW: csr_rmw = src;
            INST_SFU_CSRRS: csr_rmw = value | src;
            default:        csr_rmw = value & ~src; // INST_SFU_CSRRC
        endcase
    endfunction

    // Decode ////////////////////////////////////////////////////////////////

    // Scheduler CSRs read interface
    assign sched_csr_if.csr_rd_wid    = req_wid;

    reg [`VX_CFG_XLEN-1:0] stable_w;
    reg [NUM_LIVE-1:0]     live_sel_w;
    reg [NUM_WR-1:0]       wr_sel_w;

    always @(*) begin
        stable_w   = '0;
        live_sel_w = '0;
        wr_sel_w   = '0;
        case (req_addr)
            `VX_CSR_MVENDORID  : stable_w = `VX_CFG_XLEN'(`VX_ISA_VENDOR_ID);
            `VX_CSR_MARCHID    : stable_w = `VX_CFG_XLEN'(`VX_ISA_ARCH_ID);
            `VX_CSR_MIMPID     : stable_w = `VX_CFG_XLEN'(`VX_ISA_IMPL_ID);
            `VX_CSR_MISA       : stable_w = `VX_CFG_XLEN'({2'(`CLOG2(`VX_CFG_XLEN/16)), 30'(`VX_CFG_MISA_STD)});
        `ifdef VX_CFG_EXT_F_ENABLE
            `VX_CSR_FFLAGS     : begin live_sel_w[LIVE_FFLAGS] = 1; wr_sel_w[WR_FFLAGS] = 1; end
            `VX_CSR_FRM        : begin live_sel_w[LIVE_FRM]    = 1; wr_sel_w[WR_FRM]    = 1; end
            `VX_CSR_FCSR       : begin live_sel_w[LIVE_FCSR]   = 1; wr_sel_w[WR_FCSR]   = 1; end
        `endif
            `VX_CSR_MSCRATCH   : begin stable_w = `VX_CFG_XLEN'(sched_csr_if.mscratch); wr_sel_w[WR_MSCRATCH] = 1; end

            `VX_CSR_CTA_ID          : stable_w = `VX_CFG_XLEN'(req_cta_id);
            `VX_CSR_CTA_RANK        : live_sel_w[LIVE_CTA_RANK] = 1;
            `VX_CSR_CTA_SIZE        : live_sel_w[LIVE_CTA_SIZE] = 1;
            `VX_CSR_CTA_BLOCK_ID_X  : live_sel_w[LIVE_BLOCK_ID + 0] = 1;
            `VX_CSR_CTA_BLOCK_ID_Y  : live_sel_w[LIVE_BLOCK_ID + 1] = 1;
            `VX_CSR_CTA_BLOCK_ID_Z  : live_sel_w[LIVE_BLOCK_ID + 2] = 1;
            `VX_CSR_CTA_BLOCK_DIM_X : live_sel_w[LIVE_BLOCK_DIM + 0] = 1;
            `VX_CSR_CTA_BLOCK_DIM_Y : live_sel_w[LIVE_BLOCK_DIM + 1] = 1;
            `VX_CSR_CTA_BLOCK_DIM_Z : live_sel_w[LIVE_BLOCK_DIM + 2] = 1;
            `VX_CSR_CTA_GRID_DIM_X  : live_sel_w[LIVE_GRID_DIM + 0] = 1;
            `VX_CSR_CTA_GRID_DIM_Y  : live_sel_w[LIVE_GRID_DIM + 1] = 1;
            `VX_CSR_CTA_GRID_DIM_Z  : live_sel_w[LIVE_GRID_DIM + 2] = 1;
            `VX_CSR_CTA_LMEM_ADDR   : live_sel_w[LIVE_LMEM_ADDR] = 1;
            `VX_CSR_CTA_CLUSTER_SIZE: live_sel_w[LIVE_CLUSTER] = 1;
            `VX_CSR_CTA_ENTRY       : live_sel_w[LIVE_ENTRY] = 1;

            `VX_CSR_WARP_ID    : stable_w = `VX_CFG_XLEN'(req_wid);
            `VX_CSR_CORE_ID    : stable_w = `VX_CFG_XLEN'(core_id);
            `VX_CSR_ACTIVE_THREADS: live_sel_w[LIVE_THREADS] = 1;
            `VX_CSR_ACTIVE_WARPS: live_sel_w[LIVE_WARPS] = 1;
            `VX_CSR_NUM_THREADS: stable_w = `VX_CFG_XLEN'(`VX_CFG_NUM_THREADS);
            `VX_CSR_NUM_WARPS  : stable_w = `VX_CFG_XLEN'(`VX_CFG_NUM_WARPS);
            `VX_CSR_NUM_CORES  : stable_w = `VX_CFG_XLEN'(`VX_CFG_NUM_CORES * `VX_CFG_NUM_CLUSTERS);
            `VX_CSR_LOCAL_MEM_BASE: stable_w = `VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR);
            `VX_CSR_NUM_BARRIERS: stable_w = `VX_CFG_XLEN'(`VX_CFG_NUM_BARRIERS);

            `VX_CSR_MCYCLE     : live_sel_w[LIVE_CYCLE] = 1;
            `VX_CSR_MINSTRET   : live_sel_w[LIVE_INSTRET] = 1;
        `ifndef VX_CFG_XLEN_64
            `VX_CSR_MCYCLE + 12'h80   : live_sel_w[LIVE_CYCLE_H] = 1;
            `VX_CSR_MINSTRET + 12'h80 : live_sel_w[LIVE_INSTRET_H] = 1;
        `endif

        `ifdef VX_CFG_VM_ENABLE
            `VX_CSR_SATP       : begin live_sel_w[LIVE_SATP] = 1; wr_sel_w[WR_SATP] = 1; end
        `endif

            // Machine-mode trap CSRs (stored in the scheduler).
            `VX_CSR_MSTATUS : begin stable_w = sched_csr_if.csr_mstatus; wr_sel_w[WR_TRAP + 0] = 1; end
            `VX_CSR_MTVEC   : begin stable_w = sched_csr_if.csr_mtvec;   wr_sel_w[WR_TRAP + 1] = 1; end
            `VX_CSR_MEPC    : begin stable_w = sched_csr_if.csr_mepc;    wr_sel_w[WR_TRAP + 2] = 1; end
            `VX_CSR_MCAUSE  : begin stable_w = sched_csr_if.csr_mcause;  wr_sel_w[WR_TRAP + 3] = 1; end
            `VX_CSR_MTVAL   : begin stable_w = sched_csr_if.csr_mtval;   wr_sel_w[WR_TRAP + 4] = 1; end

            // The MPM counters are only reachable from the host; everything
            // else reads as zero.
            default:;
        endcase
    end

    wire wr_enable_w = (req_op == INST_SFU_CSRRW) || (| req_src);
    wire [`VX_CFG_XLEN-1:0] wr_data_w = csr_rmw(req_op, stable_w, req_src);

    // Registered in the cycle before the request fires. The stable values
    // cannot change for this warp in that cycle: CSR writes land at an earlier
    // request's fire, and the scheduler's own writes (mscratch at launch or
    // spawn, mepc/mcause at trap entry) target a warp with no CSR request in
    // flight. The SIMULATION check below compares the two at fire.
    reg [`VX_CFG_XLEN-1:0] stable_r, wr_data_r;
    reg [NUM_LIVE-1:0]     live_sel_r;
    reg [NUM_WR-1:0]       wr_sel_r;
    reg                    wr_enable_r;

    always @(posedge clk) begin
        stable_r    <= stable_w;
        live_sel_r  <= live_sel_w;
        wr_sel_r    <= wr_sel_w;
        wr_enable_r <= wr_enable_w;
        wr_data_r   <= wr_data_w;
    end

`ifdef SIMULATION
    always @(posedge clk) begin
        if (~reset && req_fire) begin
            `ASSERT(stable_r == stable_w && live_sel_r == live_sel_w && wr_sel_r == wr_sel_w
                 && wr_enable_r == wr_enable_w && wr_data_r == wr_data_w,
                ("%t: *** %s CSR 0x%0h changed between decode and fire (#%0d)", $time, INSTANCE_ID, req_addr, req_uuid));
        end
    end
`endif

    // Live values ///////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_F_ENABLE
    wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0] req_fcsr = fcsr_rd_data;
`endif

    wire [NUM_LIVE-1:0][`VX_CFG_XLEN-1:0] live_src;

`ifdef VX_CFG_XLEN_64
    assign live_src[LIVE_CYCLE]     = `VX_CFG_XLEN'(sched_csr_if.cycles);
    assign live_src[LIVE_CYCLE_H]   = '0;
    assign live_src[LIVE_INSTRET]   = `VX_CFG_XLEN'(sched_csr_if.instret);
    assign live_src[LIVE_INSTRET_H] = '0;
`else
    assign live_src[LIVE_CYCLE]     = sched_csr_if.cycles[31:0];
    assign live_src[LIVE_CYCLE_H]   = 32'(sched_csr_if.cycles[PERF_CTR_BITS-1:32]);
    assign live_src[LIVE_INSTRET]   = sched_csr_if.instret[31:0];
    assign live_src[LIVE_INSTRET_H] = 32'(sched_csr_if.instret[PERF_CTR_BITS-1:32]);
`endif
    assign live_src[LIVE_THREADS]   = `VX_CFG_XLEN'(sched_csr_if.thread_mask);
    assign live_src[LIVE_WARPS]     = `VX_CFG_XLEN'(sched_csr_if.active_warps);
`ifdef VX_CFG_EXT_F_ENABLE
    assign live_src[LIVE_FFLAGS]    = `VX_CFG_XLEN'(req_fcsr[`FP_FLAGS_BITS-1:0]);
    assign live_src[LIVE_FRM]       = `VX_CFG_XLEN'(req_fcsr[INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
    assign live_src[LIVE_FCSR]      = `VX_CFG_XLEN'(req_fcsr);
`else
    assign live_src[LIVE_FFLAGS]    = '0;
    assign live_src[LIVE_FRM]       = '0;
    assign live_src[LIVE_FCSR]      = '0;
`endif
    // The CTA context RAMs are addressed by the held request and return in the
    // fire cycle.
    assign live_src[LIVE_CTA_RANK]  = `VX_CFG_XLEN'(sched_csr_if.cta_rank);
    assign live_src[LIVE_CTA_SIZE]  = `VX_CFG_XLEN'(cta_ctx.cta_size);
    for (genvar i = 0; i < 3; ++i) begin : g_cta_dims
        assign live_src[LIVE_BLOCK_ID + i]  = `VX_CFG_XLEN'(cta_ctx.block_idx[i]);
        assign live_src[LIVE_BLOCK_DIM + i] = `VX_CFG_XLEN'(cta_ctx.block_dim[i]);
        assign live_src[LIVE_GRID_DIM + i]  = `VX_CFG_XLEN'(cta_ctx.grid_dim[i]);
    end
    assign live_src[LIVE_LMEM_ADDR] = `VX_CFG_XLEN'(cta_ctx.lmem_addr);
    assign live_src[LIVE_CLUSTER]   = `VX_CFG_XLEN'(cta_ctx.cluster_size);
    assign live_src[LIVE_ENTRY]     = `VX_CFG_XLEN'(to_fullPC(cta_ctx.entry));
`ifdef VX_CFG_VM_ENABLE
    assign live_src[LIVE_SATP]      = sched_csr_if.csr_satp;
`else
    assign live_src[LIVE_SATP]      = '0;
`endif

    reg [`VX_CFG_XLEN-1:0] live_data;
    always @(*) begin
        live_data = '0;
        for (integer i = 0; i < NUM_LIVE; ++i) begin
            live_data |= live_sel_r[i] ? live_src[i] : '0;
        end
    end

    assign read_data = stable_r | live_data;

    // Write /////////////////////////////////////////////////////////////////

    wire                   write_fire = req_fire && wr_enable_r;
    wire [NUM_WR-1:0]      write_sel  = {NUM_WR{write_fire}} & wr_sel_r;

    // Scheduler CSRs write interface
    assign sched_csr_if.csr_wr_valid      = write_sel[WR_MSCRATCH];
    assign sched_csr_if.csr_wr_wid        = req_wid;
    assign sched_csr_if.csr_wr_data       = `VX_CFG_MEM_ADDR_WIDTH'(wr_data_r);
    assign sched_csr_if.trap_csr_wr_valid = write_sel[WR_TRAP +: NUM_TRAP_CSRS];
    assign sched_csr_if.trap_csr_wr_data  = wr_data_r;

`ifdef VX_CFG_EXT_F_ENABLE
    // The FP flags keep moving under in-flight FP instructions, so their
    // read-modify-write stays in the fire cycle.
    assign fcsr_wr_data = (INST_FRM_BITS+`FP_FLAGS_BITS)'(csr_rmw(req_op, live_data, req_src));
    assign fcsr_wr_sel  = {write_sel[WR_FCSR], write_sel[WR_FRM], write_sel[WR_FFLAGS]};
`else
    `UNUSED_VAR ({write_sel[WR_FCSR], write_sel[WR_FRM], write_sel[WR_FFLAGS]})
`endif

`ifdef VX_CFG_VM_ENABLE
    // The per-core SATP CSR lives in the scheduler, shared by every SFU block,
    // so its read-modify-write stays in the fire cycle.
    assign sched_csr_if.satp_wr_valid = write_sel[WR_SATP];
    assign sched_csr_if.satp_wr_data  = csr_rmw(req_op, live_data, req_src);
`else
    `UNUSED_VAR (write_sel[WR_SATP])
`endif

    always @(posedge clk) begin
        if (write_fire) begin
            case (req_addr)
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
                `VX_CSR_MSCRATCH:;
                default: begin
                    `ASSERT(0, ("invalid CSR write address: %0h (#%0d)", req_addr, req_uuid));
                end
            endcase
        end
    end

endmodule
