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

// Flat-port testbench wrapper around VX_tcu_unit.
// All VX_dispatch_if and VX_commit_if signals are exposed as individual
// ports so that the Verilator C++ testbench can drive them directly.
// Only ISSUE_WIDTH == 1 is supported by this wrapper.

module VX_tcu_unit_top import VX_gpu_pkg::*, VX_tcu_pkg::*; (
    `SCOPE_IO_DECL

    input  wire clk,
    input  wire reset,

`ifdef TCU_WGMMA_ENABLE
    // LMEM request (output — driven by DUT)
    output wire                                             tcu_lmem_req_valid,
    input  wire                                             tcu_lmem_req_ready,
    output wire [`LMEM_LOG_SIZE-`CLOG2(LSU_WORD_SIZE)-`CLOG2(`LMEM_NUM_BANKS)-1:0] tcu_lmem_req_addr,
    // LMEM response (input — driven by testbench)
    input  wire                                             tcu_lmem_rsp_valid,
    input  wire [`LMEM_NUM_BANKS*`XLEN-1:0]               tcu_lmem_rsp_data,
`endif

    // Dispatch interface — slot 0 (ISSUE_WIDTH must be 1)
    input  wire                                dispatch_valid,
    output wire                                dispatch_ready,
    input  wire [UUID_WIDTH-1:0]               dispatch_uuid,
    input  wire [ISSUE_WIS_W-1:0]             dispatch_wis,
    input  wire [SIMD_IDX_W-1:0]             dispatch_sid,
    input  wire [`SIMD_WIDTH-1:0]             dispatch_tmask,
    input  wire [PC_BITS-1:0]                 dispatch_PC,
    input  wire                                dispatch_wb,
    input  wire [NUM_XREGS-1:0]              dispatch_wr_xregs,
    input  wire [NUM_REGS_BITS-1:0]           dispatch_rd,
    input  wire [BYTESEL_BITS-1:0]           dispatch_bytesel,
    input  wire [INST_OP_BITS-1:0]            dispatch_op_type,
    input  wire [4:0]                          dispatch_fmt_s,
    input  wire [4:0]                          dispatch_fmt_d,
    input  wire [3:0]                          dispatch_step_m,
    input  wire [3:0]                          dispatch_step_n,
    input  wire [3:0]                          dispatch_step_k,
    input  wire [`SIMD_WIDTH*`XLEN-1:0]       dispatch_rs1_data,
    input  wire [`SIMD_WIDTH*`XLEN-1:0]       dispatch_rs2_data,
    input  wire [`SIMD_WIDTH*`XLEN-1:0]       dispatch_rs3_data,
    input  wire                                dispatch_sop,
    input  wire                                dispatch_eop,

    // Commit interface — slot 0
    output wire                                commit_valid,
    input  wire                                commit_ready,
    output wire [UUID_WIDTH-1:0]              commit_uuid,
    output wire [NW_WIDTH-1:0]                commit_wid,
    output wire [SIMD_IDX_W-1:0]            commit_sid,
    output wire [`SIMD_WIDTH-1:0]            commit_tmask,
    output wire [PC_BITS-1:0]               commit_PC,
    output wire                               commit_wb,
    output wire [NUM_XREGS-1:0]             commit_wr_xregs,
    output wire [NUM_REGS_BITS-1:0]          commit_rd,
    output wire [BYTESEL_BITS-1:0]          commit_bytesel,
    output wire [`SIMD_WIDTH*`XLEN-1:0]     commit_data,
    output wire                               commit_sop,
    output wire                               commit_eop
);
    `STATIC_ASSERT (`ISSUE_WIDTH == 1, ("tcu_unit_top: only ISSUE_WIDTH=1 is supported"))

`ifdef TCU_WGMMA_ENABLE
    localparam TCU_LMEM_BANK_ADDR_W = `LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE) - `CLOG2(`LMEM_NUM_BANKS);

    VX_tcu_lmem_if #(
        .DATA_WIDTH (`LMEM_NUM_BANKS * `XLEN),
        .ADDR_WIDTH (TCU_LMEM_BANK_ADDR_W)
    ) tcu_lmem_if();

    assign tcu_lmem_req_valid      = tcu_lmem_if.req_valid;
    assign tcu_lmem_if.req_ready   = tcu_lmem_req_ready;
    assign tcu_lmem_req_addr       = tcu_lmem_if.req_addr;
    assign tcu_lmem_if.rsp_valid   = tcu_lmem_rsp_valid;
    assign tcu_lmem_if.rsp_data    = tcu_lmem_rsp_data;
`endif

    VX_dispatch_if dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if   commit_if  [`ISSUE_WIDTH]();

    // ---- Dispatch --------------------------------------------------------
    assign dispatch_if[0].valid = dispatch_valid;
    assign dispatch_ready = dispatch_if[0].ready;

    always_comb begin
        dispatch_if[0].data                     = '0;
        dispatch_if[0].data.uuid                = dispatch_uuid;
        dispatch_if[0].data.wis                 = dispatch_wis;
        dispatch_if[0].data.sid                 = dispatch_sid;
        dispatch_if[0].data.tmask               = dispatch_tmask;
        dispatch_if[0].data.PC                  = dispatch_PC;
        dispatch_if[0].data.wb                  = dispatch_wb;
        dispatch_if[0].data.wr_xregs            = dispatch_wr_xregs;
        dispatch_if[0].data.rd                  = dispatch_rd;
        dispatch_if[0].data.bytesel             = dispatch_bytesel;
        dispatch_if[0].data.op_type             = dispatch_op_type;
        dispatch_if[0].data.op_args.tcu.fmt_s   = dispatch_fmt_s;
        dispatch_if[0].data.op_args.tcu.fmt_d   = dispatch_fmt_d;
        dispatch_if[0].data.op_args.tcu.step_m  = dispatch_step_m;
        dispatch_if[0].data.op_args.tcu.step_n  = dispatch_step_n;
        dispatch_if[0].data.op_args.tcu.step_k  = dispatch_step_k;
        dispatch_if[0].data.rs1_data            = dispatch_rs1_data;
        dispatch_if[0].data.rs2_data            = dispatch_rs2_data;
        dispatch_if[0].data.rs3_data            = dispatch_rs3_data;
        dispatch_if[0].data.sop                 = dispatch_sop;
        dispatch_if[0].data.eop                 = dispatch_eop;
    end

    // ---- Commit ----------------------------------------------------------
    assign commit_valid              = commit_if[0].valid;
    assign commit_if[0].ready        = commit_ready;
    assign commit_uuid               = commit_if[0].data.uuid;
    assign commit_wid                = commit_if[0].data.wid;
    assign commit_sid                = commit_if[0].data.sid;
    assign commit_tmask              = commit_if[0].data.tmask;
    assign commit_PC                 = commit_if[0].data.PC;
    assign commit_wb                 = commit_if[0].data.wb;
    assign commit_wr_xregs           = commit_if[0].data.wr_xregs;
    assign commit_rd                 = commit_if[0].data.rd;
    assign commit_bytesel            = commit_if[0].data.bytesel;
    assign commit_data               = commit_if[0].data.data;
    assign commit_sop                = commit_if[0].data.sop;
    assign commit_eop                = commit_if[0].data.eop;

    // ---- DUT -------------------------------------------------------------
    VX_tcu_unit #(
        .INSTANCE_ID ("tcu_unit_top")
    ) tcu_unit (
        `SCOPE_IO_BIND (0)
        .clk         (clk),
        .reset       (reset),
    `ifdef TCU_WGMMA_ENABLE
        .tcu_lmem_if (tcu_lmem_if),
    `endif
        .dispatch_if (dispatch_if),
        .commit_if   (commit_if)
    );

endmodule
