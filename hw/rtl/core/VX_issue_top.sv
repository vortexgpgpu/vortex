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

module VX_issue_top import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "issue"
) (
    // Clock
    input wire                              clk,
    input wire                              reset,

    input wire                              decode_valid,
    input wire [UUID_WIDTH-1:0]             decode_uuid,
    input wire [NW_WIDTH-1:0]               decode_wid,
    input wire [`NUM_THREADS-1:0]           decode_tmask,
    input wire [PC_BITS-1:0]                decode_PC,
    input wire [EX_BITS-1:0]                decode_ex_type,
    input wire [INST_OP_BITS-1:0]           decode_op_type,
    input op_args_t                         decode_op_args,
    input wire                              decode_wb,
    input wire [NUM_SRC_OPDS-1:0]           decode_used_rs,
    input wire [NUM_REGS_BITS-1:0]          decode_rd,
    input wire [NUM_REGS_BITS-1:0]          decode_rs1,
    input wire [NUM_REGS_BITS-1:0]          decode_rs2,
    input wire [NUM_REGS_BITS-1:0]          decode_rs3,
    output wire                             decode_ready,

    input wire                              writeback_valid[`ISSUE_WIDTH],
    input wire [UUID_WIDTH-1:0]             writeback_uuid[`ISSUE_WIDTH],
    input wire [ISSUE_WIS_W-1:0]            writeback_wis[`ISSUE_WIDTH],
    input wire [SIMD_IDX_W-1:0]             writeback_sid[`ISSUE_WIDTH],
    input wire [`SIMD_WIDTH-1:0]            writeback_tmask[`ISSUE_WIDTH],
    input wire [PC_BITS-1:0]                writeback_PC[`ISSUE_WIDTH],
    input wire [NUM_REGS_BITS-1:0]          writeback_rd[`ISSUE_WIDTH],
    input wire [`SIMD_WIDTH-1:0][`XLEN-1:0] writeback_data[`ISSUE_WIDTH],
    input wire                              writeback_sop[`ISSUE_WIDTH],
    input wire                              writeback_eop[`ISSUE_WIDTH],

    output wire                             dispatch_valid[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [UUID_WIDTH-1:0]            dispatch_uuid[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [ISSUE_WIS_W-1:0]           dispatch_wis[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [SIMD_IDX_W-1:0]            dispatch_sid[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [`SIMD_WIDTH-1:0]           dispatch_tmask[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [PC_BITS-1:0]               dispatch_PC[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [INST_ALU_BITS-1:0]         dispatch_op_type[NUM_EX_UNITS * `ISSUE_WIDTH],
    output op_args_t                        dispatch_op_args[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire                             dispatch_wb[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [NUM_REGS_BITS-1:0]         dispatch_rd[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [`SIMD_WIDTH-1:0][`XLEN-1:0] dispatch_rs1_data[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [`SIMD_WIDTH-1:0][`XLEN-1:0] dispatch_rs2_data[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire [`SIMD_WIDTH-1:0][`XLEN-1:0] dispatch_rs3_data[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire                             dispatch_sop[NUM_EX_UNITS * `ISSUE_WIDTH],
    output wire                             dispatch_eop[NUM_EX_UNITS * `ISSUE_WIDTH],
    input wire                              dispatch_ready[NUM_EX_UNITS * `ISSUE_WIDTH],

    output wire [ISSUE_WIS_W-1:0]           issue_sched_wis[`ISSUE_WIDTH],
    output wire                             issue_sched_valid[`ISSUE_WIDTH]
);
    VX_decode_if    decode_if();
    VX_dispatch_if  dispatch_if[NUM_EX_UNITS * `ISSUE_WIDTH]();
    VX_writeback_if writeback_if[`ISSUE_WIDTH]();
    VX_issue_sched_if issue_sched_if[`ISSUE_WIDTH]();

    assign decode_if.valid = decode_valid;
    assign decode_if.data.uuid = decode_uuid;
    assign decode_if.data.wid = decode_wid;
    assign decode_if.data.tmask = decode_tmask;
    assign decode_if.data.PC = decode_PC;
    assign decode_if.data.ex_type = decode_ex_type;
    assign decode_if.data.op_type = decode_op_type;
    assign decode_if.data.op_args = decode_op_args;
    assign decode_if.data.wb = decode_wb;
    assign decode_if.data.used_rs = decode_used_rs;
    assign decode_if.data.rd = decode_rd;
    assign decode_if.data.rs1 = decode_rs1;
    assign decode_if.data.rs2 = decode_rs2;
    assign decode_if.data.rs3 = decode_rs3;
    assign decode_ready = decode_if.ready;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_writeback_if
        assign writeback_if[i].valid = writeback_valid[i];
        assign writeback_if[i].data.uuid = writeback_uuid[i];
        assign writeback_if[i].data.wis = writeback_wis[i];
        assign writeback_if[i].data.sid = writeback_sid[i];
        assign writeback_if[i].data.tmask = writeback_tmask[i];
        assign writeback_if[i].data.PC = writeback_PC[i];
        assign writeback_if[i].data.rd = writeback_rd[i];
        assign writeback_if[i].data.data = writeback_data[i];
        assign writeback_if[i].data.sop = writeback_sop[i];
        assign writeback_if[i].data.eop = writeback_eop[i];
    end

    for (genvar i = 0; i < NUM_EX_UNITS * `ISSUE_WIDTH; ++i) begin : g_dispatch_if
        assign dispatch_valid[i] = dispatch_if[i].valid;
        assign dispatch_uuid[i] = dispatch_if[i].data.uuid;
        assign dispatch_wis[i] = dispatch_if[i].data.wis;
        assign dispatch_sid[i] = dispatch_if[i].data.sid;
        assign dispatch_tmask[i] = dispatch_if[i].data.tmask;
        assign dispatch_PC[i] = dispatch_if[i].data.PC;
        assign dispatch_op_type[i] = dispatch_if[i].data.op_type;
        assign dispatch_op_args[i] = dispatch_if[i].data.op_args;
        assign dispatch_wb[i] = dispatch_if[i].data.wb;
        assign dispatch_rd[i] = dispatch_if[i].data.rd;
        assign dispatch_rs1_data[i] = dispatch_if[i].data.rs1_data;
        assign dispatch_rs2_data[i] = dispatch_if[i].data.rs2_data;
        assign dispatch_rs3_data[i] = dispatch_if[i].data.rs3_data;
        assign dispatch_sop[i] = dispatch_if[i].data.sop;
        assign dispatch_eop[i] = dispatch_if[i].data.eop;
        assign dispatch_if[i].ready = dispatch_ready[i];
    end

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_issue_sched_if
        assign issue_sched_wis[i] = issue_sched_if[i].wis;
        assign issue_sched_valid[i] = issue_sched_if[i].valid;
    end

`ifdef PERF_ENABLE
    issue_perf_t issue_perf = '0;
`endif

`ifdef SCOPE
    wire [0:0] scope_reset_w = 1'b0;
    wire [0:0] scope_bus_in_w = 1'b0;
    wire [0:0] scope_bus_out_w;
    `UNUSED_VAR (scope_bus_out_w)
`endif

    VX_issue #(
        .INSTANCE_ID (INSTANCE_ID)
    ) issue (
        `SCOPE_IO_BIND (0)
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .issue_perf     (issue_perf),
    `endif

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),
        .dispatch_if    (dispatch_if),
        .issue_sched_if (issue_sched_if)
    );

endmodule
