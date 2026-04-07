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

module VX_dispatcher import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter ISSUE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [NUM_EX_UNITS-1:0][PERF_CTR_BITS-1:0] perf_stalls,
    output wire [NUM_EX_UNITS-1:0][PERF_CTR_BITS-1:0] perf_instrs,
`endif
    // inputs
    VX_operands_if.slave    operands_if,

    // outputs
    output wire [NUM_EX_UNITS-1:0] dispatch_ready,
    VX_dispatch_if.master   dispatch_if [NUM_EX_UNITS]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (ISSUE_ID)

    localparam OUT_DATAW = $bits(dispatch_t);

    wire [NUM_EX_UNITS-1:0] operands_ready_in;
    assign operands_if.ready = operands_ready_in[operands_if.data.ex_type];

    // FU-availability feedback to scoreboard to avoid head-of-line blocking
    assign dispatch_ready = operands_ready_in;

    // Non-LSU execution units: pass operand data straight through
    for (genvar i = 0; i < NUM_EX_UNITS; ++i) begin : g_buffers
        if (i != EX_LSU) begin : g_non_lsu
            VX_elastic_buffer #(
                .DATAW   (OUT_DATAW),
                .SIZE    (2),
                .OUT_REG (1)
            ) buffer (
                .clk        (clk),
                .reset      (reset),
                .valid_in   (operands_if.valid && (operands_if.data.ex_type == EX_BITS'(i))),
                .ready_in   (operands_ready_in[i]),
                .data_in    ({
                    operands_if.data.uuid,
                    operands_if.data.wis,
                    operands_if.data.sid,
                    operands_if.data.tmask,
                    operands_if.data.PC,
                    operands_if.data.wb,
                    operands_if.data.wr_xregs,
                    operands_if.data.rd,
                    operands_if.data.bytesel,
                    operands_if.data.op_type,
                    operands_if.data.op_args,
                    operands_if.data.rs1_data,
                    operands_if.data.rs2_data,
                    operands_if.data.rs3_data,
                    operands_if.data.sop,
                    operands_if.data.eop
                }),
                .data_out   (dispatch_if[i].data),
                .valid_out  (dispatch_if[i].valid),
                .ready_out  (dispatch_if[i].ready)
            );
        end
    end

    logic [`SIMD_WIDTH-1:0][`XLEN-1:0] eff_rs1_data;
    op_args_t eff_op_args;

    // Pack-load: compute eff_rs1[lane] = rs1[lane] + rs2[lane] * uop_idx
    // uop_idx is in op_args.lsu.offset[1:0]; stride lives in rs2_data.
    // Multiply via shift-and-add on the 2-bit index — no multiplier needed.
    wire is_pack_lsu = (operands_if.data.op_args.lsu.pack != 2'b00);
    wire [1:0] pld_uop_idx = operands_if.data.op_args.lsu.offset[1:0];

    for (genvar j = 0; j < `SIMD_WIDTH; ++j) begin : g_eff_rs1
        wire [`XLEN-1:0] stride_off =
            ({`XLEN{pld_uop_idx[0]}} & (operands_if.data.rs2_data[j] << 0))
          + ({`XLEN{pld_uop_idx[1]}} & (operands_if.data.rs2_data[j] << 1));
        assign eff_rs1_data[j] = is_pack_lsu
            ? (operands_if.data.rs1_data[j] + stride_off)
            :  operands_if.data.rs1_data[j];
    end

    always_comb begin
        eff_op_args = operands_if.data.op_args;
        if (is_pack_lsu) begin
            eff_op_args.lsu.offset = '0;
        end
    end

    // LSU: substitute effective base address and cleared offset for bulk ops
    VX_elastic_buffer #(
        .DATAW   (OUT_DATAW),
        .SIZE    (2),
        .OUT_REG (1)
    ) lsu_buffer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (operands_if.valid && (operands_if.data.ex_type == EX_BITS'(EX_LSU))),
        .ready_in   (operands_ready_in[EX_LSU]),
        .data_in    ({
            operands_if.data.uuid,
            operands_if.data.wis,
            operands_if.data.sid,
            operands_if.data.tmask,
            operands_if.data.PC,
            operands_if.data.wb,
            operands_if.data.wr_xregs,
            operands_if.data.rd,
            operands_if.data.bytesel,
            operands_if.data.op_type,
            eff_op_args,
            eff_rs1_data,
            operands_if.data.rs2_data,
            operands_if.data.rs3_data,
            operands_if.data.sop,
            operands_if.data.eop
        }),
        .data_out   (dispatch_if[EX_LSU].data),
        .valid_out  (dispatch_if[EX_LSU].valid),
        .ready_out  (dispatch_if[EX_LSU].ready)
    );

`ifdef PERF_ENABLE
    reg [NUM_EX_UNITS-1:0][PERF_CTR_BITS-1:0] perf_stalls_r;
    reg [NUM_EX_UNITS-1:0][PERF_CTR_BITS-1:0] perf_instrs_r;

    wire operands_if_fire  = operands_if.valid && operands_if.ready;
    wire operands_if_stall = operands_if.valid && ~operands_if.ready;

    for (genvar i = 0; i < NUM_EX_UNITS; ++i) begin : g_perf_stalls
        always @(posedge clk) begin
            if (reset) begin
                perf_stalls_r[i] <= '0;
                perf_instrs_r[i] <= '0;
            end else begin
                perf_stalls_r[i] <= perf_stalls_r[i] + PERF_CTR_BITS'(operands_if_stall && operands_if.data.ex_type == EX_BITS'(i));
                perf_instrs_r[i] <= perf_instrs_r[i] + PERF_CTR_BITS'(operands_if_fire && operands_if.data.ex_type == EX_BITS'(i) && operands_if.data.eop);
            end
        end
        assign perf_stalls[i] = perf_stalls_r[i];
        assign perf_instrs[i] = perf_instrs_r[i];
    end
`endif

`ifdef DBG_TRACE_PIPELINE
    for (genvar ex = 0; ex < NUM_EX_UNITS; ++ex) begin : g_dispatch_trace
        always @(posedge clk) begin
            if (dispatch_if[ex].valid && dispatch_if[ex].ready) begin
                `TRACE(1, ("%t: %s dispatch: wid=%0d, sid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, wis_to_wid(dispatch_if[ex].data.wis, ISSUE_ID), dispatch_if[ex].data.sid, to_fullPC(dispatch_if[ex].data.PC)))
                VX_trace_pkg::trace_ex_type(1, ex);
                `TRACE(1, (", op="))
                VX_trace_pkg::trace_ex_op(1, ex, dispatch_if[ex].data.op_type, dispatch_if[ex].data.op_args);
                `TRACE(1, (", tmask=%b, wb=%b, wr_xregs=%b, rd=%0d, rs1_data=", dispatch_if[ex].data.tmask, dispatch_if[ex].data.wb, dispatch_if[ex].data.wr_xregs, dispatch_if[ex].data.rd))
                `TRACE_ARRAY1D(1, "0x%0h", dispatch_if[ex].data.rs1_data, `SIMD_WIDTH)
                `TRACE(1, (", rs2_data="))
                `TRACE_ARRAY1D(1, "0x%0h", dispatch_if[ex].data.rs2_data, `SIMD_WIDTH)
                `TRACE(1, (", rs3_data="))
                `TRACE_ARRAY1D(1, "0x%0h", dispatch_if[ex].data.rs3_data, `SIMD_WIDTH)
            VX_trace_pkg::trace_op_args(1, ex, dispatch_if[ex].data.op_type, dispatch_if[ex].data.op_args);
                `TRACE(1, (", sop=%b, eop=%b (#%0d)\n", dispatch_if[ex].data.sop, dispatch_if[ex].data.eop, dispatch_if[ex].data.uuid))
            end
        end
    end
`endif

endmodule
