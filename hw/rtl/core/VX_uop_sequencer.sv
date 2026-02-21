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

module VX_uop_sequencer import
`ifdef EXT_TCU_ENABLE
    VX_tcu_pkg::*,
`endif
    VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WARP_ID  = 0
) (
    input clk,
    input reset,

    VX_ibuffer_if.slave  input_if,
    VX_ibuffer_if.master output_if
);
    `UNUSED_PARAM (WARP_ID)
    `UNUSED_SPARAM (INSTANCE_ID)

    ibuffer_t base_uop_data;
    ibuffer_t dxa_uop_data;
    ibuffer_t uop_data;

    wire is_base_uop_input;
    wire is_dxa_uop_input;
    wire is_uop_input;
    wire uop_start = input_if.valid && is_uop_input;
    wire uop_next = output_if.ready;
    wire uop_done;
    wire base_uop_done;
    wire dxa_uop_done;

`ifdef EXT_TCU_ENABLE

    assign is_base_uop_input = (input_if.data.ex_type == EX_TCU && input_if.data.op_type == INST_TCU_WMMA);

    VX_tcu_uops tcu_uops (
        .clk     (clk),
        .reset   (reset),
        .ibuf_in (input_if.data),
        .ibuf_out(base_uop_data),
        .start   (uop_start && is_base_uop_input),
        .next    (uop_next),
        .done    (base_uop_done)
    );

`else

    assign is_base_uop_input = 0;
    assign base_uop_done = 0;
    assign base_uop_data = '0;

`endif

`ifdef EXT_DXA_ENABLE

    localparam DXA_OP_LAUNCH = 3'd5;
    assign is_dxa_uop_input = (input_if.data.ex_type == EX_SFU)
                            && (input_if.data.op_type == INST_SFU_DXA)
                            && (input_if.data.op_args.dxa.op == DXA_OP_LAUNCH);

    VX_dxa_uops dxa_uops (
        .clk     (clk),
        .reset   (reset),
        .ibuf_in (input_if.data),
        .ibuf_out(dxa_uop_data),
        .start   (uop_start && is_dxa_uop_input),
        .next    (uop_next),
        .done    (dxa_uop_done)
    );

`else

    assign is_dxa_uop_input = 0;
    assign dxa_uop_done = 0;
    assign dxa_uop_data = '0;

`endif

    assign is_uop_input = is_base_uop_input || is_dxa_uop_input;
    assign uop_data = is_dxa_uop_input ? dxa_uop_data : base_uop_data;
    assign uop_done = is_dxa_uop_input ? dxa_uop_done : base_uop_done;

    reg uop_active;

    always_ff @(posedge clk) begin
        if (reset) begin
            uop_active <= 0;
        end else begin
            if (uop_active) begin
                if (uop_next && uop_done) begin
                    uop_active <= 0;
                end
            end
            else if (uop_start) begin
                uop_active <= 1;
            end
        end
    end

    // output assignments
    wire uop_hold = ~uop_active && is_uop_input; // hold transition cycles to uop_active
    assign output_if.valid = uop_active ? 1'b1 : (input_if.valid && ~uop_hold);
    assign output_if.data  = uop_active ? uop_data : input_if.data;
    assign input_if.ready  = uop_active ? (output_if.ready && uop_done) : (output_if.ready && ~uop_hold);

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (output_if.valid && output_if.ready && uop_active) begin
            `TRACE(1, ("%t: %s decode: wid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, WARP_ID, to_fullPC(output_if.data.PC)))
            VX_trace_pkg::trace_ex_type(1, output_if.data.ex_type);
            `TRACE(1, (", op="))
            VX_trace_pkg::trace_ex_op(1, output_if.data.ex_type, output_if.data.op_type, output_if.data.op_args);
            `TRACE(1, (", tmask=%b, wb=%b, rd_xregs=%b, wr_xregs=%b, used_rs=%b, rd=", output_if.data.tmask, output_if.data.wb, output_if.data.rd_xregs, output_if.data.wr_xregs, output_if.data.used_rs))
            VX_trace_pkg::trace_reg_idx(1, output_if.data.rd);
            `TRACE(1, (", rs1="))
            VX_trace_pkg::trace_reg_idx(1, output_if.data.rs1);
            `TRACE(1, (", rs2="))
            VX_trace_pkg::trace_reg_idx(1, output_if.data.rs2);
            `TRACE(1, (", rs3="))
            VX_trace_pkg::trace_reg_idx(1, output_if.data.rs3);
            VX_trace_pkg::trace_op_args(1, output_if.data.ex_type, output_if.data.op_type, output_if.data.op_args);
            `TRACE(1, (", parent=#%0d", input_if.data.uuid))
            `TRACE(1, (" (#%0d)\n", output_if.data.uuid))
        end
    end
`endif

endmodule
