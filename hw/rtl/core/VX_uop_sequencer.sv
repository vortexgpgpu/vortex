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
    parameter WARP_ID = 0
) (
    input clk,
    input reset,

    VX_ibuffer_if.slave  input_if,
    VX_ibuffer_if.master output_if
);
    `UNUSED_PARAM (WARP_ID)
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam UOP_SEL_W = `LOG2UP(UOP_MAX);

    // UOP-expanders signals.
    wire [UOP_MAX-1:0]   uop_valid_in;
    ibuffer_t            uop_data_out [UOP_MAX];
    wire [UOP_CTR_W-1:0] uop_count_out [UOP_MAX];

    reg [UOP_CTR_W-1:0] uop_ctr;    // current uop index within active burst
    reg [UOP_SEL_W-1:0] sel_idx;    // registered active expander index
    reg [UOP_CTR_W-1:0] last_ctr;   // count[active] - 1, latched at start
    reg                 uop_active; // high while emitting a uop burst

    logic [UOP_SEL_W-1:0] sel_idx_n;
    wire is_uop_input;
    VX_priority_encoder #(
        .N (UOP_MAX),
        .REVERSE (1)
    ) priority_enc (
        .data_in    (uop_valid_in),
        `UNUSED_PIN (onehot_out),
        .index_out  (sel_idx_n),
        .valid_out  (is_uop_input)
    );

    // uop_start fires for exactly one cycle at the beginning of a new burst.
    // The ~uop_active guard prevents re-triggering while a burst is in flight.
    wire uop_start = input_if.valid && is_uop_input && ~uop_active;

    // uop_next: downstream accepted a uop this cycle.
    wire uop_next = output_if.ready;

    // uop_done: last uop of the burst is being presented this cycle.
    wire uop_done = (uop_ctr == last_ctr);

    // Sequential state machine: track the active burst and uop index.
    always_ff @(posedge clk) begin
        if (reset) begin
            uop_ctr    <= '0;
            sel_idx    <= '0;
            last_ctr   <= '0;
            uop_active <= 1'b0;
        end else begin
            if (uop_start) begin
                uop_ctr    <= '0;
                sel_idx    <= sel_idx_n;
                last_ctr   <= uop_count_out[sel_idx_n] - UOP_CTR_W'(1);
                uop_active <= 1'b1;
            end else if (uop_active && uop_next) begin
                // Advance or retire the burst.
                uop_ctr    <= uop_done ? '0 : (uop_ctr + UOP_CTR_W'(1));
                uop_active <= ~uop_done;
            end
        end
    end

    wire [UOP_MAX-1:0] uop_start_in;
    for (genvar i = 0; i < UOP_MAX; ++i) begin : g_start
        assign uop_start_in[i] = uop_start && uop_valid_in[i];
    end

`ifdef EXT_TCU_ENABLE
    // ------------------------------------------------------------------
    // TCU uop expander
    // ------------------------------------------------------------------
    assign uop_valid_in[UOP_TCU] = (input_if.data.ex_type == EX_TCU)
        && (input_if.data.op_type == INST_TCU_WMMA
    `ifdef TCU_SPARSE_ENABLE
        || input_if.data.op_type == INST_TCU_WMMA_SP
        || input_if.data.op_type == INST_TCU_META_STORE
    `endif
        );

    VX_tcu_uops tcu_uops (
        .clk       (clk),
        .reset     (reset),
        .ibuf_in   (input_if.data),
        .ibuf_out  (uop_data_out[UOP_TCU]),
        .start     (uop_start_in[UOP_TCU]),
        .uop_idx   (uop_ctr),
        .uop_count (uop_count_out[UOP_TCU])
    );
`endif

`ifdef EXT_DXA_ENABLE
    // ------------------------------------------------------------------
    // DXA uop expander
    // ------------------------------------------------------------------
    assign uop_valid_in[UOP_DXA] = (input_if.data.ex_type == EX_SFU)
                                && (input_if.data.op_type == INST_SFU_DXA);
    VX_dxa_uops dxa_uops (
        .clk       (clk),
        .reset     (reset),
        .ibuf_in   (input_if.data),
        .ibuf_out  (uop_data_out[UOP_DXA]),
        .start     (uop_start_in[UOP_DXA]),
        .uop_idx   (uop_ctr),
        .uop_count (uop_count_out[UOP_DXA])
    );
`endif

    wire uop_hold = is_uop_input && ~uop_active;

    assign output_if.valid = uop_active ? 1'b1 : (input_if.valid && ~uop_hold);
    assign output_if.data  = uop_active ? uop_data_out[sel_idx] : input_if.data;
    assign input_if.ready  = uop_active ? (uop_next && uop_done) : (uop_next && ~uop_hold);

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
