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

// reset all GPRs in debug mode
`ifdef SIMULATION
`ifndef NDEBUG
`define GPR_RESET
`endif
`endif

module VX_opc_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    output wire [SIMD_IDX_W-1:0] sid,
    output wire [ISSUE_WIS_W-1:0] wis,
    output reg [NUM_REGS-1:0] pending_regs,

    VX_scoreboard_if.slave  scoreboard_if,
    VX_opc_if.master        opc_if,
    VX_operands_if.master   operands_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NUM_OPDS = NUM_SRC_OPDS + 1;
    localparam SCB_DATAW = UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + NUM_OPDS + (NUM_OPDS * REG_IDX_BITS);
    localparam OUT_DATAW = UUID_WIDTH + ISSUE_WIS_W + SIMD_IDX_W + `SIMD_WIDTH + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + 1 + NR_BITS + (NUM_SRC_OPDS * `SIMD_WIDTH * `XLEN);

    localparam STATE_IDLE  = 0;
    localparam STATE_FETCH = 1;
    localparam STATE_DISPATCH = 2;

    VX_scoreboard_if staging_if();

    reg [NUM_SRC_OPDS-1:0] opds_needed, opds_needed_n;
    reg [NUM_SRC_OPDS-1:0] opds_busy, opds_busy_n;
    reg [2:0] state, state_n;
    reg [SIMD_IDX_W-1:0] simd_index, simd_index_n;

    wire scboard_fire = scoreboard_if.valid && scoreboard_if.ready;
    wire col_req_fire = opc_if.req_valid && opc_if.req_ready;
    wire col_rsp_fire = opc_if.rsp_valid;

    VX_pipe_buffer #(
        .DATAW (SCB_DATAW)
    ) stanging_buf (
        .clk      (clk),
        .reset    (reset),
        .valid_in (scoreboard_if.valid),
        .data_in  (scoreboard_if.data),
        .ready_in (scoreboard_if.ready),
        .valid_out(staging_if.valid),
        .data_out (staging_if.data),
        .ready_out(staging_if.ready)
    );

    wire output_ready;
    wire dispatched = (state == STATE_DISPATCH) && output_ready;
    wire is_last_simd = (simd_index == SIMD_IDX_W'(SIMD_COUNT-1));

    assign staging_if.ready = dispatched && is_last_simd;

    wire [NR_BITS-1:0] rs1 = to_reg_number(staging_if.data.rs1);
    wire [NR_BITS-1:0] rs2 = to_reg_number(staging_if.data.rs2);
    wire [NR_BITS-1:0] rs3 = to_reg_number(staging_if.data.rs3);
    wire [NUM_SRC_OPDS-1:0][NR_BITS-1:0] src_regs = {rs3, rs2, rs1};

    always @(*) begin
        state_n = state;
        opds_needed_n = opds_needed;
        opds_busy_n   = opds_busy;
        simd_index_n  = simd_index;
        case (state)
        STATE_IDLE: begin
            if (scboard_fire) begin
                opds_needed_n = scoreboard_if.data.used_rs;
                opds_busy_n = scoreboard_if.data.used_rs;
                if (opds_busy_n == 0) begin
                    state_n = STATE_DISPATCH;
                end else begin
                    state_n = STATE_FETCH;
                end
            end
        end
        STATE_FETCH: begin
            if (col_req_fire) begin
                opds_needed_n[opc_if.req_data.opd_id] = 0;
            end
            if (col_rsp_fire) begin
                opds_busy_n[opc_if.rsp_data.opd_id] = 0;
            end
            if (opds_busy_n == 0) begin
                state_n = STATE_DISPATCH;
            end
        end
        STATE_DISPATCH: begin
            if (output_ready) begin
                if (is_last_simd) begin
                    state_n = STATE_IDLE;
                end else begin
                    opds_needed_n = staging_if.data.used_rs;
                    opds_busy_n = staging_if.data.used_rs;
                    simd_index_n = simd_index + 1;
                    state_n = STATE_FETCH;
                end
            end
        end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
            opds_needed <= '0;
            opds_busy <= '0;
            simd_index <= 0;
        end else begin
            state <= state_n;
            opds_needed <= opds_needed_n;
            opds_busy <= opds_busy_n;
            simd_index <= simd_index_n;
        end
    end

    wire [SRC_OPD_WIDTH-1:0] opd_id;
    wire opd_fetch_valid;

    VX_priority_encoder #(
        .N (NUM_SRC_OPDS)
    ) opd_id_sel (
        .data_in (opds_needed),
        .index_out (opd_id),
        `UNUSED_PIN (onehot_out),
        .valid_out (opd_fetch_valid)
    );

    // operands fetch request
    assign opc_if.req_valid = opd_fetch_valid;
    assign opc_if.req_data.opd_id = opd_id;
    assign opc_if.req_data.sid = simd_index;
    assign opc_if.req_data.wis = staging_if.data.wis;
    assign opc_if.req_data.reg_id = src_regs[opd_id];

    // operands fetch response
    reg [NUM_SRC_OPDS-1:0][`SIMD_WIDTH-1:0][`XLEN-1:0] opd_values;
    always @(posedge clk) begin
        if (reset || dispatched) begin
            for (integer i = 0; i < NUM_SRC_OPDS; ++i) begin
                opd_values[i] <= '0;
            end
        end else begin
            if (col_rsp_fire) begin
                opd_values[opc_if.rsp_data.opd_id] <= opc_if.rsp_data.value;
            end
        end
    end

    // output scheduler info
    assign sid = simd_index;
    assign wis = staging_if.data.wis;
    always @(*) begin
        pending_regs = '0;
        for (integer i = 0; i < NUM_SRC_OPDS; ++i) begin
            if (opds_busy[i]) begin
                pending_regs[src_regs[i]] = 1;
            end
        end
    end

    // instruction dispatch
    VX_elastic_buffer #(
        .DATAW   (OUT_DATAW),
        .SIZE    (0),
        .OUT_REG (0)
    ) out_buf (
        .clk      (clk),
        .reset    (reset),
        .valid_in (state == STATE_DISPATCH),
        .data_in  ({
            staging_if.data.uuid,
            staging_if.data.wis,
            simd_index,
            staging_if.data.tmask[simd_index * `SIMD_WIDTH +: `SIMD_WIDTH],
            staging_if.data.PC,
            staging_if.data.ex_type,
            staging_if.data.op_type,
            staging_if.data.op_args,
            staging_if.data.wb,
            to_reg_number(staging_if.data.rd),
            opd_values[0],
            opd_values[1],
            opd_values[2]
        }),
        .ready_in (output_ready),
        .valid_out(operands_if.valid),
        .data_out (operands_if.data),
        .ready_out(operands_if.ready)
    );

endmodule
