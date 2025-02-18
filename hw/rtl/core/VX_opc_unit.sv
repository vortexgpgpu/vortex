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
    parameter `STRING INSTANCE_ID = "",
    parameter OUT_BUF = 3
) (
    input wire              clk,
    input wire              reset,

    output wire [ISSUE_WIS_W-1:0] wis,
    output wire [NUM_REGS-1:0] pending_regs_n,

    VX_scoreboard_if.slave  scoreboard_if,
    VX_opc_if.master        opc_if,
    VX_operands_if.master   operands_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NUM_OPDS = NUM_SRC_OPDS + 1;
    localparam SCB_DATAW = UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + NUM_OPDS + (REG_IDX_BITS * NUM_OPDS);

    localparam STATE_IDLE  = 0;
    localparam STATE_FETCH = 1;
    localparam STATE_DONE  = 2;

    VX_scoreboard_if staging_if();

    reg [NUM_SRC_OPDS-1:0] opds_needed, opds_needed_n;
    reg [NUM_SRC_OPDS-1:0] opds_busy, opds_busy_n;
    reg [2:0] state, state_n;

    wire scboard_fire  = scoreboard_if.valid && scoreboard_if.ready;
    wire col_req_fire  = opc_if.req_valid && opc_if.req_ready;
    wire col_rsp_fire  = opc_if.rsp_valid;
    wire operands_fire = operands_if.valid && operands_if.ready;

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

    wire [NR_BITS-1:0] rs1 = {staging_if.data.rs1.rtype[0], staging_if.data.rs1.id};
    wire [NR_BITS-1:0] rs2 = {staging_if.data.rs2.rtype[0], staging_if.data.rs2.id};
    wire [NR_BITS-1:0] rs3 = {staging_if.data.rs3.rtype[0], staging_if.data.rs3.id};
    wire [NUM_SRC_OPDS-1:0][NR_BITS-1:0] src_opds = {rs3, rs2, rs1};

    always @(*) begin
        state_n = state;
        opds_needed_n = opds_needed;
        opds_busy_n = opds_busy;
        case (state)
        STATE_IDLE: begin
            if (scboard_fire) begin
                opds_needed_n = scoreboard_if.data.used_rs;
                opds_busy_n = opds_needed_n;
                if (opds_busy_n == 0) begin
                    state_n = STATE_DONE;
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
                state_n = STATE_DONE;
            end
        end
        STATE_DONE: begin
            if (operands_fire) begin
                state_n = STATE_IDLE;
            end
        end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
            opds_needed <= '0;
            opds_busy <= '0;
        end else begin
            state <= state_n;
            opds_needed <= opds_needed_n;
            opds_busy <= opds_busy_n;
        end
    end

endmodule
