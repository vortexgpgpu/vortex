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

module VX_operands import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 4,
    parameter OUT_REG   = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [`PERF_CTR_BITS-1:0] perf_stalls,
`endif

    VX_writeback_if.slave   writeback_if,
    VX_scoreboard_if.slave  scoreboard_if,
    VX_operands_if.master   operands_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam NUM_SRC_REGS = 3;
    localparam REQ_SEL_BITS = `CLOG2(NUM_SRC_REGS);
    localparam REQ_SEL_WIDTH = `UP(REQ_SEL_BITS);
    localparam BANK_SEL_BITS = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam PER_BANK_REGS = `NUM_REGS / NUM_BANKS;
    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `PC_BITS + 1 + `EX_BITS + `INST_OP_BITS + `INST_ARGS_BITS + `NR_BITS + 3 * `NUM_THREADS * `XLEN;
    localparam RAM_ADDRW = `LOG2UP(`NUM_REGS * PER_ISSUE_WARPS);
    localparam PER_BANK_ADDRW = RAM_ADDRW - BANK_SEL_BITS;
    localparam XLEN_SIZE = `XLEN / 8;
    localparam BYTEENW = `NUM_THREADS * XLEN_SIZE;

    `UNUSED_VAR (writeback_if.data.sop)

    wire [NUM_SRC_REGS-1:0] req_valid_in;
    wire [NUM_SRC_REGS-1:0] req_ready_in;
    wire [NUM_SRC_REGS-1:0][PER_BANK_ADDRW-1:0] req_data_in;
    wire [NUM_SRC_REGS-1:0][BANK_SEL_WIDTH-1:0] req_bank_idx;

    wire [NUM_BANKS-1:0] gpr_rd_valid;
    wire [NUM_BANKS-1:0][PER_BANK_ADDRW-1:0] gpr_rd_addr;
    wire [NUM_BANKS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] gpr_rd_data;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_rd_req_idx;

    reg [NUM_SRC_REGS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] src_data, src_data_n;
    wire [NUM_SRC_REGS-1:0] src_valid;
    reg [NUM_SRC_REGS-1:0] data_fetched;
    reg data_ready;

    assign src_valid[0] = (scoreboard_if.data.rs1 != 0) && ~data_fetched[0];
    assign src_valid[1] = (scoreboard_if.data.rs2 != 0) && ~data_fetched[1];
    assign src_valid[2] = (scoreboard_if.data.rs3 != 0) && ~data_fetched[2];

    assign req_valid_in[0] = scoreboard_if.valid && src_valid[0];
    assign req_valid_in[1] = scoreboard_if.valid && src_valid[1];
    assign req_valid_in[2] = scoreboard_if.valid && src_valid[2];

    if (ISSUE_WIS != 0) begin
        assign req_data_in[0] = {scoreboard_if.data.wis, scoreboard_if.data.rs1[`NR_BITS-1:BANK_SEL_BITS]};
        assign req_data_in[1] = {scoreboard_if.data.wis, scoreboard_if.data.rs2[`NR_BITS-1:BANK_SEL_BITS]};
        assign req_data_in[2] = {scoreboard_if.data.wis, scoreboard_if.data.rs3[`NR_BITS-1:BANK_SEL_BITS]};
    end else begin
        assign req_data_in[0] = {scoreboard_if.data.rs1[`NR_BITS-1:BANK_SEL_BITS]};
        assign req_data_in[1] = {scoreboard_if.data.rs2[`NR_BITS-1:BANK_SEL_BITS]};
        assign req_data_in[2] = {scoreboard_if.data.rs3[`NR_BITS-1:BANK_SEL_BITS]};
    end

    if (NUM_BANKS > 1) begin
        assign req_bank_idx[0] = scoreboard_if.data.rs1[BANK_SEL_BITS-1:0];
        assign req_bank_idx[1] = scoreboard_if.data.rs2[BANK_SEL_BITS-1:0];
        assign req_bank_idx[2] = scoreboard_if.data.rs3[BANK_SEL_BITS-1:0];
    end else begin
        assign req_bank_idx = '0;
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_SRC_REGS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (PER_BANK_ADDRW),
        .PERF_CTR_BITS(`PERF_CTR_BITS),
        .OUT_BUF     (1) // single-cycle EB since ready_out=1
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
    `ifdef PERF_ENABLE
        .collisions(perf_stalls),
    `else
        `UNUSED_PIN(collisions),
    `endif
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .sel_in    (req_bank_idx),
        .ready_in  (req_ready_in),
        .valid_out (gpr_rd_valid),
        .data_out  (gpr_rd_addr),
        .sel_out   (gpr_rd_req_idx),
        .ready_out ({NUM_BANKS{1'b1}})
    );

    always @(*) begin
        src_data_n = src_data;
        for (integer b = 0; b < NUM_BANKS; ++b) begin
            if (gpr_rd_valid[b]) begin
                src_data_n[gpr_rd_req_idx[b]] = gpr_rd_data[b];
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            data_fetched <= '0;
            src_data     <= '0;
            data_ready   <= '0;
        end else begin
            if (scoreboard_if.ready) begin
                data_fetched <= '0;
                src_data     <= '0;
                data_ready   <= '0;
            end else begin
                data_fetched <= data_fetched | req_ready_in;
                src_data     <= src_data_n;
                data_ready   <= scoreboard_if.valid
                             && (~src_valid[0] || req_ready_in[0])
                             && (~src_valid[1] || req_ready_in[1])
                             && (~src_valid[2] || req_ready_in[2]);
            end
        end
    end

    wire stg_valid_in, stg_ready_in;

    assign stg_valid_in = scoreboard_if.valid && data_ready;
    assign scoreboard_if.ready = stg_ready_in && data_ready;

    // We use a toggle buffer since the input signal also toggles
    VX_toggle_buffer #(
        .DATAW (DATAW),
        .PASSTHRU (~OUT_REG)
    ) rsp_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (stg_valid_in),
        .data_in   ({
            scoreboard_if.data.uuid,
            scoreboard_if.data.wis,
            scoreboard_if.data.tmask,
            scoreboard_if.data.PC,
            scoreboard_if.data.wb,
            scoreboard_if.data.ex_type,
            scoreboard_if.data.op_type,
            scoreboard_if.data.op_args,
            scoreboard_if.data.rd,
            src_data_n[0],
            src_data_n[1],
            src_data_n[2]
        }),
        .ready_in  (stg_ready_in),
        .valid_out (operands_if.valid),
        .data_out  ({
            operands_if.data.uuid,
            operands_if.data.wis,
            operands_if.data.tmask,
            operands_if.data.PC,
            operands_if.data.wb,
            operands_if.data.ex_type,
            operands_if.data.op_type,
            operands_if.data.op_args,
            operands_if.data.rd,
            operands_if.data.rs1_data,
            operands_if.data.rs2_data,
            operands_if.data.rs3_data
        }),
        .ready_out (operands_if.ready)
    );

    wire [RAM_ADDRW-1:0] gpr_wr_addr;
    if (ISSUE_WIS != 0) begin
        assign gpr_wr_addr = {writeback_if.data.wis, writeback_if.data.rd};
    end else begin
        assign gpr_wr_addr = writeback_if.data.rd;
    end

    `ifdef GPR_RESET
        reg wr_enabled = 0;
        always @(posedge clk) begin
            if (reset) begin
                wr_enabled <= 1;
            end
        end
    `else
        wire wr_enabled = 1;
    `endif

    for (genvar b = 0; b < NUM_BANKS; ++b) begin
        wire gpr_wr_enabled;
        if (BANK_SEL_BITS != 0) begin
            assign gpr_wr_enabled = wr_enabled && writeback_if.valid
                                 && (gpr_wr_addr[BANK_SEL_BITS-1:0] == BANK_SEL_BITS'(b));
        end else begin
            assign gpr_wr_enabled = wr_enabled && writeback_if.valid;
        end

        wire [BYTEENW-1:0] wren;
        for (genvar i = 0; i < `NUM_THREADS; ++i) begin
            assign wren[i*XLEN_SIZE+:XLEN_SIZE] = {XLEN_SIZE{writeback_if.data.tmask[i]}};
        end

        VX_dp_ram #(
            .DATAW (`XLEN * `NUM_THREADS),
            .SIZE  (PER_BANK_REGS * PER_ISSUE_WARPS),
            .WRENW (BYTEENW),
        `ifdef GPR_RESET
            .INIT_ENABLE (1),
            .INIT_VALUE (0),
        `endif
            .NO_RWCHECK (1)
        ) gpr_ram (
            .clk   (clk),
            .read  (1'b1),
            .wren  (wren),
            .write (gpr_wr_enabled),
            .waddr (gpr_wr_addr[BANK_SEL_BITS +: PER_BANK_ADDRW]),
            .wdata (writeback_if.data.data),
            .raddr (gpr_rd_addr[b]),
            .rdata (gpr_rd_data[b])
        );
    end

endmodule
