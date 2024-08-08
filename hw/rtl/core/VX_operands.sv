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

module VX_operands import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 4,
    parameter OUT_BUF   = 4 // using 2-cycle EB for area reduction
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
    localparam META_DATAW = ISSUE_WIS_W + `NUM_THREADS + `PC_BITS + 1 + `EX_BITS + `INST_OP_BITS + `INST_ARGS_BITS + `NR_BITS + `UUID_WIDTH;
    localparam REGS_DATAW = `XLEN * `NUM_THREADS;
    localparam DATAW = META_DATAW + NUM_SRC_REGS * REGS_DATAW;
    localparam RAM_ADDRW = `LOG2UP(`NUM_REGS * PER_ISSUE_WARPS);
    localparam PER_BANK_ADDRW = RAM_ADDRW - BANK_SEL_BITS;
    localparam XLEN_SIZE = `XLEN / 8;
    localparam BYTEENW = `NUM_THREADS * XLEN_SIZE;

    `UNUSED_VAR (writeback_if.data.sop)

    wire [NUM_SRC_REGS-1:0] src_valid;
    wire [NUM_SRC_REGS-1:0] req_in_valid, req_in_ready;
    wire [NUM_SRC_REGS-1:0][PER_BANK_ADDRW-1:0] req_in_data;
    wire [NUM_SRC_REGS-1:0][BANK_SEL_WIDTH-1:0] req_bank_idx;

    wire [NUM_BANKS-1:0] gpr_rd_valid, gpr_rd_ready;
    wire [NUM_BANKS-1:0] gpr_rd_valid_st1, gpr_rd_valid_st2;
    wire [NUM_BANKS-1:0][PER_BANK_ADDRW-1:0] gpr_rd_addr, gpr_rd_addr_st1;
    wire [NUM_BANKS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] gpr_rd_data_st1, gpr_rd_data_st2;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_rd_req_idx, gpr_rd_req_idx_st1, gpr_rd_req_idx_st2;

    wire pipe_valid_st1, pipe_ready_st1;
    wire pipe_valid_st2, pipe_ready_st2;
    wire [META_DATAW-1:0] pipe_data, pipe_data_st1, pipe_data_st2;

    reg [NUM_SRC_REGS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] src_data_n;
    wire [NUM_SRC_REGS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] src_data_st1, src_data_st2;

    reg [NUM_SRC_REGS-1:0] data_fetched_n;
    wire [NUM_SRC_REGS-1:0] data_fetched_st1;

    reg has_collision_n;
    wire has_collision_st1;

    wire [NUM_SRC_REGS-1:0][`NR_BITS-1:0] src_regs = {scoreboard_if.data.rs3,
                                                      scoreboard_if.data.rs2,
                                                      scoreboard_if.data.rs1};

    for (genvar i = 0; i < NUM_SRC_REGS; ++i) begin
        if (ISSUE_WIS != 0) begin
            assign req_in_data[i] = {src_regs[i][`NR_BITS-1:BANK_SEL_BITS], scoreboard_if.data.wis};
        end else begin
            assign req_in_data[i] = src_regs[i][`NR_BITS-1:BANK_SEL_BITS];
        end
        if (NUM_BANKS != 1) begin
            assign req_bank_idx[i] = src_regs[i][BANK_SEL_BITS-1:0];
        end else begin
            assign req_bank_idx[i] = '0;
        end
    end

    for (genvar i = 0; i < NUM_SRC_REGS; ++i) begin
        assign src_valid[i] = (src_regs[i] != 0) && ~data_fetched_st1[i];
    end

    assign req_in_valid = {NUM_SRC_REGS{scoreboard_if.valid}} & src_valid;

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_SRC_REGS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (PER_BANK_ADDRW),
        .ARBITER     ("P"), // use priority arbiter
        .PERF_CTR_BITS(`PERF_CTR_BITS),
        .OUT_BUF     (0) // no output buffering
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN(collisions),
        .valid_in  (req_in_valid),
        .data_in   (req_in_data),
        .sel_in    (req_bank_idx),
        .ready_in  (req_in_ready),
        .valid_out (gpr_rd_valid),
        .data_out  (gpr_rd_addr),
        .sel_out   (gpr_rd_req_idx),
        .ready_out (gpr_rd_ready)
    );

    wire pipe_in_ready = pipe_ready_st1 || ~pipe_valid_st1;

    assign gpr_rd_ready = {NUM_BANKS{pipe_in_ready}};

    assign scoreboard_if.ready = pipe_in_ready && ~has_collision_n;

    wire pipe_fire_st1 = pipe_valid_st1 && pipe_ready_st1;
    wire pipe_fire_st2 = pipe_valid_st2 && pipe_ready_st2;

    always @(*) begin
        has_collision_n = 0;
        for (integer i = 0; i < NUM_SRC_REGS; ++i) begin
            for (integer j = 1; j < (NUM_SRC_REGS-i); ++j) begin
                has_collision_n |= src_valid[i]
                                && src_valid[j+i]
                                && (req_bank_idx[i] == req_bank_idx[j+i]);
            end
        end
    end

    always @(*) begin
        data_fetched_n = data_fetched_st1;
         if (scoreboard_if.ready) begin
            data_fetched_n = '0;
        end else begin
            data_fetched_n = data_fetched_st1 | req_in_ready;
        end
    end

    assign pipe_data = {
        scoreboard_if.data.wis,
        scoreboard_if.data.tmask,
        scoreboard_if.data.PC,
        scoreboard_if.data.wb,
        scoreboard_if.data.ex_type,
        scoreboard_if.data.op_type,
        scoreboard_if.data.op_args,
        scoreboard_if.data.rd,
        scoreboard_if.data.uuid
    };

    VX_pipe_register #(
        .DATAW  (1 + NUM_SRC_REGS + NUM_BANKS + META_DATAW + 1 + NUM_BANKS * (PER_BANK_ADDRW + REQ_SEL_WIDTH)),
        .RESETW (1 + NUM_SRC_REGS)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (pipe_in_ready),
        .data_in  ({scoreboard_if.valid, data_fetched_n,   gpr_rd_valid,     pipe_data,     has_collision_n,   gpr_rd_addr,     gpr_rd_req_idx}),
        .data_out ({pipe_valid_st1,      data_fetched_st1, gpr_rd_valid_st1, pipe_data_st1, has_collision_st1, gpr_rd_addr_st1, gpr_rd_req_idx_st1})
    );

    assign pipe_ready_st1 = pipe_ready_st2 || ~pipe_valid_st2;

    assign src_data_st1 = pipe_fire_st2 ? '0 : src_data_n;

    wire pipe_valid2_st1 = pipe_valid_st1 && ~has_collision_st1;

    `RESET_RELAY (pipe2_reset, reset); // needed for pipe_reg2's wide RESETW

    VX_pipe_register #(
        .DATAW  (1 + NUM_SRC_REGS * REGS_DATAW + NUM_BANKS + NUM_BANKS * REGS_DATAW + META_DATAW + NUM_BANKS * REQ_SEL_WIDTH),
        .RESETW (1 + NUM_SRC_REGS * REGS_DATAW)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (pipe2_reset),
        .enable   (pipe_ready_st1),
        .data_in  ({pipe_valid2_st1, src_data_st1, gpr_rd_valid_st1, gpr_rd_data_st1, pipe_data_st1, gpr_rd_req_idx_st1}),
        .data_out ({pipe_valid_st2,  src_data_st2, gpr_rd_valid_st2, gpr_rd_data_st2, pipe_data_st2, gpr_rd_req_idx_st2})
    );

    always @(*) begin
        src_data_n = src_data_st2;
        for (integer b = 0; b < NUM_BANKS; ++b) begin
            if (gpr_rd_valid_st2[b]) begin
                src_data_n[gpr_rd_req_idx_st2[b]] = gpr_rd_data_st2[b];
            end
        end
    end

    VX_elastic_buffer #(
        .DATAW   (DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
        .LUTRAM  (1)
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (pipe_valid_st2),
        .ready_in  (pipe_ready_st2),
        .data_in   ({
            pipe_data_st2,
            src_data_n[0],
            src_data_n[1],
            src_data_n[2]
        }),
        .data_out  ({
            operands_if.data.wis,
            operands_if.data.tmask,
            operands_if.data.PC,
            operands_if.data.wb,
            operands_if.data.ex_type,
            operands_if.data.op_type,
            operands_if.data.op_args,
            operands_if.data.rd,
            operands_if.data.uuid,
            operands_if.data.rs1_data,
            operands_if.data.rs2_data,
            operands_if.data.rs3_data
        }),
        .valid_out (operands_if.valid),
        .ready_out (operands_if.ready)
    );

    wire [PER_BANK_ADDRW-1:0] gpr_wr_addr;
    if (ISSUE_WIS != 0) begin
        assign gpr_wr_addr = {writeback_if.data.rd[`NR_BITS-1:BANK_SEL_BITS], writeback_if.data.wis};
    end else begin
        assign gpr_wr_addr = writeback_if.data.rd[`NR_BITS-1:BANK_SEL_BITS];
    end

    wire [BANK_SEL_WIDTH-1:0] gpr_wr_bank_idx;
    if (NUM_BANKS != 1) begin
        assign gpr_wr_bank_idx = writeback_if.data.rd[BANK_SEL_BITS-1:0];
    end else begin
        assign gpr_wr_bank_idx = '0;
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
            assign gpr_wr_enabled = wr_enabled
                                 && writeback_if.valid
                                 && (gpr_wr_bank_idx == BANK_SEL_BITS'(b));
        end else begin
            assign gpr_wr_enabled = wr_enabled && writeback_if.valid;
        end

        wire [BYTEENW-1:0] wren;
        for (genvar i = 0; i < `NUM_THREADS; ++i) begin
            assign wren[i*XLEN_SIZE+:XLEN_SIZE] = {XLEN_SIZE{writeback_if.data.tmask[i]}};
        end

        VX_dp_ram #(
            .DATAW (REGS_DATAW),
            .SIZE  (PER_BANK_REGS * PER_ISSUE_WARPS),
            .WRENW (BYTEENW),
         `ifdef GPR_RESET
            .RESET_RAM (1),
         `endif
            .NO_RWCHECK (1)
        ) gpr_ram (
            .clk   (clk),
            .reset (reset),
            .read  (pipe_fire_st1),
            .wren  (wren),
            .write (gpr_wr_enabled),
            .waddr (gpr_wr_addr),
            .wdata (writeback_if.data.data),
            .raddr (gpr_rd_addr_st1[b]),
            .rdata (gpr_rd_data_st1[b])
        );
    end

`ifdef PERF_ENABLE
    reg [`PERF_CTR_BITS-1:0] collisions_r;
    always @(posedge clk) begin
        if (reset) begin
            collisions_r <= '0;
        end else begin
            collisions_r <= collisions_r + `PERF_CTR_BITS'(scoreboard_if.valid && pipe_in_ready && has_collision_n);
        end
    end
    assign perf_stalls = collisions_r;
`endif

endmodule
