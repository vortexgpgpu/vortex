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
    localparam METADATAW = ISSUE_WIS_W + `NUM_THREADS + `PC_BITS + 1 + `EX_BITS + `INST_OP_BITS + `INST_ARGS_BITS + `NR_BITS;
    localparam DATAW = `UUID_WIDTH + METADATAW + 3 * `NUM_THREADS * `XLEN;
    localparam RAM_ADDRW = `LOG2UP(`NUM_REGS * PER_ISSUE_WARPS);
    localparam PER_BANK_ADDRW = RAM_ADDRW - BANK_SEL_BITS;
    localparam XLEN_SIZE = `XLEN / 8;
    localparam BYTEENW = `NUM_THREADS * XLEN_SIZE;

    `UNUSED_VAR (writeback_if.data.sop)

    wire [NUM_SRC_REGS-1:0] src_valid;
    wire [NUM_SRC_REGS-1:0] req_in_valid;
    wire [NUM_SRC_REGS-1:0] req_in_ready;
    wire [NUM_SRC_REGS-1:0][PER_BANK_ADDRW-1:0] req_in_data;
    wire [NUM_SRC_REGS-1:0][BANK_SEL_WIDTH-1:0] req_bank_idx;

    wire [NUM_BANKS-1:0] gpr_rd_valid_n, gpr_rd_ready;
    reg [NUM_BANKS-1:0] gpr_rd_valid;
    wire [NUM_BANKS-1:0][PER_BANK_ADDRW-1:0] gpr_rd_addr_n;
    reg [NUM_BANKS-1:0][PER_BANK_ADDRW-1:0] gpr_rd_addr;
    wire [NUM_BANKS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] gpr_rd_data;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_rd_req_idx_n;
    reg [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_rd_req_idx;

    wire pipe_in_ready;
    reg pipe_out_valid;
    wire pipe_out_ready;
    reg [`UUID_WIDTH-1:0] pipe_out_uuid;
    reg [METADATAW-1:0] pipe_out_data;

    reg [NUM_SRC_REGS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] src_data, src_data_n;
    reg [NUM_SRC_REGS-1:0] data_fetched;
    reg has_collision, has_collision_n;

    wire stg_in_valid, stg_in_ready;

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
        assign src_valid[i] = (src_regs[i] != 0) && ~data_fetched[i];
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
        .valid_out (gpr_rd_valid_n),
        .data_out  (gpr_rd_addr_n),
        .sel_out   (gpr_rd_req_idx_n),
        .ready_out (gpr_rd_ready)
    );

    assign gpr_rd_ready = {NUM_BANKS{stg_in_ready}};

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
        src_data_n = src_data;
        for (integer b = 0; b < NUM_BANKS; ++b) begin
            if (gpr_rd_valid[b]) begin
                src_data_n[gpr_rd_req_idx[b]] = gpr_rd_data[b];
            end
        end
    end

    wire pipe_stall = pipe_out_valid && ~pipe_out_ready;
    assign pipe_in_ready = ~pipe_stall;

    assign scoreboard_if.ready = pipe_in_ready && ~has_collision_n;

    wire stg_in_fire = stg_in_valid && stg_in_ready;

    always @(posedge clk) begin
        if (reset) begin
            pipe_out_valid <= 0;
            gpr_rd_valid   <= '0;
            data_fetched   <= '0;
            src_data       <= '0;
        end else begin
            if (~pipe_stall) begin
                pipe_out_valid <= scoreboard_if.valid;
                gpr_rd_valid   <= gpr_rd_valid_n;
                if (scoreboard_if.ready) begin
                    data_fetched <= '0;
                end else begin
                    data_fetched <= data_fetched | req_in_ready;
                end
                if (stg_in_fire) begin
                   src_data <= '0;
                end else begin
                   src_data <= src_data_n;
                end
            end
        end
        if (~pipe_stall) begin
            pipe_out_uuid  <= scoreboard_if.data.uuid;
            pipe_out_data  <= {
                scoreboard_if.data.wis,
                scoreboard_if.data.tmask,
                scoreboard_if.data.PC,
                scoreboard_if.data.wb,
                scoreboard_if.data.ex_type,
                scoreboard_if.data.op_type,
                scoreboard_if.data.op_args,
                scoreboard_if.data.rd
            };
            has_collision  <= has_collision_n;
            gpr_rd_addr    <= gpr_rd_addr_n;
            gpr_rd_req_idx <= gpr_rd_req_idx_n;
        end
    end

    assign pipe_out_ready = stg_in_ready;
    assign stg_in_valid = pipe_out_valid && ~has_collision;

    VX_elastic_buffer #(
        .DATAW   (DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
        .LUTRAM  (1)
    ) out_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (stg_in_valid),
        .ready_in  (stg_in_ready),
        .data_in   ({
            pipe_out_uuid,
            pipe_out_data,
            src_data_n[0],
            src_data_n[1],
            src_data_n[2]
        }),
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

    `ifdef GPR_RESET
        VX_dp_ram_rst #(
    `else
        VX_dp_ram #(
    `endif
            .DATAW (`XLEN * `NUM_THREADS),
            .SIZE  (PER_BANK_REGS * PER_ISSUE_WARPS),
            .ADDR_MIN ((b == 0) ? PER_ISSUE_WARPS : 0),
            .WRENW (BYTEENW),
            .NO_RWCHECK (1)
        ) gpr_ram (
            .clk   (clk),
        `ifdef GPR_RESET
            .reset (reset),
        `endif
            .read  (1'b1),
            .wren  (wren),
            .write (gpr_wr_enabled),
            .waddr (gpr_wr_addr),
            .wdata (writeback_if.data.data),
            .raddr (gpr_rd_addr[b]),
            .rdata (gpr_rd_data[b])
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
