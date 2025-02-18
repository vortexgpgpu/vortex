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

module VX_gpr_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS = 1,
    parameter NUM_BANKS = 1
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    VX_writeback_if.slave   writeback_if,
    VX_opc_if.slave         opc_if [NUM_REQS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam REQ_SEL_BITS = `CLOG2(NUM_REQS);
    localparam REQ_SEL_WIDTH = `UP(REQ_SEL_BITS);
    localparam BANK_SEL_BITS = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam GPR_BANK_DATAW = `XLEN * `SIMD_WIDTH;
    localparam GPR_BANK_SIZE = (PER_ISSUE_WARPS * NUM_REGS * SIMD_COUNT) / NUM_BANKS;
    localparam GPR_BANK_ADDRW = `CLOG2(GPR_BANK_SIZE);
    localparam BANKID_WIS_BITS = (BANK_SEL_BITS > 1 && ISSUE_WIS_BITS != 0) ? 1 : 0;
    localparam BANKID_REG_BITS = BANK_SEL_BITS - BANKID_WIS_BITS;
    localparam PER_BANK_WIS_BITS = ISSUE_WIS_BITS - BANKID_WIS_BITS;
    localparam PER_BANK_REG_BITS = NR_BITS - BANKID_REG_BITS;
    localparam PER_BANK_WIS_WIDTH = `UP(PER_BANK_WIS_BITS);
    localparam PER_BANK_REG_WIDTH = `UP(PER_BANK_REG_BITS);
    localparam OPC_REQ_DATAW = 2 + SIMD_IDX_W + PER_BANK_WIS_BITS + PER_BANK_REG_BITS;
    localparam OPC_RSP_DATAW = 2 + `SIMD_WIDTH * `XLEN;
    localparam BYTEENW = `SIMD_WIDTH * XLENB;

    wire [NUM_REQS-1:0] opc_req_valid, opc_req_ready;
    wire [NUM_REQS-1:0][OPC_REQ_DATAW-1:0] opc_req_data;
    wire [NUM_REQS-1:0][BANK_SEL_WIDTH-1:0] opc_req_bank_idx;

    wire [NUM_BANKS-1:0] gpr_req_valid, gpr_req_ready;
    wire [NUM_BANKS-1:0][OPC_REQ_DATAW-1:0] gpr_req_data;
    wire [NUM_BANKS-1:0][1:0] gpr_req_opd_id;
    wire [NUM_BANKS-1:0][SIMD_IDX_W-1:0] gpr_req_sid;
    wire [NUM_BANKS-1:0][PER_BANK_WIS_WIDTH-1:0] gpr_req_wis;
    wire [NUM_BANKS-1:0][PER_BANK_REG_WIDTH-1:0] gpr_reg_id;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_req_idx;

    wire [NUM_BANKS-1:0][`SIMD_WIDTH-1:0][`XLEN-1:0] gpr_rd_data;

    wire [NUM_BANKS-1:0] gpr_rsp_valid;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_rsp_idx;
    wire [NUM_BANKS-1:0][1:0] gpr_rsp_opd_id;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] collisions;
`endif

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_opc_req
        assign opc_req_valid[i] = opc_if[i].req_valid;
        assign opc_req_data[i] = {
            opc_if[i].req_data.opd_id,
            opc_if[i].req_data.sid,
            opc_if[i].req_data.wis[ISSUE_WIS_W-1:BANKID_WIS_BITS],
            opc_if[i].req_data.reg_id[NR_BITS-1:BANKID_REG_BITS]
        };
        `CONCAT(opc_req_bank_idx[i], opc_if[i].req_data.wis[BANKID_WIS_BITS-1:0], opc_if[i].req_data.reg_id[BANKID_REG_BITS-1:0], BANKID_WIS_BITS, BANKID_REG_BITS)
        assign opc_if[i].req_ready = opc_req_ready[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (OPC_REQ_DATAW),
        .ARBITER     ("P"),
        .OUT_BUF     (1),
        .PERF_CTR_BITS (PERF_CTR_BITS)
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
    `ifdef PERF_ENABLE
        .collisions(collisions),
    `endif
        .valid_in  (opc_req_valid),
        .data_in   (opc_req_data),
        .sel_in    (opc_req_bank_idx),
        .ready_in  (opc_req_ready),
        .valid_out (gpr_req_valid),
        .data_out  (gpr_req_data),
        .sel_out   (gpr_req_idx),
        .ready_out ('1)
    );

    wire [GPR_BANK_ADDRW-1:0] gpr_wr_addr;
    if (SIMD_IDX_BITS != 0 || PER_BANK_WIS_BITS != 0) begin : g_gpr_wr_addr
        wire [SIMD_IDX_BITS + PER_BANK_WIS_BITS-1:0] tmp;
        `CONCAT(tmp, writeback_if.data.sid, writeback_if.data.wis[ISSUE_WIS_W-1:BANKID_WIS_BITS], SIMD_IDX_BITS, PER_BANK_WIS_BITS);
        assign gpr_wr_addr = {tmp, writeback_if.data.rd[NR_BITS-1:BANKID_REG_BITS]};
    end else begin : g_gpr_wr_addr_reg
        assign gpr_wr_addr = writeback_if.data.rd[NR_BITS-1:BANKID_REG_BITS];
    end

    wire [BANK_SEL_WIDTH-1:0] gpr_wr_bank_idx;
    if (NUM_BANKS != 1) begin : g_gpr_wr_bank_idx
        `CONCAT(gpr_wr_bank_idx, writeback_if.data.wis[BANKID_WIS_BITS-1:0], writeback_if.data.rd[BANKID_REG_BITS-1:0], BANKID_WIS_BITS, BANKID_REG_BITS)
    end else begin : g_gpr_wr_bank_idx_0
        assign gpr_wr_bank_idx = '0;
    end

    wire [BYTEENW-1:0] gpr_wr_byteen;
    for (genvar i = 0; i < `SIMD_WIDTH; ++i) begin : g_gpr_wr_byteen
        assign gpr_wr_byteen[i*XLENB+:XLENB] = {XLENB{writeback_if.data.tmask[i]}};
    end

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_gpr_req_data
        assign {gpr_req_opd_id[b], gpr_req_sid[b], gpr_req_wis[b], gpr_reg_id[b]} = gpr_req_data[b];
    end

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_gpr_rams
        wire gpr_wr_enabled;
        if (BANK_SEL_BITS != 0) begin : g_gpr_wr_enabled_multibanks
            assign gpr_wr_enabled = writeback_if.valid && (gpr_wr_bank_idx == BANK_SEL_BITS'(b));
        end else begin : g_gpr_wr_enabled
            assign gpr_wr_enabled = writeback_if.valid;
        end

        wire [GPR_BANK_ADDRW-1:0] gpr_rd_addr;
        if (SIMD_IDX_BITS != 0 || PER_BANK_WIS_BITS != 0) begin : g_gpr_rd_addr
            wire [(SIMD_IDX_BITS + PER_BANK_WIS_BITS)-1:0] tmp;
            `CONCAT(tmp, gpr_req_sid[b], gpr_req_wis[b], SIMD_IDX_BITS, PER_BANK_WIS_BITS);
            assign gpr_rd_addr = {tmp, gpr_reg_id[b]};
        end else begin : g_gpr_rd_addr_reg
            assign gpr_rd_addr = gpr_reg_id[b];
        end

        VX_dp_ram #(
            .DATAW (GPR_BANK_DATAW),
            .SIZE  (GPR_BANK_SIZE),
            .WRENW (BYTEENW),
         `ifdef GPR_RESET
            .RESET_RAM (1),
         `endif
            .OUT_REG (1),
            .RDW_MODE ("R")
        ) gpr_ram (
            .clk   (clk),
            .reset (reset),
            .read  (gpr_req_valid[b]),
            .wren  (gpr_wr_byteen),
            .write (gpr_wr_enabled),
            .waddr (gpr_wr_addr),
            .wdata (writeback_if.data.data),
            .raddr (gpr_rd_addr),
            .rdata (gpr_rd_data[b])
        );

        VX_pipe_buffer #(
            .DATAW (REQ_SEL_WIDTH + 2)
        ) pipe_reg1 (
            .clk      (clk),
            .reset    (reset),
            .valid_in (gpr_req_valid[b]),
            .data_in  ({gpr_req_idx[b], gpr_req_opd_id[b]}),
            `UNUSED_PIN (ready_in),
            .valid_out(gpr_rsp_valid[b]),
            .data_out ({gpr_rsp_idx[b], gpr_rsp_opd_id[b]}),
            `UNUSED_PIN (ready_out)
        );
    end

    wire [NUM_BANKS-1:0][OPC_RSP_DATAW-1:0] gpr_rsp_data;

    `AOS_TO_ITF_RSP_V (opc, opc_if, NUM_REQS, OPC_RSP_DATAW)

    VX_stream_xpoint #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (OPC_RSP_DATAW),
        .OUT_BUF     (0) // no output buffering
    ) rsp_xpoint (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (gpr_rsp_valid),
        .data_in   (gpr_rsp_data),
        .sel_in    (gpr_rsp_idx),
        `UNUSED_PIN (ready_in),
        .valid_out (opc_rsp_valid),
        .data_out  (opc_rsp_data),
        `UNUSED_PIN (ready_out)
    );

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] collisions_r;
    always @(posedge clk) begin
        if (reset) begin
            collisions_r <= '0;
        end else begin
            collisions_r <= collisions_r + collisions;
        end
    end
    assign perf_stalls = collisions_r;
`endif

endmodule
