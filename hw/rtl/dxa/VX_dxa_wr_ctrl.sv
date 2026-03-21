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

// DXA Write Controller (refactored): thin SMEM write adapter.
// Accepts packed SMEM words from cl2smem, buffers in FIFO, drains
// to smem_bank_wr_if with sequential addressing, tracks completion.
// All data packing/compression logic moved to VX_dxa_cl2smem.

`include "VX_define.vh"

module VX_dxa_wr_ctrl import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter WR_QUEUE_DEPTH   = 16,
    parameter SMEM_BYTES       = DXA_SMEM_WORD_SIZE,
    parameter SMEM_DATAW       = SMEM_BYTES * 8,
    parameter SMEM_OFF_BITS    = `CLOG2(SMEM_BYTES),
    parameter SMEM_ADDR_WIDTH  = DXA_SMEM_ADDR_WIDTH
) (
    input  wire                        clk,
    input  wire                        reset,
`ifdef PERF_ENABLE
    output wire [31:0]                 perf_smem_writes,
`endif
    input  wire                        transfer_active,
    input  wire                        transfer_start,

    // Params from setup.
    input  wire [31:0]                 total_smem_writes,
    input  wire [`MEM_ADDR_WIDTH-1:0]  initial_smem_base,

    // SMEM word input (from cl2smem, valid/ready).
    input  wire                        smem_in_valid,
    output wire                        smem_in_ready,
    input  wire [SMEM_DATAW-1:0]       smem_in_data,
    input  wire [SMEM_BYTES-1:0]       smem_in_byteen,
    input  wire                        smem_in_last,

    // SMEM write output.
    output wire                        smem_wr_valid,
    output wire [SMEM_ADDR_WIDTH-1:0]  smem_wr_addr,
    output wire [SMEM_DATAW-1:0]       smem_wr_data,
    output wire [SMEM_BYTES-1:0]       smem_wr_byteen,
    input  wire                        smem_wr_ready,
    output wire                        smem_wr_last_pkt,

    // Completion.
    output wire                        transfer_done,
    output wire [31:0]                 wr_done_count,
    output wire                        smem_req_fire

);
    // ---- Write queue ----
    localparam WRQ_DATAW = 1 + SMEM_ADDR_WIDTH + SMEM_DATAW + SMEM_BYTES;
    localparam WRQ_SIZEW = `CLOG2(WR_QUEUE_DEPTH + 1);

    `STATIC_ASSERT(`IS_POW2(WR_QUEUE_DEPTH), ("WR_QUEUE_DEPTH must be power of 2"))

    wire [WRQ_DATAW-1:0] wrq_data_in;
    wire [WRQ_DATAW-1:0] wrq_data_out;
    wire wrq_empty, wrq_full;
    wire wrq_alm_empty, wrq_alm_full;
    wire [WRQ_SIZEW-1:0] wrq_size;
    wire wrq_push, wrq_pop;

    VX_fifo_queue #(
        .DATAW   (WRQ_DATAW),
        .DEPTH   (WR_QUEUE_DEPTH),
        .OUT_REG (1),
        .LUTRAM  (1)
    ) wr_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (wrq_push),
        .pop      (wrq_pop),
        .data_in  (wrq_data_in),
        .data_out (wrq_data_out),
        .empty    (wrq_empty),
        .alm_empty(wrq_alm_empty),
        .full     (wrq_full),
        .alm_full (wrq_alm_full),
        .size     (wrq_size)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 1: Accept SMEM words from cl2smem, assign sequential addresses
    // ════════════════════════════════════════════════════════════════════

    // SMEM address counter.
    wire [SMEM_ADDR_WIDTH-1:0] initial_smem_word = SMEM_ADDR_WIDTH'(initial_smem_base >> SMEM_OFF_BITS);
    reg [SMEM_ADDR_WIDTH-1:0]  smem_addr_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            smem_addr_r <= transfer_start ? initial_smem_word : '0;
        end else if (wrq_push) begin
            smem_addr_r <= smem_addr_r + SMEM_ADDR_WIDTH'(1);
        end
    end

    // Push to write queue.
    assign smem_in_ready = ~wrq_full;
    assign wrq_push      = smem_in_valid && smem_in_ready;
    assign wrq_data_in   = {smem_in_last, smem_addr_r, smem_in_data, smem_in_byteen};

    // ════════════════════════════════════════════════════════════════════
    // Stage 2: SMEM write output from queue head
    // ════════════════════════════════════════════════════════════════════

    wire wrq_head_last;
    wire [SMEM_ADDR_WIDTH-1:0] wrq_head_addr;
    wire [SMEM_DATAW-1:0] wrq_head_data;
    wire [SMEM_BYTES-1:0] wrq_head_byteen;

    assign {wrq_head_last, wrq_head_addr, wrq_head_data, wrq_head_byteen} = wrq_data_out;

    assign smem_wr_valid  = transfer_active && ~wrq_empty;
    assign smem_wr_addr   = wrq_head_addr;
    assign smem_wr_data   = wrq_head_data;
    assign smem_wr_byteen = wrq_head_byteen;
    assign wrq_pop        = smem_wr_valid && smem_wr_ready;
    assign smem_wr_last_pkt = wrq_pop && wrq_head_last;
    assign smem_req_fire  = wrq_pop;


    // ════════════════════════════════════════════════════════════════════
    // Stage 3: Completion Tracking
    // ════════════════════════════════════════════════════════════════════

    reg [31:0] wr_count_r;
    reg        seen_last_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wr_count_r  <= '0;
            seen_last_r <= 1'b0;
        end else if (transfer_active) begin
            if (wrq_pop) begin
                wr_count_r <= wr_count_r + 32'd1;
                if (wrq_head_last) begin
                    seen_last_r <= 1'b1;
                end
            end
        end
    end

    assign wr_done_count = wr_count_r;

    // Transfer done: all SMEM writes completed and last marker seen.
    wire [31:0] wr_count_next = wr_count_r + 32'(wrq_pop);
    wire seen_last_next = seen_last_r || (wrq_pop && wrq_head_last);

    assign transfer_done = transfer_active
                        && (wr_count_next >= total_smem_writes)
                        && seen_last_next;


    `UNUSED_VAR (wrq_alm_empty)
    `UNUSED_VAR (wrq_alm_full)
    `UNUSED_VAR (wrq_size)

`ifdef PERF_ENABLE
    // Lightweight write counter (no eff_bytes, no span, no back-to-back)
    reg [31:0] wrp_total_smem_writes_r;
    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wrp_total_smem_writes_r <= '0;
        end else if (wrq_pop) begin
            wrp_total_smem_writes_r <= wrp_total_smem_writes_r + 32'd1;
        end
    end
    assign perf_smem_writes = wrp_total_smem_writes_r + 32'(wrq_pop);
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (wrq_push) begin
                `TRACE(2, ("%t: wr_ctrl push: addr=0x%0h byteen=0x%0h last=%0b\n",
                    $time, smem_addr_r, smem_in_byteen, smem_in_last))
            end
            if (transfer_active && wrq_pop) begin
                `TRACE(2, ("%t: wr_ctrl pop: addr=0x%0h count=%0d total=%0d last=%0b done=%0b\n",
                    $time, wrq_head_addr, wr_count_next, total_smem_writes, wrq_head_last, transfer_done))
            end
        end
    end
`endif

endmodule
