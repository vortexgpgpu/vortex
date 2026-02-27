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

// Write-Back Controller for DXA non-blocking worker.
// Buffers write entries from RRS in a FIFO and issues smem write requests.
// Tracks completion via dual-condition: seen_last + write_count == total.

`include "VX_define.vh"

module VX_dxa_wbc import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter WR_QUEUE_DEPTH  = 16,
    parameter SMEM_BYTES      = DXA_SMEM_WORD_SIZE,
    parameter SMEM_DATAW      = SMEM_BYTES * 8,
    parameter SMEM_OFF_BITS   = `CLOG2(SMEM_BYTES),
    parameter SMEM_ADDR_WIDTH = DXA_SMEM_ADDR_WIDTH
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,
    input  wire                        transfer_start,
    input  wire [31:0]                 total_elements,

    // RRS → WBC write entries (valid/ready handshake).
    input  wire                        wb_valid,
    output wire                        wb_ready,
    input  wire [`MEM_ADDR_WIDTH-1:0]  wb_smem_byte_addr,
    input  wire [SMEM_DATAW-1:0]       wb_smem_data,
    input  wire [SMEM_BYTES-1:0]       wb_smem_byteen,
    input  wire                        wb_is_last,

    // smem write port.
    output wire                        smem_wr_req_valid,
    output wire [SMEM_ADDR_WIDTH-1:0]  smem_wr_req_addr,
    output wire [SMEM_DATAW-1:0]       smem_wr_req_data,
    output wire [SMEM_BYTES-1:0]       smem_wr_req_byteen,
    input  wire                        smem_wr_req_ready,
    output wire                        smem_wr_last_pkt,

    // Completion.
    output wire                        transfer_done,

    // Progress counters exposed.
    output wire [31:0]                 wr_done_count,
    output wire                        smem_req_fire
);
    localparam WRQ_DATAW = 1 + `MEM_ADDR_WIDTH + SMEM_DATAW + SMEM_BYTES;
    localparam WRQ_SIZEW = `CLOG2(WR_QUEUE_DEPTH + 1);

    `STATIC_ASSERT(`IS_POW2(WR_QUEUE_DEPTH), ("WR_QUEUE_DEPTH must be power of 2"))

    // ---- Write queue (FIFO) ----
    wire [WRQ_DATAW-1:0] wrq_data_in = {wb_is_last, wb_smem_byte_addr, wb_smem_data, wb_smem_byteen};
    wire [WRQ_DATAW-1:0] wrq_data_out;
    wire wrq_empty, wrq_full;
    wire wrq_alm_empty, wrq_alm_full;
    wire [WRQ_SIZEW-1:0] wrq_size;
    wire wrq_pop;

    VX_fifo_queue #(
        .DATAW   (WRQ_DATAW),
        .DEPTH   (WR_QUEUE_DEPTH),
        .OUT_REG (1),
        .LUTRAM  (1)
    ) wr_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (wb_valid && wb_ready),
        .pop      (wrq_pop),
        .data_in  (wrq_data_in),
        .data_out (wrq_data_out),
        .empty    (wrq_empty),
        .alm_empty(wrq_alm_empty),
        .full     (wrq_full),
        .alm_full (wrq_alm_full),
        .size     (wrq_size)
    );

    // Accept writes when queue has space (or is being popped this cycle).
    assign wb_ready = ~wrq_full || wrq_pop;

    // ---- smem write request output from queue head ----
    wire wrq_head_last;
    wire [`MEM_ADDR_WIDTH-1:0] wrq_head_smem_byte_addr;
    wire [SMEM_DATAW-1:0] wrq_head_smem_data;
    wire [SMEM_BYTES-1:0] wrq_head_smem_byteen;

    assign {
        wrq_head_last,
        wrq_head_smem_byte_addr,
        wrq_head_smem_data,
        wrq_head_smem_byteen
    } = wrq_data_out;

    assign smem_wr_req_valid = transfer_active && ~wrq_empty;
    assign smem_wr_req_addr = SMEM_ADDR_WIDTH'(wrq_head_smem_byte_addr >> SMEM_OFF_BITS);
    assign smem_wr_req_data = wrq_head_smem_data;
    assign smem_wr_req_byteen = wrq_head_smem_byteen;
    assign wrq_pop = smem_wr_req_valid && smem_wr_req_ready;
    assign smem_wr_last_pkt = wrq_pop && wrq_head_last;
    assign smem_req_fire = wrq_pop;

    // ---- Completion tracking ----
    reg [31:0] wr_count_r;
    reg        seen_last_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wr_count_r <= '0;
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

    // Transfer done when:
    //   1. All elements have been written (count >= total)
    //   2. "last" element has been seen through the queue
    //   3. Write queue is drained
    // Note: uses next-cycle values for immediate detection.
    wire [31:0] wr_count_next = wr_count_r + 32'(wrq_pop);
    wire seen_last_next = seen_last_r || (wrq_pop && wrq_head_last);
    wire [WRQ_SIZEW-1:0] wrq_size_next = WRQ_SIZEW'(wrq_size)
                                        + WRQ_SIZEW'(wb_valid && wb_ready)
                                        - WRQ_SIZEW'(wrq_pop);

    assign transfer_done = transfer_active
                        && (wr_count_next >= total_elements)
                        && seen_last_next
                        && (wrq_size_next == WRQ_SIZEW'(0));

    `UNUSED_VAR (wrq_alm_empty)
    `UNUSED_VAR (wrq_alm_full)
endmodule
