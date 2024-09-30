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

`include "VX_platform.vh"

`TRACING_OFF
module VX_fifo_queue #(
    parameter DATAW     = 1,
    parameter DEPTH     = 2,
    parameter ALM_FULL  = (DEPTH - 1),
    parameter ALM_EMPTY = 1,
    parameter OUT_REG   = 0,
    parameter LUTRAM    = 1,
    parameter SIZEW     = `CLOG2(DEPTH+1)
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             push,
    input  wire             pop,
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    output wire             empty,
    output wire             alm_empty,
    output wire             full,
    output wire             alm_full,
    output wire [SIZEW-1:0] size
);

    `STATIC_ASSERT(ALM_FULL > 0, ("alm_full must be greater than 0!"))
    `STATIC_ASSERT(ALM_FULL < DEPTH, ("alm_full must be smaller than size!"))
    `STATIC_ASSERT(ALM_EMPTY > 0, ("alm_empty must be greater than 0!"))
    `STATIC_ASSERT(ALM_EMPTY < DEPTH, ("alm_empty must be smaller than size!"))
    `STATIC_ASSERT(`IS_POW2(DEPTH), ("depth must be a power of 2!"))

    VX_pending_size #(
        .SIZE      (DEPTH),
        .ALM_EMPTY (ALM_EMPTY),
        .ALM_FULL  (ALM_FULL)
    ) pending_size (
        .clk   (clk),
        .reset (reset),
        .incr  (push),
        .decr  (pop),
        .empty (empty),
        .full  (full),
        .alm_empty(alm_empty),
        .alm_full(alm_full),
        .size  (size)
    );

    if (DEPTH == 1) begin : g_depth_1

        reg [DATAW-1:0] head_r;

        always @(posedge clk) begin
            if (push) begin
                head_r <= data_in;
            end
        end

        assign data_out = head_r;

    end else begin : g_depth_n

        localparam ADDRW = `CLOG2(DEPTH);

        if (OUT_REG != 0) begin : g_out_reg

            wire [DATAW-1:0] dout;
            reg [DATAW-1:0] dout_r;
            reg [ADDRW-1:0] wr_ptr_r;
            reg [ADDRW-1:0] rd_ptr_r;
            reg [ADDRW-1:0] rd_ptr_n_r;

            always @(posedge clk) begin
                if (reset) begin
                    wr_ptr_r   <= '0;
                    rd_ptr_r   <= '0;
                    rd_ptr_n_r <= 1;
                end else begin
                    wr_ptr_r <= wr_ptr_r + ADDRW'(push);
                    if (pop) begin
                        rd_ptr_r <= rd_ptr_n_r;
                        if (DEPTH > 2) begin
                            rd_ptr_n_r <= rd_ptr_r + ADDRW'(2);
                        end else begin // (DEPTH == 2);
                            rd_ptr_n_r <= ~rd_ptr_n_r;
                        end
                    end
                end
            end

            VX_dp_ram #(
                .DATAW  (DATAW),
                .SIZE   (DEPTH),
                .LUTRAM (LUTRAM)
            ) dp_ram (
                .clk   (clk),
                .reset (reset),
                .read  (1'b1),
                .write (push),
                .wren  (1'b1),
                .waddr (wr_ptr_r),
                .wdata (data_in),
                .raddr (rd_ptr_n_r),
                .rdata (dout)
            );

            wire going_empty = (ALM_EMPTY == 1) ? alm_empty : (size[ADDRW-1:0] == ADDRW'(1));

            always @(posedge clk) begin
                if (push && (empty || (going_empty && pop))) begin
                    dout_r <= data_in;
                end else if (pop) begin
                    dout_r <= dout;
                end
            end

            assign data_out = dout_r;

        end else begin : g_no_out_reg

            reg [ADDRW-1:0] rd_ptr_r;
            reg [ADDRW-1:0] wr_ptr_r;

            always @(posedge clk) begin
                if (reset) begin
                    rd_ptr_r <= '0;
                    wr_ptr_r <= '0;
                end else begin
                    wr_ptr_r <= wr_ptr_r + ADDRW'(push);
                    rd_ptr_r <= rd_ptr_r + ADDRW'(pop);
                end
            end

            VX_dp_ram #(
                .DATAW  (DATAW),
                .SIZE   (DEPTH),
                .LUTRAM (LUTRAM)
            ) dp_ram (
                .clk   (clk),
                .reset (reset),
                .read  (1'b1),
                .write (push),
                .wren  (1'b1),
                .waddr (wr_ptr_r),
                .wdata (data_in),
                .raddr (rd_ptr_r),
                .rdata (data_out)
            );

        end
    end

    `RUNTIME_ASSERT(~(push && ~pop) || ~full, ("%t: runtime error: incrementing full queue", $time))
    `RUNTIME_ASSERT(~(pop && ~push) || ~empty, ("%t: runtime error: decrementing empty queue", $time))

endmodule
`TRACING_ON
