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

    localparam ADDRW = `CLOG2(DEPTH);

    `STATIC_ASSERT(ALM_FULL > 0, ("alm_full must be greater than 0!"))
    `STATIC_ASSERT(ALM_FULL < DEPTH, ("alm_full must be smaller than size!"))
    `STATIC_ASSERT(ALM_EMPTY > 0, ("alm_empty must be greater than 0!"))
    `STATIC_ASSERT(ALM_EMPTY < DEPTH, ("alm_empty must be smaller than size!"))
    `STATIC_ASSERT(`IS_POW2(DEPTH), ("size must be a power of 2!"))

    if (DEPTH == 1) begin

        reg [DATAW-1:0] head_r;
        reg size_r;

        always @(posedge clk) begin
            if (reset) begin
                head_r <= '0;
                size_r <= '0;
            end else begin
                `ASSERT(~push || ~full, ("runtime error: writing to a full queue"));
                `ASSERT(~pop || ~empty, ("runtime error: reading an empty queue"));
                if (push) begin
                    if (~pop) begin
                        size_r <= 1;
                    end
                end else if (pop) begin
                    size_r <= '0;
                end
                if (push) begin
                    head_r <= data_in;
                end
            end
        end

        assign data_out  = head_r;
        assign empty     = (size_r == 0);
        assign alm_empty = 1'b1;
        assign full      = (size_r != 0);
        assign alm_full  = 1'b1;
        assign size      = size_r;

    end else begin

        reg empty_r, alm_empty_r;
        reg full_r, alm_full_r;
        reg [ADDRW-1:0] used_r;
        wire [ADDRW-1:0] used_n;

        always @(posedge clk) begin
            if (reset) begin
                empty_r     <= 1;
                alm_empty_r <= 1;
                full_r      <= 0;
                alm_full_r  <= 0;
                used_r      <= '0;
            end else begin
                `ASSERT(~(push && ~pop) || ~full, ("runtime error: incrementing full queue"));
                `ASSERT(~(pop && ~push) || ~empty, ("runtime error: decrementing empty queue"));
                if (push) begin
                    if (~pop) begin
                        empty_r <= 0;
                        if (used_r == ADDRW'(ALM_EMPTY))
                            alm_empty_r <= 0;
                        if (used_r == ADDRW'(DEPTH-1))
                            full_r <= 1;
                        if (used_r == ADDRW'(ALM_FULL-1))
                            alm_full_r <= 1;
                    end
                end else if (pop) begin
                    full_r <= 0;
                    if (used_r == ADDRW'(ALM_FULL))
                        alm_full_r <= 0;
                    if (used_r == ADDRW'(1))
                        empty_r <= 1;
                    if (used_r == ADDRW'(ALM_EMPTY+1))
                        alm_empty_r <= 1;
                end
                used_r <= used_n;
            end
        end

        if (DEPTH == 2 && LUTRAM == 0) begin

            assign used_n = used_r ^ (push ^ pop);

            if (0 == OUT_REG) begin

                reg [1:0][DATAW-1:0] shift_reg;

                always @(posedge clk) begin
                    if (push) begin
                        shift_reg[1] <= shift_reg[0];
                        shift_reg[0] <= data_in;
                    end
                end

                assign data_out = shift_reg[!used_r[0]];

            end else begin

                reg [DATAW-1:0] data_out_r;
                reg [DATAW-1:0] buffer;

                always @(posedge clk) begin
                    if (push) begin
                        buffer <= data_in;
                    end
                    if (push && (empty_r || (used_r && pop))) begin
                        data_out_r <= data_in;
                    end else if (pop) begin
                        data_out_r <= buffer;
                    end
                end

                assign data_out = data_out_r;

            end

        end else begin

            assign used_n = $signed(used_r) + ADDRW'($signed(2'(push) - 2'(pop)));

            if (0 == OUT_REG) begin

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

            end else begin

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

                wire going_empty;
                if (ALM_EMPTY == 1) begin
                    assign going_empty = alm_empty_r;
                end else begin
                    assign going_empty = (used_r == ADDRW'(1));
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

                always @(posedge clk) begin
                    if (push && (empty_r || (going_empty && pop))) begin
                        dout_r <= data_in;
                    end else if (pop) begin
                        dout_r <= dout;
                    end
                end

                assign data_out = dout_r;
            end
        end

        assign empty     = empty_r;
        assign alm_empty = alm_empty_r;
        assign full      = full_r;
        assign alm_full  = alm_full_r;
        assign size      = {full_r, used_r};
    end

endmodule
`TRACING_ON
