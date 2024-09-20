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

module VX_ipdom_stack #(
    parameter WIDTH   = 1,
    parameter DEPTH   = 1,
    parameter OUT_REG = 0,
    parameter ADDRW   = `LOG2UP(DEPTH)
) (
    input  wire             clk,
    input  wire             reset,
    input  wire [WIDTH-1:0] q0,
    input  wire [WIDTH-1:0] q1,
    output wire [WIDTH-1:0] d,
    output wire             d_set,
    output wire [ADDRW-1:0] q_ptr,
    input  wire             push,
    input  wire             pop,
    output wire             empty,
    output wire             full
);
    reg slot_set [DEPTH-1:0];

    reg [ADDRW-1:0] rd_ptr, wr_ptr;

    reg empty_r, full_r;

    wire [WIDTH-1:0] d0, d1;

    wire d_set_n = slot_set[rd_ptr];

    always @(posedge clk) begin
        if (reset) begin
            rd_ptr  <= '0;
            wr_ptr  <= '0;
            empty_r <= 1;
            full_r  <= 0;
        end else begin
            `ASSERT(~push || ~full, ("runtime error: writing to a full stack!"));
            `ASSERT(~pop || ~empty, ("runtime error: reading an empty stack!"));
            `ASSERT(~push || ~pop,  ("runtime error: push and pop in same cycle not supported!"));
            if (push) begin
                rd_ptr  <= wr_ptr;
                wr_ptr  <= wr_ptr + ADDRW'(1);
                empty_r <= 0;
                full_r  <= (ADDRW'(DEPTH-1) == wr_ptr);
            end else if (pop) begin
                wr_ptr  <= wr_ptr - ADDRW'(d_set_n);
                rd_ptr  <= rd_ptr - ADDRW'(d_set_n);
                empty_r <= (rd_ptr == 0) && (d_set_n == 1);
                full_r  <= 0;
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (WIDTH * 2),
        .SIZE    (DEPTH),
        .OUT_REG (OUT_REG ? 1 : 0),
        .LUTRAM  (OUT_REG ? 0 : 1)
    ) store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (push),
        .wren  (1'b1),
        .waddr (wr_ptr),
        .wdata ({q1, q0}),
        .raddr (rd_ptr),
        .rdata ({d1, d0})
    );

    always @(posedge clk) begin
        if (push) begin
            slot_set[wr_ptr] <= 0;
        end else if (pop) begin
            slot_set[rd_ptr] <= 1;
        end
    end

    wire d_set_r;

    VX_pipe_register #(
        .DATAW (1),
        .DEPTH (OUT_REG)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  (d_set_n),
        .data_out (d_set_r)
    );

    assign d     = d_set_r ? d0 : d1;
    assign d_set = ~d_set_r;
    assign q_ptr = wr_ptr;
    assign empty = empty_r;
    assign full  = full_r;

endmodule
