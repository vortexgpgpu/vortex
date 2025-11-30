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
    reg [ADDRW-1:0] rd_ptr, rd_ptr_n, wr_ptr;

    reg empty_r, full_r;

    wire [WIDTH-1:0] d0, d1;

    wire d_set_r;

    always @(*) begin
        rd_ptr_n = rd_ptr;
        if (push) begin
            rd_ptr_n = wr_ptr;
        end else if (pop) begin
            rd_ptr_n = rd_ptr - ADDRW'(d_set_r);
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            wr_ptr  <= '0;
            empty_r <= 1;
            full_r  <= 0;
            rd_ptr  <= '0;
        end else begin
            `ASSERT(~push || ~full, ("%t: runtime error: writing to a full stack!", $time));
            `ASSERT(~pop || ~empty, ("%t: runtime error: reading an empty stack!", $time));
            `ASSERT(~push || ~pop,  ("%t: runtime error: push and pop in same cycle not supported!", $time));
            if (push) begin
                wr_ptr  <= wr_ptr + ADDRW'(1);
                empty_r <= 0;
                full_r  <= (ADDRW'(DEPTH-1) == wr_ptr);
            end else if (pop) begin
                wr_ptr  <= wr_ptr - ADDRW'(d_set_r);
                empty_r <= (rd_ptr == 0) && d_set_r;
                full_r  <= 0;
            end
            rd_ptr <= rd_ptr_n;
        end
    end

    wire [WIDTH * 2:0] qout = push ? {1'b0, q1, q0} : {1'b1, d1, d0};

    VX_dp_ram #(
        .DATAW (1 + WIDTH * 2),
        .SIZE (DEPTH),
        .OUT_REG (1),
        .RDW_MODE ("R")
    ) ipdom_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (push || pop),
        .wren  (1'b1),
        .waddr (push ? wr_ptr : rd_ptr),
        .wdata (qout),
        .raddr (rd_ptr_n),
        .rdata ({d_set_r, d1, d0})
    );

    assign d     = d_set_r ? d0 : d1;
    assign d_set = ~d_set_r;
    assign q_ptr = wr_ptr;
    assign empty = empty_r;
    assign full  = full_r;

endmodule
