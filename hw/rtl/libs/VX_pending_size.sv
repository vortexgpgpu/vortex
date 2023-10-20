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
module VX_pending_size #(
    parameter SIZE  = 1,
    parameter INCRW = 1,
    parameter DECRW = 1,
    parameter SIZEW = `CLOG2(SIZE+1)
) (
    input wire  clk,
    input wire  reset,
    input wire [INCRW-1:0] incr,
    input wire [DECRW-1:0] decr,
    output wire empty,
    output wire full,
    output wire [SIZEW-1:0] size
);
    `STATIC_ASSERT(INCRW <= SIZEW, ("invalid parameter"))
    `STATIC_ASSERT(DECRW <= SIZEW, ("invalid parameter"))
    localparam ADDRW = `LOG2UP(SIZE);

    reg empty_r;
    reg full_r;

    if (INCRW != 1 || DECRW != 1) begin

        reg [SIZEW-1:0] size_r;
        wire [SIZEW-1:0] size_n;

        assign size_n = size_r + SIZEW'(incr) - SIZEW'(decr);

        always @(posedge clk) begin
            if (reset) begin          
                size_r  <= '0;
                empty_r <= 1;
                full_r  <= 0;
            end else begin
                size_r  <= size_n;
                empty_r <= (size_n == SIZEW'(0));
                full_r  <= (size_n == SIZEW'(SIZE));
            end
        end

        assign size = size_r;

    end else begin

        reg [ADDRW-1:0] used_r;

        always @(posedge clk) begin
            if (reset) begin          
                used_r  <= '0;
                empty_r <= 1;
                full_r  <= 0;
            end else begin            
                `ASSERT(~(incr && ~decr) || ~full, ("runtime error: incrementing full counter"));
                `ASSERT(~(decr && ~incr) || ~empty, ("runtime error: decrementing empty counter"));
                if (incr) begin
                    if (~decr) begin
                        empty_r <= 0;
                        if (used_r == ADDRW'(SIZE-1))
                            full_r <= 1;
                    end
                end else if (decr) begin
                    full_r <= 0;
                    if (used_r == ADDRW'(1))
                        empty_r <= 1;                
                end
                used_r <= $signed(used_r) + ADDRW'($signed(2'(incr) - 2'(decr)));
            end
        end

        if (SIZE > 1) begin
            if (SIZEW > ADDRW) begin
                assign size = {full_r, used_r};
            end else begin
                assign size = used_r;
            end
        end else begin
            assign size = full_r;
        end

    end

    assign empty = empty_r;
    assign full  = full_r;
  
endmodule
`TRACING_ON
