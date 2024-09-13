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
    parameter SIZE      = 1,
    parameter INCRW     = 1,
    parameter DECRW     = 1,
    parameter ALM_FULL  = (SIZE - 1),
    parameter ALM_EMPTY = 1,
    parameter SIZEW     = `CLOG2(SIZE+1)
) (
    input wire  clk,
    input wire  reset,
    input wire [INCRW-1:0] incr,
    input wire [DECRW-1:0] decr,
    output wire empty,
    output wire alm_empty,
    output wire full,
    output wire alm_full,
    output wire [SIZEW-1:0] size
);
    `STATIC_ASSERT(INCRW <= SIZEW, ("invalid parameter: %d vs %d", INCRW, SIZEW))
    `STATIC_ASSERT(DECRW <= SIZEW, ("invalid parameter: %d vs %d", DECRW, SIZEW))

    if (SIZE == 1) begin : g_size1

        reg size_r;

        always @(posedge clk) begin
            if (reset) begin
                size_r <= '0;
            end else begin
                if (incr) begin
                    if (~decr) begin
                        size_r <= 1;
                    end
                end else if (decr) begin
                    size_r <= '0;
                end
            end
        end

        assign empty     = (size_r == 0);
        assign full      = (size_r != 0);
        assign alm_empty = 1'b1;
        assign alm_full  = 1'b1;
        assign size      = size_r;

    end else begin : g_sizeN

        reg empty_r, alm_empty_r;
        reg full_r, alm_full_r;

        if (INCRW != 1 || DECRW != 1) begin : g_wide_step

            localparam SUBW = `MIN(SIZEW, `MAX(INCRW, DECRW)+1);

            logic [SIZEW-1:0] size_n, size_r;

            assign size_n = $signed(size_r) + SIZEW'($signed(SUBW'(incr) - SUBW'(decr)));

            always @(posedge clk) begin
                if (reset) begin
                    empty_r     <= 1;
                    full_r      <= 0;
                    alm_empty_r <= 1;
                    alm_full_r  <= 0;
                    size_r      <= '0;
                end else begin
                    `ASSERT((SIZEW'(incr) >= SIZEW'(decr)) || (size_n >= size_r), ("runtime error: counter overflow"));
                    `ASSERT((SIZEW'(incr) <= SIZEW'(decr)) || (size_n <= size_r), ("runtime error: counter underflow"));
                    empty_r     <= (size_n == SIZEW'(0));
                    full_r      <= (size_n == SIZEW'(SIZE));
                    alm_empty_r <= (size_n <= SIZEW'(ALM_EMPTY));
                    alm_full_r  <= (size_n >= SIZEW'(ALM_FULL));
                    size_r      <= size_n;
                end
            end

            assign size = size_r;

        end else begin : g_single_step

            localparam ADDRW = `LOG2UP(SIZE);

            reg [ADDRW-1:0] used_r;

            wire is_alm_empty   = (used_r == ADDRW'(ALM_EMPTY));
            wire is_alm_empty_n = (used_r == ADDRW'(ALM_EMPTY+1));
            wire is_alm_full    = (used_r == ADDRW'(ALM_FULL));
            wire is_alm_full_n  = (used_r == ADDRW'(ALM_FULL-1));

            always @(posedge clk) begin
                if (reset) begin
                    alm_empty_r <= 1;
                    alm_full_r  <= 0;
                end else begin
                    if (incr) begin
                        if (~decr) begin
                            if (is_alm_empty)
                                alm_empty_r <= 0;
                            if (is_alm_full_n)
                                alm_full_r <= 1;
                        end
                    end else if (decr) begin
                        if (is_alm_full)
                            alm_full_r <= 0;
                        if (is_alm_empty_n)
                            alm_empty_r <= 1;
                    end
                end
            end

            if (SIZE > 2) begin : g_sizeN

                wire is_empty_n = (used_r == ADDRW'(1));
                wire is_full_n  = (used_r == ADDRW'(SIZE-1));

                wire [1:0] push_minus_pop = {~incr & decr, incr ^ decr};

                always @(posedge clk) begin
                    if (reset) begin
                        empty_r <= 1;
                        full_r  <= 0;
                        used_r  <= '0;
                    end else begin
                        if (incr) begin
                            if (~decr) begin
                                empty_r <= 0;
                                if (is_full_n)
                                    full_r <= 1;
                            end
                        end else if (decr) begin
                            full_r <= 0;
                            if (is_empty_n)
                                empty_r <= 1;
                        end
                        used_r <= $signed(used_r) + ADDRW'($signed(push_minus_pop));
                    end
                end

            end else begin : g_size2

                always @(posedge clk) begin
                    if (reset) begin
                        empty_r <= 1;
                        full_r  <= 0;
                        used_r  <= '0;
                    end else begin
                        empty_r <= (empty_r & ~incr) | (~full_r & decr & ~incr);
                        full_r  <= (~empty_r & incr & ~decr) | (full_r & ~(decr ^ incr));
                        used_r  <= used_r ^ (incr ^ decr);
                    end
                end
            end

            if (SIZE > 1) begin : g_sizeN
                if (SIZEW > ADDRW) begin : g_not_log2
                    assign size = {full_r, used_r};
                end else begin : g_log2
                    assign size = used_r;
                end
            end else begin : g_size1
                assign size = full_r;
            end

        end

        assign empty     = empty_r;
        assign full      = full_r;
        assign alm_empty = alm_empty_r;
        assign alm_full  = alm_full_r;

    end

endmodule
`TRACING_ON
