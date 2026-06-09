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
module VX_ticket_lock #(
    parameter N    = 2,
    parameter LOGN = `LOG2UP(N)
) (
    input  wire            clk,
    input wire             reset,
    input wire             aquire_en,
    input wire             release_en,
    output wire [LOGN-1:0] acquire_id,
    output wire [LOGN-1:0] release_id,
    output wire            full,
    output wire            empty
);
    reg [LOGN-1:0] rd_ctr_r, wr_ctr_r;

    always @(posedge clk) begin
        if (reset) begin
            rd_ctr_r <= '0;
            wr_ctr_r <= '0;
        end else begin
            if (aquire_en && !full) begin
                wr_ctr_r <= wr_ctr_r + 1;
            end
            if (release_en && !empty) begin
                rd_ctr_r <= rd_ctr_r + 1;
            end
        end
    end

    VX_pending_size #(
        .SIZE  (N),
        .INCRW (1),
        .DECRW (1)
    ) pending_size (
        .clk       (clk),
        .reset     (reset),
        .incr      (aquire_en),
        .decr      (release_en),
        .empty     (empty),
        .full      (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    assign acquire_id = wr_ctr_r;
    assign release_id = rd_ctr_r;

endmodule
`TRACING_ON
