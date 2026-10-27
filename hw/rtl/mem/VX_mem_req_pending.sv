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

// Tracks the requests held between the input and output handshakes of a
// section that neither creates nor drops requests (arbiters, buffers).
module VX_mem_req_pending #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter SIZE        = 8 * NUM_OUTPUTS + 2 * NUM_INPUTS // bound on the requests the section can hold
) (
    input wire                   clk,
    input wire                   reset,
    input wire [NUM_INPUTS-1:0]  in_fire,
    input wire [NUM_OUTPUTS-1:0] out_fire,
    output wire                  empty
);
    localparam INCRW = `CLOG2(NUM_INPUTS+1);
    localparam DECRW = `CLOG2(NUM_OUTPUTS+1);

    wire [INCRW-1:0] in_cnt;
    wire [DECRW-1:0] out_cnt;
    `POP_COUNT(in_cnt, in_fire);
    `POP_COUNT(out_cnt, out_fire);

    VX_pending_size #(
        .SIZE  (SIZE),
        .INCRW (INCRW),
        .DECRW (DECRW)
    ) pending (
        .clk       (clk),
        .reset     (reset),
        .incr      (in_cnt),
        .decr      (out_cnt),
        .empty     (empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

endmodule
