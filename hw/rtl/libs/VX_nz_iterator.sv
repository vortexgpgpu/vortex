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
module VX_nz_iterator #(
  parameter DATAW = 8,  // Bit-width of each data element
  parameter KEYW  = DATAW, // Bit-width of the key
  parameter N = 4,      // Number of elements in the stream
  parameter OUT_REG = 0, // Output register
  parameter LPID_WIDTH = `LOG2UP(N)
) (
    input wire clk,
    input wire reset,
    input wire valid_in,                    // Stream input valid
    input wire [N-1:0][DATAW-1:0] data_in,  // Stream input data
    input wire next,                        // Advance iterator
    output wire valid_out,                  // Current output valid
    output reg [DATAW-1:0] data_out,        // Current output data
    output reg [LPID_WIDTH-1:0] pid,        // Index of the current element
    output reg sop,                         // Start of valid stream
    output reg eop                          // End of valid stream
);
    if (N > 1) begin : g_iterator

        reg [N-1:0] sent_mask_p;
        wire [LPID_WIDTH-1:0] start_p, end_p;

        wire [N-1:0] packet_valids;
        for (genvar i = 0; i < N; ++i) begin : g_packet_valids
            assign packet_valids[i] = (| data_in[i][KEYW-1:0]);
        end

        wire [N-1:0][LPID_WIDTH-1:0] packet_ids;
        for (genvar i = 0; i < N; ++i) begin : g_packet_ids
            assign packet_ids[i] = LPID_WIDTH'(i);
        end

        VX_find_first #(
            .N       (N),
            .DATAW   (LPID_WIDTH),
            .REVERSE (0)
        ) find_first (
            .valid_in (packet_valids & ~sent_mask_p),
            .data_in  (packet_ids),
            .data_out (start_p),
            `UNUSED_PIN (valid_out)
        );

        VX_find_first #(
            .N       (N),
            .DATAW   (LPID_WIDTH),
            .REVERSE (1)
        ) find_last (
            .valid_in (packet_valids),
            .data_in  (packet_ids),
            .data_out (end_p),
            `UNUSED_PIN (valid_out)
        );

        reg is_first_p;
        wire is_last_p = (start_p == end_p);

        wire enable = valid_in && (~valid_out || next);

        always @(posedge clk) begin
            if (reset || (enable && (is_last_p || eop))) begin
                sent_mask_p <= '0;
                is_first_p <= 1;
            end else if (enable) begin
                sent_mask_p[start_p] <= 1;
                is_first_p <= 0;
            end
        end

        VX_pipe_register #(
            .DATAW  (1 + DATAW + LPID_WIDTH + 1 + 1),
            .RESETW (1),
            .DEPTH  (OUT_REG)
        ) pipe_reg (
            .clk      (clk),
            .reset    (reset || (enable && eop)),
            .enable   (enable),
            .data_in  ({valid_in,  data_in[start_p], start_p, is_first_p, is_last_p}),
            .data_out ({valid_out, data_out,         pid,     sop,        eop})
        );

    end else begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (next)

        assign valid_out = valid_in;
        assign data_out = data_in[0];
        assign pid = 0;
        assign sop = 1;
        assign eop = 1;

    end

endmodule
`TRACING_ON
