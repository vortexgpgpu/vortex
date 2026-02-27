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

module VX_dxa_nb_line_cache #(
    parameter GMEM_ADDR_WIDTH = 26,
    parameter GMEM_DATAW      = 512,
    parameter GMEM_OFF_BITS   = 6
) (
    input  wire clk,
    input  wire reset,

    input  wire                        update_fire,
    input  wire [GMEM_ADDR_WIDTH-1:0]  update_line_addr,
    input  wire [GMEM_DATAW-1:0]       update_line_data,

    input  wire                        query_valid,
    input  wire [GMEM_ADDR_WIDTH-1:0]  query_line_addr,
    input  wire [GMEM_OFF_BITS-1:0]    query_off,
    output wire                        query_hit,
    output wire [63:0]                 query_elem_data
);
    reg                       line_valid_r;
    reg [GMEM_ADDR_WIDTH-1:0] line_addr_r;
    reg [GMEM_DATAW-1:0]      line_data_r;

    wire [31:0] query_shift_w = 32'(query_off) * 32'd8;

    assign query_hit       = query_valid && line_valid_r && (line_addr_r == query_line_addr);
    assign query_elem_data = 64'(GMEM_DATAW'(line_data_r) >> query_shift_w);

    always @(posedge clk) begin
        if (reset) begin
            line_valid_r <= 1'b0;
            line_addr_r  <= '0;
            line_data_r  <= '0;
        end else if (update_fire) begin
            line_valid_r <= 1'b1;
            line_addr_r  <= update_line_addr;
            line_data_r  <= update_line_data;
        end
    end
endmodule

