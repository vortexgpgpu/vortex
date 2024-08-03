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
module VX_index_buffer #(
    parameter DATAW  = 1,
    parameter SIZE   = 1,
    parameter LUTRAM = 1,
    parameter ADDRW  = `LOG2UP(SIZE)
) (
    input  wire             clk,
    input  wire             reset,

    output wire [ADDRW-1:0] write_addr,
    input  wire [DATAW-1:0] write_data,
    input  wire             acquire_en,

    input  wire [ADDRW-1:0] read_addr,
    output wire [DATAW-1:0] read_data,
    input  wire             release_en,

    output wire             empty,
    output wire             full
);

    VX_allocator #(
        .SIZE (SIZE)
    ) allocator (
        .clk        (clk),
        .reset      (reset),
        .acquire_en (acquire_en),
        .acquire_addr (write_addr),
        .release_en (release_en),
        .release_addr (read_addr),
        .empty      (empty),
        .full       (full)
    );

    VX_dp_ram #(
        .DATAW  (DATAW),
        .SIZE   (SIZE),
        .LUTRAM (LUTRAM)
    ) data_table (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (acquire_en),
        .wren  (1'b1),
        .waddr (write_addr),
        .wdata (write_data),
        .raddr (read_addr),
        .rdata (read_data)
    );

endmodule
`TRACING_ON
