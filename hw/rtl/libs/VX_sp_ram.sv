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
module VX_sp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter ADDR_MIN    = 0,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter NO_RWCHECK  = 0,
    parameter RW_ASSERT   = 0,
    parameter LUTRAM      = 0,
    parameter RESET_RAM   = 0,
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0,
    parameter ADDRW       = `LOG2UP(SIZE)
) (
    input wire               clk,
    input wire               reset,
    input wire               read,
    input wire               write,
    input wire [WRENW-1:0]   wren,
    input wire [ADDRW-1:0]   addr,
    input wire [DATAW-1:0]   wdata,
    output wire [DATAW-1:0]  rdata
);
    VX_dp_ram #(
        .DATAW (DATAW),
        .SIZE (SIZE),
        .ADDR_MIN (ADDR_MIN),
        .WRENW (WRENW),
        .OUT_REG (OUT_REG),
        .NO_RWCHECK (NO_RWCHECK),
        .RW_ASSERT (RW_ASSERT),
        .LUTRAM (LUTRAM),
        .RESET_RAM (RESET_RAM),
        .INIT_ENABLE (INIT_ENABLE),
        .INIT_FILE (INIT_FILE),
        .INIT_VALUE (INIT_VALUE),
        .ADDRW (ADDRW)
    ) dp_ram (
        .clk   (clk),
        .reset (reset),
        .read  (read),
        .write (write),
        .wren  (wren),
        .waddr (addr),
        .wdata (wdata),
        .raddr (addr),
        .rdata (rdata)
    );

endmodule
`TRACING_ON
