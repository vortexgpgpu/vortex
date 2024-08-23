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
module VX_dp_ram_rst #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter ADDR_MIN    = 0,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter NO_RWCHECK  = 0,
    parameter LUTRAM      = 0,
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
    input wire [ADDRW-1:0]   waddr,
    input wire [DATAW-1:0]   wdata,
    input wire [ADDRW-1:0]   raddr,
    output wire [DATAW-1:0]  rdata
);
    localparam WSELW = DATAW / WRENW;
    `STATIC_ASSERT((WRENW * WSELW == DATAW), ("invalid parameter"))

`define RAM_INITIALIZATION                         \
    if (INIT_ENABLE != 0) begin                    \
        if (INIT_FILE != "") begin                 \
            initial $readmemh(INIT_FILE, ram);     \
        end else begin                             \
            initial                                \
                for (integer i = 0; i < SIZE; ++i) \
                    ram[i] = INIT_VALUE;           \
        end                                        \
    end

    `UNUSED_VAR (read)

    // RAM emulation
    reg [DATAW-1:0] ram [ADDR_MIN:SIZE-1];
    `RAM_INITIALIZATION

    wire [DATAW-1:0] ram_n;
    for (genvar i = 0; i < WRENW; ++i) begin
        assign ram_n[i * WSELW +: WSELW] = ((WRENW == 1) | wren[i]) ? wdata[i * WSELW +: WSELW] : ram[waddr][i * WSELW +: WSELW];
    end

    if (OUT_REG != 0) begin
        reg [DATAW-1:0] rdata_r;
        always @(posedge clk) begin
            if (reset) begin
                for (integer i = 0; i < SIZE; ++i) begin
                    ram[i] <= DATAW'(INIT_VALUE);
                end
                rdata_r <= '0;
            end else begin
                if (write) begin
                    ram[waddr] <= ram_n;
                end
                if (read) begin
                    rdata_r <= ram[raddr];
                end
            end
        end
        assign rdata = rdata_r;
    end else begin
        reg [DATAW-1:0] prev_data;
        reg [ADDRW-1:0] prev_waddr;
        reg prev_write;
        always @(posedge clk) begin
            if (reset) begin
                for (integer i = 0; i < SIZE; ++i) begin
                    ram[i] <= DATAW'(INIT_VALUE);
                end
                prev_write <= 0;
                prev_data  <= '0;
                prev_waddr <= '0;
            end else begin
                if (write) begin
                    ram[waddr] <= ram_n;
                end
                prev_write <= (| wren);
                prev_data  <= ram[waddr];
                prev_waddr <= waddr;
            end
        end
        if (LUTRAM || !NO_RWCHECK) begin
            `UNUSED_VAR (prev_write)
            `UNUSED_VAR (prev_data)
            `UNUSED_VAR (prev_waddr)
            assign rdata = ram[raddr];
        end else begin
            assign rdata = (prev_write && (prev_waddr == raddr)) ? prev_data : ram[raddr];
        end
    end

endmodule
`TRACING_ON
