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
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter NO_RWCHECK  = 0,
    parameter RW_ASSERT   = 0,
    parameter RESET_RAM   = 0,
    parameter `STRING WRITE_MODE = "R", // R: read-first, W: write-first, N: no-change
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
    localparam WSELW = DATAW / WRENW;

    `STATIC_ASSERT(!(WRENW * WSELW != DATAW), ("invalid parameter"))
    `UNUSED_PARAM (RW_ASSERT)

`define RAM_INITIALIZATION \
    if (INIT_ENABLE != 0) begin : g_init \
        if (INIT_FILE != "") begin : g_file \
            initial $readmemh(INIT_FILE, ram); \
        end else begin : g_value \
            initial begin \
                for (integer i = 0; i < SIZE; ++i) begin : g_i \
                    ram[i] = INIT_VALUE; \
                end \
            end \
        end \
    end

`ifdef SYNTHESIS
`ifdef QUARTUS
    `define RAM_ARRAY   reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE   for (integer i = 0; i < WRENW; ++i) begin \
                            if (wren[i]) begin \
                                ram[addr][i] <= wdata[i * WSELW +: WSELW]; \
                            end \
                        end
`else
    `define RAM_ARRAY   reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE   for (integer i = 0; i < WRENW; ++i) begin \
                            if (wren[i]) begin \
                                ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                            end \
                        end
`endif
    if (OUT_REG) begin : g_sync
        wire cs = read || write;
        if (WRITE_MODE == "R") begin : g_read_first
            `RAM_ARRAY
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (cs) begin
                    if (write) begin
                        `RAM_WRITE
                    end
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end else if (WRITE_MODE == "W") begin : g_write_first
            `UNUSED_VAR (wren)
            `RAM_ARRAY
            `RAM_INITIALIZATION
            reg [ADDRW-1:0] addr_reg;
            always @(posedge clk) begin
                if (cs) begin
                    addr_reg <= addr;
                    if (write) begin
                        `RAM_WRITE
                    end
                end
            end
            assign rdata = ram[addr_reg];
        end else if (WRITE_MODE == "N") begin : g_no_change
            `RAM_ARRAY
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (cs) begin
                    if (write) begin
                        `RAM_WRITE
                    end else begin
                        rdata_r <= ram[addr];
                    end
                end
            end
            assign rdata = rdata_r;
        end
    end else begin : g_async
        if (NO_RWCHECK) begin : g_no_rwcehck
            `NO_RW_RAM_CHECK `RAM_ARRAY
            `RAM_INITIALIZATION
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE
                end
            end
            assign rdata = ram[addr];
        end else begin : g_rwcheck
            `RAM_ARRAY
            `RAM_INITIALIZATION
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE
                end
            end
            assign rdata = ram[addr];
        end
    end
`else
    // simulation
    reg [DATAW-1:0] ram [0:SIZE-1];
    `RAM_INITIALIZATION

    wire [DATAW-1:0] ram_n;
    for (genvar i = 0; i < WRENW; ++i) begin : g_ram_n
        assign ram_n[i * WSELW +: WSELW] = wren[i] ? wdata[i * WSELW +: WSELW] : ram[addr][i * WSELW +: WSELW];
    end

    always @(posedge clk) begin
        if (RESET_RAM && reset) begin
            for (integer i = 0; i < SIZE; ++i) begin
                ram[i] <= DATAW'(INIT_VALUE);
            end
        end else if (write) begin
            ram[addr] <= ram_n;
        end
    end

    if (OUT_REG) begin : g_sync
        if (WRITE_MODE == "R") begin : g_read_first
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read || write) begin
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end else if (WRITE_MODE == "W") begin : g_write_first
            reg [ADDRW-1:0] addr_reg;
            always @(posedge clk) begin
                if (read || write) begin
                    addr_reg <= addr;
                end
            end
            assign rdata = ram[addr_reg];
        end else if (WRITE_MODE == "N") begin : g_no_change
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read && ~write) begin
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end
    end else begin : g_async
        if (NO_RWCHECK) begin : g_no_rwcheck
            reg [DATAW-1:0] prev_data;
            reg [ADDRW-1:0] prev_waddr;
            reg prev_write;
            always @(posedge clk) begin
                if (reset) begin
                    prev_write <= 0;
                    prev_data  <= '0;
                    prev_waddr <= '0;
                end else begin
                    prev_write <= write;
                    prev_data  <= ram[addr];
                    prev_waddr <= addr;
                end
            end
            assign rdata = (prev_write && (prev_waddr == addr)) ? prev_data : ram[addr];
            if (RW_ASSERT) begin : g_rw_asert
                `RUNTIME_ASSERT(~read || (rdata == ram[addr]), ("%t: read after write hazard", $time))
            end
        end else begin : g_rwcheck
            `UNUSED_VAR (read)
            assign rdata = ram[addr];
        end
    end
`endif

endmodule
`TRACING_ON
