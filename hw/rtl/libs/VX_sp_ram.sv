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

`ifdef SIMULATION
    `define RAM_RESET_BLOCK if (RESET_RAM && reset) begin \
                                for (integer i = 0; i < SIZE; ++i) begin \
                                    ram[i] <= DATAW'(INIT_VALUE); \
                                end \
                            end else
`else
    `define RAM_RESET_BLOCK
`endif

`define RAM_WRITE_ALL   `RAM_RESET_BLOCK \
                        if (write) begin \
                            ram[addr] <= wdata; \
                        end

`ifdef QUARTUS
    `define RAM_ARRAY_WREN  reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  `RAM_RESET_BLOCK \
                            if (write) begin \
                                for (integer i = 0; i < WRENW; ++i) begin \
                                    if (wren[i]) begin \
                                        ram[addr][i] <= wdata[i * WSELW +: WSELW]; \
                                    end \
                                end \
                            end
`else
    `define RAM_ARRAY_WREN  reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  `RAM_RESET_BLOCK \
                            if (write) begin \
                                for (integer i = 0; i < WRENW; ++i) begin \
                                    if (wren[i]) begin \
                                        ram[addr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                                    end \
                                end \
                            end
`endif

`TRACING_OFF
module VX_sp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter LUTRAM      = 0,
    parameter `STRING RDW_MODE = "W", // W: write-first, R: read-first, N: no-change
    parameter RADDR_REG   = 0, // read address registered hint
    parameter RADDR_RESET = 0, // read address has reset
    parameter RDW_ASSERT  = 0,
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
    localparam WSELW = DATAW / WRENW;
    `UNUSED_PARAM (LUTRAM)
    `UNUSED_PARAM (RADDR_REG)
    `UNUSED_PARAM (RADDR_RESET)

    `STATIC_ASSERT(!(WRENW * WSELW != DATAW), ("invalid parameter"))
    `STATIC_ASSERT((RDW_MODE == "R" || RDW_MODE == "W" || RDW_MODE == "N"), ("invalid parameter"))
    `UNUSED_PARAM (RDW_ASSERT)

`ifdef SYNTHESIS
    localparam FORCE_BRAM = !LUTRAM && `FORCE_BRAM(SIZE, DATAW);
    if (OUT_REG) begin : g_sync
        if (FORCE_BRAM) begin : g_bram
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RW_RAM_CHECK `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] addr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            addr_r <= addr;
                        end
                    end
                    assign rdata = ram[addr_r];
                end else begin : g_no_wren
                    `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
                            if (write) begin
                                rdata_r <= wdata;
                            end else begin
                                rdata_r <= ram[addr];
                            end
                        end
                    end
                    assign rdata = rdata_r;
                end
            end else if (RDW_MODE == "R") begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end else if (RDW_MODE == "N") begin : g_no_change
                if (WRENW != 1) begin : g_wren
                    `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        else if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        else if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end
        end else begin : g_auto
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] addr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            addr_r <= addr;
                        end
                    end
                    assign rdata = ram[addr_r];
                end else begin : g_no_wren
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
                            if (write) begin
                                rdata_r <= wdata;
                            end else begin
                                rdata_r <= ram[addr];
                            end
                        end
                    end
                    assign rdata = rdata_r;
                end
            end else if (RDW_MODE == "R") begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end else if (RDW_MODE == "N") begin : g_no_change
                if (WRENW != 1) begin : g_wren
                    `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        else if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        else if (read) begin
                            rdata_r <= ram[addr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end
        end
    end else begin : g_async
        `UNUSED_VAR (read)
        if (FORCE_BRAM) begin : g_bram
        `ifdef ASYNC_BRAM_PATCH
            VX_async_ram_patch #(
                .DATAW      (DATAW),
                .SIZE       (SIZE),
                .WRENW      (WRENW),
                .DUAL_PORT  (0),
                .FORCE_BRAM (FORCE_BRAM),
                .RADDR_REG  (RADDR_REG),
                .RADDR_RESET(RADDR_RESET),
                .WRITE_FIRST(RDW_MODE == "W"),
                .INIT_ENABLE(INIT_ENABLE),
                .INIT_FILE  (INIT_FILE),
                .INIT_VALUE (INIT_VALUE)
            ) async_ram_patch (
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
        `else
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RW_RAM_CHECK `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                    end
                    assign rdata = ram[addr];
                end else begin : g_no_wren
                    `RW_RAM_CHECK `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                    end
                    assign rdata = ram[addr];
                end
            end else begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `NO_RW_RAM_CHECK `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                    end
                    assign rdata = ram[addr];
                end else begin : g_no_wren
                    `NO_RW_RAM_CHECK `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                    end
                    assign rdata = ram[addr];
                end
            end
        `endif
        end else begin : g_auto
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RW_RAM_CHECK `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                    end
                    assign rdata = ram[addr];
                end else begin : g_no_wren
                    `RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                    end
                    assign rdata = ram[addr];
                end
            end else begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `NO_RW_RAM_CHECK `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                    end
                    assign rdata = ram[addr];
                end else begin : g_no_wren
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                    end
                    assign rdata = ram[addr];
                end
            end
        end
    end
`else
    // simulation
    reg [DATAW-1:0] ram [0:SIZE-1];
    `RAM_INITIALIZATION

    always @(posedge clk) begin
        `RAM_WRITE_WREN
    end

    if (OUT_REG) begin : g_sync
        if (RDW_MODE == "W") begin : g_write_first
            reg [ADDRW-1:0] addr_r;
            always @(posedge clk) begin
                if (read) begin
                    addr_r <= addr;
                end
            end
            assign rdata = ram[addr_r];
        end else if (RDW_MODE == "R") begin : g_read_first
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read) begin
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end else if (RDW_MODE == "N") begin : g_no_change
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read && ~write) begin
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end
    end else begin : g_async
        `UNUSED_VAR (read)
        if (RDW_MODE == "W") begin : g_write_first
            assign rdata = ram[addr];
        end else begin : g_read_first
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
            if (RDW_ASSERT) begin : g_rw_asert
                `RUNTIME_ASSERT(~read || (rdata == ram[addr]), ("%t: read after write hazard", $time))
            end
        end
    end
`endif

endmodule
`TRACING_ON
