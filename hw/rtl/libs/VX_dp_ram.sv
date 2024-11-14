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

`ifdef QUARTUS
    `define RAM_ARRAY_WREN  reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  for (integer i = 0; i < WRENW; ++i) begin \
                                if (wren[i]) begin \
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW]; \
                                end \
                            end
`else
    `define RAM_ARRAY_WREN  reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  for (integer i = 0; i < WRENW; ++i) begin \
                                if (wren[i]) begin \
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                                end \
                            end
`endif

`TRACING_OFF
module VX_dp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter LUTRAM      = 0,
    parameter `STRING RDW_MODE = "W", // W: write-first, R: read-first, U: undefined
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
    input wire [ADDRW-1:0]   waddr,
    input wire [DATAW-1:0]   wdata,
    input wire [ADDRW-1:0]   raddr,
    output wire [DATAW-1:0]  rdata
);
    localparam WSELW = DATAW / WRENW;
    `UNUSED_PARAM (LUTRAM)

    `STATIC_ASSERT(!(WRENW * WSELW != DATAW), ("invalid parameter"))
    `STATIC_ASSERT((RDW_MODE == "R" || RDW_MODE == "W" || RDW_MODE == "U"), ("invalid parameter"))
    `UNUSED_PARAM (RDW_ASSERT)

`ifdef SYNTHESIS
    localparam FORCE_BRAM = !LUTRAM && (SIZE * DATAW >= `MAX_LUTRAM);
    if (OUT_REG) begin : g_sync
        if (FORCE_BRAM) begin : g_bram
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    (* rw_addr_collision = "yes" *) `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                `RAM_WRITE_WREN
                            end
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end else begin : g_no_wren
                    (* rw_addr_collision = "yes" *) `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                ram[waddr] <= wdata;
                            end
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end
            end else if (RDW_MODE == "R") begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                `RAM_WRITE_WREN
                            end
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                ram[waddr] <= wdata;
                            end
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end else begin : g_undefined
                if (WRENW != 1) begin : g_wren
                    `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (write) begin
                            `RAM_WRITE_WREN
                        end
                        if (read) begin
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                        if (read) begin
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end
        end else begin : g_auto
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    (* rw_addr_collision = "yes" *) `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                `RAM_WRITE_WREN
                            end
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end else begin : g_no_wren
                    (* rw_addr_collision = "yes" *) reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                ram[waddr] <= wdata;
                            end
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end
            end else if (RDW_MODE == "R") begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                `RAM_WRITE_WREN
                            end
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (read || write) begin
                            if (write) begin
                                ram[waddr] <= wdata;
                            end
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end else begin
                if (WRENW != 1) begin : g_wren
                    `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (write) begin
                            `RAM_WRITE_WREN
                        end
                        if (read) begin
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                        if (read) begin
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end
        end
    end else begin : g_async
        `UNUSED_VAR (read)
        if (FORCE_BRAM) begin : g_bram
            if (RDW_MODE == "W") begin : g_write_first
            `ifdef VIVADO
                VX_async_ram_patch #(
                    .DATAW      (DATAW),
                    .SIZE       (SIZE),
                    .WRENW      (WRENW),
                    .DUAL_PORT  (1),
                    .INIT_ENABLE(INIT_ENABLE),
                    .INIT_FILE  (INIT_FILE),
                    .INIT_VALUE (INIT_VALUE)
                ) async_ram_patch (
                    .clk   (clk),
                    .reset (reset),
                    .read  (read),
                    .write (write),
                    .wren  (wren),
                    .waddr (waddr),
                    .wdata (wdata),
                    .raddr (raddr),
                    .rdata (rdata)
                );
            `else
                if (WRENW != 1) begin : g_wren
                    `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            `RAM_WRITE_WREN
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin : g_no_wren
                    `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata = ram[raddr];
                end
            `endif
            end else begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `NO_RW_RAM_CHECK `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            `RAM_WRITE_WREN
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin : g_no_wren
                    `NO_RW_RAM_CHECK `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata = ram[raddr];
                end
            end
        end else begin : g_auto
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            `RAM_WRITE_WREN
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin : g_no_wren
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata = ram[raddr];
                end
            end else begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `NO_RW_RAM_CHECK `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            `RAM_WRITE_WREN
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin : g_no_wren
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata = ram[raddr];
                end
            end
        end
    end
`else
    // simulation
    reg [DATAW-1:0] ram [0:SIZE-1];
    `RAM_INITIALIZATION

    always @(posedge clk) begin
        if (RESET_RAM && reset) begin
            for (integer i = 0; i < SIZE; ++i) begin
                ram[i] <= DATAW'(INIT_VALUE);
            end
        end else if (write) begin
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren[i]) begin
                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
            end
        end
    end

    if (OUT_REG) begin : g_sync
        if (RDW_MODE == "W") begin : g_write_first
            reg [ADDRW-1:0] raddr_r;
            always @(posedge clk) begin
                if (read || write) begin
                    raddr_r <= raddr;
                end
            end
            assign rdata = ram[raddr_r];
        end else if (RDW_MODE == "R") begin : g_read_first
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read || write) begin
                    rdata_r <= ram[raddr];
                end
            end
            assign rdata = rdata_r;
        end else begin : g_undefined
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (read) begin
                    rdata_r <= ram[raddr];
                end
            end
            assign rdata = rdata_r;
        end
    end else begin : g_async
        `UNUSED_VAR (read)
        if (RDW_MODE == "W") begin : g_write_first
            assign rdata = ram[raddr];
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
                    prev_data  <= ram[waddr];
                    prev_waddr <= waddr;
                end
            end

            assign rdata = (prev_write && (prev_waddr == raddr)) ? prev_data : ram[raddr];
            if (RDW_ASSERT) begin : g_rw_asert
                `RUNTIME_ASSERT(~read || (rdata == ram[raddr]), ("%t: read after write hazard", $time))
            end
        end
    end
`endif

endmodule
`TRACING_ON
