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

`ifndef ASIC
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
`else
    `define RAM_INITIALIZATION
`endif

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
                        if (write_a) begin \
                            ram[addr_a] <= wdata_a; \
                        end

`ifdef QUARTUS
    `define RAM_ARRAY_WREN  reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  `RAM_RESET_BLOCK \
                            if (write_a) begin \
                                for (integer i = 0; i < WRENW; ++i) begin \
                                    if (wren_a[i]) begin \
                                        ram[addr_a][i] <= wdata_a[i * WSELW +: WSELW]; \
                                    end \
                                end \
                            end
`else
    `define RAM_ARRAY_WREN  reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  `RAM_RESET_BLOCK \
                            if (write_a) begin \
                                for (integer i = 0; i < WRENW; ++i) begin \
                                    if (wren_a[i]) begin \
                                        ram[addr_a][i * WSELW +: WSELW] <= wdata_a[i * WSELW +: WSELW]; \
                                    end \
                                end \
                            end
`endif

// A RAM with one read/write port (A) and one read-only port (B), both with a
// registered read: the two ports of a true dual-port block RAM. Port A reads
// before it writes. A port B read of the word port A writes in the same cycle
// returns an undefined value on block RAM; the caller must not issue one.
//
// The ASIC libraries have no read/write + read macro, so on ASIC the RAM is two
// VX_dp_ram copies, one per read port, that both take every write: each maps
// onto the VX_dp_ram_asic macro the synthesis flows and their SRAM cost model
// already know.
`TRACING_OFF
module VX_1rw1r_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter LUTRAM      = 0,
    parameter RESET_RAM   = 0,
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0,
    parameter ADDRW       = `LOG2UP(SIZE)
) (
    input wire               clk,
    input wire               reset,

    input wire               read_a,
    input wire               write_a,
    input wire [WRENW-1:0]   wren_a,
    input wire [ADDRW-1:0]   addr_a,
    input wire [DATAW-1:0]   wdata_a,
    output wire [DATAW-1:0]  rdata_a,

    input wire               read_b,
    input wire [ADDRW-1:0]   addr_b,
    output wire [DATAW-1:0]  rdata_b
);
    localparam WSELW = DATAW / WRENW;

    `STATIC_ASSERT(!(WRENW * WSELW != DATAW), ("invalid parameter"))

`ifdef SYNTHESIS
`ifdef ASIC
    `UNUSED_PARAM (LUTRAM)
    VX_dp_ram #(
        .DATAW       (DATAW),
        .SIZE        (SIZE),
        .WRENW       (WRENW),
        .OUT_REG     (1),
        .RDW_MODE    ("R"),
        .RESET_RAM   (RESET_RAM),
        .INIT_ENABLE (INIT_ENABLE),
        .INIT_FILE   (INIT_FILE),
        .INIT_VALUE  (INIT_VALUE)
    ) ram_a (
        .clk   (clk),
        .reset (reset),
        .read  (read_a),
        .write (write_a),
        .wren  (wren_a),
        .waddr (addr_a),
        .wdata (wdata_a),
        .raddr (addr_a),
        .rdata (rdata_a)
    );
    VX_dp_ram #(
        .DATAW       (DATAW),
        .SIZE        (SIZE),
        .WRENW       (WRENW),
        .OUT_REG     (1),
        .RDW_MODE    ("R"),
        .RESET_RAM   (RESET_RAM),
        .INIT_ENABLE (INIT_ENABLE),
        .INIT_FILE   (INIT_FILE),
        .INIT_VALUE  (INIT_VALUE)
    ) ram_b (
        .clk   (clk),
        .reset (reset),
        .read  (read_b),
        .write (write_a),
        .wren  (wren_a),
        .waddr (addr_a),
        .wdata (wdata_a),
        .raddr (addr_b),
        .rdata (rdata_b)
    );
`else
    localparam FORCE_BRAM = !LUTRAM && `FORCE_BRAM(SIZE, DATAW);
    if (FORCE_BRAM) begin : g_bram
        if (WRENW != 1) begin : g_wren
            `NO_RW_RAM_CHECK `USE_BLOCK_BRAM `RAM_ARRAY_WREN
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_a_r;
            reg [DATAW-1:0] rdata_b_r;
            always @(posedge clk) begin
                `RAM_WRITE_WREN
                if (read_a) begin
                    rdata_a_r <= ram[addr_a];
                end
            end
            always @(posedge clk) begin
                if (read_b) begin
                    rdata_b_r <= ram[addr_b];
                end
            end
            assign rdata_a = rdata_a_r;
            assign rdata_b = rdata_b_r;
        end else begin : g_no_wren
            `UNUSED_VAR (wren_a)
            `NO_RW_RAM_CHECK `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_a_r;
            reg [DATAW-1:0] rdata_b_r;
            always @(posedge clk) begin
                `RAM_WRITE_ALL
                if (read_a) begin
                    rdata_a_r <= ram[addr_a];
                end
            end
            always @(posedge clk) begin
                if (read_b) begin
                    rdata_b_r <= ram[addr_b];
                end
            end
            assign rdata_a = rdata_a_r;
            assign rdata_b = rdata_b_r;
        end
    end else if (LUTRAM) begin : g_lutram
        if (WRENW != 1) begin : g_wren
            `NO_RW_RAM_CHECK `USE_FAST_BRAM `RAM_ARRAY_WREN
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_a_r;
            reg [DATAW-1:0] rdata_b_r;
            always @(posedge clk) begin
                `RAM_WRITE_WREN
                if (read_a) begin
                    rdata_a_r <= ram[addr_a];
                end
            end
            always @(posedge clk) begin
                if (read_b) begin
                    rdata_b_r <= ram[addr_b];
                end
            end
            assign rdata_a = rdata_a_r;
            assign rdata_b = rdata_b_r;
        end else begin : g_no_wren
            `UNUSED_VAR (wren_a)
            `NO_RW_RAM_CHECK `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_a_r;
            reg [DATAW-1:0] rdata_b_r;
            always @(posedge clk) begin
                `RAM_WRITE_ALL
                if (read_a) begin
                    rdata_a_r <= ram[addr_a];
                end
            end
            always @(posedge clk) begin
                if (read_b) begin
                    rdata_b_r <= ram[addr_b];
                end
            end
            assign rdata_a = rdata_a_r;
            assign rdata_b = rdata_b_r;
        end
    end else begin : g_auto
        if (WRENW != 1) begin : g_wren
            `NO_RW_RAM_CHECK `RAM_ARRAY_WREN
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_a_r;
            reg [DATAW-1:0] rdata_b_r;
            always @(posedge clk) begin
                `RAM_WRITE_WREN
                if (read_a) begin
                    rdata_a_r <= ram[addr_a];
                end
            end
            always @(posedge clk) begin
                if (read_b) begin
                    rdata_b_r <= ram[addr_b];
                end
            end
            assign rdata_a = rdata_a_r;
            assign rdata_b = rdata_b_r;
        end else begin : g_no_wren
            `UNUSED_VAR (wren_a)
            `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
            `RAM_INITIALIZATION
            reg [DATAW-1:0] rdata_a_r;
            reg [DATAW-1:0] rdata_b_r;
            always @(posedge clk) begin
                `RAM_WRITE_ALL
                if (read_a) begin
                    rdata_a_r <= ram[addr_a];
                end
            end
            always @(posedge clk) begin
                if (read_b) begin
                    rdata_b_r <= ram[addr_b];
                end
            end
            assign rdata_a = rdata_a_r;
            assign rdata_b = rdata_b_r;
        end
    end
`endif
`else
    // simulation
    `UNUSED_PARAM (LUTRAM)
    reg [DATAW-1:0] ram [0:SIZE-1];
    `RAM_INITIALIZATION

    if (WRENW != 1) begin : g_wren
        reg [DATAW-1:0] wdata_n;
        always @(*) begin
            wdata_n = ram[addr_a];
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren_a[i]) begin
                    wdata_n[i * WSELW +: WSELW] = wdata_a[i * WSELW +: WSELW];
                end
            end
        end
        always @(posedge clk) begin
            `RAM_RESET_BLOCK
            if (write_a) begin
                ram[addr_a] <= wdata_n;
            end
        end
    end else begin : g_no_wren
        `UNUSED_VAR (wren_a)
        always @(posedge clk) begin
            `RAM_WRITE_ALL
        end
    end

    reg [DATAW-1:0] rdata_a_r;
    reg [DATAW-1:0] rdata_b_r;
    always @(posedge clk) begin
        if (read_a) begin
            rdata_a_r <= ram[addr_a];
        end
        if (read_b) begin
            rdata_b_r <= ram[addr_b];
        end
    end
    assign rdata_a = rdata_a_r;
    assign rdata_b = rdata_b_r;
`endif

endmodule
`TRACING_ON
