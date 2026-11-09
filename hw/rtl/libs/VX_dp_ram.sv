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
                        if (write) begin \
                            ram[waddr] <= wdata; \
                        end

`ifdef QUARTUS
    `define RAM_ARRAY_WREN  reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  `RAM_RESET_BLOCK \
                            if (write) begin \
                                for (integer i = 0; i < WRENW; ++i) begin \
                                    if (wren[i]) begin \
                                        ram[waddr][i] <= wdata[i * WSELW +: WSELW]; \
                                    end \
                                end \
                            end
`else
    `define RAM_ARRAY_WREN  reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE_WREN  `RAM_RESET_BLOCK \
                            if (write) begin \
                                for (integer i = 0; i < WRENW; ++i) begin \
                                    if (wren[i]) begin \
                                        ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                                    end \
                                end \
                            end
`endif

// Read forms. The form fixes the behaviour; size and LUTRAM alone pick block or
// distributed RAM.
//  OUT_REG=1, RDW_MODE "W": registered address. raddr is the address's next
//    value; rdata = ram[raddr captured on read], seeing a write on the same edge.
//  OUT_REG=1, RDW_MODE "R": registered data. rdata = ram[raddr] captured on
//    read, not seeing a write on the same edge.
//  OUT_REG=0: combinational, rdata = ram[raddr]. Never block RAM by itself: the
//    synthesis flows' RAM fold (hw/scripts/ram_fold.py) turns a block-RAM-sized
//    one into the registered-address form when its address comes from registers.
`TRACING_OFF
module VX_dp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter LUTRAM      = 0,
    parameter `STRING RDW_MODE = "W", // W: write-first, R: read-first
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
    `STATIC_ASSERT((RDW_MODE == "R" || RDW_MODE == "W"), ("invalid parameter"))

`ifdef SYNTHESIS
    localparam FORCE_BRAM = !LUTRAM && `FORCE_BRAM(SIZE, DATAW);
`ifdef ASIC
    // The macro reads synchronously, so a combinational read stays in logic.
    if (FORCE_BRAM && OUT_REG != 0) begin : g_asic
        `UNUSED_VAR (reset)
        wire [WRENW-1:0] wr_lanes = (WRENW != 1) ? wren : {WRENW{1'b1}};
        wire [DATAW-1:0] macro_rdata;
        VX_dp_ram_asic #(
            .DATAW (DATAW),
            .SIZE  (SIZE),
            .WRENW (WRENW)
        ) dp_ram_asic (
            .clk   (clk),
            .reset (1'b0),
            .read  (read),
            .write (write),
            .wren  (wr_lanes),
            .waddr (waddr),
            .wdata (wdata),
            .raddr (raddr),
            .rdata (macro_rdata)
        );
        if (RDW_MODE == "W") begin : g_write_first
            // The macro reads before it writes and then holds its output, while
            // this form reads the captured address continuously: the lanes written
            // at that address since the read are forwarded around the macro.
            reg [ADDRW-1:0] raddr_r;
            reg [WRENW-1:0] fwd_en_r;
            reg [DATAW-1:0] fwd_data_r;
            wire [ADDRW-1:0] fwd_addr = read ? raddr : raddr_r;
            wire [WRENW-1:0] fwd_lanes = {WRENW{write && waddr == fwd_addr}} & wr_lanes;
            always @(posedge clk) begin
                if (read) begin
                    raddr_r <= raddr;
                end
                fwd_en_r <= (read ? '0 : fwd_en_r) | fwd_lanes;
                for (integer i = 0; i < WRENW; ++i) begin
                    if (fwd_lanes[i]) begin
                        fwd_data_r[i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                    end
                end
            end
            for (genvar i = 0; i < WRENW; ++i) begin : g_lanes
                assign rdata[i * WSELW +: WSELW] = fwd_en_r[i] ? fwd_data_r[i * WSELW +: WSELW] : macro_rdata[i * WSELW +: WSELW];
            end
        end else begin : g_read_first
            assign rdata = macro_rdata;
        end
    end else begin : g_no_asic
`else
    if (1) begin : g_no_asic
`endif
    if (OUT_REG != 0) begin : g_sync
        if (FORCE_BRAM) begin : g_bram
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RW_RAM_CHECK `USE_BLOCK_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end else begin : g_no_wren
                    `RW_RAM_CHECK `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
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
                        `RAM_WRITE_WREN
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
                        `RAM_WRITE_ALL
                        if (read) begin
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end
            end
        end else if (LUTRAM) begin : g_lutram
            // Force distributed RAM so a small array places in fabric next to
            // its logic, instead of inferring (and shattering across) block-RAM
            // columns whose long routes dominate the path delay.
            if (RDW_MODE == "W") begin : g_write_first
                if (WRENW != 1) begin : g_wren
                    `RW_RAM_CHECK `USE_FAST_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end else begin : g_no_wren
                    `RW_RAM_CHECK `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end
            end else if (RDW_MODE == "R") begin : g_read_first
                if (WRENW != 1) begin : g_wren
                    `USE_FAST_BRAM `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            rdata_r <= ram[raddr];
                        end
                    end
                    assign rdata = rdata_r;
                end else begin : g_no_wren
                    `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [DATAW-1:0] rdata_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
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
                    `RW_RAM_CHECK `RAM_ARRAY_WREN
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_WREN
                        if (read) begin
                            raddr_r <= raddr;
                        end
                    end
                    assign rdata = ram[raddr_r];
                end else begin : g_no_wren
                    `RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    reg [ADDRW-1:0] raddr_r;
                    always @(posedge clk) begin
                        `RAM_WRITE_ALL
                        if (read) begin
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
                        `RAM_WRITE_WREN
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
                        `RAM_WRITE_ALL
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
        if (RDW_MODE == "W") begin : g_write_first
            if (WRENW != 1) begin : g_wren
                `RW_RAM_CHECK `RAM_ARRAY_WREN
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    `RAM_WRITE_WREN
                end
                assign rdata = ram[raddr];
            end else begin : g_no_wren
                `RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    `RAM_WRITE_ALL
                end
                assign rdata = ram[raddr];
            end
        end else begin : g_read_first
            if (WRENW != 1) begin : g_wren
                `NO_RW_RAM_CHECK `RAM_ARRAY_WREN
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    `RAM_WRITE_WREN
                end
                assign rdata = ram[raddr];
            end else begin : g_no_wren
                `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    `RAM_WRITE_ALL
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

    if (WRENW != 1) begin : g_wren
        reg [DATAW-1:0] wdata_n;
        always @(*) begin
            wdata_n = ram[waddr];
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren[i]) begin
                    wdata_n[i * WSELW +: WSELW] = wdata[i * WSELW +: WSELW];
                end
            end
        end
        always @(posedge clk) begin
            `RAM_RESET_BLOCK
            if (write) begin
                ram[waddr] <= wdata_n;
            end
        end
    end else begin : g_no_wren
        `UNUSED_VAR (wren)
        always @(posedge clk) begin
            `RAM_WRITE_ALL
        end
    end

    if (OUT_REG != 0) begin : g_sync
        if (RDW_MODE == "W") begin : g_write_first
            reg [ADDRW-1:0] raddr_r;
            always @(posedge clk) begin
                if (read) begin
                    raddr_r <= raddr;
                end
            end
            assign rdata = ram[raddr_r];
        end else if (RDW_MODE == "R") begin : g_read_first
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
        assign rdata = ram[raddr];
    end
`endif

endmodule
`TRACING_ON
