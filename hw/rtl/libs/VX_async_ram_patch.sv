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

`define SYNC_RAM_WF_BLOCK(__d, __re, __we, __ra, __wa) \
    `RAM_ATTRIBUTES `RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    reg [ADDRW-1:0] raddr_r; \
    always @(posedge clk) begin \
        if (__re || __we) begin \
            if (__we) begin \
                ram[__wa] <= wdata; \
            end \
            raddr_r <= __ra; \
        end \
    end \
    assign __d = ram[raddr_r]

`define SYNC_RAM_WF_WREN_BLOCK(__d, __re, __we, __ra, __wa) \
    `RAM_ATTRIBUTES `RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    reg [ADDRW-1:0] raddr_r; \
    always @(posedge clk) begin \
        if (__re || __we) begin \
            if (__we) begin \
                for (integer i = 0; i < WRENW; ++i) begin \
                    if (wren[i]) begin \
                        ram[__wa][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                    end \
                end \
            end \
            raddr_r <= __ra; \
        end \
    end \
    assign __d = ram[raddr_r]

`define SYNC_RAM_RF_BLOCK(__d, __re, __we, __ra, __wa) \
    `RAM_ATTRIBUTES reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    reg [DATAW-1:0] rdata_r; \
    always @(posedge clk) begin \
        if (__re || __we) begin \
            if (__we) begin \
                ram[__wa] <= wdata; \
            end \
            rdata_r <= ram[__ra]; \
        end \
    end \
    assign __d = rdata_r

`define SYNC_RAM_RF_WREN_BLOCK(__d, __re, __we, __ra, __wa) \
    `RAM_ATTRIBUTES reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    reg [DATAW-1:0] rdata_r; \
    always @(posedge clk) begin \
        if (__re || __we) begin \
            if (__we) begin \
                for (integer i = 0; i < WRENW; ++i) begin \
                    if (wren[i]) begin \
                        ram[__wa][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                    end \
                end \
            end \
            rdata_r <= ram[__ra]; \
        end \
    end \
    assign __d = rdata_r

`define ASYNC_RAM_BLOCK(__d, __we, __ra, __wa) \
    `RAM_ATTRIBUTES reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    always @(posedge clk) begin \
        if (__we) begin \
            ram[__wa] <= wdata; \
        end \
    end \
    assign __d = ram[__ra]

`define ASYNC_RAM_BLOCK_WREN(__d, __we, __ra, __wa) \
    `RAM_ATTRIBUTES reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    always @(posedge clk) begin \
        if (__we) begin \
            for (integer i = 0; i < WRENW; ++i) begin \
                if (wren[i]) begin \
                    ram[__wa][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                end \
            end \
        end \
    end \
    assign __d = ram[__ra]

`TRACING_OFF
module VX_async_ram_patch #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter DUAL_PORT   = 0,
    parameter FORCE_BRAM  = 0,
    parameter RADDR_REG   = 0, // read address registered hint
    parameter WRITE_FIRST = 0,
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

    `UNUSED_VAR (reset)

    (* keep = "true" *) wire [ADDRW-1:0] raddr_w, raddr_s;
    (* keep = "true" *) wire read_s, is_raddr_reg;

    assign raddr_w = raddr;

    VX_placeholder #(
        .I (ADDRW),
        .O (ADDRW + 1 + 1)
    ) placeholder (
        .in  (raddr_w),
        .out ({raddr_s, read_s, is_raddr_reg})
    );

    wire [DATAW-1:0] rdata_s;

    if (1) begin : g_sync_ram
        if (WRENW != 1) begin : g_wren
            if (FORCE_BRAM) begin : g_bram
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES `USE_BLOCK_BRAM
                    `SYNC_RAM_WF_WREN_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES `USE_BLOCK_BRAM
                    `SYNC_RAM_RF_WREN_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end else begin : g_lutram
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES
                    `SYNC_RAM_WF_WREN_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES
                    `SYNC_RAM_RF_WREN_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end
        end else begin : g_no_wren
            if (FORCE_BRAM) begin : g_bram
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES `USE_BLOCK_BRAM
                    `SYNC_RAM_WF_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES `USE_BLOCK_BRAM
                    `SYNC_RAM_RF_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end else begin : g_lutram
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES
                    `SYNC_RAM_WF_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES
                    `SYNC_RAM_RF_BLOCK(rdata_s, read_s, write, raddr_s, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end
        end
    end

    if (RADDR_REG) begin : g_raddr_reg
        `UNUSED_VAR (is_raddr_reg)
        assign rdata = rdata_s;
    end else begin : g_async_ram
        wire [DATAW-1:0] rdata_a;
        if (DUAL_PORT) begin : g_dp
            if (WRENW != 1) begin : g_wren
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES `RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK_WREN(rdata_a, write, raddr, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES `NO_RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK_WREN(rdata_a, write, raddr, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end else begin : g_no_wren
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES `RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK(rdata_a, write, raddr, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES `NO_RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK(rdata_a, write, raddr, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end
        end else begin : g_sp
            if (WRENW != 1) begin : g_wren
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES `RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK_WREN(rdata_a, write, waddr, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES `NO_RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK_WREN(rdata_a, write, waddr, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end else begin : g_no_wren
                if (WRITE_FIRST) begin : g_write_first
                    `define RAM_ATTRIBUTES `RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK(rdata_a, write, waddr, waddr);
                    `undef RAM_ATTRIBUTES
                end else begin : g_read_first
                    `define RAM_ATTRIBUTES `NO_RW_RAM_CHECK
                    `ASYNC_RAM_BLOCK(rdata_a, write, waddr, waddr);
                    `undef RAM_ATTRIBUTES
                end
            end
        end
        assign rdata = is_raddr_reg ? rdata_s : rdata_a;
    end

endmodule
`TRACING_ON
