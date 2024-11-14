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

`define RAM_WRITE_WREN  for (integer i = 0; i < WRENW; ++i) begin \
                            if (wren[i]) begin \
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                            end \
                        end

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

`define RAM_BYPASS(__d) \
    reg [DATAW-1:0] bypass_data_r; \
    reg bypass_valid_r; \
    always @(posedge clk) begin \
        bypass_valid_r <= read_s && write && (raddr_s == waddr); \
        bypass_data_r <= wdata; \
    end \
    assign __d = bypass_valid_r ? bypass_data_r : rdata_r

`TRACING_OFF
module VX_async_ram_patch #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter DUAL_PORT   = 0,
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

    // synchroneous ram

    wire [DATAW-1:0] rdata_s;

    if (WRENW != 1) begin : g_wren_sync_ram
        `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
        reg [DATAW-1:0] rdata_r;
        `RAM_INITIALIZATION
        always @(posedge clk) begin
            if (read_s || write) begin
                if (write) begin
                    `RAM_WRITE_WREN
                end
                rdata_r <= ram[raddr_s];
            end
        end
        `RAM_BYPASS(rdata_s);
    end else begin : g_no_wren_sync_ram
        `USE_BLOCK_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
        reg [DATAW-1:0] rdata_r;
        `RAM_INITIALIZATION
        `UNUSED_VAR (wren)
        always @(posedge clk) begin
            if (read_s || write) begin
                if (write) begin
                    ram[waddr] <= wdata;
                end
                rdata_r <= ram[raddr_s];
            end
        end
        `RAM_BYPASS(rdata_s);
    end

    // asynchronous ram (fallback)

    wire [DATAW-1:0] rdata_a;

    if (DUAL_PORT != 0) begin : g_dp_async_ram
         reg [DATAW-1:0] ram [0:SIZE-1];
        `RAM_INITIALIZATION
        if (WRENW != 1) begin : g_wren
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE_WREN
                end
            end
        end else begin : g_no_wren
            always @(posedge clk) begin
                if (write) begin
                    ram[waddr] <= wdata;
                end
            end
        end
        assign rdata_a = ram[raddr];
    end else begin : g_sp_async_ram
         reg [DATAW-1:0] ram [0:SIZE-1];
        `RAM_INITIALIZATION
        if (WRENW != 1) begin : g_wren
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE_WREN
                end
            end
        end else begin : g_no_wren
            always @(posedge clk) begin
                if (write) begin
                    ram[waddr] <= wdata;
                end
            end
        end
        assign rdata_a = ram[waddr];
    end

    assign rdata = is_raddr_reg ? rdata_s : rdata_a;

endmodule
`TRACING_ON
