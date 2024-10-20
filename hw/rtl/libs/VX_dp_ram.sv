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
module VX_dp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter LUTRAM      = 0,
    parameter NO_RWCHECK  = 0,
    parameter RW_ASSERT   = 0,
    parameter RESET_RAM   = 0,
    parameter RESET_OUT   = 0,
    parameter `STRING WRITE_MODE = "R", // R: read-first, W: write-first, N: no-change, U: undefined
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
    localparam USE_BRAM = !LUTRAM && ((DATAW * SIZE) >= `MAX_LUTRAM);

    `STATIC_ASSERT((WRENW * WSELW == DATAW), ("invalid parameter"))
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
    localparam `STRING RAM_STYLE_VALUE = USE_BRAM ? "block" : (LUTRAM ? "MLAB, no_rw_check" : "auto");
    localparam `STRING RAM_NO_RWCHECK_VALUE = NO_RWCHECK ? "-name add_pass_through_logic_to_inferred_rams off" : "";
    `define RAM_ARRAY (* ramstyle = RAM_STYLE_VALUE *) reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE   for (integer i = 0; i < WRENW; ++i) begin \
                            if (wren[i]) begin \
                                ram[waddr][i] <= wdata[i * WSELW +: WSELW]; \
                            end \
                        end
    `define RAM_NO_RWCHECK (* altera_attribute = RAM_NO_RWCHECK_VALUE *)
`elsif VIVADO
    localparam `STRING RAM_STYLE_VALUE = USE_BRAM ? "block" : (LUTRAM ? "distributed" : "auto");
    localparam `STRING RAM_NO_RWCHECK_VALUE = NO_RWCHECK ? "no" : "auto";
    `define RAM_ARRAY (* ram_style = RAM_STYLE_VALUE *) reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE   for (integer i = 0; i < WRENW; ++i) begin \
                            if (wren[i]) begin \
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                            end \
                        end
    `define RAM_NO_RWCHECK (* rw_addr_collision = RAM_NO_RWCHECK_VALUE *)
`else
    `define RAM_ARRAY   reg [DATAW-1:0] ram [0:SIZE-1];
    `define RAM_WRITE   for (integer i = 0; i < WRENW; ++i) begin \
                            if (wren[i]) begin \
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                            end \
                        end
    `define RAM_NO_RWCHECK
`endif
    if (OUT_REG) begin : g_out_reg
        reg [DATAW-1:0] rdata_r;
        if (WRITE_MODE == "R") begin : g_read_first
            `RAM_ARRAY
            `RAM_INITIALIZATION
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE
                end
                if (RESET_OUT && reset) begin
                    rdata_r <= INIT_VALUE;
                end else if (read || write) begin
                    rdata_r <= ram[raddr];
                end
            end
        end else if (WRITE_MODE == "W") begin : g_write_first
            `RAM_ARRAY
            `RAM_INITIALIZATION
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE
                end
                if (RESET_OUT && reset) begin
                    rdata_r <= INIT_VALUE;
                end else if (read || write) begin
                    rdata_r = ram[raddr];
                end
            end
        end else if (WRITE_MODE == "N") begin : g_no_change
            `RAM_ARRAY
            `RAM_INITIALIZATION
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE
                end
                if (RESET_OUT && reset) begin
                    rdata_r <= INIT_VALUE;
                end else if (read && ~write) begin
                    rdata_r <= ram[raddr];
                end
            end
        end else if (WRITE_MODE == "U") begin : g_undefined
            `RAM_NO_RWCHECK `RAM_ARRAY
            `RAM_INITIALIZATION
            always @(posedge clk) begin
                if (write) begin
                    `RAM_WRITE
                end
                if (RESET_OUT && reset) begin
                    rdata_r <= INIT_VALUE;
                end else if (read) begin
                    rdata_r <= ram[raddr];
                end
            end
        end else begin
            `STATIC_ASSERT(0, ("invalid write mode: %s", WRITE_MODE))
        end
        assign rdata = rdata_r;
    end else begin : g_no_out_reg
        `UNUSED_VAR (read)
        `RAM_NO_RWCHECK `RAM_ARRAY
        `RAM_INITIALIZATION
        always @(posedge clk) begin
            if (write) begin
                `RAM_WRITE
            end
        end
        assign rdata = ram[raddr];
    end
`else
    // simulation
    reg [DATAW-1:0] ram [0:SIZE-1];
    `RAM_INITIALIZATION

    wire [DATAW-1:0] ram_n;
    for (genvar i = 0; i < WRENW; ++i) begin : g_ram_n
        assign ram_n[i * WSELW +: WSELW] = wren[i] ? wdata[i * WSELW +: WSELW] : ram[waddr][i * WSELW +: WSELW];
    end

    always @(posedge clk) begin
        if (RESET_RAM && reset) begin
            for (integer i = 0; i < SIZE; ++i) begin
                ram[i] <= DATAW'(INIT_VALUE);
            end
        end else begin
            if (write) begin
                ram[waddr] <= ram_n;
            end
        end
    end

    if (OUT_REG && WRITE_MODE == "R") begin : g_read_first
        reg [DATAW-1:0] rdata_r;
        always @(posedge clk) begin
            if (RESET_OUT && reset) begin
                rdata_r <= DATAW'(INIT_VALUE);
            end else if (read || write) begin
                rdata_r <= ram[raddr];
            end
        end
        assign rdata = rdata_r;
    end else if (OUT_REG && WRITE_MODE == "W") begin : g_read_first
        reg [DATAW-1:0] rdata_r;
        always @(posedge clk) begin
            if (RESET_OUT && reset) begin
                rdata_r <= DATAW'(INIT_VALUE);
            end else if (read || write) begin
                if (write && (raddr == waddr)) begin
                    rdata_r <= ram_n;
                end else begin
                    rdata_r <= ram[raddr];
                end
            end
        end
        assign rdata = rdata_r;
    end else if (OUT_REG && WRITE_MODE == "N") begin : g_read_first
        reg [DATAW-1:0] rdata_r;
        always @(posedge clk) begin
            if (RESET_OUT && reset) begin
                rdata_r <= DATAW'(INIT_VALUE);
            end else if (read && ~write) begin
                rdata_r <= ram[raddr];
            end
        end
        assign rdata = rdata_r;
    end else begin : g_async_or_undef
        wire [DATAW-1:0] rdata_w;
        if (USE_BRAM && NO_RWCHECK) begin : g_rdata_no_bypass
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

            assign rdata_w = (prev_write && (prev_waddr == raddr)) ? prev_data : ram[raddr];
            if (RW_ASSERT) begin : g_rw_asert
                `RUNTIME_ASSERT(~read || (rdata_w == ram[raddr]), ("%t: read after write hazard", $time))
            end
        end else begin : g_rdata_with_bypass
            assign rdata_w = ram[raddr];
        end
        if (OUT_REG) begin : g_out_reg
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (RESET_OUT && reset) begin
                    rdata_r <= DATAW'(INIT_VALUE);
                end else if (read) begin
                    rdata_r <= rdata_w;
                end
            end
            assign rdata = rdata_r;
        end else begin : g_no_out_reg
            `UNUSED_VAR (read)
            assign rdata = rdata_w;
        end
    end
`endif

endmodule
`TRACING_ON
