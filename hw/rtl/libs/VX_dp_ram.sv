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
    parameter READ_ENABLE = 0,
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
    if (INIT_ENABLE != 0) begin : g_init           \
        if (INIT_FILE != "") begin : g_file        \
            initial $readmemh(INIT_FILE, ram);     \
        end else begin : g_value                   \
            initial begin                          \
                for (integer i = 0; i < SIZE; ++i) \
                    ram[i] = INIT_VALUE;           \
            end                                    \
        end                                        \
    end

    `UNUSED_PARAM (RW_ASSERT)
    `UNUSED_VAR (read)

    `RUNTIME_ASSERT((((WRENW == 1) ) || ~write) || (| wren), ("%t: invalid write enable mask", $time))

    if (OUT_REG && !READ_ENABLE) begin : g_out_reg
        `UNUSED_PARAM (NO_RWCHECK)
        reg [DATAW-1:0] rdata_r;
        wire cs = read || write;
        if (WRENW != 1) begin : g_writeen
        `ifdef QUARTUS
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (cs) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                        if (RESET_OUT && reset) begin
                            rdata_r <= '0;
                        end else begin
                            rdata_r <= ram[raddr];
                        end
                    end
                end
            end else begin : g_no_lutram
                reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (cs) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                        if (RESET_OUT && reset) begin
                            rdata_r <= '0;
                        end else begin
                            rdata_r <= ram[raddr];
                        end
                    end
                end
            end
        `else
            // default synthesis
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (cs) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                        if (RESET_OUT && reset) begin
                            rdata_r <= '0;
                        end else begin
                            rdata_r <= ram[raddr];
                        end
                    end
                end
            end else begin : g_no_lutram
                reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (cs) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                        if (RESET_OUT && reset) begin
                            rdata_r <= '0;
                        end else begin
                            rdata_r <= ram[raddr];
                        end
                    end
                end
            end
        `endif
        end else begin : g_no_writeen
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (cs) begin
                        if (write)
                            ram[waddr] <= wdata;
                        if (RESET_OUT && reset) begin
                            rdata_r <= '0;
                        end else begin
                            rdata_r <= ram[raddr];
                        end
                    end
                end

            end else begin : g_no_lutram
                reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (cs) begin
                        if (write)
                            ram[waddr] <= wdata;
                        if (RESET_OUT && reset) begin
                            rdata_r <= '0;
                        end else begin
                            rdata_r <= ram[raddr];
                        end
                    end
                end
            end
        end
        assign rdata = rdata_r;
    end else begin : g_no_out_reg
        // OUT_REG==0 || READ_ENABLE=1
        wire [DATAW-1:0] rdata_w;
    `ifdef SYNTHESIS
        if (WRENW > 1) begin : g_writeen
        `ifdef QUARTUS
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                end
                assign rdata_w = ram[raddr];
            end else begin : g_no_lutram
                if (NO_RWCHECK != 0) begin : g_no_rwcheck
                    `NO_RW_RAM_CHECK reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata_w = ram[raddr];
                end else begin : g_rwcheck
                    reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata_w = ram[raddr];
                end
            end
        `else
            // default synthesis
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                end
                assign rdata_w = ram[raddr];
            end else begin : g_no_lutram
                if (NO_RWCHECK != 0) begin : g_no_rwcheck
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata_w = ram[raddr];
                end else begin : g_rwcheck
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata_w = ram[raddr];
                end
            end
        `endif
        end else begin : g_no_writeen
            // (WRENW == 1)
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM reg [DATAW-1:0] ram [0:SIZE-1];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        ram[waddr] <= wdata;
                    end
                end
                assign rdata_w = ram[raddr];
            end else begin : g_no_lutram
                if (NO_RWCHECK != 0) begin : g_no_rwcheck
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata_w = ram[raddr];
                end else begin : g_rwcheck
                    reg [DATAW-1:0] ram [0:SIZE-1];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata_w = ram[raddr];
                end
            end
        end
    `else
        // simulation
        reg [DATAW-1:0] ram [0:SIZE-1];
        `RAM_INITIALIZATION

        wire [DATAW-1:0] ram_n;
        for (genvar i = 0; i < WRENW; ++i) begin : g_ram_n
            assign ram_n[i * WSELW +: WSELW] = ((WRENW == 1) | wren[i]) ? wdata[i * WSELW +: WSELW] : ram[waddr][i * WSELW +: WSELW];
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

        if (!LUTRAM && NO_RWCHECK) begin : g_rdata_no_bypass
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
            if (RW_ASSERT) begin : g_rw_assert
                `RUNTIME_ASSERT(~read || (rdata_w == ram[raddr]), ("%t: read after write hazard", $time))
            end
        end else begin : g_rdata_with_bypass
            assign rdata_w = ram[raddr];
        end
    `endif

        if (OUT_REG != 0) begin : g_rdata_req
            reg [DATAW-1:0] rdata_r;
            always @(posedge clk) begin
                if (READ_ENABLE && reset) begin
                    rdata_r <= '0;
                end else if (!READ_ENABLE || read) begin
                    rdata_r <= rdata_w;
                end
            end
            assign rdata = rdata_r;
        end else begin : g_rdata_comb
            assign rdata = rdata_w;
        end

    end

endmodule
`TRACING_ON
