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

`define RAM_WREN_BLOCK_ALTERA(__we__) \
    reg [WRENW-1:0][WSELW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    always @(posedge clk) begin \
        if (__we__) begin \
            for (integer i = 0; i < WRENW; ++i) begin \
                if (wren[i]) begin \
                    ram[waddr][i] <= wdata[i * WSELW +: WSELW]; \
                end \
            end \
        end \
    end

`define RAM_WREN_BLOCK_XILINX(__we__) \
    reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    always @(posedge clk) begin \
        if (__we__) begin \
            for (integer i = 0; i < WRENW; ++i) begin \
                if (wren[i]) begin \
                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW]; \
                end \
            end \
        end \
    end

`define RAM_WRITE_BLOCK(__we__) \
    reg [DATAW-1:0] ram [0:SIZE-1]; \
    `RAM_INITIALIZATION \
    always @(posedge clk) begin \
        if (__we__) begin \
            ram[waddr] <= wdata; \
        end \
    end

`define RAM_READ_BLOCK_OUT_REG(__re__) \
    always @(posedge clk) begin \
        if (__re__) begin \
            if (RESET_OUT && reset) begin \
                rdata_r <= INIT_VALUE; \
            end else begin \
                rdata_r <= ram[raddr]; \
            end \
        end \
    end

    `UNUSED_PARAM (RW_ASSERT)
    `UNUSED_VAR (read)
    `UNUSED_VAR (wren)

    if (OUT_REG) begin : g_out_reg
        reg [DATAW-1:0] rdata_r;
        if (READ_ENABLE) begin : g_readen
            if (WRENW != 1) begin : g_writeen
            `ifdef QUARTUS
                if (LUTRAM != 0) begin : g_lutram
                    `USE_FAST_BRAM `RAM_WREN_BLOCK_ALTERA(write)
                    `RAM_READ_BLOCK_OUT_REG(read)
                end else begin : g_no_lutram
                    `RAM_WREN_BLOCK_ALTERA(write)
                    `RAM_READ_BLOCK_OUT_REG(read)
                end
            `else
                // Not Quartus
                if (LUTRAM != 0) begin : g_lutram
                    `USE_FAST_BRAM `RAM_WREN_BLOCK_XILINX(write)
                    `RAM_READ_BLOCK_OUT_REG(read)
                end else begin : g_no_lutram
                    `RAM_WREN_BLOCK_XILINX(write)
                    `RAM_READ_BLOCK_OUT_REG(read)
                end
            `endif
            end else begin : g_no_writeen
                if (LUTRAM != 0) begin : g_lutram
                    `USE_FAST_BRAM `RAM_WRITE_BLOCK(write)
                    `RAM_READ_BLOCK_OUT_REG(read)
                end else begin : g_no_lutram
                    `RAM_WRITE_BLOCK(write)
                    `RAM_READ_BLOCK_OUT_REG(read)
                end
            end
        end else begin : g_no_readen
            if (WRENW != 1) begin : g_writeen
            `ifdef QUARTUS
                if (LUTRAM != 0) begin : g_lutram
                    `USE_FAST_BRAM `RAM_WREN_BLOCK_ALTERA(write)
                    `RAM_READ_BLOCK_OUT_REG(read || write)
                end else begin : g_no_lutram
                    `RAM_WREN_BLOCK_ALTERA(write)
                    `RAM_READ_BLOCK_OUT_REG(read || write)
                end
            `else
                // Not Quartus
                if (LUTRAM != 0) begin : g_lutram
                    `USE_FAST_BRAM `RAM_WREN_BLOCK_XILINX(write)
                    `RAM_READ_BLOCK_OUT_REG(read || write)
                end else begin : g_no_lutram
                    `RAM_WREN_BLOCK_XILINX(write)
                    `RAM_READ_BLOCK_OUT_REG(read || write)
                end
            `endif
            end else begin : g_no_writeen
                if (LUTRAM != 0) begin : g_lutram
                    `USE_FAST_BRAM `RAM_WRITE_BLOCK(write)
                    `RAM_READ_BLOCK_OUT_REG(read || write)
                end else begin : g_no_lutram
                    `RAM_WRITE_BLOCK(write)
                    `RAM_READ_BLOCK_OUT_REG(read || write)
                end
            end
        end
        assign rdata = rdata_r;
    end else begin : g_no_out_reg
    `ifdef SYNTHESIS
        if (WRENW > 1) begin : g_writeen
        `ifdef QUARTUS
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM `RAM_WREN_BLOCK_ALTERA(write)
                assign rdata = ram[raddr];
            end else begin : g_no_lutram
                if (NO_RWCHECK != 0) begin : g_no_rwcheck
                    `NO_RW_RAM_CHECK `RAM_WREN_BLOCK_ALTERA(write)
                    assign rdata = ram[raddr];
                end else begin : g_rwcheck
                    `RAM_WREN_BLOCK_ALTERA(write)
                    assign rdata = ram[raddr];
                end
            end
        `else
            // default synthesis
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM `RAM_WREN_BLOCK_XILINX(write)
                assign rdata = ram[raddr];
            end else begin : g_no_lutram
                if (NO_RWCHECK != 0) begin : g_no_rwcheck
                    `NO_RW_RAM_CHECK `RAM_WREN_BLOCK_XILINX(write)
                    assign rdata = ram[raddr];
                end else begin : g_rwcheck
                    `RAM_WREN_BLOCK_XILINX(write)
                    assign rdata = ram[raddr];
                end
            end
        `endif
        end else begin : g_no_writeen
            // (WRENW == 1)
            if (LUTRAM != 0) begin : g_lutram
                `USE_FAST_BRAM `RAM_WRITE_BLOCK(write)
                assign rdata = ram[raddr];
            end else begin : g_no_lutram
                if (NO_RWCHECK != 0) begin : g_no_rwcheck
                    `NO_RW_RAM_CHECK `RAM_WRITE_BLOCK(write)
                    assign rdata = ram[raddr];
                end else begin : g_rwcheck
                    `RAM_WRITE_BLOCK(write)
                    assign rdata = ram[raddr];
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

            assign rdata = (prev_write && (prev_waddr == raddr)) ? prev_data : ram[raddr];
            if (RW_ASSERT) begin : g_rw_assert
                `RUNTIME_ASSERT(~read || (rdata == ram[raddr]), ("%t: read after write hazard", $time))
            end
        end else begin : g_rdata_with_bypass
            assign rdata = ram[raddr];
        end
    `endif
    end

endmodule
`TRACING_ON
