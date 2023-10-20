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
    parameter NO_RWCHECK  = 0,
    parameter LUTRAM      = 0,    
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0,
    parameter ADDRW       = `LOG2UP(SIZE)
) ( 
    input wire               clk,
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
    if (INIT_ENABLE != 0) begin                    \
        if (INIT_FILE != "") begin                 \
            initial $readmemh(INIT_FILE, ram);     \
        end else begin                             \
            initial                                \
                for (integer i = 0; i < SIZE; ++i) \
                    ram[i] = INIT_VALUE;           \
        end                                        \
    end
    
    `UNUSED_VAR (read)

`ifdef SYNTHESIS
    if (WRENW > 1) begin
    `ifdef QUARTUS
        if (LUTRAM != 0) begin
            if (OUT_REG != 0) begin        
                reg [DATAW-1:0] rdata_r;
                `USE_FAST_BRAM reg [WRENW-1:0][WSELW-1:0] ram [SIZE-1:0];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                    if (read) begin
                        rdata_r <= ram[raddr];
                    end
                end
                assign rdata = rdata_r;
            end else begin
                `USE_FAST_BRAM reg [WRENW-1:0][WSELW-1:0] ram [SIZE-1:0];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                end
                assign rdata = ram[raddr];
            end
        end else begin
            if (OUT_REG != 0) begin
                reg [DATAW-1:0] rdata_r;
                reg [WRENW-1:0][WSELW-1:0] ram [SIZE-1:0];
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                    if (read) begin
                        rdata_r <= ram[raddr];
                    end
                end
                assign rdata = rdata_r;
            end else begin
                if (NO_RWCHECK != 0) begin
                    `NO_RW_RAM_CHECK reg [WRENW-1:0][WSELW-1:0] ram [SIZE-1:0];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin
                    reg [WRENW-1:0][WSELW-1:0] ram [SIZE-1:0];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata = ram[raddr];
                end
            end
        end
    `else
        // default synthesis
        if (LUTRAM != 0) begin
            `USE_FAST_BRAM reg [DATAW-1:0] ram [SIZE-1:0];
            `RAM_INITIALIZATION
            if (OUT_REG != 0) begin            
                reg [DATAW-1:0] rdata_r;
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                    if (read) begin
                        rdata_r <= ram[raddr];
                    end
                end
                assign rdata = rdata_r;
            end else begin
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                end
                assign rdata = ram[raddr];
            end
        end else begin
            if (OUT_REG != 0) begin
                reg [DATAW-1:0] ram [SIZE-1:0];
                reg [DATAW-1:0] rdata_r;
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        for (integer i = 0; i < WRENW; ++i) begin
                            if (wren[i])
                                ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                        end
                    end
                    if (read) begin
                        rdata_r <= ram[raddr];
                    end
                end
                assign rdata = rdata_r;
            end else begin
                if (NO_RWCHECK != 0) begin
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [SIZE-1:0];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin
                    reg [DATAW-1:0] ram [SIZE-1:0];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            for (integer i = 0; i < WRENW; ++i) begin
                                if (wren[i])
                                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                            end
                        end
                    end
                    assign rdata = ram[raddr];
                end
            end
        end
    `endif
    end else begin
        // (WRENW == 1)
        if (LUTRAM != 0) begin
            `USE_FAST_BRAM reg [DATAW-1:0] ram [SIZE-1:0];
            `RAM_INITIALIZATION
            if (OUT_REG != 0) begin            
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
            end else begin
                always @(posedge clk) begin
                    if (write) begin
                        ram[waddr] <= wdata;
                    end
                end
                assign rdata = ram[raddr];
            end
        end else begin
            if (OUT_REG != 0) begin
                reg [DATAW-1:0] ram [SIZE-1:0];
                reg [DATAW-1:0] rdata_r;
                `RAM_INITIALIZATION
                always @(posedge clk) begin
                    if (write) begin
                        ram[waddr] <= wdata;
                    end
                    if (read) begin
                        rdata_r <= ram[raddr];
                    end
                end
                assign rdata = rdata_r;
            end else begin
                if (NO_RWCHECK != 0) begin
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [SIZE-1:0];
                    `RAM_INITIALIZATION
                    always @(posedge clk) begin
                        if (write) begin
                            ram[waddr] <= wdata;
                        end
                    end
                    assign rdata = ram[raddr];
                end else begin
                    reg [DATAW-1:0] ram [SIZE-1:0];
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
    // RAM emulation
    reg [DATAW-1:0] ram [SIZE-1:0];
    `RAM_INITIALIZATION

    wire [DATAW-1:0] ram_n;
    for (genvar i = 0; i < WRENW; ++i) begin
        assign ram_n[i * WSELW +: WSELW] = ((WRENW == 1) | wren[i]) ? wdata[i * WSELW +: WSELW] : ram[waddr][i * WSELW +: WSELW];
    end

    if (OUT_REG != 0) begin        
        reg [DATAW-1:0] rdata_r;        
        always @(posedge clk) begin
            if (write) begin
                ram[waddr] <= ram_n;
            end
            if (read) begin
                rdata_r <= ram[raddr];
            end
        end
        assign rdata = rdata_r;
    end else begin        
        reg [DATAW-1:0] prev_data;
        reg [ADDRW-1:0] prev_waddr;
        reg prev_write;
        always @(posedge clk) begin
            if (write) begin
                ram[waddr] <= ram_n;
            end
            prev_write <= (| wren);
            prev_data  <= ram[waddr];
            prev_waddr <= waddr;
        end            
        if (LUTRAM || !NO_RWCHECK) begin
            `UNUSED_VAR (prev_write)
            `UNUSED_VAR (prev_data)
            `UNUSED_VAR (prev_waddr)
            assign rdata = ram[raddr];
        end else begin
            assign rdata = (prev_write && (prev_waddr == raddr)) ? prev_data : ram[raddr];
        end
    end
`endif

endmodule
`TRACING_ON
