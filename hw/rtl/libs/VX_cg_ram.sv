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

// ASIC flip-flop RAM with a clock gate per row, behind VX_dp_ram / VX_sp_ram
// for arrays below the macro size. Same read forms and timing as those modules;
// a row is clocked only on the cycle it is written, the read register only on a
// read, and every register keeps its enable so an open gate changes nothing.
`TRACING_OFF
module VX_cg_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter OUT_REG     = 0,
    parameter `STRING RDW_MODE = "W", // W: write-first, R: read-first, N: no-change
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

    `STATIC_ASSERT(!(WRENW * WSELW != DATAW), ("invalid parameter"))
    `STATIC_ASSERT((RDW_MODE == "R" || RDW_MODE == "W" || RDW_MODE == "N"), ("invalid parameter"))

    wire [WRENW-1:0] wr_lanes = (WRENW != 1) ? wren : {WRENW{1'b1}};
    if (WRENW == 1) begin : g_no_wren
        `UNUSED_VAR (wren)
    end

    wire [DATAW-1:0] rows [0:SIZE-1];

    for (genvar r = 0; r < SIZE; ++r) begin : g_rows
        wire [WRENW-1:0] row_wren = {WRENW{write && (waddr == ADDRW'(r))}} & wr_lanes;
        wire row_en = | row_wren;
        wire row_clk;
        VX_clockgate row_cg (
            .clk_in  (clk),
            .en      (row_en),
            .clk_out (row_clk)
        );
        reg [DATAW-1:0] row_r;
        always @(posedge row_clk) begin
            for (integer i = 0; i < WRENW; ++i) begin
                if (row_wren[i]) begin
                    row_r[i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
            end
        end
        assign rows[r] = row_r;
    end

    if (OUT_REG != 0) begin : g_sync
        // RDW_MODE "W" registers the address and reads the rows continuously,
        // so it sees a write on the same edge; "R" and "N" register the data.
        localparam REGW = (RDW_MODE == "W") ? ADDRW : DATAW;
        wire rd_en = (RDW_MODE == "N") ? (read && ~write) : read;
        wire [REGW-1:0] rd_in;
        reg [REGW-1:0] rd_r;
        wire rd_clk;
        if (REGW >= 4) begin : g_rd_cg
            VX_clockgate rd_cg (
                .clk_in  (clk),
                .en      (rd_en),
                .clk_out (rd_clk)
            );
        end else begin : g_rd_clk
            assign rd_clk = clk;
        end
        always @(posedge rd_clk) begin
            if (rd_en) begin
                rd_r <= rd_in;
            end
        end
        if (RDW_MODE == "W") begin : g_write_first
            assign rd_in = raddr;
            assign rdata = rows[rd_r];
        end else begin : g_read_first
            assign rd_in = rows[raddr];
            assign rdata = rd_r;
        end
    end else begin : g_async
        `UNUSED_VAR (read)
        assign rdata = rows[raddr];
    end

endmodule
`TRACING_ON
