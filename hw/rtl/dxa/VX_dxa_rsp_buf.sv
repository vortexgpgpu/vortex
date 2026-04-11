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

// DXA Response Buffer: BRAM-based 512-bit data store + arrival bitvector.
// Write port: from GMEM responses (indexed by tag).
// Read port: from smem_wr (indexed by tag).
// LUTRAM=0 forces BRAM/URAM inference, saving LUTs.

`include "VX_define.vh"

module VX_dxa_rsp_buf import VX_gpu_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter GMEM_DATAW      = `L1_LINE_SIZE * 8
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,

    // Write port: GMEM response arrival (data + arrival bit)
    input  wire                        rsp_write_en,
    input  wire [TAG_W-1:0]            rsp_write_tag,
    input  wire [GMEM_DATAW-1:0]       rsp_write_data,

    // OOB arrival (arrival bit only, no data write)
    input  wire                        oob_arrived_en,
    input  wire [TAG_W-1:0]            oob_arrived_tag,

    // Read port: smem_wr data fetch
    input  wire                        read_en,
    input  wire [TAG_W-1:0]            read_tag,
    output wire [GMEM_DATAW-1:0]       read_data,

    // Arrival bitvector
    output wire [MAX_OUTSTANDING-1:0]  rsp_arrived,

    // Clear arrival bit (from smem_wr after consuming)
    input  wire                        clear_en,
    input  wire [TAG_W-1:0]            clear_tag
);
    localparam TAG_W = `CLOG2(MAX_OUTSTANDING);

    // ---- BRAM data storage ----
    VX_dp_ram #(
        .DATAW    (GMEM_DATAW),
        .SIZE     (MAX_OUTSTANDING),
        .LUTRAM   (0),       // Force BRAM inference
        .OUT_REG  (1),       // Registered output for timing
        .RDW_MODE ("W")
    ) data_store (
        .clk   (clk),
        .reset (reset),
        .read  (read_en),
        .write (rsp_write_en),
        .wren  (1'b1),
        .raddr (read_tag),
        .waddr (rsp_write_tag),
        .wdata (rsp_write_data),
        .rdata (read_data)
    );

    // ---- Arrival bitvector ----
    reg [MAX_OUTSTANDING-1:0] arrived_r;

    always @(posedge clk) begin
        if (reset || !transfer_active) begin
            arrived_r <= '0;
        end else begin
            if (clear_en) begin
                arrived_r[clear_tag] <= 1'b0;
            end
            if (rsp_write_en) begin
                arrived_r[rsp_write_tag] <= 1'b1;
            end
            if (oob_arrived_en) begin
                arrived_r[oob_arrived_tag] <= 1'b1;
            end
        end
    end

    assign rsp_arrived = arrived_r;

endmodule
