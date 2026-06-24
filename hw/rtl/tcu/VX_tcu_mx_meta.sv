// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef VX_CFG_TCU_MX_ENABLE

module VX_tcu_mx_meta import VX_gpu_pkg::*, VX_tcu_pkg::*; (
    input wire clk,
    input wire reset,

    input wire wr_en,
    input wire [NW_WIDTH-1:0] wr_wid,
    input wire wr_axis,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] wr_data,

    input wire [NW_WIDTH-1:0] rd_wid,
    output wire [TCU_BLOCK_CAP-1:0][31:0] meta_a,
    output wire [TCU_BLOCK_CAP-1:0][31:0] meta_b
);
    localparam DATAW = TCU_BLOCK_CAP * 32;
    localparam DEPTH = `VX_CFG_NUM_WARPS;
    localparam ADDRW = `CLOG2(DEPTH);

    wire [TCU_BLOCK_CAP-1:0][31:0] wr_data32;
    for (genvar i = 0; i < TCU_BLOCK_CAP; ++i) begin : g_wr_data
        assign wr_data32[i] = 32'(wr_data[i]);
    end

    wire [`UP(ADDRW)-1:0] wr_addr = `UP(ADDRW)'(wr_wid);
    wire [`UP(ADDRW)-1:0] rd_addr = `UP(ADDRW)'(rd_wid);

    VX_dp_ram #(
        .DATAW     (DATAW),
        .SIZE      (DEPTH),
        .LUTRAM    (0),
        .OUT_REG   (0),
        .RDW_MODE  ("W"),
        .RADDR_REG (1) // rd_wid is registered!
    ) meta_a_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (wr_en && !wr_axis),
        .wren  (1'b1),
        .waddr (wr_addr),
        .wdata (wr_data32),
        .raddr (rd_addr),
        .rdata (meta_a)
    );

    VX_dp_ram #(
        .DATAW     (DATAW),
        .SIZE      (DEPTH),
        .LUTRAM    (0),
        .OUT_REG   (0),
        .RDW_MODE  ("W"),
        .RADDR_REG (1) // rd_wid is registered!
    ) meta_b_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (wr_en && wr_axis),
        .wren  (1'b1),
        .waddr (wr_addr),
        .wdata (wr_data32),
        .raddr (rd_addr),
        .rdata (meta_b)
    );

endmodule

`endif
