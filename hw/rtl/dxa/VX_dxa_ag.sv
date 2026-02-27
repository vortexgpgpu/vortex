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

// Streaming address generator for DXA non-blocking worker.
// Produces one (gmem_addr, smem_addr) pair per cycle with valid/ready
// handshake. Iterates through a 2D tile using counter-pair (cx, cy)
// to avoid hardware dividers.

`include "VX_define.vh"

module VX_dxa_ag import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_BYTES      = `L2_LINE_SIZE,
    parameter GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES),
    parameter GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - GMEM_OFF_BITS
) (
    input  wire                        clk,
    input  wire                        reset,

    // Command interface — latched on start pulse.
    input  wire                        start,
    input  wire dxa_issue_dec_t        issue_dec,
    input  wire [`MEM_ADDR_WIDTH-1:0]  gmem_base,
    input  wire [`XLEN-1:0]            smem_base,
    input  wire [4:0][`XLEN-1:0]       coords,
    input  wire [31:0]                 cfill,

    // Output element stream (valid/ready handshake).
    output wire                        ag_valid,
    input  wire                        ag_ready,
    output wire [`MEM_ADDR_WIDTH-1:0]  ag_gmem_byte_addr,
    output wire [GMEM_ADDR_WIDTH-1:0]  ag_gmem_line_addr,
    output wire [GMEM_OFF_BITS-1:0]    ag_gmem_off,
    output wire [`MEM_ADDR_WIDTH-1:0]  ag_smem_byte_addr,
    output wire                        ag_in_bounds,
    output wire                        ag_is_last,
    output wire [31:0]                 ag_elem_idx,

    // Status
    output wire                        ag_busy,
    output wire                        ag_done
);
    // Registered transfer parameters.
    reg                        active_r;
    reg [`MEM_ADDR_WIDTH-1:0]  gbase_r;
    reg [`XLEN-1:0]            smem_base_r;
    reg [31:0]                 coord0_r;
    reg [31:0]                 coord1_r;
    reg [31:0]                 size0_r;
    reg [31:0]                 size1_r;
    reg [31:0]                 stride0_r;
    reg [31:0]                 tile0_r;
    reg [31:0]                 elem_bytes_r;
    reg [31:0]                 total_r;

    // Counter-pair (cx, cy) for 2D iteration — avoids hardware dividers.
    reg [31:0]                 idx_r;
    reg [31:0]                 cx_r;
    reg [31:0]                 cy_r;

    // Combinational address computation.
    wire [31:0] cur_i0 = coord0_r + cx_r;
    wire [31:0] cur_i1 = coord1_r + cy_r;
    wire cur_in_bounds = (cur_i0 < size0_r) && (cur_i1 < size1_r);
    wire [`MEM_ADDR_WIDTH-1:0] cur_smem_byte_addr = `MEM_ADDR_WIDTH'(smem_base_r)
                                                   + `MEM_ADDR_WIDTH'(idx_r * elem_bytes_r);
    wire [`MEM_ADDR_WIDTH-1:0] cur_gmem_byte_addr = gbase_r
                                                   + `MEM_ADDR_WIDTH'(cur_i0 * elem_bytes_r)
                                                   + `MEM_ADDR_WIDTH'(cur_i1 * stride0_r);
    wire [GMEM_ADDR_WIDTH-1:0] cur_gmem_line_addr = GMEM_ADDR_WIDTH'(cur_gmem_byte_addr >> GMEM_OFF_BITS);
    wire [GMEM_OFF_BITS-1:0] cur_gmem_off = GMEM_OFF_BITS'(cur_gmem_byte_addr);
    wire cur_is_last = ((idx_r + 32'd1) >= total_r);

    // Counter wrap logic for 2D tile iteration.
    wire cx_wrap = (tile0_r == 0) || ((cx_r + 32'd1) >= tile0_r);
    wire [31:0] next_cx = cx_wrap ? 32'd0 : (cx_r + 32'd1);
    wire [31:0] next_cy = cx_wrap ? (cy_r + 32'd1) : cy_r;

    // Output assignments.
    assign ag_valid         = active_r && (idx_r < total_r);
    assign ag_gmem_byte_addr = cur_gmem_byte_addr;
    assign ag_gmem_line_addr = cur_gmem_line_addr;
    assign ag_gmem_off      = cur_gmem_off;
    assign ag_smem_byte_addr = cur_smem_byte_addr;
    assign ag_in_bounds     = cur_in_bounds;
    assign ag_is_last       = cur_is_last;
    assign ag_elem_idx      = idx_r;
    assign ag_busy          = active_r;
    assign ag_done          = active_r && (idx_r >= total_r);

    wire advance = ag_valid && ag_ready;

    always @(posedge clk) begin
        if (reset) begin
            active_r    <= 1'b0;
            idx_r       <= '0;
            cx_r        <= '0;
            cy_r        <= '0;
            gbase_r     <= '0;
            smem_base_r <= '0;
            coord0_r    <= '0;
            coord1_r    <= '0;
            size0_r     <= '0;
            size1_r     <= '0;
            stride0_r   <= '0;
            tile0_r     <= '0;
            elem_bytes_r <= '0;
            total_r     <= '0;
        end else begin
            if (start) begin
                active_r    <= 1'b1;
                gbase_r     <= gmem_base;
                smem_base_r <= smem_base;
                coord0_r    <= coords[0];
                coord1_r    <= (issue_dec.rank >= 2) ? coords[1] : 32'd0;
                size0_r     <= issue_dec.size0;
                size1_r     <= issue_dec.size1;
                stride0_r   <= issue_dec.stride0;
                tile0_r     <= issue_dec.tile0;
                elem_bytes_r <= issue_dec.elem_bytes;
                total_r     <= issue_dec.total;
                idx_r       <= 32'd0;
                cx_r        <= 32'd0;
                cy_r        <= 32'd0;
            end else if (advance) begin
                idx_r <= idx_r + 32'd1;
                cx_r  <= next_cx;
                cy_r  <= next_cy;
                if (cur_is_last) begin
                    active_r <= 1'b0;
                end
            end
        end
    end

    `UNUSED_VAR (cfill)
    `UNUSED_VAR (issue_dec)
    `UNUSED_VAR (coords[4:2])
endmodule
