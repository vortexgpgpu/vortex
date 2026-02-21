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

`include "VX_define.vh"

/* verilator lint_off UNUSEDSIGNAL */
module VX_dxa_xfer_math import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input dxa_xfer_state_t xfer_state,
    input wire [`L2_LINE_SIZE * 8-1:0] gmem_rsp_data,
    input wire [LSU_WORD_SIZE * 8-1:0] smem_rsp_data,
    output wire dxa_xfer_math_t xfer_math
);
    localparam XFER_ELEM_IDLE    = 2'd0;
    localparam XFER_ELEM_WAIT_RD = 2'd1;
    localparam XFER_ELEM_WAIT_WR = 2'd2;

    localparam GMEM_BYTES      = `L2_LINE_SIZE;
    localparam GMEM_DATAW      = GMEM_BYTES * 8;
    localparam GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES);

    localparam SMEM_BYTES      = LSU_WORD_SIZE;
    localparam SMEM_DATAW      = SMEM_BYTES * 8;
    localparam SMEM_OFF_BITS   = `CLOG2(SMEM_BYTES);

    localparam MAX_BYTES = (GMEM_BYTES > SMEM_BYTES) ? GMEM_BYTES : SMEM_BYTES;
    localparam MAX_DATAW = MAX_BYTES * 8;

    function automatic [MAX_BYTES-1:0] dxa_byte_mask(input [31:0] nbytes);
    begin
        if (nbytes >= MAX_BYTES) begin
            dxa_byte_mask = {MAX_BYTES{1'b1}};
        end else begin
            dxa_byte_mask = (MAX_BYTES'(1) << nbytes) - 1;
        end
    end
    endfunction

    wire state_idle_w    = xfer_state.active && (xfer_state.elem_state == XFER_ELEM_IDLE);
    wire state_wait_rd_w = xfer_state.active && (xfer_state.elem_state == XFER_ELEM_WAIT_RD);
    wire state_wait_wr_w = xfer_state.active && (xfer_state.elem_state == XFER_ELEM_WAIT_WR);

    wire [31:0] cur_elem_x = (xfer_state.tile0 != 0) ? (xfer_state.idx % xfer_state.tile0) : 32'd0;
    wire [31:0] cur_elem_y = (xfer_state.tile0 != 0) ? (xfer_state.idx / xfer_state.tile0) : 32'd0;
    wire [31:0] cur_i0 = xfer_state.coord0 + cur_elem_x;
    wire [31:0] cur_i1 = xfer_state.coord1 + cur_elem_y;
    wire cur_in_bounds = (cur_i0 < xfer_state.size0) && (cur_i1 < xfer_state.size1);

    wire [`MEM_ADDR_WIDTH-1:0] cur_elem_off_bytes = `MEM_ADDR_WIDTH'(xfer_state.idx * xfer_state.elem_bytes);
    wire [`MEM_ADDR_WIDTH-1:0] cur_gmem_byte_addr_w = xfer_state.gbase
                                                    + `MEM_ADDR_WIDTH'(cur_i0 * xfer_state.elem_bytes)
                                                    + `MEM_ADDR_WIDTH'(cur_i1 * xfer_state.stride0);
    wire [`MEM_ADDR_WIDTH-1:0] cur_smem_byte_addr_w = `MEM_ADDR_WIDTH'(xfer_state.smem_base) + cur_elem_off_bytes;

    wire [GMEM_OFF_BITS-1:0] cur_gmem_off = GMEM_OFF_BITS'(cur_gmem_byte_addr_w);
    wire [SMEM_OFF_BITS-1:0] cur_smem_off = SMEM_OFF_BITS'(cur_smem_byte_addr_w);
    wire [MAX_BYTES-1:0] cur_elem_mask = dxa_byte_mask(xfer_state.elem_bytes);
    wire [GMEM_BYTES-1:0] cur_gmem_byteen_w = GMEM_BYTES'(cur_elem_mask << cur_gmem_off);
    wire [SMEM_BYTES-1:0] cur_smem_byteen_w = SMEM_BYTES'(cur_elem_mask << cur_smem_off);

    wire cur_need_skip_w = state_idle_w
                        && (xfer_state.idx < xfer_state.total)
                        && xfer_state.is_s2g
                        && ~cur_in_bounds;
    wire cur_need_fill_w = state_idle_w
                        && (xfer_state.idx < xfer_state.total)
                        && ~xfer_state.is_s2g
                        && ~cur_in_bounds;
    wire cur_need_read_w = state_idle_w && (xfer_state.idx < xfer_state.total) && cur_in_bounds;
    wire cur_read_from_gmem_w = cur_need_read_w && ~xfer_state.is_s2g;
    wire cur_read_from_smem_w = cur_need_read_w && xfer_state.is_s2g;

    wire [GMEM_OFF_BITS-1:0] pending_gmem_wr_off = GMEM_OFF_BITS'(xfer_state.pending_wr_byte_addr);
    wire [SMEM_OFF_BITS-1:0] pending_smem_wr_off = SMEM_OFF_BITS'(xfer_state.pending_wr_byte_addr);
    wire [31:0] pending_gmem_wr_shift = 32'(pending_gmem_wr_off) * 32'd8;
    wire [31:0] pending_smem_wr_shift = 32'(pending_smem_wr_off) * 32'd8;
    wire [MAX_DATAW-1:0] pending_elem_data_ext = MAX_DATAW'(xfer_state.pending_elem_data);
    wire [SMEM_DATAW-1:0] pending_elem_data_smem = SMEM_DATAW'(xfer_state.pending_elem_data);
    wire [GMEM_DATAW-1:0] pending_gmem_wr_data_shifted_w = pending_elem_data_ext << pending_gmem_wr_shift;
    wire [SMEM_DATAW-1:0] pending_smem_wr_data_shifted_w = pending_elem_data_smem << pending_smem_wr_shift;
    wire [GMEM_BYTES-1:0] pending_gmem_byteen_w = GMEM_BYTES'(cur_elem_mask << pending_gmem_wr_off);
    wire [SMEM_BYTES-1:0] pending_smem_byteen_w = SMEM_BYTES'(cur_elem_mask << pending_smem_wr_off);

    wire [GMEM_OFF_BITS-1:0] pending_gmem_rd_off = GMEM_OFF_BITS'(xfer_state.pending_rd_byte_addr);
    wire [SMEM_OFF_BITS-1:0] pending_smem_rd_off = SMEM_OFF_BITS'(xfer_state.pending_rd_byte_addr);
    wire [31:0] pending_gmem_rd_shift = 32'(pending_gmem_rd_off) * 32'd8;
    wire [31:0] pending_smem_rd_shift = 32'(pending_smem_rd_off) * 32'd8;
    wire [63:0] gmem_rsp_data_shifted_w = 64'(GMEM_DATAW'(gmem_rsp_data) >> pending_gmem_rd_shift);
    wire [63:0] smem_rsp_data_shifted_w = 64'(smem_rsp_data) >> pending_smem_rd_shift;

    assign xfer_math.state_idle = state_idle_w;
    assign xfer_math.state_wait_rd = state_wait_rd_w;
    assign xfer_math.state_wait_wr = state_wait_wr_w;
    assign xfer_math.cur_gmem_byte_addr = cur_gmem_byte_addr_w;
    assign xfer_math.cur_smem_byte_addr = cur_smem_byte_addr_w;
    assign xfer_math.cur_need_skip = cur_need_skip_w;
    assign xfer_math.cur_need_fill = cur_need_fill_w;
    assign xfer_math.cur_need_read = cur_need_read_w;
    assign xfer_math.cur_read_from_gmem = cur_read_from_gmem_w;
    assign xfer_math.cur_read_from_smem = cur_read_from_smem_w;
    assign xfer_math.pending_gmem_wr_data_shifted = pending_gmem_wr_data_shifted_w;
    assign xfer_math.pending_smem_wr_data_shifted = pending_smem_wr_data_shifted_w;
    assign xfer_math.pending_gmem_byteen = pending_gmem_byteen_w;
    assign xfer_math.pending_smem_byteen = pending_smem_byteen_w;
    assign xfer_math.gmem_rsp_data_shifted = gmem_rsp_data_shifted_w;
    assign xfer_math.smem_rsp_data_shifted = smem_rsp_data_shifted_w;
    assign xfer_math.cur_gmem_byteen = cur_gmem_byteen_w;
    assign xfer_math.cur_smem_byteen = cur_smem_byteen_w;

endmodule
/* verilator lint_on UNUSEDSIGNAL */
