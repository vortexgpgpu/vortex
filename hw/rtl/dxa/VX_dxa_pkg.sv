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

`ifndef VX_DXA_PKG_VH
`define VX_DXA_PKG_VH

`include "VX_define.vh"

`IGNORE_UNUSED_BEGIN

package VX_dxa_pkg;

    import VX_gpu_pkg::*;

    localparam DXA_OP_SETUP0  = 3'd0;
    localparam DXA_OP_SETUP1  = 3'd1;
    localparam DXA_OP_COORD01 = 3'd2;
    localparam DXA_OP_COORD23 = 3'd3;
    localparam DXA_OP_ISSUE   = 3'd4;
    localparam DXA_OP_LAUNCH  = 3'd5;

    localparam DXA_REQ_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + 3 + (2 * `XLEN);
    localparam DXA_RSP_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + BAR_ADDR_W + 2;

    // Keep compatibility with existing global DXA descriptor macros.
    localparam DXA_DESC_SLOT_BITS = `CLOG2(`VX_DCR_DXA_DESC_COUNT);
    localparam DXA_DESC_SLOT_W    = `UP(DXA_DESC_SLOT_BITS);
    localparam DXA_DESC_WORD_BITS = `CLOG2(`VX_DCR_DXA_DESC_STRIDE);
    localparam DXA_DESC_WORD_W    = `UP(DXA_DESC_WORD_BITS);

    typedef struct packed {
        logic [31:0] rank;
        logic [31:0] elem_bytes;
        logic [31:0] tile0;
        logic [31:0] tile1;
        logic [31:0] size0;
        logic [31:0] size1;
        logic [31:0] stride0;
        logic [31:0] total;
        logic is_s2g;
        logic supported;
    } dxa_issue_dec_t;

    typedef struct packed {
        logic active;
        logic [NC_WIDTH-1:0] core_id;
        logic [UUID_WIDTH-1:0] uuid;
        logic [NW_WIDTH-1:0] wid;
        logic [BAR_ADDR_W-1:0] bar_addr;
        logic is_s2g;
        logic [`MEM_ADDR_WIDTH-1:0] gbase;
        logic [`XLEN-1:0] smem_base;
        logic [31:0] coord0;
        logic [31:0] coord1;
        logic [31:0] size0;
        logic [31:0] size1;
        logic [31:0] stride0;
        logic [31:0] tile0;
        logic [31:0] tile1;
        logic [31:0] elem_bytes;
        logic [31:0] cfill;
        logic [31:0] idx;
        logic [31:0] total;
        logic [1:0] elem_state;
        logic wait_rsp_from_gmem;
        logic write_to_gmem;
        logic [`MEM_ADDR_WIDTH-1:0] pending_rd_byte_addr;
        logic [`MEM_ADDR_WIDTH-1:0] pending_wr_byte_addr;
        logic [63:0] pending_elem_data;
    } dxa_xfer_state_t;

    typedef struct packed {
        logic state_idle;
        logic state_wait_rd;
        logic state_wait_wr;
        logic [`MEM_ADDR_WIDTH-1:0] cur_gmem_byte_addr;
        logic [`MEM_ADDR_WIDTH-1:0] cur_smem_byte_addr;
        logic cur_need_skip;
        logic cur_need_fill;
        logic cur_need_read;
        logic cur_read_from_gmem;
        logic cur_read_from_smem;
        logic [`L2_LINE_SIZE * 8-1:0] pending_gmem_wr_data_shifted;
        logic [XLENB * 8-1:0] pending_smem_wr_data_shifted;
        logic [`L2_LINE_SIZE-1:0] pending_gmem_byteen;
        logic [XLENB-1:0] pending_smem_byteen;
        logic [63:0] gmem_rsp_data_shifted;
        logic [63:0] smem_rsp_data_shifted;
        logic [`L2_LINE_SIZE-1:0] cur_gmem_byteen;
        logic [XLENB-1:0] cur_smem_byteen;
    } dxa_xfer_math_t;

    typedef struct packed {
        logic gmem_rd_req_fire;
        logic smem_rd_req_fire;
        logic gmem_wr_req_fire;
        logic smem_wr_req_fire;
        logic gmem_rsp_fire;
        logic smem_rsp_fire;
    } dxa_xfer_evt_t;

endpackage

`IGNORE_UNUSED_END

`endif // VX_DXA_PKG_VH
