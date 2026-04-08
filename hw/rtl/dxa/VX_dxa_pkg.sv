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

    localparam DXA_OP_SETUP   = 3'd0;
    localparam DXA_OP_COORD01 = 3'd1;
    localparam DXA_OP_COORD23 = 3'd2;
    localparam DXA_OP_ISSUE   = 3'd3;
    // Architected funct3 encoding: 0=1D, 1=2D, 2=3D, 3=4D, 4=5D.
    // All variants are expanded into micro-ops by VX_dxa_uops.

    // smem_addr(XLEN) + meta(XLEN) + coords[5](5*XLEN) = 7*XLEN total
    localparam DXA_REQ_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + (7 * `XLEN)
`ifdef EXT_DXA_MULTICAST_ENABLE
        + 1 + `NUM_WARPS  // is_multicast + cta_mask
`endif
        ;

    // Keep compatibility with existing global DXA descriptor macros.
    localparam DXA_DESC_SLOT_BITS = `CLOG2(`VX_DCR_DXA_DESC_COUNT);
    localparam DXA_DESC_SLOT_W    = `UP(DXA_DESC_SLOT_BITS);
    localparam DXA_DESC_WORD_BITS = `CLOG2(`VX_DCR_DXA_DESC_STRIDE);
    localparam DXA_DESC_WORD_W    = `UP(DXA_DESC_WORD_BITS);
    localparam DXA_DONE_ENGINE_BITS = `CLOG2(`NUM_DXA_UNITS);
    localparam DXA_DONE_ENGINE_W    = `UP(DXA_DONE_ENGINE_BITS);

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

    // typedef struct packed {
    //     logic active;
    //     logic [NC_WIDTH-1:0] core_id;
    //     logic [UUID_WIDTH-1:0] uuid;
    //     logic [NW_WIDTH-1:0] wid;
    //     logic [BAR_ADDR_W-1:0] bar_addr;
    //     logic notify_via_smem_done;
    //     logic is_s2g;
    //     logic [`MEM_ADDR_WIDTH-1:0] gbase;
    //     logic [`XLEN-1:0] smem_base;
    //     logic [31:0] coord0;
    //     logic [31:0] coord1;
    //     logic [31:0] size0;
    //     logic [31:0] size1;
    //     logic [31:0] stride0;
    //     logic [31:0] tile0;
    //     logic [31:0] tile1;
    //     logic [31:0] elem_bytes;
    //     logic [31:0] cfill;
    //     logic [31:0] idx;
    //     logic [31:0] elem_x;
    //     logic [31:0] elem_y;
    //     logic [31:0] total;
    //     logic [1:0] elem_state;
    //     logic wait_rsp_from_gmem;
    //     logic write_to_gmem;
    //     logic [`MEM_ADDR_WIDTH-1:0] pending_rd_byte_addr;
    //     logic [`MEM_ADDR_WIDTH-1:0] pending_wr_byte_addr;
    //     logic [63:0] pending_elem_data;
    // } dxa_xfer_state_t;

    // typedef struct packed {
    //     logic state_idle;
    //     logic state_wait_rd;
    //     logic state_wait_wr;
    //     logic [`MEM_ADDR_WIDTH-1:0] cur_gmem_byte_addr;
    //     logic [`MEM_ADDR_WIDTH-1:0] cur_smem_byte_addr;
    //     logic cur_need_skip;
    //     logic cur_need_fill;
    //     logic cur_need_read;
    //     logic cur_read_from_gmem;
    //     logic cur_read_from_smem;
    //     logic [`L2_LINE_SIZE * 8-1:0] pending_gmem_wr_data_shifted;
    //     logic [DXA_LMEM_WORD_SIZE * 8-1:0] pending_smem_wr_data_shifted;
    //     logic [`L2_LINE_SIZE-1:0] pending_gmem_byteen;
    //     logic [DXA_LMEM_WORD_SIZE-1:0] pending_smem_byteen;
    //     logic [63:0] gmem_rsp_data_shifted;
    //     logic [63:0] smem_rsp_data_shifted;
    //     logic [`L2_LINE_SIZE-1:0] cur_gmem_byteen;
    //     logic [DXA_LMEM_WORD_SIZE-1:0] cur_smem_byteen;
    // } dxa_xfer_math_t;

    // typedef struct packed {
    //     logic gmem_rd_req_fire;
    //     logic smem_rd_req_fire;
    //     logic gmem_wr_req_fire;
    //     logic smem_wr_req_fire;
    //     logic gmem_rsp_fire;
    //     logic smem_rsp_fire;
    // } dxa_xfer_evt_t;

    // DXA descriptor table entry — one per programmed descriptor slot.
    // Field order matches DCR word offsets (highest offset = MSB).
    // Total width = VX_DCR_DXA_DESC_STRIDE * 32 bits.
    localparam DXA_DESC_DATAW = `VX_DCR_DXA_DESC_STRIDE * 32;
    typedef struct packed {
    `ifdef EXT_DXA_MULTICAST_ENABLE
        logic [31:0] _pad0;           // offset 23 (unused)
        logic [31:0] bar_stride;      // offset 22
        logic [31:0] smem_stride;     // offset 21
    `else
        logic [2:0][31:0] _pad0;      // offsets 21-23 (unused)
    `endif
        logic [31:0] cfill;           // offset 20
        logic [31:0] tilesize4;       // offset 19
        logic [31:0] tilesize23;      // offset 18
        logic [31:0] tilesize01;      // offset 17
        logic [31:0] estride4;        // offset 16
        logic [31:0] estride3;        // offset 15
        logic [31:0] estride2;        // offset 14
        logic [31:0] estride1;        // offset 13
        logic [31:0] estride0;        // offset 12
        logic [31:0] meta;            // offset 11
        logic [31:0] stride3;         // offset 10
        logic [31:0] stride2;         // offset 9
        logic [31:0] stride1;         // offset 8
        logic [31:0] stride0;         // offset 7
        logic [31:0] size4;           // offset 6
        logic [31:0] size3;           // offset 5
        logic [31:0] size2;           // offset 4
        logic [31:0] size1;           // offset 3
        logic [31:0] size0;           // offset 2
        logic [31:0] base_hi;         // offset 1
        logic [31:0] base_lo;         // offset 0
    } dxa_desc_t;

    // Next-gen unified-engine skeleton types.
    typedef struct packed {
        logic [NC_WIDTH-1:0]      core_id;
        logic [NW_WIDTH-1:0]      wid;
        logic [BAR_ADDR_W-1:0]    bar_addr;
        logic [`MEM_ADDR_WIDTH-1:0] gmem_base;
        logic [`XLEN-1:0]         smem_base;
        logic [4:0][`XLEN-1:0]    coords;
        dxa_issue_dec_t           dec;
    } dxa_worker_cmd_t;

    typedef struct packed {
        logic [NC_WIDTH-1:0]      core_id;
        logic [NW_WIDTH-1:0]      wid;
        logic [BAR_ADDR_W-1:0]    bar_addr;
        logic                     done;
    } dxa_completion_info_t;

    typedef struct packed {
        logic [BAR_ADDR_W-1:0]    bar_addr;
    } dxa_smem_done_t;

    // ── Line-granularity refactor types ──────────────────────────────────

    // Maximum outer dimensions for tile iteration (dims 1..4 for up to 5D).
    localparam DXA_MAX_OUTER_DIMS = 4;

    // Setup parameters: precomputed constants for addr_gen, rd_ctrl, cl2smem, wr_ctrl.
    // All multiplies happen during setup; fast path uses additions only.
    typedef struct packed {
        logic [`MEM_ADDR_WIDTH-1:0]  initial_gmem_base;
        logic [`XLEN-1:0]           initial_smem_base;
        logic [31:0]                row_len_bytes;
        logic [31:0]                stride0;
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] oob_limit;
        logic [31:0]                total_rows;
        logic [31:0]                total_smem_writes;
        logic [31:0]                cfill;
        logic [31:0]                elem_bytes;
        logic [31:0]                rank;
    } dxa_setup_params_t;

endpackage

`IGNORE_UNUSED_END

`endif // VX_DXA_PKG_VH
