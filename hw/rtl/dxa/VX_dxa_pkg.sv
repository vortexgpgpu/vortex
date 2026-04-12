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

package VX_dxa_pkg;

    import VX_gpu_pkg::*;

    // DXA local-memory path: dimensioned to cover all banks in one request.
    localparam DXA_LMEM_WORD_SIZE = `LMEM_NUM_BANKS * (`XLEN / 8);
    localparam DXA_LMEM_ADDR_W = (`MEM_ADDR_WIDTH - `CLOG2(DXA_LMEM_WORD_SIZE));

    localparam DXA_DESC_SLOT_BITS = `CLOG2(`VX_DCR_DXA_DESC_COUNT);
    localparam DXA_DESC_SLOT_W    = `UP(DXA_DESC_SLOT_BITS);

    // DXA request data — core → DXA control path.
    typedef struct packed {
        logic [NC_WIDTH-1:0]      core_id;
        logic [UUID_WIDTH-1:0]    uuid;
        logic [NW_WIDTH-1:0]      wid;
        logic [`XLEN-1:0]         smem_addr;   // from lane 0 rs1
        logic [`XLEN-1:0]         meta;        // from lane 1 rs1 (desc[3:0], bar[30:4], 1[31])
        logic [4:0][`XLEN-1:0]    coords;      // [0]=lane2.rs1,[1]=lane3.rs1,[2]=lane0.rs2,[3]=lane1.rs2,[4]=lane2.rs2
        logic [`NUM_WARPS-1:0]    cta_mask;     // from rs2 lane 3
    } dxa_req_data_t;

    // Decoded launch data — dispatched from FIFO to workers.
    typedef struct packed {
        logic [NC_WIDTH-1:0]        core_id;
        logic [UUID_WIDTH-1:0]      uuid;
        logic [NW_WIDTH-1:0]        wid;
        logic [BAR_ADDR_W-1:0]      bar_addr;
        logic [DXA_DESC_SLOT_W-1:0] desc_slot;
        logic [`XLEN-1:0]           smem_addr;
        logic [4:0][`XLEN-1:0]      coords;
        logic [`NUM_WARPS-1:0]      cta_mask;
    } dxa_launch_t;

    // Descriptor table read output — all fields for one slot.
    typedef struct packed {
        logic [`MEM_ADDR_WIDTH-1:0] base_addr;
        logic [31:0] meta;
        logic [31:0] tile01;
        logic [31:0] tile23;
        logic [31:0] tile4;
        logic [31:0] cfill;
        logic [31:0] size0;
        logic [31:0] size1;
        logic [31:0] size2;
        logic [31:0] size3;
        logic [31:0] size4;
        logic [31:0] stride0;
        logic [31:0] stride1;
        logic [31:0] stride2;
        logic [31:0] stride3;
        logic [31:0] smem_stride;
    } dxa_desc_t;

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
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] strides;    // per-dim strides (stride0..stride3)
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_tiles;  // per-dim tile limits (tile1..tile4)
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] oob_limit;
        logic [31:0]                total_rows;
        logic [31:0]                total_smem_writes;
        logic [31:0]                total_bytes;
        logic [31:0]                cfill;
        logic [31:0]                elem_bytes;
        logic [31:0]                rank;
    } dxa_setup_params_t;

    task automatic trace_ex_op(input int level,
                     input [INST_OP_BITS-1:0] op_type,
                     input op_args_t op_args
    );
        `UNUSED_VAR (op_type)
        `UNUSED_VAR (op_args)
        `TRACE(level, ("DXA.ISSUE"))
    endtask

endpackage

`endif // VX_DXA_PKG_VH
