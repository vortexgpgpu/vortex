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
    localparam DXA_LMEM_WORD_SIZE = `VX_CFG_LMEM_NUM_BANKS * (`VX_CFG_XLEN / 8);
    localparam DXA_LMEM_ADDR_W = LMEM_DMA_ADDR_WIDTH;

    // LMEM byte-address width: an LMEM-relative byte offset, same as
    // VX_local_mem's ADDR_WIDTH = `CLOG2(SIZE) = `VX_CFG_LMEM_LOG_SIZE.
    localparam DXA_SMEM_ADDR_W = `VX_CFG_LMEM_LOG_SIZE;

    localparam DXA_DESC_SLOT_BITS = `CLOG2(`VX_DCR_DXA_DESC_COUNT);
    localparam DXA_DESC_SLOT_W    = `UP(DXA_DESC_SLOT_BITS);

    // DXA request data — core → DXA control path.
    typedef struct packed {
        logic [NC_WIDTH-1:0]      core_id;
        logic [UUID_WIDTH-1:0]    uuid;
        logic [NW_WIDTH-1:0]      wid;
        logic [DXA_SMEM_ADDR_W-1:0]      smem_addr;   // from lane 0 rs1; LMEM byte address
        logic [31:0]                     meta;        // from lane 1 rs1 (desc[3:0], bar[30:4], 1[31]); 32-bit ABI word
        logic [4:0][31:0]                coords;      // [0]=lane2.rs1,[1]=lane3.rs1,[2]=lane0.rs2,[3]=lane1.rs2,[4]=lane2.rs2; element indices, 32-bit ABI
        logic [`VX_CFG_NUM_WARPS-1:0]    cta_mask;     // from rs2 lane 3
    } dxa_req_data_t;

    // Descriptor table read output — all fields for one slot.
    typedef struct packed {
        logic [`VX_CFG_MEM_ADDR_WIDTH-1:0] base_addr;
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
        logic [31:0] estride2;    // WGMMA tile geometry (tcN) for Flat/BlockMajor
    } dxa_desc_t;

    // Destination SMEM layout (2-bit LAYOUT meta field). Mirrors dxa.h::Layout
    // and sim/simx/dxa/dxa_core.cpp::DestLayout.
    localparam DXA_DEST_ROWMAJOR   = 2'd0;
    localparam DXA_DEST_KMAJOR     = 2'd1;
    localparam DXA_DEST_FLAT       = 2'd2;  // sparse B candidate-pair (b_sp_flat_idx)
    localparam DXA_DEST_BLOCKMAJOR = 2'd3;  // dense B block-major (b_blockmajor_idx)

    // ── Line-granularity types ──────────────────────────────────────────

    // Maximum outer dimensions for tile iteration (dims 1..4 for up to 5D).
    localparam DXA_MAX_OUTER_DIMS = 4;

    // Setup parameters: precomputed constants for addr_gen, rd_ctrl, cl2smem, wr_ctrl.
    // All multiplies happen during setup; fast path uses additions only.
    typedef struct packed {
        logic [`VX_CFG_MEM_ADDR_WIDTH-1:0]  initial_gmem_base;
        logic [DXA_SMEM_ADDR_W-1:0]         initial_smem_base;
        logic [31:0]                row_len_bytes;
        // Rolling-cursor deltas applied at each outer-dim step:
        //   delta[0]: dim-0 step                = stride[0]
        //   delta[d>0]: dim (d-1)→d wrap event  = stride[d] - (tile[d-1]-1)*stride[d-1]
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] delta;
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_tiles;  // per-dim tile limits (tile1..tile4)
        logic [DXA_MAX_OUTER_DIMS-1:0][31:0] oob_limit;
        logic [31:0]                cfill;
        // K-major destination layout. When set, smem_wr
        // scatters one element per beat at +per_lane_stride_bytes per element,
        // producing smem[i0 * tile1 + i1 * ... ] instead of the default
        // row-major smem[i1 * tile0 + i0 * ...] layout. rank ≤ 2 only.
        logic                       dest_kmajor;
        logic [15:0]                per_lane_stride_bytes;  // = tile1 * elem_bytes
        logic [3:0]                 elem_bytes;             // 1, 2, 4, or 8
        // Tiled (Flat/BlockMajor) scatter: the per-element SMEM destination is
        // the bbuf-native index (vx_tensor.h::b_sp_flat_idx / b_blockmajor_idx).
        // All tile divisors are powers of two → log shift amounts. Derived once
        // in setup from tcN (estride2) and elem_bytes.
        logic [1:0]                 dest_mode;   // DXA_DEST_*
        logic [3:0]                 lg_ratio;    // log2(32-bit-word / elem) = 2 - log2(elem_bytes)
        logic [3:0]                 lg_tcN;      // log2(tcN)  (= log2(tcK))
        logic [3:0]                 lg_nsteps;   // log2(xtileN / tcN)
    } dxa_setup_params_t;

    // SMEM byte destination for B element (k = K-row, n = N-col) under the
    // bbuf-native Flat / BlockMajor layouts. Mirrors vx_tensor.h::b_sp_flat_idx
    // and b_blockmajor_idx; tcK == tcN for canonical WGMMA configs. Every
    // divisor is a power of two, so the body is shifts/masks/adds (no DSP).
    function automatic [DXA_SMEM_ADDR_W-1:0] dxa_tiled_dest_byte(
        input logic [1:0]  mode,
        input logic [15:0] k,
        input logic [15:0] n,
        input logic [3:0]  lg_ratio,
        input logic [3:0]  lg_tcN,
        input logic [3:0]  lg_nsteps,
        input logic [3:0]  esize       // log2(elem_bytes)
    );
        logic [15:0] tcN_mask, ratio_mask, ktck_mask, kw_mask;
        logic [15:0] k_word, elem, k_blk, kw_in, n_blk, n_in, r_in;
        logic [31:0] word_off, dest_elem;
        tcN_mask   = (16'd1 << lg_tcN) - 16'd1;
        ratio_mask = (16'd1 << lg_ratio) - 16'd1;
        n_blk      = n >> lg_tcN;
        n_in       = n & tcN_mask;
        if (mode == DXA_DEST_FLAT) begin
            // word_off = (k_blk*n_steps + n_blk)*blk_words + kw_in*tcN + n_in,
            // blk_words = tcN*(tcN*2); dest = word_off*ratio + elem.
            ktck_mask = (16'd1 << (lg_tcN + 4'd1)) - 16'd1;   // (tcN*2) - 1
            k_word    = k >> lg_ratio;
            elem      = k & ratio_mask;
            k_blk     = k_word >> (lg_tcN + 4'd1);
            kw_in     = k_word & ktck_mask;
            word_off  = ((((32'(k_blk) << lg_nsteps) + 32'(n_blk)) << (lg_tcN + lg_tcN + 4'd1))
                         + (32'(kw_in) << lg_tcN) + 32'(n_in));
            dest_elem = (word_off << lg_ratio) + 32'(elem);
        end else begin
            // BlockMajor: dest = (k_blk*n_steps + n_blk)*b_blk_elems + n_in*kw + r_in,
            // kw = tcN*ratio, b_blk_elems = kw*tcN.
            kw_mask   = (16'd1 << (lg_tcN + lg_ratio)) - 16'd1;
            k_blk     = k >> (lg_tcN + lg_ratio);
            r_in      = k & kw_mask;
            dest_elem = ((((32'(k_blk) << lg_nsteps) + 32'(n_blk)) << (lg_tcN + lg_ratio + lg_tcN))
                         + (32'(n_in) << (lg_tcN + lg_ratio)) + 32'(r_in));
        end
        dxa_tiled_dest_byte = DXA_SMEM_ADDR_W'(dest_elem << esize);
    endfunction

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
