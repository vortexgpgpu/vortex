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

// DXA Address Generator: CL-aware address + narrow metadata tokens.
// Supports 1D-5D tiles via nested odometer counters for outer dimensions.
// Inner loop: CL within a row (dim 0). Outer dims 1..4: odometer advance.
//
// Single rolling-cursor scheme: gmem_cursor_r is updated by per-wrap
// deltas precomputed in setup (delta[0]=stride[0], delta[d>0]=stride[d]-
// (tile[d-1]-1)*stride[d-1]).

`include "VX_define.vh"

module VX_dxa_addr_gen import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_LINE_SIZE  = `VX_CFG_L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE)
) (
    input  wire                        clk,
    input  wire                        reset,

    // Start + setup params (latched on start).
    input  wire                        start,
    input  dxa_setup_params_t          setup_params,

    // CL request output (valid/ready).
    output wire                        out_valid,
    input  wire                        out_ready,
    output wire [GMEM_ADDR_WIDTH-1:0]  out_cl_addr,
    output wire [DXA_SMEM_ADDR_W-1:0]  out_smem_byte_addr,
    output wire [CL_OFF_BITS-1:0]      out_byte_offset,
    output wire [CL_OFF_BITS:0]        out_valid_length,
    output wire                        out_oob,
    output wire                        out_last,

    // Pass-through params for downstream (stable during transfer).
    output wire [31:0]                 out_cfill,
    output wire                        out_dest_kmajor,
    output wire [15:0]                 out_per_lane_stride_bytes,
    output wire [3:0]                  out_elem_bytes
);
    localparam CL_OFF_BITS = `CLOG2(GMEM_LINE_SIZE);

    `STATIC_ASSERT(`IS_POW2(GMEM_LINE_SIZE), ("GMEM_LINE_SIZE must be power of 2"))

    // ---- Registered state ----
    reg                        active_r;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0]  gmem_cursor_r;        // Current row's GMEM base
    reg [31:0]                 row_len_r;            // row_len_bytes (constant)
    reg [31:0]                 line_idx_r;           // CL index within current row

    // Outer-dim odometer.
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_count_r;   // Current index per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_tile_r;    // Tile limit per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] delta_r;       // Wrap delta per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] oob_limit_r;   // OOB limit per dim

    // SMEM byte address tracking.
    reg [DXA_SMEM_ADDR_W-1:0]  smem_byte_addr_r;
    // K-major scatter-mode state.
    //   km_row_base_r: SMEM base for the current outer-dim row (= initial +
    //                  dim_count[0] * elem_bytes). Updates only at row wrap.
    //   km_dest_kmajor_r / km_per_lane_stride / km_elem_bytes: stable params.
    reg                        km_dest_kmajor_r;
    reg [15:0]                 km_per_lane_stride_r;
    reg [3:0]                  km_elem_bytes_r;
    // per_lane_stride / elem_bytes, held stable per transfer. elem_bytes is a
    // power of 2 (per_lane_stride = tile1 * elem_bytes), so the divide is an
    // exact right shift by log2(elem_bytes) — no runtime divider on the path.
    reg [31:0]                 km_tile1_r;
    wire [1:0] start_km_esize = (setup_params.elem_bytes == 4'd8) ? 2'd3
                              : (setup_params.elem_bytes == 4'd4) ? 2'd2
                              : (setup_params.elem_bytes == 4'd2) ? 2'd1 : 2'd0;
    reg [DXA_SMEM_ADDR_W-1:0]  km_row_base_r;

    // Pass-through latched params.
    reg [31:0]                 cfill_r;

    assign out_cfill = cfill_r;
    assign out_dest_kmajor = km_dest_kmajor_r;
    assign out_per_lane_stride_bytes = km_per_lane_stride_r;
    assign out_elem_bytes = km_elem_bytes_r;

    // ---- Per-row geometry (combinatorial from current row state) ----
    wire [CL_OFF_BITS-1:0] first_off = gmem_cursor_r[CL_OFF_BITS-1:0];
    wire [31:0] bytes_span = 32'(first_off) + row_len_r;
    wire [31:0] num_lines  = (bytes_span + GMEM_LINE_SIZE - 1) >> CL_OFF_BITS;

    // ---- Current CL address ----
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] first_cl_base = {gmem_cursor_r[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS],
                                                 {CL_OFF_BITS{1'b0}}};
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cur_cl_byte_addr = first_cl_base
        + (`VX_CFG_MEM_ADDR_WIDTH'(line_idx_r) << CL_OFF_BITS);
    wire [GMEM_ADDR_WIDTH-1:0] cur_cl_addr = cur_cl_byte_addr[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS];

    // ---- Narrow token: byte_offset + valid_length ----
    wire [CL_OFF_BITS-1:0] cur_byte_offset = (line_idx_r == 0) ? first_off : '0;

    wire [31:0] total_end = 32'(first_off) + row_len_r;
    wire [CL_OFF_BITS-1:0] last_end_off = total_end[CL_OFF_BITS-1:0];
    wire last_aligned = (last_end_off == '0);
    wire is_last_line = (line_idx_r + 1 >= num_lines);

    wire [CL_OFF_BITS:0] valid_end = is_last_line
        ? (last_aligned ? (CL_OFF_BITS+1)'(GMEM_LINE_SIZE) : {1'b0, last_end_off})
        : (CL_OFF_BITS+1)'(GMEM_LINE_SIZE);

    wire [CL_OFF_BITS:0] valid_start = (line_idx_r == 0) ? {1'b0, first_off} : '0;
    wire [CL_OFF_BITS:0] cur_valid_length = valid_end - valid_start;

    // ---- OOB detection: any outer dim exceeds its limit ----
    wire is_oob = (dim_count_r[0] >= oob_limit_r[0])
               || (dim_count_r[1] >= oob_limit_r[1])
               || (dim_count_r[2] >= oob_limit_r[2])
               || (dim_count_r[3] >= oob_limit_r[3]);

    // ---- Last-row detection: all outer dims at their max ----
    wire is_last_outer = (dim_count_r[0] + 1 >= dim_tile_r[0])
                      && (dim_count_r[1] + 1 >= dim_tile_r[1])
                      && (dim_count_r[2] + 1 >= dim_tile_r[2])
                      && (dim_count_r[3] + 1 >= dim_tile_r[3]);

    // ---- Output ----
    assign out_valid          = active_r;
    assign out_cl_addr        = cur_cl_addr;
    assign out_smem_byte_addr = smem_byte_addr_r;
    assign out_byte_offset    = cur_byte_offset;
    assign out_valid_length   = cur_valid_length;
    assign out_oob            = is_oob;
    assign out_last           = is_last_line && is_last_outer;

    // ---- Advance logic ----
    wire advance = out_valid && out_ready;

    // Which delta to apply on a row-boundary step. Selects delta[0..3]
    // based on which outer dim wraps. Computed from current dim_count_r
    // vs dim_tile_r.
    wire dim0_steps = (dim_count_r[0] + 1 < dim_tile_r[0]);
    wire dim1_steps = (dim_count_r[1] + 1 < dim_tile_r[1]);
    wire dim2_steps = (dim_count_r[2] + 1 < dim_tile_r[2]);
    // dim3 must step otherwise (we already checked is_last_outer).

    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] step_delta =
        dim0_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[0]) :
        dim1_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[1]) :
        dim2_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[2]) :
                     `VX_CFG_MEM_ADDR_WIDTH'(delta_r[3]);

    // K-major SMEM step within a row: per-element stride * num elements in
    // this CL = valid_length * (per_lane_stride / esize). km_tile1_r holds the
    // (setup-stable) division result, so only the multiply remains on the path.
    wire [31:0] km_step_in_row = 32'(cur_valid_length) * km_tile1_r;

    always @(posedge clk) begin
        if (reset) begin
            active_r         <= 1'b0;
            gmem_cursor_r    <= '0;
            line_idx_r       <= '0;
            smem_byte_addr_r <= '0;
            // Reset K-major params so out_dest_kmajor (a mux selector
            // downstream in smem_wr) doesn't carry X before the first
            // pipeline_start. Without this, fb_word_data / fb_word_byteen
            // muxes propagate X into the bus even though transfer_active=0.
            km_dest_kmajor_r <= 1'b0;
            km_per_lane_stride_r <= '0;
            km_elem_bytes_r  <= '0;
            km_tile1_r       <= '0;
            km_row_base_r    <= '0;
            for (int d = 0; d < DXA_MAX_OUTER_DIMS; d++) begin
                dim_count_r[d] <= '0;
            end
        end else if (start) begin
            active_r         <= 1'b1;
            gmem_cursor_r    <= setup_params.initial_gmem_base;
            row_len_r        <= setup_params.row_len_bytes;
            line_idx_r       <= '0;
            cfill_r          <= setup_params.cfill;
            smem_byte_addr_r <= setup_params.initial_smem_base;
            km_dest_kmajor_r <= setup_params.dest_kmajor;
            km_per_lane_stride_r <= setup_params.per_lane_stride_bytes;
            km_elem_bytes_r  <= setup_params.elem_bytes;
            km_tile1_r       <= (setup_params.elem_bytes == 4'd0) ? 32'd0
                              : (32'(setup_params.per_lane_stride_bytes) >> start_km_esize);
            km_row_base_r    <= setup_params.initial_smem_base;
            for (int d = 0; d < DXA_MAX_OUTER_DIMS; d++) begin
                dim_count_r[d] <= '0;
                dim_tile_r[d]  <= setup_params.dim_tiles[d];
                delta_r[d]     <= setup_params.delta[d];
                oob_limit_r[d] <= setup_params.oob_limit[d];
            end
        end else if (advance) begin
            // SMEM byte address advance:
            //   Row-major: linear += valid_length.
            //   K-major:   per-CL += valid_length * tile1 (scatter each
            //              element to its column position in the destination
            //              layout). On row wrap, the address is overridden
            //              below to the next column's row base.
            if (km_dest_kmajor_r) begin
                smem_byte_addr_r <= smem_byte_addr_r + DXA_SMEM_ADDR_W'(km_step_in_row);
            end else begin
                smem_byte_addr_r <= smem_byte_addr_r + DXA_SMEM_ADDR_W'(cur_valid_length);
            end

            if (is_last_line && is_last_outer) begin
                // All done.
                active_r <= 1'b0;
            end else if (is_last_line) begin
                // Last CL of current row → advance odometer + cursor.
                line_idx_r    <= '0;
                gmem_cursor_r <= gmem_cursor_r + step_delta;
                // K-major: SMEM jumps back to the next row base (initial +
                // (i1+1) * esize). Maintained via km_row_base_r which we bump
                // by esize at each row wrap. Row-major path keeps its linear
                // cursor — the override above is unconditional, so it would
                // step PAST the row base; we restore it here for K-major.
                if (km_dest_kmajor_r) begin
                    smem_byte_addr_r <= km_row_base_r + DXA_SMEM_ADDR_W'(km_elem_bytes_r);
                    km_row_base_r    <= km_row_base_r + DXA_SMEM_ADDR_W'(km_elem_bytes_r);
                end
                if (dim0_steps) begin
                    dim_count_r[0] <= dim_count_r[0] + 1;
                end else begin
                    dim_count_r[0] <= '0;
                    if (dim1_steps) begin
                        dim_count_r[1] <= dim_count_r[1] + 1;
                    end else begin
                        dim_count_r[1] <= '0;
                        if (dim2_steps) begin
                            dim_count_r[2] <= dim_count_r[2] + 1;
                        end else begin
                            dim_count_r[2] <= '0;
                            dim_count_r[3] <= dim_count_r[3] + 1;
                        end
                    end
                end
            end else begin
                line_idx_r <= line_idx_r + 1;
            end
        end
    end

    `UNUSED_VAR (cur_cl_byte_addr[CL_OFF_BITS-1:0])
    `UNUSED_VAR (total_end[31:CL_OFF_BITS])
    // km_step_in_row is LMEM-bounded; only the low DXA_SMEM_ADDR_W bits feed smem_byte_addr_r.
    `UNUSED_VAR (km_step_in_row[31:DXA_SMEM_ADDR_W])

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && advance) begin
            $write("DXA_PIPE,%0d,AG_OUT,addr=0x%0h,off=%0d,len=%0d,oob=%0d,last=%0d,dim={%0d,%0d,%0d,%0d},cl=%0d,smem=0x%0h\n",
                $time, cur_cl_addr, cur_byte_offset, cur_valid_length, is_oob, out_last,
                dim_count_r[0], dim_count_r[1], dim_count_r[2], dim_count_r[3],
                line_idx_r, smem_byte_addr_r);
        end
    end
`endif

endmodule
