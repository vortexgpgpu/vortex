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
// Two-loop structure. The inner loop (CL within a row) advances every beat
// and is pure counters/flags — no 32-bit compares, no DSP on the path:
//   cl_addr_r   : GMEM CL address, ++1 / beat
//   lines_left_r: CLs remaining in the row, --1 / beat; is_last_line == (==1)
//   is_first_r  : current CL is the row's first
//   smem_byte_addr_r += a registered step
// The outer odometer (gmem_cursor, dim counters, num_lines, deltas) changes
// only at a row wrap, so its adds/compares sit on once-per-row paths.
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
    localparam VLEN_W      = CL_OFF_BITS + 1;

    `STATIC_ASSERT(`IS_POW2(GMEM_LINE_SIZE), ("GMEM_LINE_SIZE must be power of 2"))

    // ---- Registered state ----
    reg                        active_r;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] gmem_cursor_r;  // Current row's GMEM byte base
    reg [31:0]                 row_len_r;            // row_len_bytes (constant)

    // Inner-loop counters/flags (advance every beat).
    reg [GMEM_ADDR_WIDTH-1:0]  cl_addr_r;            // Current CL address (GMEM)
    reg [31:0]                 lines_left_r;         // CLs remaining in current row
    reg                        is_first_r;           // current CL is the row's first
    reg [VLEN_W-1:0]           last_vend_r;          // valid_end for the row's last CL

    // Outer-dim odometer.
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_count_r;   // Current index per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_last_r;    // Last index per dim (tile-1)
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

    wire [CL_OFF_BITS-1:0] first_off = gmem_cursor_r[CL_OFF_BITS-1:0];

    // ---- Narrow token: byte_offset + valid_length ----
    // is_last_line is a decremented trip-count reaching 1 — a register-fed
    // equality, so only a small mux/subtract precedes the SMEM-step add/DSP.
    wire is_last_line = (lines_left_r == 32'd1);

    wire [CL_OFF_BITS-1:0] cur_byte_offset = is_first_r ? first_off : '0;

    wire [VLEN_W-1:0] valid_end   = is_last_line ? last_vend_r : VLEN_W'(GMEM_LINE_SIZE);
    wire [VLEN_W-1:0] valid_start = is_first_r ? {1'b0, first_off} : '0;
    wire [VLEN_W-1:0] cur_valid_length = valid_end - valid_start;

    // ---- OOB detection: any outer dim exceeds its limit ----
    wire is_oob = (dim_count_r[0] >= oob_limit_r[0])
               || (dim_count_r[1] >= oob_limit_r[1])
               || (dim_count_r[2] >= oob_limit_r[2])
               || (dim_count_r[3] >= oob_limit_r[3]);

    // ---- Last-row detection: all outer dims at their last index ----
    // An odometer index reaches its bound by equality, not magnitude: with
    // dim_count in [0, tile-1], (dim_count+1 >= tile) ⟺ (dim_count == tile-1).
    // Equality is ~2 LUT levels vs a CARRY8 chain, which keeps the once-per-row
    // step_delta / num_lines wrap path short (see docs/proposals/dxa_addr_gen_timing.md).
    wire dim0_last = (dim_count_r[0] == dim_last_r[0]);
    wire dim1_last = (dim_count_r[1] == dim_last_r[1]);
    wire dim2_last = (dim_count_r[2] == dim_last_r[2]);
    wire dim3_last = (dim_count_r[3] == dim_last_r[3]);
    wire is_last_outer = dim0_last && dim1_last && dim2_last && dim3_last;

    // ---- Output ----
    assign out_valid          = active_r;
    assign out_cl_addr        = cl_addr_r;
    assign out_smem_byte_addr = smem_byte_addr_r;
    assign out_byte_offset    = cur_byte_offset;
    assign out_valid_length   = cur_valid_length;
    assign out_oob            = is_oob;
    assign out_last           = is_last_line && is_last_outer;

    // ---- Advance logic ----
    wire advance = out_valid && out_ready;

    // Which delta to apply on a row-boundary step. Selects delta[0..3]
    // based on which outer dim wraps (the first not yet at its last index).
    wire dim0_steps = ~dim0_last;
    wire dim1_steps = ~dim1_last;
    wire dim2_steps = ~dim2_last;
    // dim3 must step otherwise (we already checked is_last_outer).

    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] step_delta =
        dim0_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[0]) :
        dim1_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[1]) :
        dim2_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[2]) :
                     `VX_CFG_MEM_ADDR_WIDTH'(delta_r[3]);

    // K-major SMEM step = (elements in this CL) * per_lane_stride/esize.
    //   Interior CL: valid_length == LINE, so step == tile1 << log2(LINE) — a
    //                shift, no multiply.
    //   First CL:    (LINE - first_off) * tile1 — one multiply, fed directly by
    //                registers (first_off, km_tile1_r), both row-stable.
    //   Last CL:     don't-care; the row wrap overrides smem_byte_addr_r below.
    // The per-beat path is therefore a 2:1 mux of these, never the long
    // valid_length cone in front of the multiply (see
    // docs/proposals/dxa_addr_gen_timing.md).
    wire [VLEN_W-1:0] cl0_len      = VLEN_W'(GMEM_LINE_SIZE) - {1'b0, first_off};
    wire [31:0] km_step_full  = km_tile1_r << CL_OFF_BITS;
    wire [31:0] km_step_first = 32'(cl0_len) * km_tile1_r;
    wire [31:0] km_step       = is_first_r ? km_step_first : km_step_full;

    // Span (first_off + row_len) for the row about to be entered, and its CL
    // count. At start it comes from setup params; at a row wrap from the
    // post-step cursor (reusing the gmem_cursor_r adder).
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] next_cursor = gmem_cursor_r + step_delta;
    wire [31:0] start_span = 32'(setup_params.initial_gmem_base[CL_OFF_BITS-1:0])
                           + setup_params.row_len_bytes;
    wire [31:0] wrap_span  = 32'(next_cursor[CL_OFF_BITS-1:0]) + row_len_r;

    function automatic logic [31:0] calc_num_lines(input logic [31:0] span);
        calc_num_lines = (span + GMEM_LINE_SIZE - 1) >> CL_OFF_BITS;
    endfunction
    function automatic logic [VLEN_W-1:0] calc_last_vend(input logic [CL_OFF_BITS-1:0] off);
        calc_last_vend = (off == '0) ? VLEN_W'(GMEM_LINE_SIZE) : {1'b0, off};
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            active_r         <= 1'b0;
            gmem_cursor_r    <= '0;
            cl_addr_r        <= '0;
            lines_left_r     <= '0;
            is_first_r       <= 1'b0;
            last_vend_r      <= '0;
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
            cl_addr_r        <= setup_params.initial_gmem_base[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS];
            lines_left_r     <= calc_num_lines(start_span);
            is_first_r       <= 1'b1;
            last_vend_r      <= calc_last_vend(start_span[CL_OFF_BITS-1:0]);
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
                dim_last_r[d]  <= setup_params.dim_tiles[d] - 32'd1;
                delta_r[d]     <= setup_params.delta[d];
                oob_limit_r[d] <= setup_params.oob_limit[d];
            end
        end else if (advance) begin
            // SMEM byte address advance:
            //   Row-major: linear += valid_length.
            //   K-major:   per-CL += km_step (scatter each element to its column
            //              position). On a row wrap the address is overridden
            //              below to the next column's row base.
            if (km_dest_kmajor_r) begin
                smem_byte_addr_r <= smem_byte_addr_r + DXA_SMEM_ADDR_W'(km_step);
            end else begin
                smem_byte_addr_r <= smem_byte_addr_r + DXA_SMEM_ADDR_W'(cur_valid_length);
            end

            if (is_last_line && is_last_outer) begin
                // All done.
                active_r <= 1'b0;
            end else if (is_last_line) begin
                // Last CL of current row → advance odometer + cursor. The new
                // row's geometry is recomputed from the post-step cursor; the
                // next beat is the new row's CL0.
                gmem_cursor_r <= next_cursor;
                cl_addr_r     <= next_cursor[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS];
                lines_left_r  <= calc_num_lines(wrap_span);
                is_first_r    <= 1'b1;
                last_vend_r   <= calc_last_vend(wrap_span[CL_OFF_BITS-1:0]);
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
                // Next CL of the same row.
                cl_addr_r    <= cl_addr_r + 1;
                lines_left_r <= lines_left_r - 1;
                is_first_r   <= 1'b0;
            end
        end
    end

    // km_step is LMEM-bounded; only the low DXA_SMEM_ADDR_W bits feed smem_byte_addr_r.
    `UNUSED_VAR (km_step[31:DXA_SMEM_ADDR_W])

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && advance) begin
            $write("DXA_PIPE,%0d,AG_OUT,addr=0x%0h,off=%0d,len=%0d,oob=%0d,last=%0d,dim={%0d,%0d,%0d,%0d},left=%0d,smem=0x%0h\n",
                $time, cl_addr_r, cur_byte_offset, cur_valid_length, is_oob, out_last,
                dim_count_r[0], dim_count_r[1], dim_count_r[2], dim_count_r[3],
                lines_left_r, smem_byte_addr_r);
        end
    end
`endif

endmodule
