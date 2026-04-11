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
// Per-dim GMEM offset accumulators avoid runtime multiplies.

`include "VX_define.vh"

module VX_dxa_addr_gen import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_LINE_SIZE  = `L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE)
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
    output wire [`MEM_ADDR_WIDTH-1:0]  out_smem_byte_addr,
    output wire [CL_OFF_BITS-1:0]      out_byte_offset,
    output wire [CL_OFF_BITS:0]        out_valid_length,
    output wire                        out_oob,
    output wire                        out_last,

    // Pass-through params for downstream (stable during transfer).
    output wire [31:0]                 out_cfill,
    output wire [31:0]                 out_total_smem_writes
);
    localparam CL_OFF_BITS = `CLOG2(GMEM_LINE_SIZE);

    `STATIC_ASSERT(`IS_POW2(GMEM_LINE_SIZE), ("GMEM_LINE_SIZE must be power of 2"))

    // ---- Registered state ----
    reg                        active_r;
    reg [`MEM_ADDR_WIDTH-1:0]  initial_gmem_base_r; // Base GMEM addr (with coord offsets)
    reg [31:0]                 row_len_r;            // row_len_bytes (constant)
    reg [31:0]                 line_idx_r;            // CL index within current row

    // Nested odometer counters for outer dimensions (dims 1..4).
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]           dim_count_r;   // Current index per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]           dim_tile_r;    // Tile limit per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]           stride_r;      // Stride per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][`MEM_ADDR_WIDTH-1:0] dim_offset_r;  // Accumulated GMEM offset per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]           oob_limit_r;   // OOB limit per dim

    // SMEM byte address tracking
    reg [`MEM_ADDR_WIDTH-1:0]  smem_byte_addr_r;

    // Pass-through latched params
    reg [31:0]                 cfill_r;
    reg [31:0]                 total_smem_writes_r;

    assign out_cfill = cfill_r;
    assign out_total_smem_writes = total_smem_writes_r;

    // ---- GMEM base: sum of initial base + per-dim offsets ----
    wire [`MEM_ADDR_WIDTH-1:0] gmem_base = initial_gmem_base_r
        + dim_offset_r[0] + dim_offset_r[1]
        + dim_offset_r[2] + dim_offset_r[3];

    // ---- Per-row geometry (combinatorial from current row state) ----
    wire [CL_OFF_BITS-1:0] first_off = gmem_base[CL_OFF_BITS-1:0];
    wire [31:0] bytes_span = 32'(first_off) + row_len_r;
    wire [31:0] num_lines  = (bytes_span + GMEM_LINE_SIZE - 1) >> CL_OFF_BITS;

    // ---- Current CL address ----
    wire [`MEM_ADDR_WIDTH-1:0] first_cl_base = {gmem_base[`MEM_ADDR_WIDTH-1:CL_OFF_BITS],
                                                 {CL_OFF_BITS{1'b0}}};
    wire [`MEM_ADDR_WIDTH-1:0] cur_cl_byte_addr = first_cl_base
        + (`MEM_ADDR_WIDTH'(line_idx_r) << CL_OFF_BITS);
    wire [GMEM_ADDR_WIDTH-1:0] cur_cl_addr = cur_cl_byte_addr[`MEM_ADDR_WIDTH-1:CL_OFF_BITS];

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
    assign out_valid        = active_r;
    assign out_cl_addr      = cur_cl_addr;
    assign out_smem_byte_addr = smem_byte_addr_r;
    assign out_byte_offset  = cur_byte_offset;
    assign out_valid_length = cur_valid_length;
    assign out_oob          = is_oob;
    assign out_last         = is_last_line && is_last_outer;

    // ---- Advance logic ----
    wire advance = out_valid && out_ready;

    // Assume correct input: total_rows > 0.
    `RUNTIME_ASSERT(!start || (setup_params.total_rows != 0), ("DXA addr_gen: total_rows is 0"))

    always @(posedge clk) begin
        if (reset) begin
            active_r          <= 1'b0;
            initial_gmem_base_r <= '0;
            line_idx_r        <= '0;
            smem_byte_addr_r  <= '0;
            for (int d = 0; d < DXA_MAX_OUTER_DIMS; d++) begin
                dim_count_r[d]  <= '0;
                dim_offset_r[d] <= '0;
            end
        end else if (start) begin
            active_r            <= 1'b1;
            initial_gmem_base_r <= setup_params.initial_gmem_base;
            row_len_r           <= setup_params.row_len_bytes;
            line_idx_r          <= '0;
            cfill_r             <= setup_params.cfill;
            total_smem_writes_r <= setup_params.total_smem_writes;
            smem_byte_addr_r    <= `MEM_ADDR_WIDTH'(setup_params.initial_smem_base);
            for (int d = 0; d < DXA_MAX_OUTER_DIMS; d++) begin
                dim_count_r[d]  <= '0;
                dim_tile_r[d]   <= setup_params.dim_tiles[d];
                stride_r[d]     <= setup_params.strides[d];
                dim_offset_r[d] <= '0;
                oob_limit_r[d]  <= setup_params.oob_limit[d];
            end
        end else if (advance) begin
            // Advance SMEM byte address by valid_length
            smem_byte_addr_r <= smem_byte_addr_r + `MEM_ADDR_WIDTH'(cur_valid_length);

            if (is_last_line && is_last_outer) begin
                // All done
                active_r <= 1'b0;
            end else if (is_last_line) begin
                // Last CL of current row → advance outer dim odometer
                line_idx_r <= '0;
                // Ripple-carry odometer: advance dim 0, cascade wraps
                if (dim_count_r[0] + 1 < dim_tile_r[0]) begin
                    dim_count_r[0]  <= dim_count_r[0] + 1;
                    dim_offset_r[0] <= dim_offset_r[0] + `MEM_ADDR_WIDTH'(stride_r[0]);
                end else begin
                    dim_count_r[0]  <= '0;
                    dim_offset_r[0] <= '0;
                    if (dim_count_r[1] + 1 < dim_tile_r[1]) begin
                        dim_count_r[1]  <= dim_count_r[1] + 1;
                        dim_offset_r[1] <= dim_offset_r[1] + `MEM_ADDR_WIDTH'(stride_r[1]);
                    end else begin
                        dim_count_r[1]  <= '0;
                        dim_offset_r[1] <= '0;
                        if (dim_count_r[2] + 1 < dim_tile_r[2]) begin
                            dim_count_r[2]  <= dim_count_r[2] + 1;
                            dim_offset_r[2] <= dim_offset_r[2] + `MEM_ADDR_WIDTH'(stride_r[2]);
                        end else begin
                            dim_count_r[2]  <= '0;
                            dim_offset_r[2] <= '0;
                            // dim[3] must advance (we already checked is_last_outer)
                            dim_count_r[3]  <= dim_count_r[3] + 1;
                            dim_offset_r[3] <= dim_offset_r[3] + `MEM_ADDR_WIDTH'(stride_r[3]);
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
    `UNUSED_VAR (setup_params.initial_smem_base)
    `UNUSED_VAR (setup_params.elem_bytes)
    `UNUSED_VAR (setup_params.rank)
    `UNUSED_VAR (setup_params.total_bytes)
    `UNUSED_VAR (setup_params.total_rows)

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
