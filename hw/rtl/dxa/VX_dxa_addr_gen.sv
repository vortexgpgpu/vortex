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

// DXA Address Generator: CL-aware address generator replacing tile_iter.
// Emits per-CL requests with byte-valid bitmasks.
// Two nested counters: outer=row, inner=CL within row.
// Zero runtime multiplies (stride-based address advancement).

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
    output wire [GMEM_LINE_SIZE-1:0]   out_byte_mask,
    output wire                        out_oob,
    output wire                        out_last,

    // Pass-through params for downstream (stable during transfer).
    output wire [31:0]                 out_cfill,
    output wire [31:0]                 out_total_lmem_writes
);
    localparam CL_OFF_BITS = `CLOG2(GMEM_LINE_SIZE);

    `STATIC_ASSERT(`IS_POW2(GMEM_LINE_SIZE), ("GMEM_LINE_SIZE must be power of 2"))

    // ---- Registered state ----
    reg                        active_r;
    reg [`MEM_ADDR_WIDTH-1:0]  gmem_base_r;     // Current row's GMEM base addr
    reg [31:0]                 row_count_r;
    reg [31:0]                 total_rows_r;
    reg [31:0]                 row_len_r;        // row_len_bytes (constant)
    reg [31:0]                 stride0_r;
    reg [31:0]                 oob_limit_r;
    reg [31:0]                 line_idx_r;       // CL index within current row

    // Pass-through latched params
    reg [31:0]                 cfill_r;
    reg [31:0]                 total_lmem_writes_r;

    assign out_cfill = cfill_r;
    assign out_total_lmem_writes = total_lmem_writes_r;

    // ---- Per-row geometry (combinatorial from current row state) ----
    wire [CL_OFF_BITS-1:0] first_off = gmem_base_r[CL_OFF_BITS-1:0];
    wire [31:0] bytes_span = 32'(first_off) + row_len_r;
    wire [31:0] num_lines  = (bytes_span + GMEM_LINE_SIZE - 1) >> CL_OFF_BITS;

    // ---- Current CL address ----
    wire [`MEM_ADDR_WIDTH-1:0] first_cl_base = {gmem_base_r[`MEM_ADDR_WIDTH-1:CL_OFF_BITS],
                                                 {CL_OFF_BITS{1'b0}}};
    wire [`MEM_ADDR_WIDTH-1:0] cur_cl_byte_addr = first_cl_base
        + (`MEM_ADDR_WIDTH'(line_idx_r) << CL_OFF_BITS);
    wire [GMEM_ADDR_WIDTH-1:0] cur_cl_addr = cur_cl_byte_addr[`MEM_ADDR_WIDTH-1:CL_OFF_BITS];

    // ---- Byte mask computation ----
    // Valid byte range within CL: [valid_start, valid_end)
    wire [CL_OFF_BITS:0] valid_start = (line_idx_r == 0)
        ? {1'b0, first_off} : '0;

    wire [31:0] total_end = 32'(first_off) + row_len_r;
    wire [CL_OFF_BITS-1:0] last_end_off = total_end[CL_OFF_BITS-1:0];
    wire last_aligned = (last_end_off == '0);
    wire is_last_line = (line_idx_r + 1 >= num_lines);

    wire [CL_OFF_BITS:0] valid_end = is_last_line
        ? (last_aligned ? (CL_OFF_BITS+1)'(GMEM_LINE_SIZE) : {1'b0, last_end_off})
        : (CL_OFF_BITS+1)'(GMEM_LINE_SIZE);

    // OOB detection
    wire is_oob = (row_count_r >= oob_limit_r);
    wire is_last_row = (row_count_r + 1 >= total_rows_r);

    // Generate byte mask: bits [valid_start, valid_end) set.
    wire [GMEM_LINE_SIZE-1:0] byte_mask_w;
    for (genvar i = 0; i < GMEM_LINE_SIZE; ++i) begin : g_mask
        assign byte_mask_w[i] = ((CL_OFF_BITS+1)'(i) >= valid_start)
                              && ((CL_OFF_BITS+1)'(i) < valid_end);
    end

    // ---- Output ----
    assign out_valid     = active_r;
    assign out_cl_addr   = cur_cl_addr;
    assign out_byte_mask = byte_mask_w;
    assign out_oob       = is_oob;
    assign out_last      = is_last_line && is_last_row;

    // ---- Advance logic ----
    wire advance = out_valid && out_ready;

    always @(posedge clk) begin
        if (reset) begin
            active_r    <= 1'b0;
            gmem_base_r <= '0;
            row_count_r <= '0;
            line_idx_r  <= '0;
        end else if (start) begin
            active_r     <= (setup_params.total_rows != 0);
            gmem_base_r  <= setup_params.initial_gmem_base;
            row_count_r  <= '0;
            total_rows_r <= setup_params.total_rows;
            row_len_r    <= setup_params.row_len_bytes;
            stride0_r    <= setup_params.stride0;
            oob_limit_r  <= setup_params.oob_limit[0];
            line_idx_r   <= '0;
            cfill_r      <= setup_params.cfill;
            total_lmem_writes_r <= setup_params.total_lmem_writes;
        end else if (advance) begin
            if (is_last_line && is_last_row) begin
                active_r <= 1'b0;
            end else if (is_last_line) begin
                // Next row
                row_count_r <= row_count_r + 1;
                gmem_base_r <= gmem_base_r + `MEM_ADDR_WIDTH'(stride0_r);
                line_idx_r  <= '0;
            end else begin
                line_idx_r <= line_idx_r + 1;
            end
        end
    end

    `UNUSED_VAR (cur_cl_byte_addr[CL_OFF_BITS-1:0])
    `UNUSED_VAR (total_end[31:CL_OFF_BITS])
    `UNUSED_VAR (setup_params.initial_lmem_base)
    `UNUSED_VAR (setup_params.elem_bytes)
    `UNUSED_VAR (setup_params.rank)
    `UNUSED_VAR (setup_params.oob_limit[DXA_MAX_OUTER_DIMS-1:1])

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && advance) begin
            $write("DXA_PIPE,%0d,AG_OUT,addr=0x%0h,mask=0x%0h,oob=%0d,last=%0d,row=%0d,cl=%0d\n",
                $time, cur_cl_addr, byte_mask_w, is_oob, out_last, row_count_r, line_idx_r);
        end
    end
`endif

endmodule
