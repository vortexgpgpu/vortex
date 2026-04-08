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

// DXA Setup Phase: precomputes all constants for addr_gen, rd_ctrl, cl2smem, wr_ctrl.
// Uses a single registered DSP multiply, sequenced by an FSM.
// Runs once per transfer launch (~4-6 cycles for 2D).

`include "VX_define.vh"

module VX_dxa_setup import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter SMEM_BYTES = DXA_LMEM_WORD_SIZE
) (
    input  wire                        clk,
    input  wire                        reset,

    // Trigger: pulse high for one cycle to start setup.
    input  wire                        start,

    // Raw descriptor fields (valid when start is asserted).
    input  wire dxa_issue_dec_t        issue_dec,
    input  wire [`MEM_ADDR_WIDTH-1:0]  gmem_base,
    input  wire [`XLEN-1:0]            smem_base,
    input  wire [4:0][`XLEN-1:0]       coords,
    input  wire [31:0]                 cfill,

    // Output: precomputed parameters (stable after setup_done).
    output wire                        setup_done,
    output wire dxa_setup_params_t     setup_params
);
    // ---- FSM states ----
    // Sequence of multiply operations for 2D:
    //   S0: row_len_bytes     = tile0 * elem_bytes
    //   S1: gmem_off0         = coord0 * elem_bytes      (add to gbase)
    //   S2: gmem_off1         = coord1 * stride0          (add to gbase)
    //   S3: total_bytes       = total_rows * row_len_bytes
    //   S4: done (compute total_smem_writes from total_bytes)
    localparam S_IDLE        = 3'd0;
    localparam S_ROW_LEN     = 3'd1;
    localparam S_GMEM_OFF0   = 3'd2;
    localparam S_GMEM_OFF1   = 3'd3;
    localparam S_TOTAL_WRITES= 3'd4;
    localparam S_DONE        = 3'd5;

    reg [2:0] state_r;

    // Registered DSP multiply: result available next cycle.
    reg  [31:0] mul_a_r, mul_b_r;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0] mul_result_w = 64'(mul_a_r) * 64'(mul_b_r);
    /* verilator lint_on UNUSEDSIGNAL */
    reg  [31:0] mul_out_r;
    always @(posedge clk) begin
        mul_out_r <= mul_result_w[31:0];
    end

    // Registered setup parameters.
    reg [`MEM_ADDR_WIDTH-1:0] r_initial_gmem_base;
    reg [`XLEN-1:0]           r_initial_smem_base;
    reg [31:0]                r_row_len_bytes;
    reg [31:0]                r_stride0;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] r_oob_limit;
    reg [31:0]                r_total_rows;
    reg [31:0]                r_total_smem_writes;
    reg [31:0]                r_cfill;
    reg [31:0]                r_elem_bytes;
    reg [31:0]                r_rank;

    // Latched inputs (captured on start).
    reg [31:0] lat_elem_bytes, lat_stride0;
    reg [31:0] lat_coord0, lat_coord1;
    reg [31:0] lat_size1;
    reg [`MEM_ADDR_WIDTH-1:0] lat_gbase;

    // SMEM offset bits for total_smem_writes computation.
    localparam SMEM_OFF_BITS = `CLOG2(SMEM_BYTES);

    always @(posedge clk) begin
        if (reset) begin
            state_r <= S_IDLE;
            mul_a_r <= '0;
            mul_b_r <= '0;
        end else begin
            case (state_r)
            S_IDLE: begin
                if (start) begin
                    // Latch inputs.
                    lat_elem_bytes <= issue_dec.elem_bytes;
                    lat_stride0    <= issue_dec.stride0;
                    lat_coord0     <= coords[0][31:0];
                    lat_coord1     <= (issue_dec.rank >= 2) ? coords[1][31:0] : 32'd0;
                    lat_size1      <= issue_dec.size1;
                    lat_gbase      <= gmem_base;
                    r_initial_smem_base <= smem_base;
                    r_cfill        <= cfill;
                    r_elem_bytes   <= issue_dec.elem_bytes;
                    r_rank         <= issue_dec.rank;
                    r_stride0      <= issue_dec.stride0;
                    r_total_rows   <= issue_dec.tile1;  // For 2D: total_rows = tile1.

                    // OOB limits for outer dims (only dim1 used for 2D).
                    r_oob_limit[0] <= (issue_dec.size1 > ((issue_dec.rank >= 2) ? coords[1][31:0] : 32'd0))
                                    ? (issue_dec.size1 - ((issue_dec.rank >= 2) ? coords[1][31:0] : 32'd0))
                                    : 32'd0;
                    for (int d = 1; d < DXA_MAX_OUTER_DIMS; d++) begin
                        r_oob_limit[d] <= 32'hFFFF_FFFF; // Unused dims: never OOB.
                    end

                    // Start multiply: tile0 * elem_bytes.
                    mul_a_r <= issue_dec.tile0;
                    mul_b_r <= issue_dec.elem_bytes;
                    state_r <= S_ROW_LEN;
                end
            end

            S_ROW_LEN: begin
                // mul_out_r will have tile0 * elem_bytes next cycle.
                // Start next multiply: coord0 * elem_bytes.
                mul_a_r <= lat_coord0;
                mul_b_r <= lat_elem_bytes;
                state_r <= S_GMEM_OFF0;
            end

            S_GMEM_OFF0: begin
                // Capture row_len_bytes = tile0 * elem_bytes (from previous cycle).
                r_row_len_bytes <= mul_out_r;
                // Start next multiply: coord1 * stride0.
                mul_a_r <= lat_coord1;
                mul_b_r <= lat_stride0;
                state_r <= S_GMEM_OFF1;
            end

            S_GMEM_OFF1: begin
                // Capture gmem_base += coord0 * elem_bytes.
                r_initial_gmem_base <= lat_gbase + `MEM_ADDR_WIDTH'(mul_out_r);
                // Start next multiply: total_rows * row_len_bytes = total_bytes.
                mul_a_r <= r_total_rows;
                mul_b_r <= r_row_len_bytes;
                state_r <= S_TOTAL_WRITES;
            end

            S_TOTAL_WRITES: begin
                // Capture gmem_base += coord1 * stride0.
                r_initial_gmem_base <= r_initial_gmem_base + `MEM_ADDR_WIDTH'(mul_out_r);
                state_r <= S_DONE;
            end

            S_DONE: begin
                // mul_out_r = total_rows * row_len_bytes = total_bytes.
                // Compute packed total_smem_writes = ceil((total_bytes + smem_off) / SMEM_BYTES).
                // With write packing, multiple rows' data may pack into one SMEM word.
                r_total_smem_writes <= (mul_out_r + 32'(r_initial_smem_base[SMEM_OFF_BITS-1:0])
                                        + SMEM_BYTES - 1) >> SMEM_OFF_BITS;
                state_r <= S_IDLE;
            end

            default: state_r <= S_IDLE;
            endcase
        end
    end

    // Output assignments.
    // Delay setup_done by 1 cycle so that registers written in S_DONE
    // (e.g. r_total_smem_writes) are stable when downstream captures them.
    reg setup_done_r;
    always @(posedge clk) begin
        if (reset) begin
            setup_done_r <= 1'b0;
        end else begin
            setup_done_r <= (state_r == S_DONE);
        end
    end
    assign setup_done = setup_done_r;

    assign setup_params.initial_gmem_base  = r_initial_gmem_base;
    assign setup_params.initial_smem_base  = r_initial_smem_base;
    assign setup_params.row_len_bytes      = r_row_len_bytes;
    assign setup_params.stride0            = r_stride0;
    assign setup_params.oob_limit          = r_oob_limit;
    assign setup_params.total_rows         = r_total_rows;
    assign setup_params.total_smem_writes  = r_total_smem_writes;
    assign setup_params.cfill              = r_cfill;
    assign setup_params.elem_bytes         = r_elem_bytes;
    assign setup_params.rank               = r_rank;

    `UNUSED_VAR (issue_dec.is_s2g)
    `UNUSED_VAR (issue_dec.supported)
    `UNUSED_VAR (issue_dec.total)
    `UNUSED_VAR (issue_dec.size0)
    `UNUSED_VAR (coords[4:2])
    `UNUSED_VAR (lat_size1)

endmodule
