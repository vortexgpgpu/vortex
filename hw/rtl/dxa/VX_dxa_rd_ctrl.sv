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

// DXA Read Controller (OOO v2 — scoreboard + response FIFO):
//   - Uses VX_dxa_scoreboard for metadata (offset, length, smem_addr) per slot.
//   - GMEM responses are stored in a response FIFO along with compact metadata.
//   - Slots are released immediately on response arrival (not on drain).
//   - Credit counter prevents response FIFO overflow; gmem_rd_rsp_ready NEVER
//     depends on FIFO fullness.
//   - OOB entries bypass directly to output with priority over FIFO drain.

`include "VX_define.vh"

module VX_dxa_rd_ctrl import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING  = 8,
    parameter GMEM_BYTES       = `L1_LINE_SIZE,
    parameter GMEM_OFF_BITS    = `CLOG2(GMEM_BYTES),
    parameter GMEM_ADDR_WIDTH  = `MEM_ADDR_WIDTH - GMEM_OFF_BITS,
    parameter GMEM_DATAW       = GMEM_BYTES * 8,
    parameter GMEM_TAG_VALUEW  = L1_MEM_ARB_TAG_WIDTH - `UP(UUID_WIDTH)
) (
    input  wire                        clk,
    input  wire                        reset,
`ifdef PERF_ENABLE
    output wire [31:0]                 perf_gmem_reqs,
    output wire [31:0]                 perf_gmem_span_cycles,
`endif
    input  wire                        transfer_active,

    // CL input (from dedup, valid/ready).
    input  wire                        cl_in_valid,
    output wire                        cl_in_ready,
    input  wire [GMEM_ADDR_WIDTH-1:0]  cl_in_addr,
    input  wire [GMEM_BYTES-1:0]       cl_in_byte_mask,
    input  wire                        cl_in_oob,
    input  wire                        cl_in_last,
    input  wire [`MEM_ADDR_WIDTH-1:0]  cl_in_smem_byte_addr,

    // Params from setup (stable during transfer).
    input  wire [31:0]                 cfill,

    // GMEM bus (read req/rsp).
    output wire                        gmem_rd_req_valid,
    output wire [GMEM_ADDR_WIDTH-1:0]  gmem_rd_req_addr,
    output wire [GMEM_TAG_VALUEW-1:0]  gmem_rd_req_tag,
    input  wire                        gmem_rd_req_ready,
    input  wire                        gmem_rd_rsp_valid,
    input  wire [GMEM_DATAW-1:0]       gmem_rd_rsp_data,
    input  wire [GMEM_TAG_VALUEW-1:0]  gmem_rd_rsp_tag,
    output wire                        gmem_rd_rsp_ready,

    // CL output (to cl2smem, valid/ready).
    output wire                        cl_out_valid,
    input  wire                        cl_out_ready,
    output wire [GMEM_DATAW-1:0]       cl_out_data,
    output wire [GMEM_BYTES-1:0]       cl_out_byte_mask,
    output wire                        cl_out_last,
    output wire [`MEM_ADDR_WIDTH-1:0]  cl_out_smem_byte_addr,

    // Progress events.
    output wire                        gmem_req_fire,
    output wire                        rsp_fire,
    output wire                        stall_no_slot,

    // Completion signal: all CLs emitted.
    output wire                        all_cls_done
);
    localparam RD_SLOT_BITS = `CLOG2(MAX_OUTSTANDING);
    localparam RD_SLOT_W    = `UP(RD_SLOT_BITS);

    `STATIC_ASSERT(`IS_POW2(MAX_OUTSTANDING), ("MAX_OUTSTANDING must be power of 2"))
    `STATIC_ASSERT(GMEM_TAG_VALUEW >= RD_SLOT_W, ("gmem tag too narrow for slot encoding"))

    // ════════════════════════════════════════════════════════════════════
    // Response FIFO — stores data + compact metadata from scoreboard
    // ════════════════════════════════════════════════════════════════════
    // Payload: {is_last, cl_data, offset, length, smem_addr}
    localparam RSP_FIFO_DATAW = 1 + GMEM_DATAW + GMEM_OFF_BITS + (GMEM_OFF_BITS+1) + `MEM_ADDR_WIDTH;
    localparam RSP_FIFO_DEPTH = MAX_OUTSTANDING;
    localparam RSP_FIFO_SIZEW = `CLOG2(RSP_FIFO_DEPTH + 1);

    wire [RSP_FIFO_DATAW-1:0] rsp_fifo_data_in, rsp_fifo_data_out;
    wire rsp_fifo_empty, rsp_fifo_full;
    wire rsp_fifo_push, rsp_fifo_pop;
    wire [RSP_FIFO_SIZEW-1:0] rsp_fifo_size;
    wire rsp_fifo_alm_empty, rsp_fifo_alm_full;

    // rsp_fifo: wide (~530-bit) GMEM response payload. Pre-fix config was
    //   `OUT_REG=0, LUTRAM=1`, which placed 512-bit cl_data in distributed
    //   RAM with an async read path — the dominant LUT consumer inside
    //   rd_ctrl on U55C (~14K LUTs). Two orthogonal fixes applied:
    //     1. OUT_REG=1 — add an output register stage so downstream
    //        cl2smem/wr_ctrl consumers see a register boundary, not a
    //        530-bit comb mux from LUTRAM. Show-ahead FIFO semantics are
    //        preserved by VX_fifo_queue's empty->non-empty bypass path.
    //     2. LUTRAM=0 — move the underlying DP-RAM storage to BRAM so the
    //        wide payload lives in block RAM (a few BRAM18/36 tiles) rather
    //        than burning ~14K LUTs on the wide LUTRAM mux tree. BRAM with
    //        OUT_REG=1 matches the existing VX_fifo_queue drain timing (1
    //        cycle data_out_r → downstream).
    //   Total cost: 1 cycle of drain latency (negligible — DMA throughput
    //   is bound by GMEM read latency, not FIFO drain), plus a few BRAM
    //   tiles (plenty of headroom on U55C).
    VX_fifo_queue #(
        .DATAW   (RSP_FIFO_DATAW),
        .DEPTH   (RSP_FIFO_DEPTH),
        .OUT_REG (1),
        .LUTRAM  (0)
    ) rsp_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (rsp_fifo_push),
        .pop      (rsp_fifo_pop),
        .data_in  (rsp_fifo_data_in),
        .data_out (rsp_fifo_data_out),
        .empty    (rsp_fifo_empty),
        .alm_empty(rsp_fifo_alm_empty),
        .full     (rsp_fifo_full),
        .alm_full (rsp_fifo_alm_full),
        .size     (rsp_fifo_size)
    );

    `UNUSED_VAR (rsp_fifo_alm_empty)
    `UNUSED_VAR (rsp_fifo_alm_full)
    `UNUSED_VAR (rsp_fifo_size)

    // Unpack response FIFO output.
    wire                       rsp_fifo_is_last;
    wire [GMEM_DATAW-1:0]     rsp_fifo_cl_data;
    wire [GMEM_OFF_BITS-1:0]  rsp_fifo_offset;
    wire [GMEM_OFF_BITS:0]    rsp_fifo_length;
    wire [`MEM_ADDR_WIDTH-1:0] rsp_fifo_smem_addr;

    assign {rsp_fifo_is_last, rsp_fifo_cl_data, rsp_fifo_offset,
            rsp_fifo_length, rsp_fifo_smem_addr} = rsp_fifo_data_out;

    // ════════════════════════════════════════════════════════════════════
    // Scoreboard — metadata-only (offset, length, smem_addr)
    // ════════════════════════════════════════════════════════════════════

    wire rd_free_found;
    wire [RD_SLOT_W-1:0] rd_free_slot;

    wire [RD_SLOT_W-1:0] rsp_slot = RD_SLOT_W'(gmem_rd_rsp_tag[RD_SLOT_W-1:0]);
    wire rsp_slot_busy;
    wire [GMEM_OFF_BITS-1:0] rsp_sb_offset;
    wire [GMEM_OFF_BITS:0]   rsp_sb_length;
    wire [`MEM_ADDR_WIDTH-1:0] rsp_sb_smem_addr;

    wire alloc_fire_w;
    wire sb_release_fire;

    // Allocation metadata extraction: find first set bit (offset) and popcount (length).
    wire [GMEM_OFF_BITS-1:0] alloc_offset;
    wire                     alloc_mask_valid;
    VX_priority_encoder #(.N(GMEM_BYTES)) alloc_ctz (
        .data_in   (cl_in_byte_mask),
        .index_out (alloc_offset),
        .valid_out (alloc_mask_valid),
        `UNUSED_PIN(onehot_out)
    );

    wire [GMEM_OFF_BITS:0] alloc_length;
    VX_popcount #(.N(GMEM_BYTES)) alloc_pc (
        .data_in  (cl_in_byte_mask),
        .data_out (alloc_length)
    );

    `UNUSED_VAR (alloc_mask_valid)

    VX_dxa_scoreboard #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .GMEM_OFF_BITS   (GMEM_OFF_BITS),
        .SMEM_ADDR_W     (`MEM_ADDR_WIDTH)
    ) scoreboard (
        .clk            (clk),
        .reset          (reset),
        .free_found     (rd_free_found),
        .free_slot      (rd_free_slot),
        .alloc_fire     (alloc_fire_w),
        .alloc_slot     (rd_free_slot),
        .alloc_offset   (alloc_offset),
        .alloc_length   (alloc_length),
        .alloc_smem_addr(cl_in_smem_byte_addr),
        .rsp_slot       (rsp_slot),
        .rsp_slot_busy  (rsp_slot_busy),
        .rsp_offset     (rsp_sb_offset),
        .rsp_length     (rsp_sb_length),
        .rsp_smem_addr  (rsp_sb_smem_addr),
        .release_fire   (sb_release_fire),
        .release_slot   (rsp_slot)
    );

    // Sidecar register for is_last (not stored in scoreboard).
    reg [MAX_OUTSTANDING-1:0] slot_is_last_r;

    always @(posedge clk) begin
        if (reset || !transfer_active) begin
            slot_is_last_r <= '0;
        end else if (alloc_fire_w) begin
            slot_is_last_r[rd_free_slot] <= cl_in_last;
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Credit counter — prevents response FIFO overflow
    // ════════════════════════════════════════════════════════════════════
    localparam CREDIT_W = `CLOG2(RSP_FIFO_DEPTH + 1);
    reg [CREDIT_W-1:0] credit_r;

    always @(posedge clk) begin
        if (reset || !transfer_active)
            credit_r <= CREDIT_W'(RSP_FIFO_DEPTH);
        else begin
            case ({alloc_fire_w, rsp_fifo_pop})
                2'b10: credit_r <= credit_r - CREDIT_W'(1);
                2'b01: credit_r <= credit_r + CREDIT_W'(1);
                default: ;
            endcase
        end
    end

    wire has_credit = (credit_r > 0);

    // ════════════════════════════════════════════════════════════════════
    // Replicate cfill as GMEM-width data for OOB
    // ════════════════════════════════════════════════════════════════════
    wire [GMEM_DATAW-1:0] cfill_replicated;
    for (genvar i = 0; i < GMEM_BYTES / 4; ++i) begin : g_cfill
        assign cfill_replicated[i*32 +: 32] = cfill;
    end

    // ════════════════════════════════════════════════════════════════════
    // Input classification
    // ════════════════════════════════════════════════════════════════════
    // OOB bypasses slot table, emitted directly.
    // Normal requires a free slot AND credit (room in response FIFO).
    wire oob_emit = cl_in_valid && cl_in_oob;
    wire want_normal = cl_in_valid && !cl_in_oob && rd_free_found && has_credit;
    wire accept_normal = want_normal && gmem_rd_req_ready;

    // OOB has priority over FIFO drain. Since oob_emit suppresses fifo_emit,
    // the output path drives OOB data, so cl_out_ready is for the OOB handshake.
    wire accept_oob = oob_emit && cl_out_ready;

    wire cl_accept = accept_oob || accept_normal;
    assign cl_in_ready = cl_accept;

    // ════════════════════════════════════════════════════════════════════
    // GMEM request
    // ════════════════════════════════════════════════════════════════════
    assign gmem_rd_req_valid = want_normal;
    assign gmem_rd_req_addr  = cl_in_addr;
    assign gmem_rd_req_tag   = GMEM_TAG_VALUEW'(rd_free_slot);
    assign alloc_fire_w      = accept_normal;
    assign gmem_req_fire     = alloc_fire_w;

    // ════════════════════════════════════════════════════════════════════
    // GMEM response — push to response FIFO, release slot immediately
    // ════════════════════════════════════════════════════════════════════
    // CRITICAL: gmem_rd_rsp_ready MUST NOT depend on rsp_fifo_full.
    // Credit counter guarantees space exists.
    wire rsp_accept = gmem_rd_rsp_valid && transfer_active && rsp_slot_busy;

    assign gmem_rd_rsp_ready = transfer_active;  // NEVER depends on FIFO fullness
    assign rsp_fire = rsp_accept;

    // Release slot on response arrival (not on drain).
    assign sb_release_fire = rsp_accept;

    // Pack response FIFO input.
    assign rsp_fifo_push = rsp_accept;
    assign rsp_fifo_data_in = {slot_is_last_r[rsp_slot], gmem_rd_rsp_data,
                               rsp_sb_offset, rsp_sb_length, rsp_sb_smem_addr};

    // ════════════════════════════════════════════════════════════════════
    // Output — OOB bypass has priority over FIFO drain
    // ════════════════════════════════════════════════════════════════════
    wire fifo_emit = !rsp_fifo_empty && !oob_emit;
    assign cl_out_valid = oob_emit || fifo_emit;
    assign rsp_fifo_pop = fifo_emit && cl_out_ready;

    // Reconstruct byte mask from offset + length.
    wire [GMEM_BYTES-1:0] reconstructed_mask;
    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off CMPCONST */
    for (genvar i = 0; i < GMEM_BYTES; ++i) begin : g_mask_recon
        assign reconstructed_mask[i] = ((GMEM_OFF_BITS+1)'(i) >= {1'b0, rsp_fifo_offset})
                                    && ((GMEM_OFF_BITS+1)'(i) < (GMEM_OFF_BITS+1)'(rsp_fifo_offset) + rsp_fifo_length);
    end
    /* verilator lint_on CMPCONST */
    /* verilator lint_on WIDTHEXPAND */

    assign cl_out_data          = oob_emit ? cfill_replicated    : rsp_fifo_cl_data;
    assign cl_out_byte_mask     = oob_emit ? cl_in_byte_mask     : reconstructed_mask;
    assign cl_out_last          = oob_emit ? cl_in_last          : rsp_fifo_is_last;
    assign cl_out_smem_byte_addr = oob_emit ? cl_in_smem_byte_addr : rsp_fifo_smem_addr;

    // ════════════════════════════════════════════════════════════════════
    // Completion tracking — issue/done counters
    // ════════════════════════════════════════════════════════════════════
    reg        seen_last_input_r;
    reg [31:0] issue_count_r;
    reg [31:0] done_count_r;

    always @(posedge clk) begin
        if (reset || !transfer_active) begin
            seen_last_input_r <= 1'b0;
            issue_count_r     <= 32'd0;
            done_count_r      <= 32'd0;
        end else begin
            if (cl_accept && cl_in_last) begin
                seen_last_input_r <= 1'b1;
            end
            if (accept_oob || alloc_fire_w) begin
                issue_count_r <= issue_count_r + 32'd1;
            end
            if (cl_out_valid && cl_out_ready) begin
                done_count_r <= done_count_r + 32'd1;
            end
        end
    end

    assign all_cls_done = transfer_active
                       && seen_last_input_r
                       && (issue_count_r == done_count_r)
                       && rsp_fifo_empty;

    // ════════════════════════════════════════════════════════════════════
    // Progress events + assertions
    // ════════════════════════════════════════════════════════════════════
    assign stall_no_slot = cl_in_valid && !cl_in_oob && !rd_free_found;

    `UNUSED_VAR (gmem_rd_rsp_tag[GMEM_TAG_VALUEW-1:RD_SLOT_W])

    `RUNTIME_ASSERT(!(rsp_accept) || rsp_slot_busy,
        ("dxa rd_ctrl: gmem rsp to non-busy slot"))

    `RUNTIME_ASSERT(credit_r <= CREDIT_W'(RSP_FIFO_DEPTH),
        ("dxa rd_ctrl credit overflow"))

    `RUNTIME_ASSERT(!(rsp_fifo_push && rsp_fifo_full),
        ("dxa rd_ctrl rsp_fifo overflow"))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && transfer_active) begin
            if (alloc_fire_w) begin
                $write("DXA_PIPE,%0d,GMEM_REQ,addr=0x%0h,slot=%0d,smem=0x%0h\n",
                    $time, cl_in_addr, rd_free_slot, cl_in_smem_byte_addr);
            end
            if (rsp_accept) begin
                $write("DXA_PIPE,%0d,GMEM_RSP,slot=%0d\n",
                    $time, rsp_slot);
            end
            if (cl_out_valid && cl_out_ready) begin
                if (oob_emit) begin
                    $write("DXA_PIPE,%0d,RC_OUT,type=oob,mask=0x%0h,smem=0x%0h,last=%0d\n",
                        $time, cl_out_byte_mask, cl_out_smem_byte_addr, cl_out_last);
                end else begin
                    $write("DXA_PIPE,%0d,RC_OUT,type=fifo,mask=0x%0h,smem=0x%0h,last=%0d\n",
                        $time, cl_out_byte_mask, cl_out_smem_byte_addr, cl_out_last);
                end
            end
        end
    end
`endif

`ifdef PERF_ENABLE
    reg [31:0] rdp_cycle_ctr_r;
    reg [31:0] rdp_total_gmem_req_r;
    reg [31:0] rdp_first_req_cycle_r;
    reg [31:0] rdp_last_rsp_cycle_r;
    reg        rdp_has_req_r;
    always @(posedge clk) begin
        if (reset || !transfer_active) begin
            rdp_cycle_ctr_r       <= '0;
            rdp_total_gmem_req_r  <= '0;
            rdp_first_req_cycle_r <= '0;
            rdp_last_rsp_cycle_r  <= '0;
            rdp_has_req_r         <= 1'b0;
        end else begin
            rdp_cycle_ctr_r <= rdp_cycle_ctr_r + 32'd1;
            if (alloc_fire_w) begin
                rdp_total_gmem_req_r <= rdp_total_gmem_req_r + 32'd1;
                if (!rdp_has_req_r) begin
                    rdp_first_req_cycle_r <= rdp_cycle_ctr_r;
                    rdp_has_req_r <= 1'b1;
                end
            end
            if (rsp_accept) begin
                rdp_last_rsp_cycle_r <= rdp_cycle_ctr_r;
            end
        end
    end
    assign perf_gmem_reqs        = rdp_total_gmem_req_r;
    assign perf_gmem_span_cycles = (rdp_has_req_r && rdp_last_rsp_cycle_r >= rdp_first_req_cycle_r)
                                 ? (rdp_last_rsp_cycle_r - rdp_first_req_cycle_r + 32'd1) : 32'd0;
`endif

endmodule
