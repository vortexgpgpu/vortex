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

// DXA Read Controller: accepts CL entries from dedup, issues pipelined
// GMEM reads via slot table, and emits CL data with byte masks to
// cl2lmem. Dedup guarantees each CL address appears at most once.
//
// Two input paths:
//   1. OOB:    emit cfill immediately (no GMEM read)
//   2. Normal: allocate slot, issue GMEM read (pipelined, up to MAX_OUTSTANDING)
//
// A Response Reorder Buffer (ROB) ensures GMEM responses are emitted
// to cl2lmem in request-issue order, regardless of GMEM response order.

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

    // CL output (to cl2lmem, valid/ready).
    output wire                        cl_out_valid,
    input  wire                        cl_out_ready,
    output wire [GMEM_DATAW-1:0]       cl_out_data,
    output wire [GMEM_BYTES-1:0]       cl_out_byte_mask,
    output wire                        cl_out_last,

    // Progress events.
    output wire                        gmem_req_fire,
    output wire                        rsp_fire,
    output wire                        stall_no_slot
);
    localparam RD_SLOT_BITS = `CLOG2(MAX_OUTSTANDING);
    localparam RD_SLOT_W    = `UP(RD_SLOT_BITS);

    `STATIC_ASSERT(`IS_POW2(MAX_OUTSTANDING), ("MAX_OUTSTANDING must be power of 2"))
    `STATIC_ASSERT(GMEM_TAG_VALUEW >= RD_SLOT_W, ("gmem tag too narrow for slot encoding"))

    // ---- Output FIFO ----
    localparam OUT_FIFO_DATAW = GMEM_DATAW + GMEM_BYTES + 1;
    localparam OUT_FIFO_DEPTH = 4;
    localparam OUT_FIFO_SIZEW = `CLOG2(OUT_FIFO_DEPTH + 1);

    wire [OUT_FIFO_DATAW-1:0] ofifo_data_in, ofifo_data_out;
    wire ofifo_empty, ofifo_full;
    wire ofifo_alm_empty, ofifo_alm_full;
    wire [OUT_FIFO_SIZEW-1:0] ofifo_size;
    wire ofifo_push, ofifo_pop;

    VX_fifo_queue #(
        .DATAW   (OUT_FIFO_DATAW),
        .DEPTH   (OUT_FIFO_DEPTH),
        .OUT_REG (1),
        .LUTRAM  (1)
    ) out_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (ofifo_push),
        .pop      (ofifo_pop),
        .data_in  (ofifo_data_in),
        .data_out (ofifo_data_out),
        .empty    (ofifo_empty),
        .alm_empty(ofifo_alm_empty),
        .full     (ofifo_full),
        .alm_full (ofifo_alm_full),
        .size     (ofifo_size)
    );

    `UNUSED_VAR (ofifo_alm_empty)
    `UNUSED_VAR (ofifo_alm_full)
    `UNUSED_VAR (ofifo_size)

    // Output from FIFO to cl2lmem.
    assign cl_out_valid = ~ofifo_empty;
    assign {cl_out_last, cl_out_byte_mask, cl_out_data} = ofifo_data_out;
    assign ofifo_pop = cl_out_valid && cl_out_ready;

    // ---- Slot table ----
    wire rd_free_found;
    wire [RD_SLOT_W-1:0] rd_free_slot;
    wire line_inflight_found;
    wire rsp_slot_busy;
    wire [GMEM_ADDR_WIDTH-1:0] rsp_gmem_line_addr;
    wire [GMEM_OFF_BITS-1:0] rsp_gmem_off;
    wire rsp_is_last;

    wire [RD_SLOT_W-1:0] rsp_slot = RD_SLOT_W'(gmem_rd_rsp_tag[RD_SLOT_W-1:0]);

    wire alloc_fire_w;
    wire release_fire_w;

    VX_dxa_nb_slot_table #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_OFF_BITS   (GMEM_OFF_BITS)
    ) slot_table (
        .clk                 (clk),
        .reset               (reset),
        .q_line_addr         (cl_in_addr),
        .q_inflight_found    (line_inflight_found),
        .q_free_found        (rd_free_found),
        .q_free_slot         (rd_free_slot),
        .rsp_slot            (rsp_slot),
        .rsp_slot_busy       (rsp_slot_busy),
        .rsp_gmem_line_addr  (rsp_gmem_line_addr),
        .rsp_gmem_off        (rsp_gmem_off),
        .rsp_is_last         (rsp_is_last),
        .alloc_fire          (alloc_fire_w),
        .alloc_slot          (rd_free_slot),
        .alloc_gmem_line_addr(cl_in_addr),
        .alloc_gmem_off      ('0),
        .alloc_is_last       (cl_in_last),
        .release_fire        (release_fire_w),
        .release_slot        (rsp_slot)
    );

    // Per-slot byte mask storage.
    reg [GMEM_BYTES-1:0] slot_byte_mask_r [MAX_OUTSTANDING-1:0];

    // ---- Response Reorder Buffer (ROB) ----
    // GMEM responses may arrive out of order (e.g. due to cache bank
    // interleaving). The ROB ensures they are emitted to cl2lmem in
    // the order requests were issued, preserving row ordering in SMEM.
    //
    // alloc_seq_r: sequence counter incremented per request issue.
    // drain_seq_r: sequence counter incremented per ROB drain.
    // Each slot records its sequence via slot_seq_r[slot].
    // On response, data is stored at rob_*[seq]. Drain emits from
    // drain_seq_r when rob_valid_r[drain_seq_idx] is set.

    reg [MAX_OUTSTANDING-1:0]       rob_valid_r;
    reg [RD_SLOT_W-1:0]             slot_seq_r  [MAX_OUTSTANDING-1:0];
    reg [RD_SLOT_W:0]               alloc_seq_r;
    reg [RD_SLOT_W:0]               drain_seq_r;

    wire [RD_SLOT_W-1:0] alloc_seq_idx = alloc_seq_r[RD_SLOT_W-1:0];
    wire [RD_SLOT_W-1:0] drain_seq_idx = drain_seq_r[RD_SLOT_W-1:0];

    wire rob_empty = (alloc_seq_r == drain_seq_r);
    wire rob_full  = (alloc_seq_r[RD_SLOT_W] != drain_seq_r[RD_SLOT_W])
                  && (alloc_seq_r[RD_SLOT_W-1:0] == drain_seq_r[RD_SLOT_W-1:0]);

    // ROB data/mask/last stored in LUTRAM via VX_dp_ram.
    localparam ROB_ENTRY_DATAW = 1 + GMEM_BYTES + GMEM_DATAW;

    wire rsp_accept;  // forward declaration
    wire [ROB_ENTRY_DATAW-1:0] rob_rdata;

    VX_dp_ram #(
        .DATAW  (ROB_ENTRY_DATAW),
        .SIZE   (MAX_OUTSTANDING),
        .LUTRAM (1)
    ) rob_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (rsp_accept),
        .wren  (1'b1),
        .waddr (slot_seq_r[rsp_slot]),
        .wdata ({rsp_is_last, slot_byte_mask_r[rsp_slot], gmem_rd_rsp_data}),
        .raddr (drain_seq_idx),
        .rdata (rob_rdata)
    );

    wire                  rob_entry_last;
    wire [GMEM_BYTES-1:0] rob_entry_mask;
    wire [GMEM_DATAW-1:0] rob_entry_data;
    assign {rob_entry_last, rob_entry_mask, rob_entry_data} = rob_rdata;

    // ---- Replicate cfill as GMEM-width data for OOB ----
    wire [GMEM_DATAW-1:0] cfill_replicated;
    for (genvar i = 0; i < GMEM_BYTES / 4; ++i) begin : g_cfill
        assign cfill_replicated[i*32 +: 32] = cfill;
    end

    // ---- Input classification ----
    // OOB: use cfill, emit directly. Needs ROB drained (output ordering).
    // Normal: issue GMEM read, can pipeline; blocked when ROB full.
    wire accept_oob    = cl_in_valid && cl_in_oob && rob_empty && !ofifo_full;
    // want_normal: valid signal for GMEM request (must NOT depend on gmem_rd_req_ready).
    wire want_normal   = cl_in_valid && !cl_in_oob && rd_free_found && !rob_full;
    wire accept_normal = want_normal && gmem_rd_req_ready;

    wire cl_accept = accept_oob || accept_normal;

    assign cl_in_ready = cl_accept;

    // ---- GMEM request ----
    // Valid must be independent of ready to avoid combinational deadlock.
    assign gmem_rd_req_valid = want_normal;
    assign gmem_rd_req_addr  = cl_in_addr;
    assign gmem_rd_req_tag   = GMEM_TAG_VALUEW'(rd_free_slot);
    assign alloc_fire_w      = accept_normal;
    assign gmem_req_fire     = alloc_fire_w;

    // Store per-slot byte mask and sequence on allocation.
    always @(posedge clk) begin
        if (alloc_fire_w) begin
            slot_byte_mask_r[rd_free_slot] <= cl_in_byte_mask;
            slot_seq_r[rd_free_slot]       <= alloc_seq_idx;
        end
    end

    // Allocation sequence counter.
    always @(posedge clk) begin
        if (reset || ~transfer_active) begin
            alloc_seq_r <= '0;
        end else if (alloc_fire_w) begin
            alloc_seq_r <= alloc_seq_r + (RD_SLOT_W+1)'(1);
        end
    end

    // ---- GMEM response handling ----
    // Accept response and store in ROB (slot is released immediately).
    // No backpressure needed: each outstanding read has a reserved ROB entry.
    assign rsp_accept = gmem_rd_rsp_valid && transfer_active && rsp_slot_busy;
    assign release_fire_w = rsp_accept;
    assign rsp_fire       = release_fire_w;

    assign gmem_rd_rsp_ready = transfer_active && rsp_slot_busy;

    // ---- ROB fill and drain ----
    wire rob_head_valid = rob_valid_r[drain_seq_idx];
    wire rob_drain      = rob_head_valid && !ofifo_full;

    // Direct emit (OOB only): mutually exclusive with ROB drain
    // because direct_emit requires rob_empty.
    wire direct_emit = accept_oob;
    wire [GMEM_DATAW-1:0] direct_data = cfill_replicated;
    wire [GMEM_BYTES-1:0]  direct_mask = cl_in_byte_mask;
    wire                   direct_last = cl_in_last;

    // ROB valid tracking (data/mask/last stored in rob_store LUTRAM).
    always @(posedge clk) begin
        if (reset || ~transfer_active) begin
            rob_valid_r <= '0;
        end else begin
            if (rsp_accept) begin
                rob_valid_r[slot_seq_r[rsp_slot]] <= 1'b1;
            end
            if (rob_drain) begin
                rob_valid_r[drain_seq_idx] <= 1'b0;
            end
        end
    end

    // Drain sequence counter.
    always @(posedge clk) begin
        if (reset || ~transfer_active) begin
            drain_seq_r <= '0;
        end else if (rob_drain) begin
            drain_seq_r <= drain_seq_r + (RD_SLOT_W+1)'(1);
        end
    end

    // ---- Output FIFO push ----
    assign ofifo_push = direct_emit || rob_drain;
    assign ofifo_data_in = direct_emit
        ? {direct_last, direct_mask, direct_data}
        : {rob_entry_last, rob_entry_mask, rob_entry_data};

    // ---- Progress events ----
    assign stall_no_slot = cl_in_valid && !cl_in_oob && !rd_free_found;

    `UNUSED_VAR (line_inflight_found)
    `UNUSED_VAR (rsp_gmem_line_addr)
    `UNUSED_VAR (rsp_gmem_off)
    `UNUSED_VAR (gmem_rd_rsp_tag[GMEM_TAG_VALUEW-1:RD_SLOT_W])

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && transfer_active) begin
            if (alloc_fire_w) begin
                $write("DXA_PIPE,%0d,GMEM_REQ,addr=0x%0h,slot=%0d,seq=%0d\n",
                    $time, cl_in_addr, rd_free_slot, alloc_seq_idx);
            end
            if (release_fire_w) begin
                $write("DXA_PIPE,%0d,GMEM_RSP,slot=%0d,seq=%0d\n",
                    $time, rsp_slot, slot_seq_r[rsp_slot]);
            end
            if (ofifo_push) begin
                if (direct_emit) begin
                    $write("DXA_PIPE,%0d,RC_OUT,type=oob,mask=0x%0h,last=%0d\n",
                        $time, direct_mask, direct_last);
                end else begin
                    $write("DXA_PIPE,%0d,RC_OUT,type=rob,seq=%0d,mask=0x%0h,last=%0d\n",
                        $time, drain_seq_idx, rob_entry_mask,
                        rob_entry_last);
                end
            end
        end
    end
`endif

    `RUNTIME_ASSERT(~(gmem_rd_rsp_valid && gmem_rd_rsp_ready) || rsp_slot_busy,
        ("invalid dxa rd_ctrl gmem rsp slot"))

`ifdef PERF_ENABLE
    // Lightweight counters (no per-slot timestamps, no eff_bytes, no $display)
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
            if (release_fire_w) begin
                rdp_last_rsp_cycle_r <= rdp_cycle_ctr_r;
            end
        end
    end
    assign perf_gmem_reqs        = rdp_total_gmem_req_r;
    assign perf_gmem_span_cycles = (rdp_has_req_r && rdp_last_rsp_cycle_r >= rdp_first_req_cycle_r)
                                 ? (rdp_last_rsp_cycle_r - rdp_first_req_cycle_r + 32'd1) : 32'd0;
`endif

endmodule
