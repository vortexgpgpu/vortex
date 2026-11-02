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

`include "VX_define.vh"

`ifdef VX_CFG_TCU_TMEM_ENABLE

// Per-CTA TMEM allocator: a first-fit free list + a live-allocation CAM
// over the column dimension
//
// alloc() only ever reserves a column range, it does not
// initialize TMEM contents
//
// Single already-arbitrated request per cycle — the caller,
// VX_tcu_tmem.sv, is responsible for picking one winner among BLOCK_SIZE
// competing blocks before driving these ports
//
// Errors (free list exhausted, allocator sizing exhausted, unknown
// handle on dealloc, or a cta_id's cached allocation size mismatching a
// repeat request) are RUNTIME_ASSERT-fatal
//
// Barrier dealloc: a handle's range is only actually returned to the free
// list once WARPGROUP_SIZE dealloc calls have been seen for it. This
// assumes CTA size == warpgroup size, matching Blackwell tcgen05 (fixed
// 4-warp warpgroups) and matching WGMMA's existing scope.

module VX_tcu_tmem_alloc import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID  = "",
    parameter         WARPGROUP_SIZE = `VX_CFG_NUM_TCU_BLOCKS,
    localparam        MAX_CONCURRENT_CTAS = `VX_CFG_NUM_WARPS / WARPGROUP_SIZE,
    localparam        NUM_ENTRIES         = MAX_CONCURRENT_CTAS + 1
) (
    input wire clk,
    input wire reset,

    input  wire                         req_valid,
    input  wire                         req_is_dealloc,
    input  wire [NCTA_WIDTH-1:0]        req_cta_id,
    input  wire [TCU_TMEM_COL_BITS:0]   req_ncols,
    input  wire [TCU_TMEM_COL_BITS-1:0] req_handle,

    output wire                         resp_valid,
    output wire [TCU_TMEM_COL_BITS-1:0] resp_handle,

    // Live-allocation table, exposed read-only so callers can bounds-check
    // TMEM_LD/TMEM_ST addresses against live ranges
    output wire                         live_valid_out  [NUM_ENTRIES],
    output wire [TCU_TMEM_COL_BITS-1:0] live_handle_out [NUM_ENTRIES],
    output wire [TCU_TMEM_COL_BITS:0]   live_ncols_out  [NUM_ENTRIES]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam ENTRY_BITS           = `CLOG2(NUM_ENTRIES);
    localparam COLW                 = TCU_TMEM_COL_BITS;
    localparam SIZEW                = TCU_TMEM_COL_BITS + 1;

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    logic               free_valid [NUM_ENTRIES];
    logic [COLW-1:0]    free_start [NUM_ENTRIES];
    logic [SIZEW-1:0]   free_size  [NUM_ENTRIES];

    logic                  live_valid         [NUM_ENTRIES];
    logic [NCTA_WIDTH-1:0] live_cta_id        [NUM_ENTRIES];
    logic [COLW-1:0]       live_handle        [NUM_ENTRIES];
    logic [SIZEW-1:0]      live_ncols         [NUM_ENTRIES];
    logic [7:0]            live_dealloc_count [NUM_ENTRIES];

    for (genvar i = 0; i < NUM_ENTRIES; ++i) begin : g_live_export
        assign live_valid_out[i]  = live_valid[i];
        assign live_handle_out[i] = live_handle[i];
        assign live_ncols_out[i]  = live_ncols[i];
    end

    // -----------------------------------------------------------------------
    // Combinational searches
    // -----------------------------------------------------------------------

    // ALLOC idempotency: does req_cta_id already have a live entry
    logic                cta_match_found;
    logic [ENTRY_BITS-1:0] cta_match_idx;
    always_comb begin
        cta_match_found = 1'b0;
        cta_match_idx   = '0;
        for (int i = 0; i < NUM_ENTRIES; ++i) begin
            if (live_valid[i] && (live_cta_id[i] == req_cta_id)) begin
                cta_match_found = 1'b1;
                cta_match_idx   = ENTRY_BITS'(i);
            end
        end
    end

    // ALLOC first-fit: first free entry large enough for req_ncols
    logic                  free_fit_found;
    logic [ENTRY_BITS-1:0] free_fit_idx;
    always_comb begin
        free_fit_found = 1'b0;
        free_fit_idx   = '0;
        for (int i = 0; i < NUM_ENTRIES; ++i) begin
            if (!free_fit_found && free_valid[i] && (free_size[i] >= req_ncols)) begin
                free_fit_found = 1'b1;
                free_fit_idx   = ENTRY_BITS'(i);
            end
        end
    end

    // ALLOC: first invalid CAM slot to register a new live entry into
    logic                  cam_slot_found;
    logic [ENTRY_BITS-1:0] cam_slot_idx;
    always_comb begin
        cam_slot_found = 1'b0;
        cam_slot_idx   = '0;
        for (int i = 0; i < NUM_ENTRIES; ++i) begin
            if (!cam_slot_found && !live_valid[i]) begin
                cam_slot_found = 1'b1;
                cam_slot_idx   = ENTRY_BITS'(i);
            end
        end
    end

    // DEALLOC: find the live entry by handle (exact match)
    logic                  handle_match_found;
    logic [ENTRY_BITS-1:0] handle_match_idx;
    always_comb begin
        handle_match_found = 1'b0;
        handle_match_idx   = '0;
        for (int i = 0; i < NUM_ENTRIES; ++i) begin
            if (live_valid[i] && (live_handle[i] == req_handle)) begin
                handle_match_found = 1'b1;
                handle_match_idx   = ENTRY_BITS'(i);
            end
        end
    end

    // DEALLOC-completing: is the range about to be freed adjacent to an
    // existing free entry on the left and/or right (coalescing)
    wire [COLW-1:0]  dealloc_handle = live_handle[handle_match_idx];
    wire [SIZEW-1:0] dealloc_ncols  = live_ncols[handle_match_idx];
    wire [SIZEW-1:0] dealloc_end    = {1'b0, dealloc_handle} + dealloc_ncols;

    logic                  left_adj_found;
    logic [ENTRY_BITS-1:0] left_adj_idx;
    logic                  right_adj_found;
    logic [ENTRY_BITS-1:0] right_adj_idx;
    always_comb begin
        left_adj_found  = 1'b0;
        left_adj_idx    = '0;
        right_adj_found = 1'b0;
        right_adj_idx   = '0;
        for (int i = 0; i < NUM_ENTRIES; ++i) begin
            if (free_valid[i]) begin
                if (({1'b0, free_start[i]} + free_size[i]) == {1'b0, dealloc_handle}) begin
                    left_adj_found = 1'b1;
                    left_adj_idx   = ENTRY_BITS'(i);
                end
                if ({1'b0, free_start[i]} == dealloc_end) begin
                    right_adj_found = 1'b1;
                    right_adj_idx   = ENTRY_BITS'(i);
                end
            end
        end
    end

    // DEALLOC-completing, non-adjacent case: first invalid free-list slot
    logic                  free_slot_found;
    logic [ENTRY_BITS-1:0] free_slot_idx;
    always_comb begin
        free_slot_found = 1'b0;
        free_slot_idx   = '0;
        for (int i = 0; i < NUM_ENTRIES; ++i) begin
            if (!free_slot_found && !free_valid[i]) begin
                free_slot_found = 1'b1;
                free_slot_idx   = ENTRY_BITS'(i);
            end
        end
    end

    // -----------------------------------------------------------------------
    // Request classification + response
    // -----------------------------------------------------------------------
    wire req_alloc   = req_valid && !req_is_dealloc;
    wire req_dealloc = req_valid && req_is_dealloc;

    wire alloc_is_repeat = req_alloc && cta_match_found;
    wire alloc_is_fresh  = req_alloc && !cta_match_found;

    // This dealloc is the last of WARPGROUP_SIZE calls expected
    wire dealloc_completes = req_dealloc && handle_match_found
        && ((live_dealloc_count[handle_match_idx] + 8'd1) >= 8'(WARPGROUP_SIZE));

    // A fresh ALLOC this cycle can't be satisfied yet — either no free range
    // is large enough or there's no free CAM slot. Stall by withholding
    // resp_valid so caller doesn't fire this cycle.
    wire alloc_would_stall = alloc_is_fresh && (!free_fit_found || !cam_slot_found);

    assign resp_valid  = req_valid && !alloc_would_stall;
    assign resp_handle = alloc_is_repeat ? live_handle[cta_match_idx]
                        : alloc_is_fresh ? free_start[free_fit_idx]
                        : '0;

    // -----------------------------------------------------------------------
    // Diagnostics — kernel bugs, not recoverable conditions
    // -----------------------------------------------------------------------
    `RUNTIME_ASSERT (~alloc_is_repeat || (live_ncols[cta_match_idx] == req_ncols),
        ("%s: TMEM_ALLOC ncols mismatch for cta_id %0d (existing=%0d, requested=%0d) — one CTA can only hold one live allocation",
         INSTANCE_ID, req_cta_id, live_ncols[cta_match_idx], req_ncols))

    // A request wider than TMEM could ever satisfy
    `RUNTIME_ASSERT (~req_alloc || (SIZEW'(req_ncols) <= SIZEW'(TCU_TMEM_COLS)),
        ("%s: TMEM_ALLOC request for cta_id %0d (ncols=%0d) exceeds TCU_TMEM_COLS (%0d) — unsatisfiable",
         INSTANCE_ID, req_cta_id, req_ncols, TCU_TMEM_COLS))

    `RUNTIME_ASSERT (~req_dealloc || handle_match_found,
        ("%s: TMEM_DEALLOC unknown handle %0d", INSTANCE_ID, req_handle))

    `RUNTIME_ASSERT (~dealloc_completes || left_adj_found || right_adj_found || free_slot_found,
        ("%s: TMEM dealloc free-list return failed for handle %0d — no free-list slot (allocator undersized)",
         INSTANCE_ID, dealloc_handle))

    // -----------------------------------------------------------------------
    // Registered state updates
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_ENTRIES; ++i) begin
                live_valid[i] <= 1'b0;
                free_valid[i]   <= 1'b0;
            end
            free_valid[0] <= 1'b1;
            free_start[0] <= '0;
            free_size[0]  <= SIZEW'(TCU_TMEM_COLS);
        end else begin
            // Fresh ALLOC: consume the matched free range, register a new
            // CAM entry
            if (alloc_is_fresh && free_fit_found && cam_slot_found) begin
                if (free_size[free_fit_idx] == req_ncols) begin
                    free_valid[free_fit_idx] <= 1'b0;
                end else begin
                    free_start[free_fit_idx] <= free_start[free_fit_idx] + COLW'(req_ncols);
                    free_size[free_fit_idx]  <= free_size[free_fit_idx] - req_ncols;
                end
                live_valid[cam_slot_idx]         <= 1'b1;
                live_cta_id[cam_slot_idx]        <= req_cta_id;
                live_handle[cam_slot_idx]        <= free_start[free_fit_idx];
                live_ncols[cam_slot_idx]         <= req_ncols;
                live_dealloc_count[cam_slot_idx] <= '0;
            end

            // DEALLOC: bump this handle's dealloc count; once it reaches
            // WARPGROUP_SIZE, invalidate the CAM entry and return its
            // range to the free list, coalescing with an adjacent free
            // entry where possible
            if (req_dealloc && handle_match_found) begin
                if (dealloc_completes) begin
                    live_valid[handle_match_idx] <= 1'b0;
                    if (left_adj_found && right_adj_found) begin
                        free_size[left_adj_idx]   <= free_size[left_adj_idx] + dealloc_ncols + free_size[right_adj_idx];
                        free_valid[right_adj_idx] <= 1'b0;
                    end else if (left_adj_found) begin
                        free_size[left_adj_idx] <= free_size[left_adj_idx] + dealloc_ncols;
                    end else if (right_adj_found) begin
                        free_start[right_adj_idx] <= dealloc_handle;
                        free_size[right_adj_idx]  <= free_size[right_adj_idx] + dealloc_ncols;
                    end else if (free_slot_found) begin
                        free_valid[free_slot_idx] <= 1'b1;
                        free_start[free_slot_idx] <= dealloc_handle;
                        free_size[free_slot_idx]  <= dealloc_ncols;
                    end
                end else begin
                    live_dealloc_count[handle_match_idx] <= live_dealloc_count[handle_match_idx] + 8'd1;
                end
            end
        end
    end

endmodule

`endif // VX_CFG_TCU_TMEM_ENABLE
