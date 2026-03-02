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

// Read Request Scheduler for DXA non-blocking worker.
// Credit-based flow control with tag-encoded smem destination.
// Integrates slot table, line cache, and inflight dedup from existing
// VX_dxa_nb_* submodules.

`include "VX_define.vh"

module VX_dxa_rrs import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter GMEM_BYTES      = `L2_LINE_SIZE,
    parameter GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES),
    parameter GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - GMEM_OFF_BITS,
    parameter GMEM_DATAW      = GMEM_BYTES * 8,
    parameter GMEM_TAG_VALUEW = L1_MEM_ARB_TAG_WIDTH - `UP(UUID_WIDTH),
    parameter SMEM_BYTES      = DXA_SMEM_WORD_SIZE,
    parameter SMEM_DATAW      = SMEM_BYTES * 8,
    parameter SMEM_OFF_BITS   = `CLOG2(SMEM_BYTES),
    parameter WR_QUEUE_DEPTH  = 16
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,

    // AG → RRS element stream (valid/ready handshake).
    input  wire                        ag_valid,
    output wire                        ag_ready,
    input  wire [`MEM_ADDR_WIDTH-1:0]  ag_gmem_byte_addr,
    input  wire [GMEM_ADDR_WIDTH-1:0]  ag_gmem_line_addr,
    input  wire [GMEM_OFF_BITS-1:0]    ag_gmem_off,
    input  wire [`MEM_ADDR_WIDTH-1:0]  ag_smem_byte_addr,
    input  wire                        ag_in_bounds,
    input  wire                        ag_is_last,
    input  wire [31:0]                 ag_elem_idx,
    input  wire [31:0]                 cfill,
    input  wire [31:0]                 elem_bytes,

    // gmem read port.
    output wire                        gmem_rd_req_valid,
    output wire [GMEM_ADDR_WIDTH-1:0]  gmem_rd_req_addr,
    output wire [GMEM_TAG_VALUEW-1:0]  gmem_rd_req_tag,
    input  wire                        gmem_rd_req_ready,
    input  wire                        gmem_rd_rsp_valid,
    input  wire [GMEM_DATAW-1:0]       gmem_rd_rsp_data,
    input  wire [GMEM_TAG_VALUEW-1:0]  gmem_rd_rsp_tag,
    output wire                        gmem_rd_rsp_ready,

    // RRS → WBC write entries.
    output wire                        wb_valid,
    output wire [`MEM_ADDR_WIDTH-1:0]  wb_smem_byte_addr,
    output wire [SMEM_DATAW-1:0]       wb_smem_data,
    output wire [SMEM_BYTES-1:0]       wb_smem_byteen,
    output wire                        wb_is_last,
    input  wire                        wb_ready,

    // Progress counters exposed.
    output wire                        gmem_req_fire,
    output wire                        rsp_fire,
    output wire                        stall_inflight,
    output wire                        stall_no_slot,
    output wire                        stall_rsp_backpressure
);
    localparam RD_SLOT_BITS = `CLOG2(MAX_OUTSTANDING);
    localparam RD_SLOT_W    = `UP(RD_SLOT_BITS);
    `UNUSED_PARAM (WR_QUEUE_DEPTH)

    `STATIC_ASSERT(`IS_POW2(MAX_OUTSTANDING), ("MAX_OUTSTANDING must be power of 2"))
    `STATIC_ASSERT(GMEM_TAG_VALUEW >= RD_SLOT_W, ("gmem tag too narrow for slot encoding"))

    // ---- Byte-enable mask helper ----
    function automatic [SMEM_BYTES-1:0] smem_elem_mask(input [31:0] nbytes);
    begin
        if (nbytes >= SMEM_BYTES) begin
            smem_elem_mask = {SMEM_BYTES{1'b1}};
        end else begin
            smem_elem_mask = (SMEM_BYTES'(1) << nbytes) - 1;
        end
    end
    endfunction

    // ---- Slot table ----
    wire [RD_SLOT_W-1:0] rsp_slot = RD_SLOT_W'(gmem_rd_rsp_tag[RD_SLOT_W-1:0]);
    wire rd_free_found;
    wire [RD_SLOT_W-1:0] rd_free_slot;
    wire line_inflight_found;
    wire rsp_slot_busy;
    wire [`MEM_ADDR_WIDTH-1:0] rsp_smem_byte_addr;
    wire [GMEM_ADDR_WIDTH-1:0] rsp_gmem_line_addr;
    wire [GMEM_OFF_BITS-1:0] rsp_gmem_off;
    wire rsp_last;

    wire gmem_req_fire_w;
    wire rsp_fire_w;

    VX_dxa_nb_slot_table #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .MEM_ADDR_WIDTH  (`MEM_ADDR_WIDTH),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_OFF_BITS   (GMEM_OFF_BITS)
    ) slot_table (
        .clk                 (clk),
        .reset               (reset),
        .q_line_addr         (ag_gmem_line_addr),
        .q_inflight_found    (line_inflight_found),
        .q_free_found        (rd_free_found),
        .q_free_slot         (rd_free_slot),
        .rsp_slot            (rsp_slot),
        .rsp_slot_busy       (rsp_slot_busy),
        .rsp_smem_byte_addr  (rsp_smem_byte_addr),
        .rsp_gmem_line_addr  (rsp_gmem_line_addr),
        .rsp_gmem_off        (rsp_gmem_off),
        .rsp_is_last         (rsp_last),
        .alloc_fire          (gmem_req_fire_w),
        .alloc_slot          (rd_free_slot),
        .alloc_smem_byte_addr(ag_smem_byte_addr),
        .alloc_gmem_line_addr(ag_gmem_line_addr),
        .alloc_gmem_off      (ag_gmem_off),
        .alloc_is_last       (ag_is_last),
        .release_fire        (rsp_fire_w),
        .release_slot        (rsp_slot)
    );

    // ---- Line cache (single-entry, for consecutive-element dedup) ----
    wire cache_hit;
    wire [63:0] cache_elem_data;

    wire rd_need_fill = ag_valid && ~ag_in_bounds;
    wire rd_need_read = ag_valid && ag_in_bounds;

    VX_dxa_nb_line_cache #(
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_DATAW      (GMEM_DATAW),
        .GMEM_OFF_BITS   (GMEM_OFF_BITS)
    ) line_cache (
        .clk             (clk),
        .reset           (reset),
        .update_fire     (rsp_fire_w),
        .update_line_addr(rsp_gmem_line_addr),
        .update_line_data(gmem_rd_rsp_data),
        .query_valid     (rd_need_read),
        .query_line_addr (ag_gmem_line_addr),
        .query_off       (ag_gmem_off),
        .query_hit       (cache_hit),
        .query_elem_data (cache_elem_data)
    );

    // ---- Response unpack (gmem rsp → smem write entry) ----
    wire [1 + `MEM_ADDR_WIDTH + SMEM_DATAW + SMEM_BYTES - 1:0] rsp_wrq_data;
    VX_dxa_nb_rsp_unpack #(
        .MEM_ADDR_WIDTH (`MEM_ADDR_WIDTH),
        .GMEM_DATAW     (GMEM_DATAW),
        .GMEM_OFF_BITS  (GMEM_OFF_BITS),
        .SMEM_BYTES     (SMEM_BYTES)
    ) rsp_unpack (
        .gmem_rsp_data     (gmem_rd_rsp_data),
        .rsp_gmem_off      (rsp_gmem_off),
        .rsp_smem_byte_addr(rsp_smem_byte_addr),
        .rsp_last          (rsp_last),
        .elem_bytes        (elem_bytes),
        .rsp_wrq_data      (rsp_wrq_data)
    );

    // ---- Write entry construction ----
    // Three sources can produce write entries:
    //   1. gmem read response (rsp_fire)
    //   2. line cache hit (cache_fire)
    //   3. OOB fill (fill_fire)
    // Priority: rsp > cache > fill (to drain responses quickly).

    wire rsp_push_req = gmem_rd_rsp_valid && transfer_active && rsp_slot_busy;
    assign rsp_fire_w = rsp_push_req && wb_ready;

    wire cache_fire = cache_hit && ~rsp_push_req && wb_ready;

    wire fill_fire = rd_need_fill && ~rsp_push_req && ~cache_fire && wb_ready;

    // gmem read request: need to read, no cache hit, no inflight dup, have slot.
    wire gmem_issue = rd_need_read && ~cache_hit && ~line_inflight_found && rd_free_found;
    wire stall_inflight_w = rd_need_read && ~cache_hit && line_inflight_found;
    wire stall_no_slot_w = rd_need_read && ~cache_hit && ~line_inflight_found && ~rd_free_found;
    wire stall_rsp_backpressure_w = rsp_push_req && ~wb_ready;
    assign gmem_rd_req_valid = gmem_issue;
    assign gmem_req_fire_w = gmem_issue && gmem_rd_req_ready;

    // AG advances when any of the three paths fire or when a gmem read is issued.
    wire rd_advance = fill_fire || cache_fire || gmem_req_fire_w;
    assign ag_ready = rd_advance;

    // ---- Fill data computation ----
    wire [SMEM_OFF_BITS-1:0] fill_smem_off = SMEM_OFF_BITS'(ag_smem_byte_addr);
    wire [31:0] fill_smem_shift = 32'(fill_smem_off) * 32'd8;
    wire [SMEM_DATAW-1:0] fill_smem_data = SMEM_DATAW'(64'(cfill)) << fill_smem_shift;
    wire [SMEM_BYTES-1:0] fill_smem_byteen = SMEM_BYTES'(smem_elem_mask(elem_bytes) << fill_smem_off);

    // ---- Cache hit data computation ----
    wire [SMEM_OFF_BITS-1:0] cache_smem_off = SMEM_OFF_BITS'(ag_smem_byte_addr);
    wire [31:0] cache_smem_shift = 32'(cache_smem_off) * 32'd8;
    wire [SMEM_DATAW-1:0] cache_smem_data = SMEM_DATAW'(cache_elem_data) << cache_smem_shift;
    wire [SMEM_BYTES-1:0] cache_smem_byteen = SMEM_BYTES'(smem_elem_mask(elem_bytes) << cache_smem_off);

    // ---- gmem request outputs ----
    assign gmem_rd_req_addr = ag_gmem_line_addr;
    assign gmem_rd_req_tag = GMEM_TAG_VALUEW'(rd_free_slot);

    // ---- gmem response ready ----
    assign gmem_rd_rsp_ready = transfer_active && rsp_slot_busy && wb_ready;

    // ---- Write entry mux (rsp > cache > fill) ----
    wire rsp_wrq_last;
    wire [`MEM_ADDR_WIDTH-1:0] rsp_wrq_smem_addr;
    wire [SMEM_DATAW-1:0] rsp_wrq_smem_data;
    wire [SMEM_BYTES-1:0] rsp_wrq_smem_byteen;
    assign {rsp_wrq_last, rsp_wrq_smem_addr, rsp_wrq_smem_data, rsp_wrq_smem_byteen} = rsp_wrq_data;

    assign wb_valid = rsp_fire_w || cache_fire || fill_fire;
    assign wb_smem_byte_addr = rsp_fire_w ? rsp_wrq_smem_addr
                             : `MEM_ADDR_WIDTH'(ag_smem_byte_addr);
    assign wb_smem_data = rsp_fire_w ? rsp_wrq_smem_data
                        : (cache_fire ? cache_smem_data : fill_smem_data);
    assign wb_smem_byteen = rsp_fire_w ? rsp_wrq_smem_byteen
                          : (cache_fire ? cache_smem_byteen : fill_smem_byteen);
    assign wb_is_last = rsp_fire_w ? rsp_wrq_last
                      : ag_is_last;

    // ---- Progress event outputs ----
    assign gmem_req_fire = gmem_req_fire_w;
    assign rsp_fire = rsp_fire_w;
    assign stall_inflight = stall_inflight_w;
    assign stall_no_slot = stall_no_slot_w;
    assign stall_rsp_backpressure = stall_rsp_backpressure_w;

    `UNUSED_VAR (ag_elem_idx)
    `UNUSED_VAR (ag_gmem_byte_addr)
    `UNUSED_VAR (gmem_rd_rsp_tag[GMEM_TAG_VALUEW-1:RD_SLOT_W])


    `RUNTIME_ASSERT(~(gmem_rd_rsp_valid && gmem_rd_rsp_ready) || rsp_slot_busy,
        ("invalid dxa rrs gmem rsp slot"))
endmodule
