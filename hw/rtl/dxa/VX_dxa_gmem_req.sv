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

// DXA GMEM Request Issuer & In-Flight Tracker (Phase 4b — direct drain).
//
// Manages a slot-table-indexed pool of MAX_OUTSTANDING in-flight tags. On
// each accept, allocates a free slot and writes per-tag metadata. GMEM
// responses (out-of-order from the L1) are streamed directly to smem_wr
// via the `sw_*` channel — no rsp_buf BRAM. OOB entries skip the bus and
// are presented to smem_wr via an oob_pending[] bitvector.
//
// Tag reuse is per-slot via a busy[] bitvector and outstanding counter;
// release order is independent of issue order (OOO drain in smem_wr).

`include "VX_define.vh"

module VX_dxa_gmem_req import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(`VX_CFG_L1_LINE_SIZE),
    parameter GMEM_TAG_WIDTH  = L1_MEM_ARB_TAG_WIDTH,
    parameter CL_OFF_BITS     = `CLOG2(`VX_CFG_L1_LINE_SIZE),
    parameter SMEM_ADDR_W     = `VX_CFG_MEM_ADDR_WIDTH
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,

    // UUID for GMEM request tagging.
    input  wire [UUID_WIDTH-1:0]       active_uuid,

    // From addr_gen (valid/ready).
    input  wire                        ag_valid,
    output wire                        ag_ready,
    input  wire [GMEM_ADDR_WIDTH-1:0]  ag_cl_addr,
    input  wire [SMEM_ADDR_W-1:0]      ag_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      ag_byte_offset,
    input  wire [CL_OFF_BITS:0]        ag_valid_length,
    input  wire                        ag_oob,
    input  wire                        ag_last,

    // GMEM bus interface (reads only).
    VX_mem_bus_if.master               gmem_bus_if,

    // Direct drain channel to smem_wr (replaces rsp_buf + FIFO).
    output wire                        sw_valid,
    input  wire                        sw_ready,
    output wire [TAG_W-1:0]            sw_tag,
    output wire [GMEM_DATAW-1:0]       sw_data,
    output wire [SMEM_ADDR_W-1:0]      sw_smem_byte_addr,
    output wire [CL_OFF_BITS-1:0]      sw_byte_offset,
    output wire [CL_OFF_BITS:0]        sw_valid_length,
    output wire                        sw_oob,
    output wire                        sw_last,
    output wire [SEQ_W-1:0]            sw_outstanding,  // # slots busy (incl. presented-to-smem_wr).

    // Resource release (from smem_wr) — per-tag, OOO.
    input  wire                        release_en,
    input  wire [TAG_W-1:0]            release_tag,

    // Progress events.
    output wire                        gmem_req_fire,
    output wire                        stall_no_slot

`ifdef PERF_ENABLE
    ,
    output wire [31:0]                 perf_gmem_reqs,
    output wire [31:0]                 perf_gmem_span_cycles
`endif
);
    localparam TAG_W = `CLOG2(MAX_OUTSTANDING);
    localparam SEQ_W = `CLOG2(MAX_OUTSTANDING + 1);
    localparam GMEM_BYTES = 2**CL_OFF_BITS;
    localparam GMEM_DATAW = GMEM_BYTES * 8;
    localparam GMEM_TAG_VALUEW = GMEM_TAG_WIDTH - UUID_WIDTH;

    `STATIC_ASSERT(GMEM_TAG_VALUEW >= TAG_W, ("gmem tag too narrow for slot encoding"))

    // ════════════════════════════════════════════════════════════════════
    // Slot table: per-tag metadata.
    // ════════════════════════════════════════════════════════════════════
    typedef struct packed {
        logic [SMEM_ADDR_W-1:0]   smem_byte_addr;
        logic [CL_OFF_BITS-1:0]   byte_offset;
        logic [CL_OFF_BITS:0]     valid_length;
        logic                     oob;
        logic                     last;
    } slot_t;

    slot_t slot_table_r [MAX_OUTSTANDING];

    // ════════════════════════════════════════════════════════════════════
    // Per-tag occupancy bookkeeping.
    //   busy[T]         : slot is currently in flight (issued, not released).
    //   oob_pending[T]  : OOB slot that hasn't been presented to smem_wr yet.
    //   outstanding_count_r : # of set bits in busy.
    // ════════════════════════════════════════════════════════════════════
    reg [MAX_OUTSTANDING-1:0] busy_r;
    reg [MAX_OUTSTANDING-1:0] oob_pending_r;
    reg [SEQ_W-1:0]           outstanding_count_r;

    // Free-tag picker: priority-encode the inverse of busy_r.
    wire [TAG_W-1:0] free_tag;
    wire             have_free_tag;
    VX_priority_encoder #(
        .N (MAX_OUTSTANDING)
    ) free_pe (
        .data_in   (~busy_r),
        .index_out (free_tag),
        .valid_out (have_free_tag),
        `UNUSED_PIN (onehot_out)
    );

    // OOB-tag picker: priority-encode oob_pending_r.
    wire [TAG_W-1:0] oob_tag;
    wire             have_oob;
    VX_priority_encoder #(
        .N (MAX_OUTSTANDING)
    ) oob_pe (
        .data_in   (oob_pending_r),
        .index_out (oob_tag),
        .valid_out (have_oob),
        `UNUSED_PIN (onehot_out)
    );

    // ════════════════════════════════════════════════════════════════════
    // Issue logic
    // ════════════════════════════════════════════════════════════════════

    wire can_alloc = have_free_tag;

    wire normal_fire = ag_valid && ~ag_oob && can_alloc && gmem_bus_if.req_ready;
    wire oob_fire    = ag_valid &&  ag_oob && can_alloc;
    wire accept      = normal_fire || oob_fire;

    wire [TAG_W-1:0] alloc_tag = free_tag;

    assign ag_ready      = accept;
    assign gmem_req_fire = normal_fire;
    assign stall_no_slot = ag_valid && ~ag_oob && ~have_free_tag;

    // ════════════════════════════════════════════════════════════════════
    // Direct-drain channel arbitration.
    // Prefer real GMEM responses (so the bus never stalls when smem_wr is
    // ready). OOB-synthetic CLs fill in cycles when there's no bus rsp.
    // ════════════════════════════════════════════════════════════════════

    wire bus_rsp_present = gmem_bus_if.rsp_valid;
    wire [TAG_W-1:0] bus_rsp_tag = TAG_W'(gmem_bus_if.rsp_data.tag.value);
    wire             present_oob = ~bus_rsp_present && have_oob;

    wire [TAG_W-1:0] present_tag = bus_rsp_present ? bus_rsp_tag : oob_tag;
    slot_t           present_slot;
    assign present_slot = slot_table_r[present_tag];

    assign sw_valid          = bus_rsp_present || present_oob;
    assign sw_tag            = present_tag;
    assign sw_data           = gmem_bus_if.rsp_data.data;  // don't-care when sw_oob.
    assign sw_smem_byte_addr = present_slot.smem_byte_addr;
    assign sw_byte_offset    = present_slot.byte_offset;
    assign sw_valid_length   = present_slot.valid_length;
    assign sw_oob            = present_slot.oob;
    assign sw_last           = present_slot.last;
    assign sw_outstanding    = outstanding_count_r;

    // The bus is accepted by smem_wr (via sw_ready) only when we present
    // a real rsp; OOB presentations don't touch the bus port.
    assign gmem_bus_if.rsp_ready = sw_ready && bus_rsp_present;

    // ════════════════════════════════════════════════════════════════════
    // GMEM bus wiring (read-only requests)
    // ════════════════════════════════════════════════════════════════════
    assign gmem_bus_if.req_valid       = ag_valid && ~ag_oob && can_alloc;
    assign gmem_bus_if.req_data.rw     = 1'b0;
    assign gmem_bus_if.req_data.addr   = ag_cl_addr;
    assign gmem_bus_if.req_data.data   = '0;
    assign gmem_bus_if.req_data.byteen = {GMEM_BYTES{1'b1}};
    assign gmem_bus_if.req_data.attr   = '0;
    assign gmem_bus_if.req_data.tag.uuid  = active_uuid;
    assign gmem_bus_if.req_data.tag.value = GMEM_TAG_VALUEW'(alloc_tag);

    // ════════════════════════════════════════════════════════════════════
    // Sequential update
    // ════════════════════════════════════════════════════════════════════
    wire        oob_present_fire = present_oob && sw_ready;
    wire [MAX_OUTSTANDING-1:0] busy_set        = accept ? (MAX_OUTSTANDING'(1) << alloc_tag) : '0;
    wire [MAX_OUTSTANDING-1:0] busy_clr        = release_en ? (MAX_OUTSTANDING'(1) << release_tag) : '0;
    wire [MAX_OUTSTANDING-1:0] oob_pending_set = oob_fire   ? (MAX_OUTSTANDING'(1) << alloc_tag) : '0;
    wire [MAX_OUTSTANDING-1:0] oob_pending_clr = oob_present_fire ? (MAX_OUTSTANDING'(1) << oob_tag) : '0;

    always @(posedge clk) begin
        if (reset) begin
            busy_r              <= '0;
            oob_pending_r       <= '0;
            outstanding_count_r <= '0;
        end else begin
            if (accept) begin
                slot_table_r[alloc_tag].smem_byte_addr <= ag_smem_byte_addr;
                slot_table_r[alloc_tag].byte_offset    <= ag_byte_offset;
                slot_table_r[alloc_tag].valid_length   <= ag_valid_length;
                slot_table_r[alloc_tag].oob            <= ag_oob;
                slot_table_r[alloc_tag].last           <= ag_last;
            end

            busy_r        <= (busy_r        | busy_set       ) & ~busy_clr;
            oob_pending_r <= (oob_pending_r | oob_pending_set) & ~oob_pending_clr;

            // Outstanding count: increments on accept, decrements on release.
            case ({accept, release_en})
                2'b10: outstanding_count_r <= outstanding_count_r + SEQ_W'(1);
                2'b01: outstanding_count_r <= outstanding_count_r - SEQ_W'(1);
                default: ;
            endcase
        end
    end

`ifdef PERF_ENABLE
    reg [31:0] rdp_total_gmem_req_r;
    reg [31:0] rdp_cycle_ctr_r;
    reg [31:0] rdp_first_req_cycle_r;
    reg [31:0] rdp_last_rsp_cycle_r;
    reg        rdp_has_req_r;

    always @(posedge clk) begin
        if (reset || !transfer_active) begin
            rdp_total_gmem_req_r  <= '0;
            rdp_cycle_ctr_r       <= '0;
            rdp_first_req_cycle_r <= '0;
            rdp_last_rsp_cycle_r  <= '0;
            rdp_has_req_r         <= 1'b0;
        end else begin
            rdp_cycle_ctr_r <= rdp_cycle_ctr_r + 32'd1;
            if (normal_fire) begin
                rdp_total_gmem_req_r <= rdp_total_gmem_req_r + 32'd1;
                if (!rdp_has_req_r) begin
                    rdp_first_req_cycle_r <= rdp_cycle_ctr_r;
                    rdp_has_req_r <= 1'b1;
                end
            end
            if (release_en) begin
                rdp_last_rsp_cycle_r <= rdp_cycle_ctr_r;
            end
        end
    end

    assign perf_gmem_reqs        = rdp_total_gmem_req_r;
    assign perf_gmem_span_cycles = (rdp_has_req_r && rdp_last_rsp_cycle_r >= rdp_first_req_cycle_r)
                                 ? (rdp_last_rsp_cycle_r - rdp_first_req_cycle_r + 32'd1) : 32'd0;
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && transfer_active) begin
            if (normal_fire) begin
                $write("DXA_PIPE,%0d,GMEM_REQ,addr=0x%0h,tag=%0d,smem=0x%0h\n",
                    $time, ag_cl_addr, alloc_tag, ag_smem_byte_addr);
            end
            if (oob_fire) begin
                $write("DXA_PIPE,%0d,GMEM_OOB,tag=%0d,smem=0x%0h,last=%0d\n",
                    $time, alloc_tag, ag_smem_byte_addr, ag_last);
            end
        end
    end
`endif

    `UNUSED_VAR (transfer_active)
    `UNUSED_VAR (gmem_bus_if.req_data.tag.value[GMEM_TAG_VALUEW-1:TAG_W])

endmodule
