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

// DXA GMEM Request Issuer & In-Flight Tracker.
// Uses VX_allocator for O(1) tag management.
// Uses VX_pending_size for credit-based flow control.
// Uses VX_fifo_queue to track in-flight metadata in issue order.
// OOB entries bypass GMEM read and enter FIFO directly.
// Drives VX_mem_bus_if directly (absorbs bus wiring from worker).

`include "VX_define.vh"

module VX_dxa_gmem_req import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(`L1_LINE_SIZE),
    parameter GMEM_TAG_WIDTH  = L1_MEM_ARB_TAG_WIDTH,
    parameter CL_OFF_BITS     = `CLOG2(`L1_LINE_SIZE),
    parameter SMEM_ADDR_W     = `MEM_ADDR_WIDTH
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

    // In-flight FIFO output (to smem_wr).
    output wire                        fifo_valid,
    input  wire                        fifo_pop,
    output wire [TAG_W-1:0]            fifo_tag,
    output wire [SMEM_ADDR_W-1:0]      fifo_smem_byte_addr,
    output wire [CL_OFF_BITS-1:0]      fifo_byte_offset,
    output wire [CL_OFF_BITS:0]        fifo_valid_length,
    output wire                        fifo_oob,
    output wire                        fifo_last,

    // Resource release (from smem_wr).
    input  wire                        release_en,
    input  wire [TAG_W-1:0]            release_tag,

    // Set arrival for OOB entries (no GMEM response expected).
    output wire                        oob_arrived_en,
    output wire [TAG_W-1:0]            oob_arrived_tag,

    // GMEM response forwarding (to rsp_buf).
    output wire                        gmem_rsp_valid,
    output wire [TAG_W-1:0]            gmem_rsp_tag,
    output wire [GMEM_DATAW-1:0]       gmem_rsp_data,

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
    localparam GMEM_BYTES = 2**CL_OFF_BITS;
    localparam GMEM_DATAW = GMEM_BYTES * 8;
    localparam GMEM_TAG_VALUEW = GMEM_TAG_WIDTH - UUID_WIDTH;

    `STATIC_ASSERT(GMEM_TAG_VALUEW >= TAG_W, ("gmem tag too narrow for slot encoding"))

    // ════════════════════════════════════════════════════════════════════
    // VX_allocator: O(1) tag acquisition/release
    // ════════════════════════════════════════════════════════════════════
    wire                alloc_empty, alloc_full;
    wire [TAG_W-1:0]    alloc_tag;
    wire                alloc_acquire;

    VX_allocator #(
        .SIZE (MAX_OUTSTANDING)
    ) tag_alloc (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (alloc_acquire),
        .acquire_addr (alloc_tag),
        .release_en   (release_en),
        .release_addr (release_tag),
        .empty        (alloc_empty),
        .full         (alloc_full)
    );

    `UNUSED_VAR (alloc_empty)

    // ════════════════════════════════════════════════════════════════════
    // VX_pending_size: credit-based flow control
    // ════════════════════════════════════════════════════════════════════
    wire pending_full;

    VX_pending_size #(
        .SIZE (MAX_OUTSTANDING)
    ) pending_ctr (
        .clk   (clk),
        .reset (reset),
        .incr  (alloc_acquire),
        .decr  (release_en),
        .full  (pending_full),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    // ════════════════════════════════════════════════════════════════════
    // VX_fifo_queue: in-flight metadata tracking (issue order)
    // ════════════════════════════════════════════════════════════════════
    localparam FIFO_DATAW = TAG_W + SMEM_ADDR_W + CL_OFF_BITS + (CL_OFF_BITS+1) + 1 + 1;

    wire fifo_full, fifo_empty;
    wire fifo_push;
    wire [FIFO_DATAW-1:0] fifo_data_in, fifo_data_out;

    VX_fifo_queue #(
        .DATAW   (FIFO_DATAW),
        .DEPTH   (MAX_OUTSTANDING),
        .OUT_REG (0),
        .LUTRAM  (1)
    ) inflight_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (fifo_push),
        .pop      (fifo_pop),
        .data_in  (fifo_data_in),
        .data_out (fifo_data_out),
        .empty    (fifo_empty),
        .full     (fifo_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    // Unpack FIFO output.
    assign {fifo_tag, fifo_smem_byte_addr, fifo_byte_offset,
            fifo_valid_length, fifo_oob, fifo_last} = fifo_data_out;
    assign fifo_valid = ~fifo_empty;

    // ════════════════════════════════════════════════════════════════════
    // Issue logic
    // ════════════════════════════════════════════════════════════════════

    wire can_alloc = ~alloc_full && ~pending_full && ~fifo_full;

    wire normal_fire = ag_valid && ~ag_oob && can_alloc && gmem_bus_if.req_ready;
    wire oob_fire    = ag_valid &&  ag_oob && can_alloc;

    wire accept = normal_fire || oob_fire;

    assign ag_ready      = accept;
    assign alloc_acquire = accept;
    assign fifo_push     = accept;
    assign fifo_data_in  = {alloc_tag, ag_smem_byte_addr, ag_byte_offset,
                            ag_valid_length, ag_oob, ag_last};

    // OOB arrival: set rsp_arrived immediately.
    assign oob_arrived_en  = oob_fire;
    assign oob_arrived_tag = alloc_tag;

    assign gmem_req_fire = normal_fire;
    assign stall_no_slot = ag_valid && ~ag_oob && alloc_full;

    // ════════════════════════════════════════════════════════════════════
    // GMEM bus wiring (read-only requests)
    // ════════════════════════════════════════════════════════════════════

    assign gmem_bus_if.req_valid       = ag_valid && ~ag_oob && can_alloc;
    assign gmem_bus_if.req_data.rw     = 1'b0;
    assign gmem_bus_if.req_data.addr   = ag_cl_addr;
    assign gmem_bus_if.req_data.data   = '0;
    assign gmem_bus_if.req_data.byteen = {GMEM_BYTES{1'b1}};
    assign gmem_bus_if.req_data.flags  = '0;
    assign gmem_bus_if.req_data.tag.uuid  = active_uuid;
    assign gmem_bus_if.req_data.tag.value = GMEM_TAG_VALUEW'(alloc_tag);

    // GMEM responses: always accept when active.
    assign gmem_bus_if.rsp_ready = transfer_active;

    // Forward GMEM response to rsp_buf.
    assign gmem_rsp_valid = gmem_bus_if.rsp_valid && transfer_active;
    assign gmem_rsp_tag   = TAG_W'(gmem_bus_if.rsp_data.tag.value);
    assign gmem_rsp_data  = gmem_bus_if.rsp_data.data;

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

    `UNUSED_VAR (gmem_bus_if.req_data.tag.value[GMEM_TAG_VALUEW-1:TAG_W])

endmodule
