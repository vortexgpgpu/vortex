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
//
// Manages a slot-table-indexed pool of MAX_OUTSTANDING in-flight tags. On
// each accept, allocates a free slot and writes per-tag metadata. GMEM
// responses (out-of-order from the L1) pass through a one-entry stage,
// where their slot is read from the slot RAM, to smem_wr on the `sw_*`
// channel — no rsp_buf. OOB entries skip the bus and are presented to
// smem_wr through a FIFO of their tags.
//
// Released tags are reused in release order through a free-tag FIFO;
// release order is independent of issue order (OOO drain in smem_wr). Each
// slot records its transfer, so the reads of consecutive transfers share
// the pool.

`include "VX_define.vh"

module VX_dxa_gmem_req import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter NUM_XFERS       = 2,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(`VX_CFG_L1_LINE_SIZE),
    parameter GMEM_TAG_WIDTH  = SOCKET_MEM_TAG_WIDTH,
    parameter CL_OFF_BITS     = `CLOG2(`VX_CFG_L1_LINE_SIZE),
    parameter SMEM_ADDR_W     = DXA_SMEM_ADDR_W,
    localparam TAG_W          = `CLOG2(MAX_OUTSTANDING),
    localparam XFER_W         = `LOG2UP(NUM_XFERS),
    localparam GMEM_BYTES     = 2**CL_OFF_BITS,
    localparam GMEM_DATAW     = GMEM_BYTES * 8
) (
    input  wire                        clk,
    input  wire                        reset,

    // UUID for GMEM request tagging.
    input  wire [UUID_WIDTH-1:0]       active_uuid,

    // From addr_gen (valid/ready), for transfer ag_xfer.
    input  wire [XFER_W-1:0]           ag_xfer,
    input  wire                        ag_valid,
    output wire                        ag_ready,
    input  wire [GMEM_ADDR_WIDTH-1:0]  ag_cl_addr,
    input  wire [SMEM_ADDR_W-1:0]      ag_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      ag_byte_offset,
    input  wire [CL_OFF_BITS:0]        ag_valid_length,
    input  wire                        ag_oob,
    input  wire [15:0]                 ag_k_row,    // tiled scatter: K-row of this CL
    input  wire [15:0]                 ag_n_base,   // tiled scatter: N index of CL element 0

    // GMEM bus interface (reads only).
    VX_mem_bus_if.master               gmem_bus_if,

    // Direct drain channel to smem_wr (replaces rsp_buf + FIFO).
    output wire                        sw_valid,
    input  wire                        sw_ready,
    output wire [TAG_W-1:0]            sw_tag,
    output wire [XFER_W-1:0]           sw_xfer,
    output wire [GMEM_DATAW-1:0]       sw_data,
    output wire [SMEM_ADDR_W-1:0]      sw_smem_byte_addr,
    output wire [CL_OFF_BITS-1:0]      sw_byte_offset,
    output wire [CL_OFF_BITS:0]        sw_valid_length,
    output wire                        sw_oob,
    output wire [15:0]                 sw_k_row,
    output wire [15:0]                 sw_n_base,

    // Resource release (from smem_wr) — per-tag, OOO.
    input  wire                        release_en,
    input  wire [TAG_W-1:0]            release_tag,

    // Progress events.
    output wire                        gmem_req_fire,
    output wire                        stall_no_slot
);
    localparam GMEM_TAG_VALUEW = GMEM_TAG_WIDTH - UUID_WIDTH;

    // Pre-register bus: the combinational request datapath drives this, then a
    // register slice flops it out onto gmem_bus_if so this module presents a
    // clean registered producer boundary (terminates the addr-gen cone here).
    VX_mem_bus_if #(
        .DATA_SIZE (GMEM_BYTES),
        .TAG_WIDTH (GMEM_TAG_WIDTH)
    ) mem_bus_w ();

    `STATIC_ASSERT(GMEM_TAG_VALUEW >= TAG_W, ("gmem tag too narrow for slot encoding"))

    // ════════════════════════════════════════════════════════════════════
    // Slot table: per-tag metadata.
    // ════════════════════════════════════════════════════════════════════
    typedef struct packed {
        logic [XFER_W-1:0]        xfer;
        logic [SMEM_ADDR_W-1:0]   smem_byte_addr;
        logic [CL_OFF_BITS-1:0]   byte_offset;
        logic [CL_OFF_BITS:0]     valid_length;
        logic                     oob;
        logic [15:0]              k_row;     // tiled scatter
        logic [15:0]              n_base;    // tiled scatter
    } slot_t;


    // ════════════════════════════════════════════════════════════════════
    // Tag pools. The pool is sized to the memory latency, so tags are handed
    // out in FIFO order rather than picked by a pool-wide priority encoder:
    // never-used tags count up from 0, released tags recycle through a FIFO.
    // OOB lines wait for smem_wr in a FIFO of their tags, in issue order.
    // ════════════════════════════════════════════════════════════════════
    reg [TAG_W:0]    fresh_r;
    wire             fresh_avail = ~fresh_r[TAG_W];
    wire [TAG_W-1:0] recycled_tag;
    wire             recycled_empty;
    wire             accept;

    VX_fifo_queue #(
        .DATAW (TAG_W),
        .DEPTH (MAX_OUTSTANDING)
    ) free_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (release_en),
        .pop      (accept && ~fresh_avail),
        .data_in  (release_tag),
        .data_out (recycled_tag),
        .empty    (recycled_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    wire             have_free_tag = fresh_avail || ~recycled_empty;
    wire [TAG_W-1:0] free_tag      = fresh_avail ? fresh_r[TAG_W-1:0] : recycled_tag;

    always @(posedge clk) begin
        if (reset) begin
            fresh_r <= '0;
        end else if (accept && fresh_avail) begin
            fresh_r <= fresh_r + (TAG_W+1)'(1);
        end
    end

    wire [TAG_W-1:0] oob_tag;
    wire             oob_empty;
    wire             oob_fire;
    wire             oob_take;

    VX_fifo_queue #(
        .DATAW (TAG_W),
        .DEPTH (MAX_OUTSTANDING)
    ) oob_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (oob_fire),
        .pop      (oob_take),
        .data_in  (free_tag),
        .data_out (oob_tag),
        .empty    (oob_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    wire have_oob = ~oob_empty;

    // ════════════════════════════════════════════════════════════════════
    // Issue logic
    // ════════════════════════════════════════════════════════════════════

    wire can_alloc = have_free_tag;

    wire normal_fire = ag_valid && ~ag_oob && can_alloc && mem_bus_w.req_ready;
    assign oob_fire  = ag_valid &&  ag_oob && can_alloc;
    assign accept    = normal_fire || oob_fire;

    wire [TAG_W-1:0] alloc_tag = free_tag;

    assign ag_ready      = accept;
    assign gmem_req_fire = normal_fire;
    assign stall_no_slot = ag_valid && ~ag_oob && ~have_free_tag;

    // ════════════════════════════════════════════════════════════════════
    // Response stage. The slot table is sized to the memory latency, so it
    // lives in a RAM with a registered read: a CL spends one cycle here while
    // its slot is read. A real GMEM response has priority (so the bus never
    // stalls while smem_wr is ready); OOB-synthetic CLs fill the other cycles.
    // ════════════════════════════════════════════════════════════════════

    wire bus_rsp_present = mem_bus_w.rsp_valid;
    wire [TAG_W-1:0] bus_rsp_tag = TAG_W'(mem_bus_w.rsp_data.tag.value);
    wire             in_valid    = bus_rsp_present || have_oob;
    wire [TAG_W-1:0] in_tag      = bus_rsp_present ? bus_rsp_tag : oob_tag;

    reg                  stg_valid_r;
    reg [TAG_W-1:0]      stg_tag_r;
    reg [GMEM_DATAW-1:0] stg_data_r;

    wire stg_load = in_valid && (~stg_valid_r || sw_ready);
    assign oob_take = stg_load && ~bus_rsp_present;

    slot_t alloc_slot, stg_slot;
    assign alloc_slot.xfer           = ag_xfer;
    assign alloc_slot.smem_byte_addr = ag_smem_byte_addr;
    assign alloc_slot.byte_offset    = ag_byte_offset;
    assign alloc_slot.valid_length   = ag_valid_length;
    assign alloc_slot.oob            = ag_oob;
    assign alloc_slot.k_row          = ag_k_row;
    assign alloc_slot.n_base         = ag_n_base;

    VX_dp_ram #(
        .DATAW    ($bits(slot_t)),
        .SIZE     (MAX_OUTSTANDING),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) slot_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (accept),
        .wren  (1'b1),
        .waddr (alloc_tag),
        .wdata (alloc_slot),
        .raddr (stg_load ? in_tag : stg_tag_r),
        .rdata (stg_slot)
    );

    always @(posedge clk) begin
        if (reset) begin
            stg_valid_r <= 1'b0;
        end else if (stg_load) begin
            stg_valid_r <= 1'b1;
        end else if (sw_ready) begin
            stg_valid_r <= 1'b0;
        end
        if (stg_load) begin
            stg_tag_r  <= in_tag;
            stg_data_r <= mem_bus_w.rsp_data.data;  // don't-care for an OOB CL
        end
    end

    assign sw_valid          = stg_valid_r;
    assign sw_tag            = stg_tag_r;
    assign sw_xfer           = stg_slot.xfer;
    assign sw_data           = stg_data_r;
    assign sw_smem_byte_addr = stg_slot.smem_byte_addr;
    assign sw_byte_offset    = stg_slot.byte_offset;
    assign sw_valid_length   = stg_slot.valid_length;
    assign sw_oob            = stg_slot.oob;
    assign sw_k_row          = stg_slot.k_row;
    assign sw_n_base         = stg_slot.n_base;

    assign mem_bus_w.rsp_ready = stg_load && bus_rsp_present;

    // ════════════════════════════════════════════════════════════════════
    // GMEM bus wiring (read-only requests)
    // ════════════════════════════════════════════════════════════════════
    assign mem_bus_w.req_valid       = ag_valid && ~ag_oob && can_alloc;
    assign mem_bus_w.req_data.rw     = 1'b0;
    assign mem_bus_w.req_data.addr   = ag_cl_addr;
    assign mem_bus_w.req_data.data   = '0;
    assign mem_bus_w.req_data.byteen = {GMEM_BYTES{1'b1}};
    assign mem_bus_w.req_data.attr   = '0;
    assign mem_bus_w.req_data.tag.uuid  = active_uuid;
    assign mem_bus_w.req_data.tag.value = GMEM_TAG_VALUEW'(alloc_tag);

    VX_mem_bus_slice #(
        .DATA_SIZE   (GMEM_BYTES),
        .TAG_WIDTH   (GMEM_TAG_WIDTH),
        .REQ_OUT_BUF (3),
        .RSP_OUT_BUF (0)
    ) gmem_out_slice (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (mem_bus_w),
        .bus_out_if (gmem_bus_if)
    );

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (normal_fire) begin
                $write("DXA_PIPE,%0d,GMEM_REQ,addr=0x%0h,tag=%0d,xfer=%0d,smem=0x%0h\n",
                    $time, ag_cl_addr, alloc_tag, ag_xfer, ag_smem_byte_addr);
            end
            if (oob_fire) begin
                $write("DXA_PIPE,%0d,GMEM_OOB,tag=%0d,xfer=%0d,smem=0x%0h\n",
                    $time, alloc_tag, ag_xfer, ag_smem_byte_addr);
            end
        end
    end
`endif

    if (GMEM_TAG_VALUEW > TAG_W) begin : g_tag_unused
        `UNUSED_VAR (mem_bus_w.req_data.tag.value[GMEM_TAG_VALUEW-1:TAG_W])
    end

endmodule
