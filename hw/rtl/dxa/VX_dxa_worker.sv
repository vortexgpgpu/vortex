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

// DXA worker: purely structural — only wire declarations and module
// instantiations. All logic lives in the 5 submodules:
//   setup → addr_gen → gmem_req → rsp_buf + smem_wr
// Plus watchdog for debug.

`include "VX_define.vh"

module VX_dxa_worker import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WORKER_ID = 0,
    parameter GMEM_TAG_WIDTH = L1_MEM_ARB_TAG_WIDTH
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dxa_worker_req_if.slave req_if,
    VX_mem_bus_if.master gmem_bus_if,
    VX_mem_bus_if.master smem_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_SPARAM (WORKER_ID)

    localparam GMEM_BYTES      = `VX_CFG_L1_LINE_SIZE;
    localparam GMEM_DATAW      = GMEM_BYTES * 8;
    localparam GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES);
    localparam GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - GMEM_OFF_BITS;

    localparam SMEM_BYTES      = DXA_LMEM_WORD_SIZE;
    localparam SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W;

    localparam MAX_OUTSTANDING = `VX_CFG_DXA_MAX_INFLIGHT;
    localparam TAG_W = `CLOG2(MAX_OUTSTANDING);
    localparam SEQ_W = `CLOG2(MAX_OUTSTANDING + 1);

    // ════════════════════════════════════════════════════════════════════
    // Inter-module wires
    // ════════════════════════════════════════════════════════════════════

    // setup → downstream
    wire                        transfer_active;
    wire                        pipeline_start;
    dxa_setup_params_t          setup_params;
    wire [NC_WIDTH-1:0]         active_core_id;
    wire [UUID_WIDTH-1:0]       active_uuid;
    wire [NW_WIDTH-1:0]         active_wid;
    wire [BAR_ADDR_W-1:0]       active_bar_addr;
    wire                        active_notify_smem_done;
    wire                        active_is_multicast;
    wire [`VX_CFG_NUM_WARPS-1:0]       active_cta_mask;
    wire [31:0]                 active_smem_stride;

    // addr_gen → gmem_req
    wire                        ag_valid;
    wire                        ag_ready;
    wire [GMEM_ADDR_WIDTH-1:0]  ag_cl_addr;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0]  ag_smem_byte_addr;
    wire [GMEM_OFF_BITS-1:0]    ag_byte_offset;
    wire [GMEM_OFF_BITS:0]      ag_valid_length;
    wire                        ag_oob;
    wire                        ag_last;
    wire [31:0]                 ag_cfill;
    wire                        ag_dest_kmajor;
    wire [15:0]                 ag_per_lane_stride_bytes;
    wire [3:0]                  ag_elem_bytes;

    // gmem_req → smem_wr (direct-drain channel; replaces rsp_buf + FIFO).
    wire                        sw_valid;
    wire                        sw_ready;
    wire [TAG_W-1:0]            sw_tag;
    wire [GMEM_DATAW-1:0]       sw_data;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0]  sw_smem_byte_addr;
    wire [GMEM_OFF_BITS-1:0]    sw_byte_offset;
    wire [GMEM_OFF_BITS:0]      sw_valid_length;
    wire                        sw_oob;
    wire                        sw_last;
    wire [SEQ_W-1:0]            sw_outstanding;

    // smem_wr → gmem_req (release)
    wire                        sw_release_en;
    wire [TAG_W-1:0]            sw_release_tag;

    // smem_wr → setup (completion)
    wire                        transfer_done;
    wire [31:0]                 wr_done_count;
    wire                        smem_req_fire;

    // gmem_req → watchdog
    wire                        gmem_req_fire;
    wire                        stall_no_slot;

`ifdef PERF_ENABLE
    wire [31:0]                 perf_gmem_reqs;
    wire [31:0]                 perf_gmem_span_cycles;
    wire [31:0]                 perf_lmem_writes;
`endif

    // ════════════════════════════════════════════════════════════════════
    // Stage 1: Setup
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_setup setup (
        .clk                  (clk),
        .reset                (reset),
        .req_valid            (req_if.valid),
        .req_ready            (req_if.ready),
        .req_data             (req_if.req_data),
        .desc_data            (req_if.desc_data),
        .transfer_done        (transfer_done),
        .transfer_active      (transfer_active),
        .pipeline_start       (pipeline_start),
        .setup_params         (setup_params),
        .active_core_id       (active_core_id),
        .active_uuid          (active_uuid),
        .active_wid           (active_wid),
        .active_bar_addr      (active_bar_addr),
        .active_notify_smem_done (active_notify_smem_done),
        .active_is_multicast  (active_is_multicast),
        .active_cta_mask      (active_cta_mask),
        .active_smem_stride   (active_smem_stride)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 2: Address Generator
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_addr_gen #(
        .GMEM_LINE_SIZE  (GMEM_BYTES),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH)
    ) addr_gen (
        .clk                  (clk),
        .reset                (reset),
        .start                (pipeline_start),
        .setup_params         (setup_params),
        .out_valid            (ag_valid),
        .out_ready            (ag_ready),
        .out_cl_addr          (ag_cl_addr),
        .out_smem_byte_addr   (ag_smem_byte_addr),
        .out_byte_offset      (ag_byte_offset),
        .out_valid_length     (ag_valid_length),
        .out_oob              (ag_oob),
        .out_last             (ag_last),
        .out_cfill            (ag_cfill),
        .out_dest_kmajor      (ag_dest_kmajor),
        .out_per_lane_stride_bytes (ag_per_lane_stride_bytes),
        .out_elem_bytes       (ag_elem_bytes)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 3: GMEM Request Issuer
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_gmem_req #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_TAG_WIDTH  (GMEM_TAG_WIDTH),
        .CL_OFF_BITS     (GMEM_OFF_BITS),
        .SMEM_ADDR_W     (`VX_CFG_MEM_ADDR_WIDTH)
    ) gmem_req (
        .clk                (clk),
        .reset              (reset),
    `ifdef PERF_ENABLE
        .perf_gmem_reqs       (perf_gmem_reqs),
        .perf_gmem_span_cycles(perf_gmem_span_cycles),
    `endif
        .transfer_active    (transfer_active),
        .active_uuid        (active_uuid),
        .ag_valid           (ag_valid),
        .ag_ready           (ag_ready),
        .ag_cl_addr         (ag_cl_addr),
        .ag_smem_byte_addr  (ag_smem_byte_addr),
        .ag_byte_offset     (ag_byte_offset),
        .ag_valid_length    (ag_valid_length),
        .ag_oob             (ag_oob),
        .ag_last            (ag_last),
        .gmem_bus_if        (gmem_bus_if),
        .sw_valid           (sw_valid),
        .sw_ready           (sw_ready),
        .sw_tag             (sw_tag),
        .sw_data            (sw_data),
        .sw_smem_byte_addr  (sw_smem_byte_addr),
        .sw_byte_offset     (sw_byte_offset),
        .sw_valid_length    (sw_valid_length),
        .sw_oob             (sw_oob),
        .sw_last            (sw_last),
        .sw_outstanding     (sw_outstanding),
        .release_en         (sw_release_en),
        .release_tag        (sw_release_tag),
        .gmem_req_fire      (gmem_req_fire),
        .stall_no_slot      (stall_no_slot)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 4: SMEM Writer (rsp_buf is gone — gmem_req drives smem_wr
    // directly via the sw_* channel; smem_wr's pend slot + deferred-last
    // register handle OOO drain).
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_smem_wr #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .CL_SIZE         (GMEM_BYTES),
        .SMEM_WORD_SIZE  (SMEM_BYTES),
        .SMEM_ADDR_WIDTH (SMEM_ADDR_WIDTH),
        .GMEM_DATAW      (GMEM_DATAW)
    ) smem_wr (
        .clk                   (clk),
        .reset                 (reset),
    `ifdef PERF_ENABLE
        .perf_lmem_writes      (perf_lmem_writes),
    `endif
        .transfer_active       (transfer_active),
        .transfer_start        (pipeline_start),
        .cfill                 (ag_cfill),
        .active_core_id        (active_core_id),
        .active_uuid           (active_uuid),
        .active_bar_addr       (active_bar_addr),
        .active_notify_smem_done (active_notify_smem_done),
        .sw_valid              (sw_valid),
        .sw_ready              (sw_ready),
        .sw_tag                (sw_tag),
        .sw_data               (sw_data),
        .sw_smem_byte_addr     (sw_smem_byte_addr),
        .sw_byte_offset        (sw_byte_offset),
        .sw_valid_length       (sw_valid_length),
        .sw_oob                (sw_oob),
        .sw_last               (sw_last),
        .sw_outstanding        (sw_outstanding),
        .release_en            (sw_release_en),
        .release_tag           (sw_release_tag),
        .smem_bus_if           (smem_bus_if),
        .transfer_done         (transfer_done),
        .wr_done_count         (wr_done_count),
        .smem_req_fire         (smem_req_fire),
        .is_multicast          (active_is_multicast),
        .cta_mask              (active_cta_mask),
        .smem_stride           (active_smem_stride),
        .dest_kmajor           (ag_dest_kmajor),
        .per_lane_stride_bytes (ag_per_lane_stride_bytes),
        .elem_bytes            (ag_elem_bytes)
    );

    // ════════════════════════════════════════════════════════════════════
    // Watchdog
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_watchdog #(
        .INSTANCE_ID (INSTANCE_ID)
    ) watchdog (
        .clk              (clk),
        .reset            (reset),
        .transfer_active  (transfer_active),
        .gmem_req_fire    (gmem_req_fire),
        .gmem_rsp_valid   (sw_valid && sw_ready),  // any CL delivered to smem_wr
        .smem_req_fire    (smem_req_fire),
        .active_core_id   (active_core_id),
        .active_wid       (active_wid),
        .active_bar_addr  (active_bar_addr)
    );

    // ════════════════════════════════════════════════════════════════════
    // Perf counters
    // ════════════════════════════════════════════════════════════════════

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_transfers_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_reads_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_dedup_r;
    reg [PERF_CTR_BITS-1:0] perf_lmem_writes_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_lt_r;
    always @(posedge clk) begin
        if (reset) begin
            perf_transfers_r   <= '0;
            perf_gmem_reads_r  <= '0;
            perf_gmem_dedup_r  <= '0;
            perf_lmem_writes_r <= '0;
            perf_gmem_lt_r     <= '0;
        end else begin
            if (transfer_active && transfer_done) begin
                perf_transfers_r   <= perf_transfers_r + PERF_CTR_BITS'(1);
                perf_gmem_reads_r  <= perf_gmem_reads_r + PERF_CTR_BITS'(perf_gmem_reqs);
                perf_lmem_writes_r <= perf_lmem_writes_r + PERF_CTR_BITS'(perf_lmem_writes);
                perf_gmem_lt_r     <= perf_gmem_lt_r + PERF_CTR_BITS'(perf_gmem_span_cycles);
            end
        end
    end
    assign dxa_perf.transfers    = perf_transfers_r;
    assign dxa_perf.gmem_reads   = perf_gmem_reads_r;
    assign dxa_perf.gmem_dedup   = perf_gmem_dedup_r;
    assign dxa_perf.lmem_writes  = perf_lmem_writes_r;
    assign dxa_perf.gmem_latency = perf_gmem_lt_r;
`endif

    `UNUSED_VAR (stall_no_slot)
`ifndef DBG_TRACE_DXA
    `UNUSED_VAR (wr_done_count)
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (pipeline_start) begin
                `TRACE(1, ("%t: %s setup-done: core=%0d, wid=%0d, bar=%0d\n",
                    $time, INSTANCE_ID, active_core_id, active_wid, active_bar_addr))
                $write("DXA_TL,%0d,SETUP_DONE,core=%0d,wid=%0d,bar=%0d\n",
                    $time, active_core_id, active_wid, active_bar_addr);
            end
            if (transfer_active) begin
                if (gmem_req_fire) begin
                    `TRACE(2, ("%t: %s gmem-req-fire\n", $time, INSTANCE_ID))
                end
                if (sw_valid && sw_ready) begin
                    `TRACE(2, ("%t: %s sw-deliver: tag=%0d, oob=%0d, last=%0d\n",
                        $time, INSTANCE_ID, sw_tag, sw_oob, sw_last))
                end
                if (smem_req_fire) begin
                    `TRACE(2, ("%t: %s smem-wr-fire: count=%0d\n",
                        $time, INSTANCE_ID, wr_done_count))
                end
            end
            if (transfer_active && transfer_done) begin
                `TRACE(1, ("%t: %s done: core=%0d, wid=%0d, bar=%0d, wr_count=%0d\n",
                    $time, INSTANCE_ID, active_core_id, active_wid,
                    active_bar_addr, wr_done_count))
                $write("DXA_TL,%0d,XFER_DONE,core=%0d,wid=%0d,bar=%0d,wr_count=%0d\n",
                    $time, active_core_id, active_wid, active_bar_addr,
                    wr_done_count);
            end
        end
    end
`endif

endmodule
