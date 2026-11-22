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

// DXA worker: purely structural — wire declarations and module
// instantiations. All logic lives in the submodules:
//   setup → addr_gen → gmem_req → smem_wr + watchdog

`include "VX_define.vh"

module VX_dxa_worker import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WORKER_ID = 0,
    parameter GMEM_TAG_WIDTH = SOCKET_MEM_TAG_WIDTH
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
    localparam NUM_XFERS = `VX_CFG_DXA_MAX_XFERS;
    localparam TAG_W = `CLOG2(MAX_OUTSTANDING);
    localparam SEQ_W = `CLOG2(MAX_OUTSTANDING + 1);
    localparam XFER_W = `LOG2UP(NUM_XFERS);
    // the outstanding-request queue and its tags need a power of two of at least 2
    `STATIC_ASSERT (`IS_POW2(MAX_OUTSTANDING) && (MAX_OUTSTANDING >= 2), ("invalid parameter"))
    `STATIC_ASSERT (`IS_POW2(NUM_XFERS) && (NUM_XFERS >= 2), ("invalid parameter"))

    // ════════════════════════════════════════════════════════════════════
    // Inter-module wires
    // ════════════════════════════════════════════════════════════════════

    // setup → downstream
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

    // transfer table
    wire                        xfer_avail;
    wire [XFER_W-1:0]           issue_xfer;
    wire [XFER_W-1:0]           ctx_pend_raddr;
    dxa_xfer_t                  ctx_pend_rdata;
    wire [31:0]                 ctx_cap_cfill;
    wire                        ctx_cap_scatter;
    wire                        ctx_cap_tiled;
    wire [3:0]                  ctx_cap_lg_tcN;
`ifdef DBG_TRACE_DXA
    wire [NC_WIDTH-1:0]         trace_done_core_id;
    wire [BAR_ADDR_W-1:0]       trace_done_bar_addr;
`endif
    wire [NUM_XFERS-1:0]        xfer_issued;
    wire [NUM_XFERS-1:0][SEQ_W-1:0] xfer_pending;
    wire                        xfer_busy;

    // addr_gen → gmem_req
    wire                        ag_valid;
    wire                        ag_ready;
    wire [GMEM_ADDR_WIDTH-1:0]  ag_cl_addr;
    wire [DXA_SMEM_ADDR_W-1:0]  ag_smem_byte_addr;
    wire [GMEM_OFF_BITS-1:0]    ag_byte_offset;
    wire [GMEM_OFF_BITS:0]      ag_valid_length;
    wire                        ag_oob;
    wire                        ag_last;
    wire [15:0]                 ag_k_row;
    wire [15:0]                 ag_n_base;

    // the issuing transfer's last line is accepted
    wire                        issue_done = ag_valid && ag_ready && ag_last;

    // gmem_req → smem_wr (direct-drain channel).
    wire                        sw_valid;
    wire                        sw_ready;
    wire [TAG_W-1:0]            sw_tag;
    wire [XFER_W-1:0]           sw_xfer;
    wire [GMEM_DATAW-1:0]       sw_data;
    wire [DXA_SMEM_ADDR_W-1:0]  sw_smem_byte_addr;
    wire [GMEM_OFF_BITS-1:0]    sw_byte_offset;
    wire [GMEM_OFF_BITS:0]      sw_valid_length;
    wire                        sw_oob;
    wire [15:0]                 sw_k_row;
    wire [15:0]                 sw_n_base;

    // smem_wr → gmem_req / table (release)
    wire                        sw_release_en;
    wire [TAG_W-1:0]            sw_release_tag;
    wire [XFER_W-1:0]           sw_release_xfer;

    // smem_wr → table (completion)
    wire                        xfer_done;
    wire [XFER_W-1:0]           xfer_done_idx;
    wire                        smem_req_fire;

    // gmem_req → watchdog
    wire                        gmem_req_fire;
    wire                        stall_no_slot;

`ifdef PERF_ENABLE
    wire [31:0]                 perf_xfer_latency;
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
        .issue_done           (issue_done),
        .xfer_avail           (xfer_avail),
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
    // Transfer table: contexts of the transfers in flight
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_xfer_table #(
        .NUM_XFERS       (NUM_XFERS),
        .MAX_OUTSTANDING (MAX_OUTSTANDING)
    ) xfer_table (
        .clk                    (clk),
        .reset                  (reset),
        .alloc_ready            (xfer_avail),
        .alloc_en               (pipeline_start),
        .alloc_core_id          (active_core_id),
        .alloc_uuid             (active_uuid),
        .alloc_bar_addr         (active_bar_addr),
        .alloc_notify_smem_done (active_notify_smem_done),
        .alloc_is_multicast     (active_is_multicast),
        .alloc_cta_mask         (active_cta_mask),
        .alloc_smem_stride      (active_smem_stride),
        .alloc_params           (setup_params),
        .issue_xfer             (issue_xfer),
        .issue_en               (ag_valid && ag_ready),
        .issue_last             (ag_last),
        .release_en             (sw_release_en),
        .release_xfer           (sw_release_xfer),
        .done_en                (xfer_done),
        .done_xfer              (xfer_done_idx),
        .pend_raddr             (ctx_pend_raddr),
        .pend_rdata             (ctx_pend_rdata),
        .cap_raddr              (sw_xfer),
        .cap_cfill              (ctx_cap_cfill),
        .cap_scatter            (ctx_cap_scatter),
        .cap_tiled              (ctx_cap_tiled),
        .cap_lg_tcN             (ctx_cap_lg_tcN),
        .xfer_issued            (xfer_issued),
        .xfer_pending           (xfer_pending),
        .busy                   (xfer_busy)
    `ifdef PERF_ENABLE
        ,
        .done_latency           (perf_xfer_latency)
    `endif
    `ifdef DBG_TRACE_DXA
        ,
        .done_core_id           (trace_done_core_id),
        .done_bar_addr          (trace_done_bar_addr)
    `endif
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
        .out_k_row            (ag_k_row),
        .out_n_base           (ag_n_base)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 3: GMEM Request Issuer
    // ════════════════════════════════════════════════════════════════════

    // The request issuer registers its own gmem output (VX_mem_bus_slice), so
    // gmem_bus_if is already a clean flopped producer boundary here.
    VX_dxa_gmem_req #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .NUM_XFERS       (NUM_XFERS),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_TAG_WIDTH  (GMEM_TAG_WIDTH),
        .CL_OFF_BITS     (GMEM_OFF_BITS),
        .SMEM_ADDR_W     (DXA_SMEM_ADDR_W)
    ) gmem_req (
        .clk                (clk),
        .reset              (reset),
        .active_uuid        (active_uuid),
        .ag_xfer            (issue_xfer),
        .ag_valid           (ag_valid),
        .ag_ready           (ag_ready),
        .ag_cl_addr         (ag_cl_addr),
        .ag_smem_byte_addr  (ag_smem_byte_addr),
        .ag_byte_offset     (ag_byte_offset),
        .ag_valid_length    (ag_valid_length),
        .ag_oob             (ag_oob),
        .ag_k_row           (ag_k_row),
        .ag_n_base          (ag_n_base),
        .gmem_bus_if        (gmem_bus_if),
        .sw_valid           (sw_valid),
        .sw_ready           (sw_ready),
        .sw_tag             (sw_tag),
        .sw_xfer            (sw_xfer),
        .sw_data            (sw_data),
        .sw_smem_byte_addr  (sw_smem_byte_addr),
        .sw_byte_offset     (sw_byte_offset),
        .sw_valid_length    (sw_valid_length),
        .sw_oob             (sw_oob),
        .sw_k_row           (sw_k_row),
        .sw_n_base          (sw_n_base),
        .release_en         (sw_release_en),
        .release_tag        (sw_release_tag),
        .gmem_req_fire      (gmem_req_fire),
        .stall_no_slot      (stall_no_slot)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 4: SMEM Writer (gmem_req drives smem_wr directly via the sw_*
    // channel; each CL reads its transfer's context from the table).
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_smem_wr #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .NUM_XFERS       (NUM_XFERS),
        .CL_SIZE         (GMEM_BYTES),
        .SMEM_WORD_SIZE  (SMEM_BYTES),
        .SMEM_ADDR_WIDTH (SMEM_ADDR_WIDTH),
        .GMEM_DATAW      (GMEM_DATAW)
    ) smem_wr (
        .clk                   (clk),
        .reset                 (reset),
        .pend_raddr            (ctx_pend_raddr),
        .pend_rdata            (ctx_pend_rdata),
        .cap_cfill             (ctx_cap_cfill),
        .cap_scatter           (ctx_cap_scatter),
        .cap_tiled             (ctx_cap_tiled),
        .cap_lg_tcN            (ctx_cap_lg_tcN),
        .xfer_issued           (xfer_issued),
        .xfer_pending          (xfer_pending),
        .sw_valid              (sw_valid),
        .sw_ready              (sw_ready),
        .sw_tag                (sw_tag),
        .sw_xfer               (sw_xfer),
        .sw_data               (sw_data),
        .sw_smem_byte_addr     (sw_smem_byte_addr),
        .sw_byte_offset        (sw_byte_offset),
        .sw_valid_length       (sw_valid_length),
        .sw_oob                (sw_oob),
        .sw_k_row              (sw_k_row),
        .sw_n_base             (sw_n_base),
        .release_en            (sw_release_en),
        .release_tag           (sw_release_tag),
        .release_xfer          (sw_release_xfer),
        .smem_bus_if           (smem_bus_if),
        .xfer_done             (xfer_done),
        .xfer_done_idx         (xfer_done_idx),
        .smem_req_fire         (smem_req_fire)
    );

    // ════════════════════════════════════════════════════════════════════
    // Watchdog
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_watchdog #(
        .INSTANCE_ID (INSTANCE_ID)
    ) watchdog (
        .clk              (clk),
        .reset            (reset),
        .transfer_active  (xfer_busy),
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
    reg [PERF_CTR_BITS-1:0] perf_lmem_writes_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_lt_r;
    reg [PERF_CTR_BITS-1:0] perf_noslot_r;
    always @(posedge clk) begin
        if (reset) begin
            perf_transfers_r   <= '0;
            perf_gmem_reads_r  <= '0;
            perf_lmem_writes_r <= '0;
            perf_gmem_lt_r     <= '0;
            perf_noslot_r      <= '0;
        end else begin
            if (stall_no_slot) begin
                perf_noslot_r <= perf_noslot_r + PERF_CTR_BITS'(1);
            end
            if (gmem_req_fire) begin
                perf_gmem_reads_r <= perf_gmem_reads_r + PERF_CTR_BITS'(1);
            end
            if (smem_req_fire) begin
                perf_lmem_writes_r <= perf_lmem_writes_r + PERF_CTR_BITS'(1);
            end
            if (xfer_done) begin
                perf_transfers_r <= perf_transfers_r + PERF_CTR_BITS'(1);
                perf_gmem_lt_r   <= perf_gmem_lt_r + PERF_CTR_BITS'(perf_xfer_latency);
            end
        end
    end
    assign dxa_perf.transfers    = perf_transfers_r;
    assign dxa_perf.gmem_reads   = perf_gmem_reads_r;
    assign dxa_perf.gmem_dedup   = '0;
    assign dxa_perf.lmem_writes  = perf_lmem_writes_r;
    assign dxa_perf.gmem_latency = perf_gmem_lt_r;
    assign dxa_perf.noslot_stalls = perf_noslot_r;
`else
    `UNUSED_VAR (stall_no_slot)
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (pipeline_start) begin
                `TRACE(1, ("%t: %s setup-done: core=%0d, wid=%0d, bar=%0d, xfer=%0d\n",
                    $time, INSTANCE_ID, active_core_id, active_wid, active_bar_addr, issue_xfer))
                $write("DXA_TL,%0d,SETUP_DONE,core=%0d,wid=%0d,bar=%0d\n",
                    $time, active_core_id, active_wid, active_bar_addr);
            end
            if (gmem_req_fire) begin
                `TRACE(2, ("%t: %s gmem-req-fire\n", $time, INSTANCE_ID))
            end
            if (sw_valid && sw_ready) begin
                `TRACE(2, ("%t: %s sw-deliver: tag=%0d, xfer=%0d, oob=%0d\n",
                    $time, INSTANCE_ID, sw_tag, sw_xfer, sw_oob))
            end
            if (smem_req_fire) begin
                `TRACE(2, ("%t: %s smem-wr-fire\n", $time, INSTANCE_ID))
            end
            if (xfer_done) begin
                `TRACE(1, ("%t: %s done: core=%0d, bar=%0d, xfer=%0d\n",
                    $time, INSTANCE_ID, trace_done_core_id,
                    trace_done_bar_addr, xfer_done_idx))
                $write("DXA_TL,%0d,XFER_DONE,core=%0d,bar=%0d\n",
                    $time, trace_done_core_id, trace_done_bar_addr);
            end
        end
    end
`endif

endmodule
