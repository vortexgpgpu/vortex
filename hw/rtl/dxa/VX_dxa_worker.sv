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

// DXA worker: orchestrates 5-stage pipeline:
//   addr_gen → dedup → rd_ctrl → cl2smem → wr_ctrl
// Stateless single-transaction executor: accept launch when idle, run to
// completion, signal done, return to idle.

`include "VX_define.vh"

module VX_dxa_worker import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WORKER_ID = 0
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_transfers,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_reads,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_dedup,
    output wire [PERF_CTR_BITS-1:0] perf_lmem_writes,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_lt,
`endif

    // Launch interface (from unified_engine dispatch)
    input wire                          launch_valid,
    output wire                         launch_ready,
    input wire [NC_WIDTH-1:0]           launch_core_id,
    input wire [UUID_WIDTH-1:0]         launch_uuid,
    input wire [NW_WIDTH-1:0]           launch_wid,
    input wire [BAR_ADDR_W-1:0]         launch_bar_addr,
    input wire [DXA_DESC_SLOT_W-1:0]    launch_desc_slot,
    input wire [`XLEN-1:0]              launch_smem_addr,
    input wire [4:0][`XLEN-1:0]         launch_coords,

    // Descriptor table read port (shared desc_table in unified_engine)
    output wire [DXA_DESC_SLOT_W-1:0] issue_desc_slot_out,
    input wire [`MEM_ADDR_WIDTH-1:0] issue_base_addr,
    input wire [31:0] issue_desc_meta,
    input wire [31:0] issue_desc_tile01,
    input wire [31:0] issue_desc_tile23,
    input wire [31:0] issue_desc_tile4,
    input wire [31:0] issue_desc_cfill,
    input wire [31:0] issue_size0_raw,
    input wire [31:0] issue_size1_raw,
    input wire [31:0] issue_stride0_raw,

`ifdef EXT_DXA_MULTICAST_ENABLE
    input wire                       launch_is_multicast,
    input wire [`NUM_WARPS-1:0]      launch_cta_mask,
    input wire [31:0]                issue_smem_stride,
    input wire [31:0]                issue_bar_stride,
`endif

    VX_mem_bus_if.master gmem_bus_if,
    VX_mem_bus_if.master smem_bus_if,
    output wire [NC_WIDTH-1:0] smem_core_id,

    output wire worker_idle
);

    `UNUSED_SPARAM (WORKER_ID)

    localparam GMEM_BYTES      = `L1_LINE_SIZE;
    localparam GMEM_DATAW      = GMEM_BYTES * 8;
    localparam GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES);
    localparam GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - GMEM_OFF_BITS;
    localparam GMEM_TAG_VALUEW = L1_MEM_ARB_TAG_WIDTH - `UP(UUID_WIDTH);

    localparam SMEM_BYTES      = DXA_LMEM_WORD_SIZE;
    localparam SMEM_DATAW      = SMEM_BYTES * 8;
    localparam SMEM_OFF_BITS   = `CLOG2(SMEM_BYTES);
    localparam SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_WIDTH;
    localparam BANK_ADDR_WIDTH = DXA_LMEM_BANK_ADDR_WIDTH;

`ifdef DXA_NB_MAX_OUTSTANDING
    localparam MAX_OUTSTANDING = `DXA_NB_MAX_OUTSTANDING;
`else
    localparam MAX_OUTSTANDING = 8;
`endif
`ifdef DXA_NB_WR_QUEUE_DEPTH
    localparam WR_QUEUE_DEPTH  = `DXA_NB_WR_QUEUE_DEPTH;
`else
    localparam WR_QUEUE_DEPTH  = 16;
`endif

    `UNUSED_SPARAM (INSTANCE_ID)

    // ---- Desc slot drives desc_table read port ----
    assign issue_desc_slot_out = launch_desc_slot;

    // ---- Issue decode (combinatorial from desc_table outputs) ----
    dxa_issue_dec_t issue_dec;

    VX_dxa_issue_decode #(
        .GMEM_BYTES(GMEM_BYTES),
        .SMEM_BYTES(SMEM_BYTES)
    ) issue_decode (
        .issue_desc_meta  (issue_desc_meta),
        .issue_desc_tile01(issue_desc_tile01),
        .issue_size0_raw  (issue_size0_raw),
        .issue_size1_raw  (issue_size1_raw),
        .issue_stride0_raw(issue_stride0_raw),
        .issue_dec        (issue_dec)
    );

    // ---- Transfer state ----
    // FSM: IDLE → DESC_WAIT → SETUP → ACTIVE → IDLE
    // DESC_WAIT absorbs 1-cycle BRAM read latency from desc_table.
    localparam TS_IDLE      = 3'd0;
    localparam TS_DESC_WAIT = 3'd1;
    localparam TS_SETUP     = 3'd2;
    localparam TS_ACTIVE    = 3'd3;

    reg [2:0]              ts_state_r;
    reg [NC_WIDTH-1:0]     active_core_id_r;
    reg [UUID_WIDTH-1:0]   active_uuid_r;
    reg [NW_WIDTH-1:0]     active_wid_r;
    reg [BAR_ADDR_W-1:0]   active_bar_addr_r;
    reg                    active_notify_smem_done_r;
    // Latched launch-interface signals (held stable across DESC_WAIT/SETUP).
    reg [`XLEN-1:0]        active_smem_addr_r;
    reg [4:0][`XLEN-1:0]   active_coords_r;
`ifdef EXT_DXA_MULTICAST_ENABLE
    reg                    active_is_multicast_r;
    reg [`NUM_WARPS-1:0]   active_cta_mask_r;
    reg [31:0]             active_smem_stride_r;
    reg [31:0]             active_bar_stride_r;
`endif

    wire active_r = (ts_state_r == TS_ACTIVE);

    // ---- Launch acceptance ----
    // CRITICAL: launch_supported MUST NOT be checked at the IDLE→DESC_WAIT
    // transition. issue_dec is computed combinationally from desc_table BRAM
    // outputs which have 1-cycle latency. When the unified engine pops the
    // FIFO every cycle (any worker idle), the BRAM input changes faster than
    // its output, so during IDLE the BRAM output reflects the PREVIOUS FIFO
    // entry's slot — not the current launch's slot. Checking issue_dec.* in
    // IDLE silently drops valid launches whose previous neighbor in the FIFO
    // had supported=0 stale data, causing cores to wait on a barrier whose
    // DXA completion never fires (manifests as a multi-core scheduler stall
    // — see VX_scheduler.sv timeout). The supported check is correctly
    // performed in TS_DESC_WAIT below, where BRAM rdata is now valid for
    // THIS worker's active slot (raddr was captured at the IDLE edge).
    wire launch_use_nb = ~issue_dec.is_s2g;
    wire launch_supported = launch_use_nb && issue_dec.supported && (issue_dec.total != 0);
    wire launch_valid_cmd = launch_valid;  // accept unconditionally; validate in DESC_WAIT
    wire launch_invalid_cmd = (ts_state_r == TS_DESC_WAIT) && ~launch_supported;

    assign launch_ready = (ts_state_r == TS_IDLE);

    // ---- Setup phase ----
    // setup_start fires in DESC_WAIT state, after BRAM output is valid.
    wire setup_start = (ts_state_r == TS_DESC_WAIT);
    wire setup_done;
    dxa_setup_params_t setup_params;

    VX_dxa_setup #(
        .SMEM_BYTES(SMEM_BYTES)
    ) setup (
        .clk         (clk),
        .reset       (reset),
        .start       (setup_start),
        .issue_dec   (issue_dec),
        .gmem_base   (issue_base_addr),
        .smem_base   (active_smem_addr_r),
        .coords      (active_coords_r),
        .cfill       (issue_desc_cfill),
        .setup_done  (setup_done),
        .setup_params(setup_params)
    );

    // ---- Pipeline start trigger ----
    wire pipeline_start = setup_done;

    // ════════════════════════════════════════════════════════════════════
    // Stage 1: Address Generator (replaces tile_iter)
    // ════════════════════════════════════════════════════════════════════

    wire ag_valid, ag_ready;
    wire [GMEM_ADDR_WIDTH-1:0] ag_cl_addr;
    wire [GMEM_BYTES-1:0] ag_byte_mask;
    wire ag_oob, ag_last;
    wire ag_new_row;
    wire [31:0] ag_cfill, ag_total_smem_writes;

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
        .out_byte_mask        (ag_byte_mask),
        .out_oob              (ag_oob),
        .out_last             (ag_last),
        .out_new_row          (ag_new_row),
        .out_cfill            (ag_cfill),
        .out_total_smem_writes(ag_total_smem_writes)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 2: Intra-Row Dedup
    // ════════════════════════════════════════════════════════════════════

    wire dd_valid, dd_ready;
    wire [GMEM_ADDR_WIDTH-1:0] dd_cl_addr;
    wire [GMEM_BYTES-1:0] dd_byte_mask;
    wire dd_oob, dd_last;

    VX_dxa_dedup #(
        .GMEM_LINE_SIZE  (GMEM_BYTES),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH)
    ) dedup (
        .clk           (clk),
        .reset         (reset),
        .in_valid      (ag_valid),
        .in_ready      (ag_ready),
        .in_cl_addr    (ag_cl_addr),
        .in_byte_mask  (ag_byte_mask),
        .in_oob        (ag_oob),
        .in_last       (ag_last),
        .in_new_row    (ag_new_row),
        .out_valid     (dd_valid),
        .out_ready     (dd_ready),
        .out_cl_addr   (dd_cl_addr),
        .out_byte_mask (dd_byte_mask),
        .out_oob       (dd_oob),
        .out_last      (dd_last)
    `ifdef PERF_ENABLE
        ,
        .perf_dedup_hit(dd_perf_dedup_hit)
    `endif
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 2.5: SMEM Address Tracker (per-CL SMEM byte address)
    // ════════════════════════════════════════════════════════════════════

    wire [`MEM_ADDR_WIDTH-1:0] dd_smem_byte_addr;
    VX_dxa_smem_addr_tracker #(
        .GMEM_BYTES  (GMEM_BYTES),
        .SMEM_ADDR_W (`MEM_ADDR_WIDTH)
    ) smem_addr_tracker (
        .clk              (clk),
        .reset            (reset),
        .start            (pipeline_start),
        .initial_smem_base(`MEM_ADDR_WIDTH'(setup_params.initial_smem_base)),
        .valid            (dd_valid),
        .ready            (dd_ready),
        .byte_mask        (dd_byte_mask),
        .smem_byte_addr   (dd_smem_byte_addr)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 3: Read Controller (GMEM reads, out-of-order emission)
    // ════════════════════════════════════════════════════════════════════

    wire rc_gmem_rd_req_valid;
    wire [GMEM_ADDR_WIDTH-1:0] rc_gmem_rd_req_addr;
    wire [GMEM_TAG_VALUEW-1:0] rc_gmem_rd_req_tag;
    wire rc_gmem_req_fire, rc_rsp_fire, rc_stall_no_slot;
    wire rc_all_cls_done;

    wire rc_cl_out_valid, rc_cl_out_ready;
    wire [GMEM_DATAW-1:0] rc_cl_out_data;
    wire [GMEM_BYTES-1:0] rc_cl_out_byte_mask;
    wire rc_cl_out_last;
    wire [`MEM_ADDR_WIDTH-1:0] rc_cl_out_smem_byte_addr;

    VX_dxa_rd_ctrl #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .GMEM_BYTES      (GMEM_BYTES),
        .GMEM_OFF_BITS   (GMEM_OFF_BITS),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_DATAW      (GMEM_DATAW),
        .GMEM_TAG_VALUEW (GMEM_TAG_VALUEW)
    ) rd_ctrl (
        .clk              (clk),
        .reset            (reset),
        .transfer_active  (active_r),
        .cl_in_valid      (dd_valid),
        .cl_in_ready      (dd_ready),
        .cl_in_addr       (dd_cl_addr),
        .cl_in_byte_mask  (dd_byte_mask),
        .cl_in_oob        (dd_oob),
        .cl_in_last       (dd_last),
        .cl_in_smem_byte_addr(dd_smem_byte_addr),
        .cfill            (ag_cfill),
        .gmem_rd_req_valid(rc_gmem_rd_req_valid),
        .gmem_rd_req_addr (rc_gmem_rd_req_addr),
        .gmem_rd_req_tag  (rc_gmem_rd_req_tag),
        .gmem_rd_req_ready(gmem_bus_if.req_ready),
        .gmem_rd_rsp_valid(gmem_bus_if.rsp_valid),
        .gmem_rd_rsp_data (gmem_bus_if.rsp_data.data),
        .gmem_rd_rsp_tag  (GMEM_TAG_VALUEW'(gmem_bus_if.rsp_data.tag.value)),
        .gmem_rd_rsp_ready(gmem_bus_if.rsp_ready),
        .cl_out_valid     (rc_cl_out_valid),
        .cl_out_ready     (rc_cl_out_ready),
        .cl_out_data      (rc_cl_out_data),
        .cl_out_byte_mask (rc_cl_out_byte_mask),
        .cl_out_last      (rc_cl_out_last),
        .cl_out_smem_byte_addr(rc_cl_out_smem_byte_addr),
        .gmem_req_fire    (rc_gmem_req_fire),
        .rsp_fire         (rc_rsp_fire),
        .stall_no_slot    (rc_stall_no_slot),
        .all_cls_done     (rc_all_cls_done)
    `ifdef PERF_ENABLE
        ,
        .perf_gmem_reqs       (rc_perf_gmem_reqs),
        .perf_gmem_span_cycles(rc_perf_gmem_span_cycles)
    `endif
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 4: CL-to-SMEM data format converter
    // ════════════════════════════════════════════════════════════════════

    wire cs_valid, cs_ready;
    wire [SMEM_DATAW-1:0] cs_data;
    wire [SMEM_BYTES-1:0] cs_byteen;
    wire cs_last;
    wire [SMEM_ADDR_WIDTH-1:0] cs_smem_word_addr;
    wire cs_idle;

    VX_dxa_cl2smem #(
        .CL_SIZE        (GMEM_BYTES),
        .SMEM_WORD_SIZE (SMEM_BYTES),
        .SMEM_ADDR_WIDTH(SMEM_ADDR_WIDTH)
    ) cl2smem (
        .clk                 (clk),
        .reset               (reset),
        .start               (pipeline_start),
        .cl_in_valid         (rc_cl_out_valid),
        .cl_in_ready         (rc_cl_out_ready),
        .cl_in_data          (rc_cl_out_data),
        .cl_in_byte_mask     (rc_cl_out_byte_mask),
        .cl_in_last          (rc_cl_out_last),
        .cl_in_smem_byte_addr(rc_cl_out_smem_byte_addr),
        .smem_out_valid      (cs_valid),
        .smem_out_ready      (cs_ready),
        .smem_out_data       (cs_data),
        .smem_out_byteen     (cs_byteen),
        .smem_out_last       (cs_last),
        .smem_out_word_addr  (cs_smem_word_addr),
        .idle                (cs_idle)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 5: Write Controller (SMEM write adapter)
    // ════════════════════════════════════════════════════════════════════

    wire wc_smem_wr_valid;
    wire [SMEM_ADDR_WIDTH-1:0] wc_smem_wr_addr;
    wire [SMEM_DATAW-1:0] wc_smem_wr_data;
    wire [SMEM_BYTES-1:0] wc_smem_wr_byteen;
    wire wc_smem_wr_last_pkt;
    wire wc_transfer_done;
    wire [31:0] wc_wr_done_count;
`ifdef EXT_DXA_MULTICAST_ENABLE
    wire wc_mc_cta_done;
    wire [31:0] wc_mc_cta_bar_offset;
    `UNUSED_VAR (wc_mc_cta_bar_offset)
`endif
    wire wc_smem_req_fire;
    wire wc_smem_wr_ready;

    VX_dxa_wr_ctrl #(
        .WR_QUEUE_DEPTH  (WR_QUEUE_DEPTH),
        .SMEM_BYTES      (SMEM_BYTES),
        .SMEM_DATAW      (SMEM_DATAW),
        .SMEM_OFF_BITS   (SMEM_OFF_BITS),
        .SMEM_ADDR_WIDTH (SMEM_ADDR_WIDTH)
    ) wr_ctrl (
        .clk               (clk),
        .reset             (reset),
        .transfer_active   (active_r),
        .transfer_start    (pipeline_start),
        .total_smem_writes (ag_total_smem_writes),
        .total_bytes       (setup_params.total_bytes),
        .initial_smem_base (`MEM_ADDR_WIDTH'(setup_params.initial_smem_base)),
        .all_cls_done      (rc_all_cls_done),
        .cl2smem_idle      (cs_idle),
        .smem_in_valid     (cs_valid),
        .smem_in_ready     (cs_ready),
        .smem_in_data      (cs_data),
        .smem_in_byteen    (cs_byteen),
        .smem_in_last      (cs_last),
        .smem_in_word_addr (cs_smem_word_addr),
        .smem_wr_valid     (wc_smem_wr_valid),
        .smem_wr_addr      (wc_smem_wr_addr),
        .smem_wr_data      (wc_smem_wr_data),
        .smem_wr_byteen    (wc_smem_wr_byteen),
        .smem_wr_ready     (wc_smem_wr_ready),
        .smem_wr_last_pkt  (wc_smem_wr_last_pkt),
        .transfer_done     (wc_transfer_done),
        .wr_done_count     (wc_wr_done_count),
        .smem_req_fire     (wc_smem_req_fire)
    `ifdef EXT_DXA_MULTICAST_ENABLE
        ,
        .is_multicast      (active_is_multicast_r),
        .cta_mask          (active_cta_mask_r),
        .smem_stride       (active_smem_stride_r),
        .bar_stride        (active_bar_stride_r),
        .mc_cta_done       (wc_mc_cta_done),
        .mc_cta_bar_offset (wc_mc_cta_bar_offset)
    `endif
    `ifdef PERF_ENABLE
        ,
        .perf_lmem_writes      (wc_perf_lmem_writes)
    `endif
    );

    assign wc_smem_wr_ready = smem_bus_if.req_ready;

    // ---- gmem bus wiring ----
    assign gmem_bus_if.req_valid     = rc_gmem_rd_req_valid;
    assign gmem_bus_if.req_data.rw   = 1'b0;
    assign gmem_bus_if.req_data.addr = rc_gmem_rd_req_addr;
    assign gmem_bus_if.req_data.data = '0;
    assign gmem_bus_if.req_data.byteen = {GMEM_BYTES{1'b1}};
    assign gmem_bus_if.req_data.flags  = '0;
    assign gmem_bus_if.req_data.tag.uuid = active_uuid_r;
    assign gmem_bus_if.req_data.tag.value = rc_gmem_rd_req_tag;

    // ---- smem write via VX_mem_bus_if ----
    assign smem_bus_if.req_valid       = wc_smem_wr_valid;
    assign smem_bus_if.req_data.rw     = 1'b1;
    assign smem_bus_if.req_data.addr   = BANK_ADDR_WIDTH'(wc_smem_wr_addr);
    assign smem_bus_if.req_data.data   = wc_smem_wr_data;
    assign smem_bus_if.req_data.byteen = wc_smem_wr_byteen;
    assign smem_bus_if.req_data.tag    = '0;
    assign smem_bus_if.rsp_ready       = 1'b0;

    // Completion flags: {last_pkt, bar_addr}.
`ifdef EXT_DXA_MULTICAST_ENABLE
    wire smem_wr_flags_last = active_notify_smem_done_r && (
        active_is_multicast_r ? wc_mc_cta_done : wc_smem_wr_last_pkt);
    wire [BAR_ADDR_W-1:0] smem_wr_flags_bar = active_is_multicast_r
        ? BAR_ADDR_W'(active_bar_addr_r + BAR_ADDR_W'(wc_mc_cta_bar_offset))
        : active_bar_addr_r;
    assign smem_bus_if.req_data.flags = {smem_wr_flags_last, smem_wr_flags_bar};
`elsif EXT_DXA_ENABLE
    wire smem_wr_flags_last = wc_smem_wr_last_pkt && active_notify_smem_done_r;
    assign smem_bus_if.req_data.flags = {smem_wr_flags_last, active_bar_addr_r};
`else
    assign smem_bus_if.req_data.flags = {wc_smem_wr_last_pkt, {BAR_ADDR_W{1'b0}}};
`endif

    // Core-id sideband for routing in DXA core router.
    assign smem_core_id = active_core_id_r;

    // ---- Transfer FSM (IDLE → DESC_WAIT → SETUP → ACTIVE → IDLE) ----

    always @(posedge clk) begin
        if (reset) begin
            ts_state_r              <= TS_IDLE;
            active_core_id_r        <= '0;
            active_uuid_r           <= '0;
            active_wid_r            <= '0;
            active_bar_addr_r       <= '0;
            active_notify_smem_done_r <= 1'b0;
        end else begin
            case (ts_state_r)
            TS_IDLE: begin
                if (launch_valid_cmd) begin
                    // Accept launch, issue desc_table BRAM read.
                    // Data will be valid next cycle (DESC_WAIT).
                    ts_state_r          <= TS_DESC_WAIT;
                    active_core_id_r    <= launch_core_id;
                    active_uuid_r       <= launch_uuid;
                    active_wid_r        <= launch_wid;
                    active_bar_addr_r   <= launch_bar_addr;
                `ifdef EXT_DXA_ENABLE
                    active_notify_smem_done_r <= 1'b1;
                `else
                    active_notify_smem_done_r <= 1'b0;
                `endif
                    // Latch launch-interface signals before they disappear.
                    active_smem_addr_r  <= launch_smem_addr;
                    active_coords_r     <= launch_coords;
                end
            end
            TS_DESC_WAIT: begin
                // BRAM output now valid for THIS worker's active slot.
                // Validate launch_supported here (deferred from IDLE — see
                // comment near launch_valid_cmd above). If unsupported, abort
                // back to IDLE without doing the transfer.
                if (~launch_supported) begin
                    ts_state_r <= TS_IDLE;
                end else begin
                `ifdef EXT_DXA_MULTICAST_ENABLE
                    active_is_multicast_r <= launch_is_multicast;
                    active_cta_mask_r     <= launch_cta_mask;
                    active_smem_stride_r  <= issue_smem_stride;
                    active_bar_stride_r   <= issue_bar_stride;
                `endif
                    ts_state_r <= TS_SETUP;
                end
            end
            TS_SETUP: begin
                if (setup_done) begin
                    ts_state_r <= TS_ACTIVE;
                end
            end
            TS_ACTIVE: begin
                if (wc_transfer_done) begin
                    ts_state_r <= TS_IDLE;
                end
            end
            default: ts_state_r <= TS_IDLE;
            endcase
        end
    end

    assign worker_idle = (ts_state_r == TS_IDLE);


    // ---- Progress watchdog ----
    reg [31:0] stall_ctr_r;
    wire active_no_progress = active_r
                           && ~rc_gmem_req_fire
                           && ~rc_rsp_fire
                           && ~wc_smem_req_fire;

    always @(posedge clk) begin
        if (reset || ~active_no_progress) begin
            stall_ctr_r <= '0;
        end else begin
            stall_ctr_r <= stall_ctr_r + 32'd1;
        end
    end

    `RUNTIME_ASSERT(stall_ctr_r < STALL_TIMEOUT, (
        "*** %s worker no-progress: core=%0d, wid=%0d, bar=%0d",
        INSTANCE_ID, active_core_id_r, active_wid_r, active_bar_addr_r))

`ifndef DBG_TRACE_DXA
    `UNUSED_VAR (wc_wr_done_count)
`endif
    `UNUSED_VAR (rc_stall_no_slot)
    `UNUSED_VAR (wc_smem_wr_addr[SMEM_ADDR_WIDTH-1:BANK_ADDR_WIDTH])
    `UNUSED_VAR (issue_desc_tile23)
    `UNUSED_VAR (issue_desc_tile4)
    `UNUSED_VAR (launch_desc_slot)
`ifndef DBG_TRACE_DXA
    `UNUSED_VAR (launch_invalid_cmd)
`endif
`ifndef EXT_DXA_ENABLE
    `UNUSED_VAR (active_bar_addr_r)
    `UNUSED_VAR (active_notify_smem_done_r)
`endif

`ifdef PERF_ENABLE
    // Wire declarations for submodule perf outputs
    wire [31:0] rc_perf_gmem_reqs;
    wire [31:0] rc_perf_gmem_span_cycles;
    wire [31:0] wc_perf_lmem_writes;
    wire        dd_perf_dedup_hit;
    // Accumulated DXA perf counters (never reset, sum across all transfers)
    reg [PERF_CTR_BITS-1:0] perf_transfers_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_reads_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_dedup_r;
    reg [PERF_CTR_BITS-1:0] perf_lmem_writes_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_lt_r;
    always @(posedge clk) begin
        if (reset) begin
            perf_transfers_r  <= '0;
            perf_gmem_reads_r <= '0;
            perf_gmem_dedup_r <= '0;
            perf_lmem_writes_r <= '0;
            perf_gmem_lt_r    <= '0;
        end else begin
            if (active_r && wc_transfer_done) begin
                perf_transfers_r  <= perf_transfers_r + PERF_CTR_BITS'(1);
                perf_gmem_reads_r <= perf_gmem_reads_r + PERF_CTR_BITS'(rc_perf_gmem_reqs);
                perf_lmem_writes_r <= perf_lmem_writes_r + PERF_CTR_BITS'(wc_perf_lmem_writes);
                perf_gmem_lt_r    <= perf_gmem_lt_r + PERF_CTR_BITS'(rc_perf_gmem_span_cycles);
            end
            if (dd_perf_dedup_hit) begin
                perf_gmem_dedup_r <= perf_gmem_dedup_r + PERF_CTR_BITS'(1);
            end
        end
    end
    assign perf_transfers  = perf_transfers_r;
    assign perf_gmem_reads = perf_gmem_reads_r;
    assign perf_gmem_dedup = perf_gmem_dedup_r;
    assign perf_lmem_writes = perf_lmem_writes_r;
    assign perf_gmem_lt    = perf_gmem_lt_r;
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (setup_start) begin
                `TRACE(1, ("%t: %s start: core=%0d, wid=%0d, bar=%0d, total=%0d, elem=%0d, gbase=0x%0h, smem=0x%0h, desc=%0d\n",
                    $time, INSTANCE_ID, launch_core_id, launch_wid, launch_bar_addr,
                    issue_dec.total, issue_dec.elem_bytes, issue_base_addr, launch_smem_addr, launch_desc_slot))
                $write("DXA_TL,%0d,XFER_START,core=%0d,wid=%0d,bar=%0d,total=%0d,elem=%0d\n",
                    $time, launch_core_id, launch_wid, launch_bar_addr,
                    issue_dec.total, issue_dec.elem_bytes);
            end
            if (launch_invalid_cmd) begin
                `TRACE(1, ("%t: %s launch-unsupported: core=%0d, wid=%0d, bar=%0d, supported=%0d, is_s2g=%0d, total=%0d\n",
                    $time, INSTANCE_ID, launch_core_id, launch_wid, launch_bar_addr,
                    issue_dec.supported, issue_dec.is_s2g, issue_dec.total))
            end
            if (pipeline_start) begin
                $write("DXA_TL,%0d,SETUP_DONE,core=%0d,wid=%0d,bar=%0d\n",
                    $time, active_core_id_r, active_wid_r, active_bar_addr_r);
            end
            if (active_r) begin
                if (rc_gmem_req_fire) begin
                    `TRACE(2, ("%t: %s gmem-req: addr=0x%0h\n",
                        $time, INSTANCE_ID, rc_gmem_rd_req_addr))
                end
                if (rc_rsp_fire) begin
                    `TRACE(2, ("%t: %s gmem-rsp: tag=%0d\n",
                        $time, INSTANCE_ID, gmem_bus_if.rsp_data.tag.value))
                end
                if (gmem_bus_if.rsp_valid && ~gmem_bus_if.rsp_ready) begin
                    `TRACE(2, ("%t: %s gmem-rsp-BLOCKED: tag=%0d\n",
                        $time, INSTANCE_ID, gmem_bus_if.rsp_data.tag.value))
                end
                if (wc_smem_req_fire) begin
                    `TRACE(2, ("%t: %s smem-wr: addr=0x%0h, count=%0d\n",
                        $time, INSTANCE_ID, wc_smem_wr_addr, wc_wr_done_count))
                end
            end
            if (active_r && wc_transfer_done) begin
                `TRACE(1, ("%t: %s done: core=%0d, wid=%0d, bar=%0d, wr_count=%0d\n",
                    $time, INSTANCE_ID, active_core_id_r, active_wid_r,
                    active_bar_addr_r, wc_wr_done_count))
                $write("DXA_TL,%0d,XFER_DONE,core=%0d,wid=%0d,bar=%0d,wr_count=%0d\n",
                    $time, active_core_id_r, active_wid_r, active_bar_addr_r,
                    wc_wr_done_count);
            end
        end
    end
`endif

endmodule
