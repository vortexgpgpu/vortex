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

// DXA worker: orchestrates AG→RRS→WBC pipeline for non-blocking transfers.
// Single-transaction service: accept ISSUE when idle, run to completion,
// signal done through SMEM-tag completion path, return to idle.

`include "VX_define.vh"

module VX_dxa_worker import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WORKER_ID = 0
) (
    input wire clk,
    input wire reset,

    VX_dxa_req_bus_if.slave dxa_req_bus_if,

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

    VX_mem_bus_if.master gmem_bus_if,
    VX_dxa_bank_wr_if.master smem_bank_wr_if,
    output wire [NC_WIDTH-1:0] smem_core_id,

    output wire worker_idle
);

    `UNUSED_SPARAM (WORKER_ID)

    localparam GMEM_BYTES      = `L1_LINE_SIZE;
    localparam GMEM_DATAW      = GMEM_BYTES * 8;
    localparam GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES);
    localparam GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - GMEM_OFF_BITS;
    localparam GMEM_TAG_VALUEW = L1_MEM_ARB_TAG_WIDTH - `UP(UUID_WIDTH);

    localparam SMEM_BYTES      = DXA_SMEM_WORD_SIZE;
    localparam SMEM_DATAW      = SMEM_BYTES * 8;
    localparam SMEM_OFF_BITS   = `CLOG2(SMEM_BYTES);
    localparam SMEM_ADDR_WIDTH = DXA_SMEM_ADDR_WIDTH;
    // Bank-native output params
    localparam NUM_BANKS       = `LMEM_NUM_BANKS;
    localparam BANK_WORD_SIZE  = `XLEN / 8;
    localparam BANK_WORD_WIDTH = BANK_WORD_SIZE * 8;
    localparam BANK_ADDR_WIDTH = DXA_SMEM_BANK_ADDR_WIDTH;

    localparam DXA_CTX_COUNT   = `NUM_CORES * `NUM_WARPS;
    localparam DXA_CTX_BITS    = `UP(`CLOG2(DXA_CTX_COUNT));

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

    // ---- Request buffering and decode ----
    wire req_valid;
    wire [DXA_REQ_DATAW-1:0] req_data_raw;
    wire req_accept;
    wire req_fire;
    wire [NC_WIDTH-1:0] req_core_id;
    wire [UUID_WIDTH-1:0] req_uuid;
    wire [NW_WIDTH-1:0] req_wid;
    wire [2:0] req_op;
    wire [`XLEN-1:0] req_rs1;
    wire [`XLEN-1:0] req_rs2;

    VX_elastic_buffer #(
        .DATAW (DXA_REQ_DATAW),
        .SIZE  (32)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (dxa_req_bus_if.req_valid),
        .ready_in  (dxa_req_bus_if.req_ready),
        .data_in   (dxa_req_bus_if.req_data),
        .valid_out (req_valid),
        .ready_out (req_accept),
        .data_out  (req_data_raw)
    );

    assign {req_core_id, req_uuid, req_wid, req_op, req_rs1, req_rs2} = req_data_raw;

    // ---- Context table (issue state) ----
    wire [DXA_CTX_BITS-1:0] req_core_ofs = DXA_CTX_BITS'(req_core_id) * DXA_CTX_BITS'(`NUM_WARPS);
    wire [DXA_CTX_BITS-1:0] req_ctx_idx = req_core_ofs + DXA_CTX_BITS'(req_wid);

    wire setup0_is_packed = req_rs2[31];
    wire [BAR_ADDR_W-1:0] req_bar_addr;
    if (`NUM_WARPS > 1) begin : g_bar_addr_w
        assign req_bar_addr = setup0_is_packed
                            ? {req_rs2[4 +: NW_BITS], req_rs2[20 +: NB_BITS]}
                            : {req_rs2[NW_BITS-1:0], req_rs2[16 +: NB_BITS]};
    end else begin : g_bar_addr_wo
        assign req_bar_addr = setup0_is_packed
                            ? req_rs2[20 +: NB_BITS]
                            : req_rs2[16 +: NB_BITS];
    end

    wire [BAR_ADDR_W-1:0] issue_bar_addr;
    wire [DXA_DESC_SLOT_W-1:0] issue_desc_slot;
    wire [`XLEN-1:0] issue_smem_addr;
    wire [4:0][`XLEN-1:0] issue_coords;
    wire issue_ctx_valid;

    VX_dxa_ctx_table #(
        .DXA_CTX_COUNT     (DXA_CTX_COUNT),
        .DXA_CTX_BITS      (DXA_CTX_BITS)
    ) ctx_table (
        .clk            (clk),
        .reset          (reset),
        .req_fire       (req_fire),
        .req_op         (req_op),
        .req_ctx_idx    (req_ctx_idx),
        .req_rs1        (req_rs1),
        .req_rs2        (req_rs2),
        .req_bar_addr   (req_bar_addr),
        .issue_bar_addr (issue_bar_addr),
        .issue_desc_slot(issue_desc_slot),
        .issue_smem_addr(issue_smem_addr),
        .issue_coords   (issue_coords),
        .issue_ctx_valid(issue_ctx_valid)
    );

    assign issue_desc_slot_out = issue_desc_slot;

    // ---- Issue decode ----
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
    reg  active_r;
    reg  [NC_WIDTH-1:0]    active_core_id_r;
    reg  [UUID_WIDTH-1:0]  active_uuid_r;
    reg  [NW_WIDTH-1:0]    active_wid_r;
    reg  [BAR_ADDR_W-1:0]  active_bar_addr_r;
    reg                    active_notify_smem_done_r;
    reg  [31:0]            active_total_r;
    reg  [31:0]            active_cfill_r;
    reg  [31:0]            active_elem_bytes_r;

    // ---- Request acceptance ----
    wire req_is_issue = (req_op == DXA_OP_ISSUE);
    wire req_use_nb = ~issue_dec.is_s2g;  // NB pipeline handles g2s only (for now)
    wire req_issue_ctx_valid = req_is_issue && issue_ctx_valid;
    wire req_issue_supported = req_use_nb && issue_dec.supported && (issue_dec.total != 0);
    wire req_valid_cmd = req_valid && req_issue_ctx_valid && req_issue_supported;
    wire req_invalid_cmd = req_valid && req_issue_ctx_valid && ~req_issue_supported;
    wire xfer_done_fire;

    // Accept when: not an issue, or issue context is valid and worker is idle.
    assign req_accept = ~req_valid
                     || ~req_is_issue
                     || (issue_ctx_valid && ~active_r);
    assign req_fire = req_valid && req_accept;

    // ---- AG instance ----
    wire ag_start;
    wire ag_valid_w, ag_ready_w;
    wire [`MEM_ADDR_WIDTH-1:0] ag_gmem_byte_addr_w;
    wire [GMEM_ADDR_WIDTH-1:0] ag_gmem_line_addr_w;
    wire [GMEM_OFF_BITS-1:0] ag_gmem_off_w;
    wire [`MEM_ADDR_WIDTH-1:0] ag_smem_byte_addr_w;
    wire ag_in_bounds_w, ag_is_last_w;
    wire [31:0] ag_elem_idx_w;
    wire ag_busy_w, ag_done_w;

    // AG latches inputs on ag_start; issue_* signals are valid at that time.
    VX_dxa_ag #(
        .GMEM_BYTES     (GMEM_BYTES),
        .GMEM_OFF_BITS  (GMEM_OFF_BITS),
        .GMEM_ADDR_WIDTH(GMEM_ADDR_WIDTH)
    ) ag (
        .clk             (clk),
        .reset           (reset),
        .start           (ag_start),
        .issue_dec       (issue_dec),
        .gmem_base       (issue_base_addr),
        .smem_base       (issue_smem_addr),
        .coords          (issue_coords),
        .cfill           (issue_desc_cfill),
        .ag_valid        (ag_valid_w),
        .ag_ready        (ag_ready_w),
        .ag_gmem_byte_addr(ag_gmem_byte_addr_w),
        .ag_gmem_line_addr(ag_gmem_line_addr_w),
        .ag_gmem_off     (ag_gmem_off_w),
        .ag_smem_byte_addr(ag_smem_byte_addr_w),
        .ag_in_bounds    (ag_in_bounds_w),
        .ag_is_last      (ag_is_last_w),
        .ag_elem_idx     (ag_elem_idx_w),
        .ag_busy         (ag_busy_w),
        .ag_done         (ag_done_w)
    );

    // ---- RRS instance ----
    wire rrs_wb_valid, rrs_wb_ready;
    wire [`MEM_ADDR_WIDTH-1:0] rrs_wb_smem_byte_addr;
    wire [SMEM_DATAW-1:0] rrs_wb_smem_data;
    wire [SMEM_BYTES-1:0] rrs_wb_smem_byteen;
    wire rrs_wb_is_last;
    wire rrs_gmem_req_fire, rrs_rsp_fire;
    wire rrs_stall_inflight;
    wire rrs_stall_no_slot;
    wire rrs_stall_rsp_backpressure;

    wire rrs_gmem_rd_req_valid;
    wire [GMEM_ADDR_WIDTH-1:0] rrs_gmem_rd_req_addr;
    wire [GMEM_TAG_VALUEW-1:0] rrs_gmem_rd_req_tag;

    VX_dxa_rrs #(
        .MAX_OUTSTANDING(MAX_OUTSTANDING),
        .GMEM_BYTES     (GMEM_BYTES),
        .GMEM_OFF_BITS  (GMEM_OFF_BITS),
        .GMEM_ADDR_WIDTH(GMEM_ADDR_WIDTH),
        .GMEM_DATAW     (GMEM_DATAW),
        .GMEM_TAG_VALUEW(GMEM_TAG_VALUEW),
        .SMEM_BYTES     (SMEM_BYTES),
        .SMEM_DATAW     (SMEM_DATAW),
        .SMEM_OFF_BITS  (SMEM_OFF_BITS),
        .WR_QUEUE_DEPTH (WR_QUEUE_DEPTH)
    ) rrs (
        .clk              (clk),
        .reset            (reset),
        .transfer_active  (active_r),
        .ag_valid         (ag_valid_w),
        .ag_ready         (ag_ready_w),
        .ag_gmem_byte_addr(ag_gmem_byte_addr_w),
        .ag_gmem_line_addr(ag_gmem_line_addr_w),
        .ag_gmem_off      (ag_gmem_off_w),
        .ag_smem_byte_addr(ag_smem_byte_addr_w),
        .ag_in_bounds     (ag_in_bounds_w),
        .ag_is_last       (ag_is_last_w),
        .ag_elem_idx      (ag_elem_idx_w),
        .cfill            (active_cfill_r),
        .elem_bytes       (active_elem_bytes_r),
        .gmem_rd_req_valid(rrs_gmem_rd_req_valid),
        .gmem_rd_req_addr (rrs_gmem_rd_req_addr),
        .gmem_rd_req_tag  (rrs_gmem_rd_req_tag),
        .gmem_rd_req_ready(gmem_bus_if.req_ready),
        .gmem_rd_rsp_valid(gmem_bus_if.rsp_valid),
        .gmem_rd_rsp_data (gmem_bus_if.rsp_data.data),
        .gmem_rd_rsp_tag  (GMEM_TAG_VALUEW'(gmem_bus_if.rsp_data.tag.value)),
        .gmem_rd_rsp_ready(gmem_bus_if.rsp_ready),
        .wb_valid         (rrs_wb_valid),
        .wb_smem_byte_addr(rrs_wb_smem_byte_addr),
        .wb_smem_data     (rrs_wb_smem_data),
        .wb_smem_byteen   (rrs_wb_smem_byteen),
        .wb_is_last       (rrs_wb_is_last),
        .wb_ready         (rrs_wb_ready),
        .gmem_req_fire    (rrs_gmem_req_fire),
        .rsp_fire         (rrs_rsp_fire),
        .stall_inflight   (rrs_stall_inflight),
        .stall_no_slot    (rrs_stall_no_slot),
        .stall_rsp_backpressure(rrs_stall_rsp_backpressure)
    );

    // ---- WBC instance ----
    wire wbc_smem_wr_valid;
    wire [SMEM_ADDR_WIDTH-1:0] wbc_smem_wr_addr;
    wire [SMEM_DATAW-1:0] wbc_smem_wr_data;
    wire [SMEM_BYTES-1:0] wbc_smem_wr_byteen;
    wire wbc_smem_wr_last_pkt;
    wire wbc_transfer_done;
    wire [31:0] wbc_wr_done_count;
    wire wbc_smem_req_fire;
    wire wbc_smem_wr_ready;

    VX_dxa_wbc #(
        .WR_QUEUE_DEPTH (WR_QUEUE_DEPTH),
        .SMEM_BYTES     (SMEM_BYTES),
        .SMEM_DATAW     (SMEM_DATAW),
        .SMEM_OFF_BITS  (SMEM_OFF_BITS),
        .SMEM_ADDR_WIDTH(SMEM_ADDR_WIDTH)
    ) wbc (
        .clk              (clk),
        .reset            (reset),
        .transfer_active  (active_r),
        .transfer_start   (ag_start),
        .total_elements   (active_total_r),
        .wb_valid         (rrs_wb_valid),
        .wb_ready         (rrs_wb_ready),
        .wb_smem_byte_addr(rrs_wb_smem_byte_addr),
        .wb_smem_data     (rrs_wb_smem_data),
        .wb_smem_byteen   (rrs_wb_smem_byteen),
        .wb_is_last       (rrs_wb_is_last),
        .smem_wr_req_valid(wbc_smem_wr_valid),
        .smem_wr_req_addr (wbc_smem_wr_addr),
        .smem_wr_req_data (wbc_smem_wr_data),
        .smem_wr_req_byteen(wbc_smem_wr_byteen),
        .smem_wr_req_ready(wbc_smem_wr_ready),
        .smem_wr_last_pkt (wbc_smem_wr_last_pkt),
        .transfer_done    (wbc_transfer_done),
        .wr_done_count    (wbc_wr_done_count),
        .smem_req_fire    (wbc_smem_req_fire)
    );

    assign wbc_smem_wr_ready = smem_bank_wr_if.wr_ready;

    // ---- gmem bus wiring ----
    assign gmem_bus_if.req_valid   = rrs_gmem_rd_req_valid;
    assign gmem_bus_if.req_data.rw = 1'b0;
    assign gmem_bus_if.req_data.addr = rrs_gmem_rd_req_addr;
    assign gmem_bus_if.req_data.data = '0;
    assign gmem_bus_if.req_data.byteen = {GMEM_BYTES{1'b1}};
    assign gmem_bus_if.req_data.flags  = '0;
    assign gmem_bus_if.req_data.tag.uuid = active_uuid_r;
    assign gmem_bus_if.req_data.tag.value = rrs_gmem_rd_req_tag;

    // ---- smem bank-native write (write-only for g2s) ----
    // DXA_SMEM_WORD_SIZE == NUM_BANKS * BANK_WORD_SIZE by construction.
    // Each DXA word spans exactly one row across all banks: sub-word b → bank b.
    // Bank address is the same for all banks (the DXA word address low bits).
    wire [BANK_ADDR_WIDTH-1:0] smem_bank_addr = BANK_ADDR_WIDTH'(wbc_smem_wr_addr);

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_bank_wr
        assign smem_bank_wr_if.wr_valid[b]  = wbc_smem_wr_valid
            && (|wbc_smem_wr_byteen[b * BANK_WORD_SIZE +: BANK_WORD_SIZE]);
        assign smem_bank_wr_if.wr_addr[b]   = smem_bank_addr;
        assign smem_bank_wr_if.wr_data[b]   = wbc_smem_wr_data[b * BANK_WORD_WIDTH +: BANK_WORD_WIDTH];
        assign smem_bank_wr_if.wr_byteen[b] = wbc_smem_wr_byteen[b * BANK_WORD_SIZE +: BANK_WORD_SIZE];
    end

    // Completion tag: {last_pkt, bar_addr}. Suppress last_pkt when completion
    // notification is not requested (e.g. gmem completion path instead).
`ifdef EXT_DXA_ENABLE
    wire smem_wr_tag_last = wbc_smem_wr_last_pkt && active_notify_smem_done_r;
    assign smem_bank_wr_if.wr_tag = {smem_wr_tag_last, active_bar_addr_r};
`else
    assign smem_bank_wr_if.wr_tag = {wbc_smem_wr_last_pkt, {BAR_ADDR_W{1'b0}}};
`endif

    // Core-id sideband for routing in DXA core router.
    assign smem_core_id = active_core_id_r;

    // ---- Transfer FSM (IDLE → ACTIVE → IDLE) ----
    assign ag_start = req_valid_cmd && ~active_r;
    assign xfer_done_fire = active_r && wbc_transfer_done;

    always @(posedge clk) begin
        if (reset) begin
            active_r            <= 1'b0;
            active_core_id_r    <= '0;
            active_uuid_r       <= '0;
            active_wid_r        <= '0;
            active_bar_addr_r   <= '0;
            active_notify_smem_done_r <= 1'b0;
            active_total_r      <= '0;
            active_cfill_r      <= '0;
            active_elem_bytes_r <= '0;
        end else begin
            // Start new transfer when idle and valid command arrives.
            if (ag_start) begin
                active_r            <= 1'b1;
                active_core_id_r    <= req_core_id;
                active_uuid_r       <= req_uuid;
                active_wid_r        <= req_wid;
                active_bar_addr_r   <= issue_bar_addr;
            `ifdef EXT_DXA_ENABLE
                active_notify_smem_done_r <= DXA_DONE_META_ENABLE;
            `else
                active_notify_smem_done_r <= 1'b0;
            `endif
                active_total_r      <= issue_dec.total;
                active_cfill_r      <= issue_desc_cfill;
                active_elem_bytes_r <= issue_dec.elem_bytes;
            end
            // Transfer completion.
            if (xfer_done_fire) begin
                active_r <= 1'b0;
            end
        end
    end

    assign worker_idle = ~active_r;

    // ---- Profiling counters (reason-coded stalls) ----
    reg [31:0] prof_active_cycles_r;
    reg [31:0] prof_issue_block_cycles_r;
    reg [31:0] prof_ag_block_cycles_r;
    reg [31:0] prof_gmem_req_block_cycles_r;
    reg [31:0] prof_gmem_rsp_block_cycles_r;
    reg [31:0] prof_smem_req_block_cycles_r;
    reg [31:0] prof_inflight_block_cycles_r;
    reg [31:0] prof_no_slot_block_cycles_r;
    reg [31:0] prof_rsp_backpressure_cycles_r;
    reg [31:0] prof_gmem_req_fire_r;
    reg [31:0] prof_gmem_rsp_fire_r;
    reg [31:0] prof_smem_req_fire_r;

    always @(posedge clk) begin
        if (reset || ag_start) begin
            prof_active_cycles_r <= '0;
            prof_issue_block_cycles_r <= '0;
            prof_ag_block_cycles_r <= '0;
            prof_gmem_req_block_cycles_r <= '0;
            prof_gmem_rsp_block_cycles_r <= '0;
            prof_smem_req_block_cycles_r <= '0;
            prof_inflight_block_cycles_r <= '0;
            prof_no_slot_block_cycles_r <= '0;
            prof_rsp_backpressure_cycles_r <= '0;
            prof_gmem_req_fire_r <= '0;
            prof_gmem_rsp_fire_r <= '0;
            prof_smem_req_fire_r <= '0;
        end else begin
            if (active_r) begin
                prof_active_cycles_r <= prof_active_cycles_r + 32'd1;
            end
            if (req_valid && req_is_issue && ~req_accept) begin
                prof_issue_block_cycles_r <= prof_issue_block_cycles_r + 32'd1;
            end
            if (ag_valid_w && ~ag_ready_w) begin
                prof_ag_block_cycles_r <= prof_ag_block_cycles_r + 32'd1;
            end
            if (rrs_gmem_rd_req_valid && ~gmem_bus_if.req_ready) begin
                prof_gmem_req_block_cycles_r <= prof_gmem_req_block_cycles_r + 32'd1;
            end
            if (gmem_bus_if.rsp_valid && ~gmem_bus_if.rsp_ready) begin
                prof_gmem_rsp_block_cycles_r <= prof_gmem_rsp_block_cycles_r + 32'd1;
            end
            if (wbc_smem_wr_valid && ~smem_bank_wr_if.wr_ready) begin
                prof_smem_req_block_cycles_r <= prof_smem_req_block_cycles_r + 32'd1;
            end
            if (rrs_stall_inflight) begin
                prof_inflight_block_cycles_r <= prof_inflight_block_cycles_r + 32'd1;
            end
            if (rrs_stall_no_slot) begin
                prof_no_slot_block_cycles_r <= prof_no_slot_block_cycles_r + 32'd1;
            end
            if (rrs_stall_rsp_backpressure) begin
                prof_rsp_backpressure_cycles_r <= prof_rsp_backpressure_cycles_r + 32'd1;
            end
            if (rrs_gmem_req_fire) begin
                prof_gmem_req_fire_r <= prof_gmem_req_fire_r + 32'd1;
            end
            if (rrs_rsp_fire) begin
                prof_gmem_rsp_fire_r <= prof_gmem_rsp_fire_r + 32'd1;
            end
            if (wbc_smem_req_fire) begin
                prof_smem_req_fire_r <= prof_smem_req_fire_r + 32'd1;
            end
        end
    end

    // ---- Progress watchdog ----
    reg [31:0] stall_ctr_r;
    wire active_no_progress = active_r
                           && ~rrs_gmem_req_fire
                           && ~rrs_rsp_fire
                           && ~wbc_smem_req_fire;

    always @(posedge clk) begin
        if (reset || ~active_no_progress) begin
            stall_ctr_r <= '0;
        end else begin
            stall_ctr_r <= stall_ctr_r + 32'd1;
        end
    end

    `RUNTIME_ASSERT(stall_ctr_r < STALL_TIMEOUT, (
        "*** %s worker no-progress: core=%0d wid=%0d bar=%0d",
        INSTANCE_ID, active_core_id_r, active_wid_r, active_bar_addr_r))

    `UNUSED_VAR (ag_busy_w)
    `UNUSED_VAR (ag_done_w)
    `UNUSED_VAR (wbc_wr_done_count)
    `UNUSED_VAR (wbc_smem_wr_addr[SMEM_ADDR_WIDTH-1:BANK_ADDR_WIDTH])
    `UNUSED_VAR (issue_desc_tile23)
    `UNUSED_VAR (issue_desc_tile4)
    `UNUSED_VAR (issue_desc_slot)
`ifndef DBG_TRACE_DXA
    `UNUSED_VAR (req_invalid_cmd)
`endif
`ifndef EXT_DXA_ENABLE
    `UNUSED_VAR (active_bar_addr_r)
    `UNUSED_VAR (active_notify_smem_done_r)
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (req_fire && req_is_issue) begin
                `TRACE(1, ("%t: %s issue-fire: core=%0d wid=%0d bar=%0d ctx_ok=%0d supported=%0d valid_cmd=%0d invalid_cmd=%0d active=%0d total=%0d is_s2g=%0d\n",
                    $time, INSTANCE_ID, req_core_id, req_wid, issue_bar_addr, issue_ctx_valid, req_issue_supported,
                    req_valid_cmd, req_invalid_cmd, active_r, issue_dec.total, issue_dec.is_s2g))
            end
            if (req_valid && req_is_issue && ~req_accept) begin
                `TRACE(1, ("%t: %s issue-stall: core=%0d wid=%0d bar=%0d ctx_ok=%0d supported=%0d active=%0d\n",
                    $time, INSTANCE_ID, req_core_id, req_wid, issue_bar_addr, issue_ctx_valid, req_issue_supported,
                    active_r))
            end
            if (ag_start) begin
                `TRACE(1, ("%t: %s start: core=%0d wid=%0d bar=%0d total=%0d elem=%0d\n",
                    $time, INSTANCE_ID, req_core_id, req_wid, issue_bar_addr,
                    issue_dec.total, issue_dec.elem_bytes))
            end
            if (active_r && wbc_transfer_done) begin
                `TRACE(1, ("%t: %s done: core=%0d wid=%0d bar=%0d wr_count=%0d\n",
                    $time, INSTANCE_ID, active_core_id_r, active_wid_r,
                    active_bar_addr_r, wbc_wr_done_count))
            end
        end
    end
`endif

`ifdef DBG_TRACE_DXA_PROFILE
    always @(posedge clk) begin
        if (~reset && xfer_done_fire) begin
            $display("%t: %s DXA_PROFILE core=%0d wid=%0d bar=%0d active=%0d issue_blk=%0d ag_blk=%0d gmem_req_blk=%0d gmem_rsp_blk=%0d smem_req_blk=%0d inflight_blk=%0d noslot_blk=%0d rsp_bp=%0d req_fire=%0d rsp_fire=%0d smem_fire=%0d",
                $time, INSTANCE_ID, active_core_id_r, active_wid_r, active_bar_addr_r,
                prof_active_cycles_r, prof_issue_block_cycles_r, prof_ag_block_cycles_r,
                prof_gmem_req_block_cycles_r, prof_gmem_rsp_block_cycles_r, prof_smem_req_block_cycles_r,
                prof_inflight_block_cycles_r, prof_no_slot_block_cycles_r, prof_rsp_backpressure_cycles_r,
                prof_gmem_req_fire_r, prof_gmem_rsp_fire_r, prof_smem_req_fire_r);
        end
    end
`endif

endmodule
