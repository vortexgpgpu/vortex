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

// DXA SMEM Writer: synchronizes in-flight FIFO with BRAM responses,
// barrel-shifts CL data, and drains SMEM words to the bus.
// Drives VX_mem_bus_if directly (absorbs bus wiring + completion flags).
//
// Pipeline: IDLE → FETCH (1 cycle BRAM read) → DRAIN (word-by-word SMEM writes)
// OOB entries use cfill-replicated data (no BRAM read needed but still 1 cycle).

`include "VX_define.vh"

module VX_dxa_smem_wr import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter CL_SIZE         = `L1_LINE_SIZE,
    parameter SMEM_WORD_SIZE  = DXA_LMEM_WORD_SIZE,
    parameter SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W,
    parameter GMEM_DATAW      = CL_SIZE * 8
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,
    input  wire                        transfer_start,

    // cfill (stable during transfer).
    input  wire [31:0]                 cfill,

    // Metadata for SMEM bus tag + completion flags.
    input  wire [NC_WIDTH-1:0]         active_core_id,
    input  wire [BAR_ADDR_W-1:0]       active_bar_addr,
    input  wire                        active_notify_smem_done,

    // In-flight FIFO interface (from gmem_req).
    input  wire                        fifo_valid,
    output wire                        fifo_pop,
    input  wire [TAG_W-1:0]            fifo_tag,
    input  wire [`MEM_ADDR_WIDTH-1:0]  fifo_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      fifo_byte_offset,
    input  wire [CL_OFF_BITS:0]        fifo_valid_length,
    input  wire                        fifo_oob,
    input  wire                        fifo_last,

    // Response buffer interface.
    input  wire [MAX_OUTSTANDING-1:0]  rsp_arrived,
    output wire                        rsp_read_en,
    output wire [TAG_W-1:0]            rsp_read_tag,
    input  wire [GMEM_DATAW-1:0]       rsp_read_data,
    output wire                        rsp_clear_en,
    output wire [TAG_W-1:0]            rsp_clear_tag,

    // Resource release to gmem_req.
    output wire                        release_en,
    output wire [TAG_W-1:0]            release_tag,

    // SMEM bus interface (writes only).
    VX_mem_bus_if.master               smem_bus_if,

    // Completion.
    output wire                        transfer_done,
    output wire [31:0]                 wr_done_count,
    output wire                        smem_req_fire,

    // Multicast (always available; active when is_multicast is set).
    input  wire                        is_multicast,
    input  wire [`NUM_WARPS-1:0]       cta_mask,
    input  wire [31:0]                 smem_stride

`ifdef PERF_ENABLE
    ,
    output wire [31:0]                 perf_lmem_writes
`endif
);
    localparam CL_OFF_BITS  = `CLOG2(CL_SIZE);
    localparam SMEM_OFF_W   = `CLOG2(SMEM_WORD_SIZE);
    localparam SMEM_DATAW   = SMEM_WORD_SIZE * 8;
    localparam TAG_W        = `CLOG2(MAX_OUTSTANDING);
    localparam FILL_CAP     = CL_SIZE + SMEM_WORD_SIZE;
    localparam FILL_W       = `CLOG2(FILL_CAP + 1);

    localparam ENGINE_VALUE_W = LMEM_DXA_ENGINE_TAG_W - UUID_WIDTH;
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;

    // ════════════════════════════════════════════════════════════════════
    // FSM: IDLE → FETCH → DRAIN
    // ════════════════════════════════════════════════════════════════════
    localparam S_IDLE  = 2'd0;
    localparam S_FETCH = 2'd1;
    localparam S_DRAIN = 2'd2;

    reg [1:0] state_r;

    // ════════════════════════════════════════════════════════════════════
    // FIFO head synchronization
    // ════════════════════════════════════════════════════════════════════
    wire head_ready = fifo_valid && (fifo_oob || rsp_arrived[fifo_tag]);

    // ════════════════════════════════════════════════════════════════════
    // Cfill replication
    // ════════════════════════════════════════════════════════════════════
    wire [GMEM_DATAW-1:0] cfill_replicated;
    for (genvar i = 0; i < CL_SIZE / 4; ++i) begin : g_cfill
        assign cfill_replicated[i*32 +: 32] = cfill;
    end

    // ════════════════════════════════════════════════════════════════════
    // Latched CL metadata (captured on FETCH entry)
    // ════════════════════════════════════════════════════════════════════
    reg [TAG_W-1:0]            lat_tag_r;
    reg [`MEM_ADDR_WIDTH-1:0]  lat_smem_byte_addr_r;
    reg [CL_OFF_BITS-1:0]      lat_byte_offset_r;
    reg [CL_OFF_BITS:0]        lat_valid_length_r;
    reg                        lat_oob_r;
    reg                        lat_last_r;

    // ════════════════════════════════════════════════════════════════════
    // Fill buffer (per-CL drain)
    // ════════════════════════════════════════════════════════════════════
    reg [FILL_CAP*8-1:0]       fb_data_r /*verilator split_var*/;
    reg [FILL_W-1:0]           fb_level_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_word_addr_r;
    reg [SMEM_OFF_W-1:0]       fb_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_start_word_r;

    // Fill buffer output (register-driven).
    wire has_full_word    = (fb_level_r >= FILL_W'(SMEM_WORD_SIZE));
    wire has_last_partial = !has_full_word && (fb_level_r > 0);
    wire drain_valid      = (state_r == S_DRAIN) && (has_full_word || has_last_partial);
    wire drain_will_empty = (fb_level_r <= FILL_W'(SMEM_WORD_SIZE));

    wire drain_fire = drain_valid && smem_wr_ready_internal;

    // SMEM word data and byteen from fill buffer.
    wire [SMEM_DATAW-1:0] fb_word_data = fb_data_r[SMEM_DATAW-1:0];
    wire is_first_word = (fb_word_addr_r == fb_start_word_r);

    wire [SMEM_WORD_SIZE-1:0] fb_word_byteen;
    for (genvar i = 0; i < SMEM_WORD_SIZE; ++i) begin : g_byteen
        wire byte_has_data = (FILL_W'(i) < fb_level_r);
        if (i < SMEM_WORD_SIZE - 1) begin : g_with_offset
            wire byte_is_offset = is_first_word && (SMEM_OFF_W'(i) < fb_byte_offset_r);
            assign fb_word_byteen[i] = byte_has_data && !byte_is_offset;
        end else begin : g_no_offset
            assign fb_word_byteen[i] = byte_has_data;
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Barrel shift: compress valid CL bytes to position 0
    // ════════════════════════════════════════════════════════════════════

    wire [GMEM_DATAW-1:0] fetch_cl_data = lat_oob_r ? cfill_replicated : rsp_read_data;
    wire [GMEM_DATAW-1:0] compressed_data = (lat_valid_length_r != 0)
        ? (fetch_cl_data >> {lat_byte_offset_r, 3'b000})
        : '0;

    // SMEM byte decomposition for fill buffer loading.
    wire [SMEM_OFF_W-1:0]      new_smem_byte_off = lat_smem_byte_addr_r[SMEM_OFF_W-1:0];
    wire [SMEM_ADDR_WIDTH-1:0] new_start_word    = SMEM_ADDR_WIDTH'(lat_smem_byte_addr_r >> SMEM_OFF_W);
    wire [FILL_W-1:0]          new_fill_level    = FILL_W'(new_smem_byte_off) + FILL_W'(lat_valid_length_r);
    wire [FILL_W+2:0]          new_bit_offset    = {FILL_W'(new_smem_byte_off), 3'b000};

    // ════════════════════════════════════════════════════════════════════
    // FSM + fill buffer state update
    // ════════════════════════════════════════════════════════════════════

    wire do_fetch = (state_r == S_IDLE) && head_ready;
    assign fifo_pop = do_fetch;

    assign rsp_read_en  = do_fetch && ~fifo_oob;
    assign rsp_read_tag = fifo_tag;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            state_r          <= S_IDLE;
            fb_data_r        <= '0;
            fb_level_r       <= '0;
            fb_word_addr_r   <= '0;
            fb_byte_offset_r <= '0;
            fb_start_word_r  <= '0;
            lat_tag_r        <= '0;
            lat_oob_r        <= 1'b0;
            lat_last_r       <= 1'b0;
        end else begin
            case (state_r)
            S_IDLE: begin
                if (head_ready) begin
                    lat_tag_r            <= fifo_tag;
                    lat_smem_byte_addr_r <= fifo_smem_byte_addr;
                    lat_byte_offset_r    <= fifo_byte_offset;
                    lat_valid_length_r   <= fifo_valid_length;
                    lat_oob_r            <= fifo_oob;
                    lat_last_r           <= fifo_last;
                    state_r              <= S_FETCH;
                end
            end
            S_FETCH: begin
                fb_data_r        <= (FILL_CAP*8)'(compressed_data) << new_bit_offset;
                fb_level_r       <= new_fill_level;
                fb_word_addr_r   <= new_start_word;
                fb_byte_offset_r <= new_smem_byte_off;
                fb_start_word_r  <= new_start_word;
                state_r          <= S_DRAIN;
            end
            S_DRAIN: begin
                if (drain_fire) begin
                    if (drain_will_empty) begin
                        fb_data_r  <= '0;
                        fb_level_r <= '0;
                        state_r    <= S_IDLE;
                    end else begin
                        fb_data_r      <= fb_data_r >> (SMEM_WORD_SIZE * 8);
                        fb_level_r     <= fb_level_r - FILL_W'(SMEM_WORD_SIZE);
                        fb_word_addr_r <= fb_word_addr_r + SMEM_ADDR_WIDTH'(1);
                    end
                end
            end
            default: state_r <= S_IDLE;
            endcase
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Resource release
    // ════════════════════════════════════════════════════════════════════
    wire drain_complete = (state_r == S_DRAIN) && drain_fire && drain_will_empty;

    assign rsp_clear_en  = drain_complete;
    assign rsp_clear_tag = lat_tag_r;
    assign release_en    = drain_complete;
    assign release_tag   = lat_tag_r;

    // ════════════════════════════════════════════════════════════════════
    // SMEM write output (with optional multicast replay)
    // ════════════════════════════════════════════════════════════════════

    wire                       smem_wr_valid;
    wire [SMEM_ADDR_WIDTH-1:0] smem_wr_addr;
    wire                       smem_wr_last_pkt;

    localparam MC_NW_BITS = `CLOG2(`NUM_WARPS);
    wire [SMEM_ADDR_WIDTH-1:0] smem_stride_words = SMEM_ADDR_WIDTH'(smem_stride >> SMEM_OFF_W);

    reg [`NUM_WARPS-1:0] replay_remaining_r;
    wire [MC_NW_BITS-1:0] replay_next_idx;
    wire replay_has_remaining;

    VX_priority_encoder #(
        .N (`NUM_WARPS)
    ) replay_pe (
        .data_in   (replay_remaining_r),
        .index_out (replay_next_idx),
        .valid_out (replay_has_remaining),
        `UNUSED_PIN (onehot_out)
    );

    wire [SMEM_ADDR_WIDTH-1:0] replay_addr = fb_word_addr_r
        + SMEM_ADDR_WIDTH'(replay_next_idx) * smem_stride_words;
    wire replay_is_last = replay_has_remaining
        && (replay_remaining_r == (`NUM_WARPS'(1) << replay_next_idx));

    wire mc_write_valid = transfer_active && drain_valid
                       && (!is_multicast || replay_has_remaining);
    wire mc_write_fire = mc_write_valid && smem_bus_if.req_ready;
    wire smem_wr_ready_internal = is_multicast
        ? (smem_bus_if.req_ready && (!replay_has_remaining || replay_is_last))
        : smem_bus_if.req_ready;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            replay_remaining_r <= '0;
        end else if (transfer_active && is_multicast) begin
            if (replay_remaining_r == '0 && drain_valid) begin
                replay_remaining_r <= cta_mask;
            end else if (mc_write_fire && replay_has_remaining) begin
                replay_remaining_r <= replay_remaining_r & ~(`NUM_WARPS'(1) << replay_next_idx);
            end
        end
    end

    assign smem_wr_valid   = mc_write_valid;
    assign smem_wr_addr    = is_multicast ? replay_addr : fb_word_addr_r;
    assign smem_req_fire   = mc_write_fire;

    wire is_last_drain = lat_last_r && drain_will_empty;
    assign smem_wr_last_pkt = is_last_drain && (!is_multicast || replay_is_last);

    // Completion flags: bar_stride hardcoded to 1.
    wire smem_wr_flags_last = active_notify_smem_done && (
        is_multicast ? (mc_write_fire && is_last_drain) : smem_wr_last_pkt);
    wire [BAR_ADDR_W-1:0] smem_wr_flags_bar = is_multicast
        ? BAR_ADDR_W'(active_bar_addr + BAR_ADDR_W'(replay_next_idx))
        : active_bar_addr;

    // ════════════════════════════════════════════════════════════════════
    // SMEM bus wiring
    // ════════════════════════════════════════════════════════════════════

    assign smem_bus_if.req_valid       = smem_wr_valid;
    assign smem_bus_if.req_data.rw     = 1'b1;
    assign smem_bus_if.req_data.addr   = smem_wr_addr;
    assign smem_bus_if.req_data.data   = fb_word_data;
    assign smem_bus_if.req_data.byteen = fb_word_byteen;
    assign smem_bus_if.req_data.flags  = {smem_wr_flags_last, smem_wr_flags_bar};
    assign smem_bus_if.req_data.tag.uuid  = '0;
    assign smem_bus_if.req_data.tag.value = SMEM_TAG_VALUE_W'(active_core_id) << ENGINE_VALUE_W;
    assign smem_bus_if.rsp_ready       = 1'b0;

    `UNUSED_VAR (smem_bus_if.rsp_valid)
    `UNUSED_VAR (smem_bus_if.rsp_data)


    // ════════════════════════════════════════════════════════════════════
    // Completion tracking
    // ════════════════════════════════════════════════════════════════════
    reg [31:0] wr_count_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wr_count_r <= '0;
        end else if (smem_req_fire) begin
            wr_count_r <= wr_count_r + 32'd1;
        end
    end

    assign wr_done_count = wr_count_r;
    assign transfer_done = transfer_active && smem_req_fire && smem_wr_last_pkt;

`ifdef PERF_ENABLE
    reg [31:0] wrp_total_lmem_writes_r;
    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wrp_total_lmem_writes_r <= '0;
        end else if (smem_req_fire) begin
            wrp_total_lmem_writes_r <= wrp_total_lmem_writes_r + 32'd1;
        end
    end
    assign perf_lmem_writes = wrp_total_lmem_writes_r + 32'(smem_req_fire);
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (do_fetch) begin
                $write("DXA_PIPE,%0d,SW_FETCH,tag=%0d,oob=%0d,smem=0x%0h,off=%0d,len=%0d,last=%0d\n",
                    $time, fifo_tag, fifo_oob, fifo_smem_byte_addr,
                    fifo_byte_offset, fifo_valid_length, fifo_last);
            end
            if (smem_req_fire) begin
                $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,last=%0d\n",
                    $time, smem_wr_addr, fb_word_byteen, smem_wr_last_pkt);
            end
        end
    end
`endif

endmodule
