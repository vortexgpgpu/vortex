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

// VX_raster_unit — per-core SFU PE for raster ops. Sub-ops share INST_SFU_RASTER,
// discriminated by op_args.raster:
//   is_begin     — vx_rast_begin: pulse the producer (no bus round-trip).
//   is_fwd_run   — vx_frag_fetch: pop the next covered-quad wave from the cluster
//                  raster bus and DMA-stage the per-lane frag_payload_t into the
//                  warp's own LMEM (rs1 = __local_mem() base), then return a
//                  scoreboarded drained flag (rd: 1 = producer drained → the
//                  persistent fragment worker exits its loop). No bcoord CSRs, no
//                  pos_mask sentinel — the doctrine-clean (C2/C3) handoff.
//
// vx_frag_fetch is synchronous (completes when a wave is staged or the producer
// drains) — it always completes, so the head-of-line SFU switch never
// permanently blocks the OM ops the worker issues next.

`include "VX_raster_define.vh"

module VX_raster_unit import VX_gpu_pkg::*, VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // SFU PE-style interfaces
    VX_execute_if.slave    execute_if,
    VX_result_if.master    result_if,

    // Cluster-side raster bus (slave — agent pops descriptors)
    VX_raster_bus_if.slave raster_bus_if,

    // FWD payload-stage write port (vx_frag_fetch → LMEM DMA agent)
    VX_mem_bus_if.master                       fwd_dma_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    wire is_begin_op = execute_if.data.op_args.raster.is_begin;
    wire is_fetch_op = ~is_begin_op;   // funct3=3 vx_frag_fetch (funct3=4 begin)

    // ── LMEM DMA payload geometry (matches frag_payload_t padded to a whole
    //    number of DMA lines so each lane is line-aligned) ──
    localparam LMEM_LOG       = `VX_CFG_LMEM_LOG_SIZE;
    localparam DMA_LINE_WORDS = `VX_CFG_LMEM_NUM_BANKS;
    localparam DMA_LINE_BYTES = DMA_LINE_WORDS * LSU_WORD_SIZE;
    localparam DMA_ADDR_W     = LMEM_LOG - `CLOG2(DMA_LINE_BYTES);
    localparam PAYLOAD_WORDS  = 14;                                       // pos_mask,pid,bcoord[3][4]
    localparam LANE_LINES     = (PAYLOAD_WORDS + DMA_LINE_WORDS - 1) / DMA_LINE_WORDS;
    localparam LANE_WORDS     = LANE_LINES * DMA_LINE_WORDS;
    localparam LANE_BYTES     = LANE_WORDS * LSU_WORD_SIZE;               // = sizeof(frag_payload_t) padded
    localparam LANE_W         = `LOG2UP(NUM_LANES);
    localparam LINE_W         = `LOG2UP(LANE_LINES);

    // ── frag-fetch DMA-stage FSM ──
    localparam [0:0] S_IDLE = 1'd0, S_STAGE = 1'd1;
    reg state;
    reg [LANE_W-1:0] dma_lane_r;
    reg [LINE_W-1:0] dma_line_r;

    wire raster_rsp_valid, raster_rsp_ready;

    // Producer state visible to the fetch op.
    wire bus_valid   = raster_bus_if.req_valid;
    wire bus_drained = raster_bus_if.req_data.done;

    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] lmem_base = execute_if.data.rs1_data[0];

    // Quick-pop: capture a covered (non-drained) frag_fetch wave + free the bus
    // in one cycle (legacy-like), then DMA-stage from the latch. Holding the bus
    // through the multi-cycle stage would stall the cluster raster arb's fan-out.
    wire fetch_capture = is_fetch_op && execute_if.valid
                      && (state == S_IDLE) && bus_valid && ~bus_drained;

    // Latched wave (DMA-stage source) + warp LMEM base.
    raster_stamp_t [NUM_LANES-1:0] wave_r;
    reg [LMEM_LOG-1:0]             lmem_off_r;

    // current lane/line payload image: [0]=pos_mask,[1]=pid,[2..13]=bcoord,[14..15]=pad
    raster_stamp_t cur_stamp;
    assign cur_stamp = wave_r[dma_lane_r];
    wire [31:0] pos_mask_w = {cur_stamp.pos_y, cur_stamp.pos_x, cur_stamp.mask};
    wire [LANE_WORDS-1:0][31:0] lane_words;
    assign lane_words[0] = pos_mask_w;
    assign lane_words[1] = 32'(cur_stamp.pid);
    for (genvar a = 0; a < 3; ++a) begin : g_bc_a
        for (genvar c = 0; c < 4; ++c) begin : g_bc_c
            assign lane_words[2 + a*4 + c] = cur_stamp.bcoords[a][c];
        end
    end
    for (genvar w = PAYLOAD_WORDS; w < LANE_WORDS; ++w) begin : g_pad
        assign lane_words[w] = 32'd0;
    end

    wire [DMA_LINE_WORDS-1:0][31:0] beat_words;
    for (genvar b = 0; b < DMA_LINE_WORDS; ++b) begin : g_beat
        assign beat_words[b] = lane_words[(dma_line_r << `CLOG2(DMA_LINE_WORDS)) + b];
    end
    wire [31:0] beat_line32 = (32'(lmem_off_r) >> `CLOG2(DMA_LINE_BYTES))
                            + 32'(dma_lane_r) * 32'(LANE_LINES) + 32'(dma_line_r);
    wire [DMA_ADDR_W-1:0] beat_line = DMA_ADDR_W'(beat_line32);

    wire lane_covered  = (| cur_stamp.mask);
    wire last_line     = (dma_line_r == LINE_W'(LANE_LINES - 1));
    wire last_lane     = (dma_lane_r == LANE_W'(NUM_LANES - 1));
    wire beat_fire     = (state == S_STAGE) && lane_covered && fwd_dma_if.req_valid && fwd_dma_if.req_ready;
    wire skip_lane     = (state == S_STAGE) && ~lane_covered;
    wire stage_done    = (state == S_STAGE) && last_lane
                      && (~lane_covered || (beat_fire && last_line));

    // ── DMA request drive (covered lanes only) ──
    assign fwd_dma_if.req_valid       = (state == S_STAGE) && lane_covered;
    assign fwd_dma_if.req_data.rw     = 1'b1;
    assign fwd_dma_if.req_data.addr   = ($bits(fwd_dma_if.req_data.addr))'(beat_line);
    assign fwd_dma_if.req_data.data   = ($bits(fwd_dma_if.req_data.data))'(beat_words);
    assign fwd_dma_if.req_data.byteen = {($bits(fwd_dma_if.req_data.byteen)){1'b1}};
    assign fwd_dma_if.req_data.attr   = '0;
    assign fwd_dma_if.req_data.tag    = '0;
    assign fwd_dma_if.rsp_ready       = 1'b1;

    // ── op completion ──
    // A drained fetch completes immediately in IDLE (rd=1, worker exits); a
    // covered fetch completes when its payload stage finishes (rd=0).
    wire fetch_drained = is_fetch_op && execute_if.valid
                      && (state == S_IDLE) && bus_valid && bus_drained;
    wire fetch_complete = fetch_drained || stage_done;

    assign raster_rsp_valid = is_begin_op ? execute_if.valid
                            : fetch_complete;

    assign execute_if.ready = is_begin_op ? raster_rsp_ready
                            : (fetch_complete && raster_rsp_ready);

    // Bus pop: drained fetch on complete; covered fetch at capture (quick-pop,
    // freeing the arb during the DMA stage).
    assign raster_bus_if.req_ready = (fetch_drained && raster_rsp_ready) || fetch_capture;
    assign raster_bus_if.begin_pulse = execute_if.valid && execute_if.ready && is_begin_op;

    always @(posedge clk) begin
        if (reset) begin
            state      <= S_IDLE;
            dma_lane_r <= '0;
            dma_line_r <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    // quick-pop: latch the wave + LMEM base, free the bus, stage next
                    if (fetch_capture) begin
                        wave_r     <= raster_bus_if.req_data.stamps;
                        lmem_off_r <= lmem_base[LMEM_LOG-1:0];
                        state      <= S_STAGE;
                        dma_lane_r <= '0;
                        dma_line_r <= '0;
                    end
                end
                S_STAGE: begin
                    if (stage_done && raster_rsp_ready) begin
                        state <= S_IDLE;
                    end else if (skip_lane) begin
                        if (~last_lane) dma_lane_r <= dma_lane_r + LANE_W'(1);
                    end else if (beat_fire) begin
                        if (last_line) begin
                            dma_line_r <= '0;
                            if (~last_lane) dma_lane_r <= dma_lane_r + LANE_W'(1);
                        end else begin
                            dma_line_r <= dma_line_r + LINE_W'(1);
                        end
                    end
                end
            endcase
        end
    end

    // ── result word ──
    //   frag_fetch: rd = drained flag (broadcast to all lanes); begin: 0.
    wire [NUM_LANES-1:0][31:0] response_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_response_data
        assign response_data[i] = is_begin_op ? 32'd0 : {31'd0, fetch_drained};
    end

    sfu_result_t rsp_data_in;
    assign rsp_data_in.header = execute_if.data.header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        assign rsp_data_in.data[i] = `VX_CFG_XLEN'(response_data[i]);
    end

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (raster_rsp_valid),
        .ready_in  (raster_rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: %s raster-op: wid=%0d, begin=%b fetch=%b drained=%b (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid,
                is_begin_op, is_fetch_op, raster_bus_if.req_data.done, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
