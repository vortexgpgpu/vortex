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

`include "VX_define.vh"

// NUM_WARPS is the number of warps this unit fetches for; their ids arrive in
// the low bits of schedule_if.data.wid and leave the same way on fetch_if.
module VX_fetch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS
) (
    `SCOPE_IO_DECL

    input  wire             clk,
    input  wire             reset,

`ifdef PERF_ENABLE
    output fetch_perf_t     fetch_perf,
`endif

    // Icache interface
    VX_mem_bus_if.master    icache_bus_if,
    // the icache is being invalidated: so is the L0
    input wire              icache_flush,

    // inputs
    VX_schedule_if.slave    schedule_if,

    // outputs
    VX_fetch_if.master      fetch_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)

    localparam WID_W = `LOG2UP(NUM_WARPS);
    `STATIC_ASSERT (WID_W == ICACHE_TAG_ID_BITS, ("invalid parameter"))

    wire [WID_W-1:0] sched_wid = schedule_if.data.wid[WID_W-1:0];
    if (NW_WIDTH > WID_W) begin : g_sched_wid_unused
        `UNUSED_VAR (schedule_if.data.wid[NW_WIDTH-1:WID_W])
    end

    wire icache_req_valid;
    wire icache_req_ready;
    wire [ICACHE_ADDR_WIDTH-1:0] icache_req_addr;
    // Width matches the elastic_buffer payload below — the per-port
    // tag bits VX_mem_bus_arb adds downstream live in ICACHE_TAG_WIDTH but
    // are not driven by this stage.
    wire [ICACHE_FETCH_TAG_WIDTH-1:0] icache_req_tag;
    wire [WID_W-1:0]             icache_req_wid;
    wire [UUID_WIDTH-1:0]        icache_req_uuid;

    wire [UUID_WIDTH-1:0]   rsp_uuid;
    wire [PC_BITS-1:0]      rsp_PC;
    wire [`VX_CFG_NUM_THREADS-1:0] rsp_tmask;
    wire [WID_W-1:0]        req_tag;
    wire [WID_W-1:0]        rsp_tag;
    wire [NCTA_WIDTH-1:0]   rsp_cta_id;

    wire icache_req_fire = icache_req_valid && icache_req_ready;

    assign req_tag = sched_wid;

    assign {rsp_uuid, rsp_tag} = icache_bus_if.rsp_data.tag;

    VX_dp_ram #(
        .DATAW (PC_BITS + `VX_CFG_NUM_THREADS + NCTA_WIDTH),
        .SIZE  (NUM_WARPS),
        .RDW_MODE ("R"),
        .LUTRAM (1)
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (icache_req_fire),
        .wren  (1'b1),
        .waddr (req_tag),
        .wdata ({schedule_if.data.PC, schedule_if.data.tmask, schedule_if.data.cta_id}),
        .raddr (rsp_tag),
        .rdata ({rsp_PC, rsp_tmask, rsp_cta_id})
    );

    `RUNTIME_ASSERT((!schedule_if.valid || schedule_if.data.PC != 0),
        ("invalid PC=0x%0h, wid=%0d, tmask=%b (#%0d)", to_fullPC(schedule_if.data.PC), schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.uuid))

`ifdef VX_CFG_EXT_C_ENABLE
    `UNUSED_VAR (icache_flush)
    // ------------------------------------------------------------------------
    // RVC path: VX_decompressor + follow-up request mux
    // ------------------------------------------------------------------------

    wire [PC_BITS-1:0]      icache_req_PC;
    wire [`VX_CFG_NUM_THREADS-1:0] icache_req_tmask;
    wire [WID_W-1:0]        rsp_wid;

    wire                    follow_req_valid;
    wire [PC_BITS-1:0]      follow_req_PC;
    wire [`VX_CFG_NUM_THREADS-1:0] follow_req_tmask;
    wire [WID_W-1:0]        follow_req_wid;
    wire [UUID_WIDTH-1:0]   follow_req_uuid;

    wire sched_buffered_match;
    wire sched_buffered;
    assign rsp_wid = rsp_tag;

    // One-outstanding-per-warp (Invariant B): a warp with an icache request in
    // flight must not issue another, else its two responses alias the single
    // tag_store slot. Set on request fire, cleared on response fire.
    reg [NUM_WARPS-1:0] inflight;
    wire icache_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
    always @(posedge clk) begin
        if (reset) begin
            inflight <= '0;
        end else begin
            for (int w = 0; w < NUM_WARPS; ++w) begin
                if (icache_req_fire && (icache_req_wid == WID_W'(w)))
                    inflight[w] <= 1'b1;
                else if (icache_rsp_fire && (rsp_wid == WID_W'(w)))
                    inflight[w] <= 1'b0;
            end
        end
    end
    `UNUSED_VAR (icache_req_tmask)

    // ibuffer occupancy is already gated by VX_warp_sched (schedule_warps
    // masks out warps with full ibufs), so schedule_if.valid implies space.
    // Suppress with the UNGATED sched_buffered (not the st0-gated match): a
    // word already in the decompressor buffer must never be re-fetched, even
    // while stage 0 is stalled, else the icache returns a duplicate word.
    // NOTE: do NOT gate on inflight/BUF_32HI here — the scheduler stalls a warp
    // from schedule until decode, so it never re-presents an in-flight or
    // mid-straddle warp on its own; de-asserting ready for such a warp would
    // instead wedge the (non-rotating) scheduler on it. One-outstanding is thus
    // enforced by the scheduler's stall, not by this suppression.
    wire sched_req_valid = schedule_if.valid && ~sched_buffered;

    // Follow-up has priority; scheduler PC otherwise. Address is 4-byte
    // aligned (the decompressor uses PC[1] to select halfword).
    assign icache_req_valid = follow_req_valid || sched_req_valid;
    assign icache_req_PC    = follow_req_valid ? follow_req_PC    : schedule_if.data.PC;
    assign icache_req_tmask = follow_req_valid ? follow_req_tmask : schedule_if.data.tmask;
    assign icache_req_wid   = follow_req_valid ? follow_req_wid   : sched_wid;
    assign icache_req_uuid  = follow_req_valid ? follow_req_uuid  : schedule_if.data.uuid;
    // NOTE: bind the full PC to a net before part-selecting — Vivado synthesis
    // rejects a bit-select applied directly to a function-call result.
    wire [`VX_CFG_XLEN-1:0] icache_req_full_pc = to_fullPC(icache_req_PC);
    assign icache_req_addr  = icache_req_full_pc[ICACHE_ADDR_WIDTH+1 : 2];
    assign icache_req_tag   = {icache_req_uuid, icache_req_wid};
    // The icache is word-addressed: PC[1:0] (sub-word) is intentionally
    // dropped from icache_req_addr.
    `UNUSED_VAR (icache_req_full_pc)

    // Scheduler is "ready" when icache accepts the request OR when the
    // decompressor already has the data buffered.
    assign schedule_if.ibuf_pop = fetch_if.ibuf_pop;
    // Ack only when the scheduler request actually fires (not merely when the
    // icache is ready) — with the added inflight/BUF_32HI suppression a bare
    // icache_req_ready would falsely retire a suppressed request.
    assign schedule_if.ready = (sched_req_valid && ~follow_req_valid && icache_req_ready)
                             || sched_buffered_match;

    VX_decompressor #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_WARPS   (NUM_WARPS)
    ) decompressor (
        .clk                  (clk),
        .reset                (reset),
        .sched_valid          (schedule_if.valid),
        .sched_PC             (schedule_if.data.PC),
        .sched_tmask          (schedule_if.data.tmask),
        .sched_cta_id         (schedule_if.data.cta_id),
        .sched_wid            (sched_wid),
        .sched_buffered_match (sched_buffered_match),
        .sched_buffered       (sched_buffered),
        .inflight             (inflight),
        .rsp_valid            (icache_bus_if.rsp_valid),
        .rsp_word             (icache_bus_if.rsp_data.data),
        .rsp_PC               (rsp_PC),
        .rsp_tmask            (rsp_tmask),
        .rsp_cta_id           (rsp_cta_id),
        .rsp_wid              (rsp_wid),
        .rsp_uuid             (rsp_uuid),
        .rsp_ready            (icache_bus_if.rsp_ready),
        .follow_req_valid     (follow_req_valid),
        .follow_req_PC        (follow_req_PC),
        .follow_req_tmask     (follow_req_tmask),
        .follow_req_wid       (follow_req_wid),
        .follow_req_uuid      (follow_req_uuid),
        .fetch_if             (fetch_if)
    );

`else // !VX_CFG_EXT_C_ENABLE
    // ------------------------------------------------------------------------
    // Direct path (no RVC): scheduler PC → L0 or icache request → fetch_if.
    // ------------------------------------------------------------------------

    localparam PC_LSB    = `VX_CFG_XLEN - PC_BITS;
    localparam FW_WORDS  = ICACHE_WORD_SIZE / 4;
    localparam FW_WORDSW = `CLOG2(FW_WORDS);

    assign icache_req_addr  = schedule_if.data.PC[`CLOG2(ICACHE_WORD_SIZE)-PC_LSB +: ICACHE_ADDR_WIDTH];
    assign icache_req_wid   = sched_wid;
    assign icache_req_uuid  = schedule_if.data.uuid;
    assign icache_req_tag   = {icache_req_uuid, icache_req_wid};
    assign schedule_if.ibuf_pop = fetch_if.ibuf_pop;

    // The instruction at a PC in a fetch word, and the word after it when PC is
    // the first of an aligned pair (a vx_rwin fuses with it).
    wire [31:0] rsp_instr, rsp_instr2;
    if (FW_WORDS > 1) begin : g_rsp_words
        wire [FW_WORDSW-1:0] rsp_idx  = rsp_PC[2-PC_LSB +: FW_WORDSW];
        wire [FW_WORDSW-1:0] rsp_idx2 = rsp_idx | FW_WORDSW'(1);
        assign rsp_instr  = icache_bus_if.rsp_data.data[rsp_idx * 32 +: 32];
        assign rsp_instr2 = rsp_idx[0] ? 32'd0 : icache_bus_if.rsp_data.data[rsp_idx2 * 32 +: 32];
    end else begin : g_rsp_word
        assign rsp_instr  = icache_bus_if.rsp_data.data[0 +: 32];
        assign rsp_instr2 = 32'd0;
    end

    if (ICACHE_L0_LINES != 0) begin : g_l0
        // L0: whole icache lines (the fetch word is a line), fully associative,
        // round-robin replacement. A hit skips the icache and reaches decode the
        // next cycle, ahead of icache responses; a miss installs the line its
        // response brings.
        localparam N  = ICACHE_L0_LINES;
        localparam NW = `LOG2UP(N);
        localparam LW = ICACHE_WORD_SIZE * 8;
        `STATIC_ASSERT (ICACHE_WORD_SIZE == ICACHE_LINE_SIZE, ("the L0 fetches whole lines"))

        reg [N-1:0]                         l0_valid;
        reg [N-1:0][ICACHE_ADDR_WIDTH-1:0]  l0_tag;
        reg [N-1:0][LW-1:0]                 l0_data;
        reg [NW-1:0]                        l0_victim;

        wire [N-1:0] sched_match, rsp_match;
        wire [ICACHE_ADDR_WIDTH-1:0] rsp_line = rsp_PC[`CLOG2(ICACHE_WORD_SIZE)-PC_LSB +: ICACHE_ADDR_WIDTH];
        for (genvar i = 0; i < N; ++i) begin : g_match
            assign sched_match[i] = l0_valid[i] && (l0_tag[i] == icache_req_addr);
            assign rsp_match[i]   = l0_valid[i] && (l0_tag[i] == rsp_line);
        end
        wire sched_hit = (| sched_match);

        reg [LW-1:0] hit_line;
        always @(*) begin
            hit_line = '0;
            for (integer i = 0; i < N; ++i) begin
                if (sched_match[i]) hit_line |= l0_data[i];
            end
        end
        wire [FW_WORDSW-1:0] hit_idx  = schedule_if.data.PC[2-PC_LSB +: FW_WORDSW];
        wire [FW_WORDSW-1:0] hit_idx2 = hit_idx | FW_WORDSW'(1);
        wire [31:0] hit_instr  = hit_line[hit_idx * 32 +: 32];
        wire [31:0] hit_instr2 = hit_idx[0] ? 32'd0 : hit_line[hit_idx2 * 32 +: 32];

        // one hit on its way to decode
        reg                            hit_valid_r;
        reg [31:0]                     hit_instr_r, hit_instr2_r;
        reg [PC_BITS-1:0]              hit_PC_r;
        reg [`VX_CFG_NUM_THREADS-1:0]  hit_tmask_r;
        reg [NCTA_WIDTH-1:0]           hit_cta_id_r;
        reg [WID_W-1:0]                hit_wid_r;
        reg [UUID_WIDTH-1:0]           hit_uuid_r;

        wire hit_fire = hit_valid_r && fetch_if.ready;
        wire hit_room = ~hit_valid_r || hit_fire;
        wire hit_take = schedule_if.valid && sched_hit && hit_room;

        assign icache_req_valid  = schedule_if.valid && ~sched_hit;
        assign schedule_if.ready = sched_hit ? hit_room : icache_req_ready;

        wire rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
        wire rsp_install = rsp_fire && ~(| rsp_match);

        always @(posedge clk) begin
            if (reset || icache_flush) begin
                l0_valid  <= '0;
                l0_victim <= '0;
            end else if (rsp_install) begin
                l0_valid[l0_victim] <= 1'b1;
                l0_victim <= (l0_victim == NW'(N-1)) ? '0 : (l0_victim + NW'(1));
            end
            if (rsp_install) begin
                l0_tag[l0_victim]  <= rsp_line;
                l0_data[l0_victim] <= icache_bus_if.rsp_data.data;
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                hit_valid_r <= 1'b0;
            end else if (hit_take) begin
                hit_valid_r <= 1'b1;
            end else if (hit_fire) begin
                hit_valid_r <= 1'b0;
            end
            if (hit_take) begin
                hit_instr_r  <= hit_instr;
                hit_instr2_r <= hit_instr2;
                hit_PC_r     <= schedule_if.data.PC;
                hit_tmask_r  <= schedule_if.data.tmask;
                hit_cta_id_r <= schedule_if.data.cta_id;
                hit_wid_r    <= sched_wid;
                hit_uuid_r   <= schedule_if.data.uuid;
            end
        end

        assign fetch_if.valid       = hit_valid_r || icache_bus_if.rsp_valid;
        assign fetch_if.data.tmask  = hit_valid_r ? hit_tmask_r : rsp_tmask;
        assign fetch_if.data.wid    = NW_WIDTH'(hit_valid_r ? hit_wid_r : rsp_tag);
        assign fetch_if.data.cta_id = hit_valid_r ? hit_cta_id_r : rsp_cta_id;
        assign fetch_if.data.PC     = hit_valid_r ? hit_PC_r : rsp_PC;
        assign fetch_if.data.instr  = hit_valid_r ? hit_instr_r : rsp_instr;
    `ifdef VX_CFG_EXT_RWIN_ENABLE
        assign fetch_if.data.instr2 = hit_valid_r ? hit_instr2_r : rsp_instr2;
    `else
        `UNUSED_VAR ({hit_instr2_r, rsp_instr2})
    `endif
        assign fetch_if.data.uuid   = hit_valid_r ? hit_uuid_r : rsp_uuid;
        assign icache_bus_if.rsp_ready = fetch_if.ready && ~hit_valid_r;
    end else begin : g_no_l0
        `UNUSED_VAR (icache_flush)
        assign icache_req_valid  = schedule_if.valid;
        assign schedule_if.ready = icache_req_ready;

        assign fetch_if.valid       = icache_bus_if.rsp_valid;
        assign fetch_if.data.tmask  = rsp_tmask;
        assign fetch_if.data.wid    = NW_WIDTH'(rsp_tag);
        assign fetch_if.data.cta_id = rsp_cta_id;
        assign fetch_if.data.PC     = rsp_PC;
        assign fetch_if.data.instr  = rsp_instr;
    `ifdef VX_CFG_EXT_RWIN_ENABLE
        assign fetch_if.data.instr2 = rsp_instr2;
    `else
        `UNUSED_VAR (rsp_instr2)
    `endif
        assign fetch_if.data.uuid   = rsp_uuid;
        assign icache_bus_if.rsp_ready = fetch_if.ready;
    end
`endif

    // ------------------------------------------------------------------------
    // Shared icache request elastic buffer + drives
    // ------------------------------------------------------------------------

    VX_elastic_buffer #(
        .DATAW   (ICACHE_ADDR_WIDTH + ICACHE_FETCH_TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (1) // external bus should be registered
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (icache_req_valid),
        .ready_in  (icache_req_ready),
        .data_in   ({icache_req_addr, icache_req_tag}),
        .data_out  ({icache_bus_if.req_data.addr, icache_bus_if.req_data.tag}),
        .valid_out (icache_bus_if.req_valid),
        .ready_out (icache_bus_if.req_ready)
    );

    assign icache_bus_if.req_data.attr   = '0;
    assign icache_bus_if.req_data.rw     = 1'b0;
    assign icache_bus_if.req_data.byteen = '1;
    assign icache_bus_if.req_data.data   = '0;

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_fetch_stalls;

    wire icache_req_stall = icache_req_valid && ~icache_req_ready;

    always @(posedge clk) begin
        if (reset) begin
            perf_fetch_stalls <= '0;
        end else begin
            perf_fetch_stalls <= perf_fetch_stalls + PERF_CTR_BITS'(icache_req_stall);
        end
    end

    assign fetch_perf.stalls = perf_fetch_stalls;
`endif

`ifdef SCOPE
    `SCOPE_IO_SWITCH (1);
`ifdef DBG_SCOPE_FETCH
    wire schedule_fire = schedule_if.valid && schedule_if.ready;
    wire icache_bus_req_fire = icache_bus_if.req_valid && icache_bus_if.req_ready;
    wire icache_bus_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 1, 6, 3, (
            UUID_WIDTH + NW_WIDTH + `VX_CFG_NUM_THREADS + PC_BITS +
            UUID_WIDTH + ICACHE_WORD_SIZE + ICACHE_ADDR_WIDTH +
            UUID_WIDTH + (ICACHE_WORD_SIZE * 8)
        ), {
            schedule_if.valid,
            schedule_if.ready,
            icache_bus_if.req_valid,
            icache_bus_if.req_ready,
            icache_bus_if.rsp_valid,
            icache_bus_if.rsp_ready
        }, {
            schedule_fire,
            icache_bus_req_fire,
            icache_bus_rsp_fire
        }, {
            schedule_if.data.uuid, schedule_if.data.wid, schedule_if.data.tmask, schedule_if.data.PC,
            icache_bus_if.req_data.tag.uuid, icache_bus_if.req_data.byteen, icache_bus_if.req_data.addr,
            icache_bus_if.rsp_data.tag.uuid, icache_bus_if.rsp_data.data
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_FETCH
    ila_fetch ila_fetch_inst (
        .clk    (clk),
        .probe0 ({schedule_if.valid, schedule_if.data, schedule_if.ready}),
        .probe1 ({icache_bus_if.req_valid, icache_bus_if.req_data, icache_bus_if.req_ready}),
        .probe2 ({icache_bus_if.rsp_valid, icache_bus_if.rsp_data, icache_bus_if.rsp_ready})
    );
`endif
`endif


`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (schedule_if.valid && schedule_if.ready) begin
            `TRACE(1, ("%t: %s req: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b (#%0d)\n", $time, INSTANCE_ID, schedule_if.data.wid, schedule_if.data.cta_id, to_fullPC(schedule_if.data.PC), schedule_if.data.tmask, schedule_if.data.uuid))
        end
        if (fetch_if.valid && fetch_if.ready) begin
            `TRACE(1, ("%t: %s rsp: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, instr=0x%0h (#%0d)\n", $time, INSTANCE_ID, fetch_if.data.wid, fetch_if.data.cta_id, to_fullPC(fetch_if.data.PC), fetch_if.data.tmask, fetch_if.data.instr, fetch_if.data.uuid))
        end
    end
`endif

endmodule
