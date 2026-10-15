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

// The timing of foldsim's memory endpoint as a DRAM bank model: the model of
// sim/common/dram_bank_model.h (DramBankAxi over DramBankModel), decision for
// decision, for one AXI4 port and one DRAM channel.
//
// foldsim_mem gives it each target cycle's requests when the cycle finishes;
// it answers with the port's state for the next one: whether a read or write
// may be taken, and the read and write acknowledgement R and B present. The
// DRAM clocks of a target cycle run in host cycles while the target evaluates
// it, and the cycle does not finish before they have (`done`): requests of a
// cycle reach the channel in the next one, and a read departs at least
// tCL + tBL clocks after it was reserved, so a cycle's clocks depend only on
// what earlier cycles gave.
//
// The calendars are bitmaps over a ring of 2^HLOG2 DRAM clocks in distributed
// RAM (foldsim_mem_bmap): which clocks hold a command, the row-command bus,
// and per pseudo-channel and bank group the clocks a read, a write or an ACT
// may not take -- each command, once reserved, marks the clocks its
// constraints forbid to later ones, so placing a command is a first-free
// search over 64 clocks a host cycle. A reservation reaches at most
// 2^HLOG2 - 64 clocks ahead: the ring's last word is the one being retired.
// Times of the bank table and the controller are 16-bit, modulo; a time too
// old to constrain anything is moved up before it could wrap.
module foldsim_mem_bank #(
    parameter ID_W   = 4,
    parameter SLOTS  = 32,
    parameter WSLOTS = 16,
    parameter HLOG2  = 9,
    parameter BURSTS = 2,
    parameter ROW_W  = 20,
    parameter INQ    = 64,
    parameter SW     = $clog2(SLOTS)
) (
    input  wire                  clk,
    input  wire                  reset,

    // The bank model is the port's timing from the next target reset on.
    input  wire                  enable,

    input  wire [15:0]           cfg_data_rate,
    input  wire [15:0]           cfg_core_clock,
    input  wire [15:0]           cfg_latency,
    input  wire [15:0]           cfg_queue,
    input  wire [7:0]            cfg_rq,
    input  wire [7:0]            cfg_wq,
    input  wire [7:0]            cfg_wr_high,
    input  wire [7:0]            cfg_wr_low,
    // [1:0] log2 pseudo-channels, [3:2] log2 bank groups, [5:4] log2 banks per
    // group, [9:6] log2 bursts per row, [11:10] log2 bursts per row visit,
    // [12] an ACT holds the row bus for two clocks.
    input  wire [12:0]           cfg_geom,
    // tBL, tCCD_S, tCCD_L, tCL, tCWL, tRCD_RD, tRCD_WR, tRP, tRAS, tRC, tRTP,
    // tWR, tRRD_S, tRRD_L, tFAW, tWTR_S, tWTR_L, read-to-write extra: DRAM
    // clocks, 12 bits each.
    input  wire [18*12-1:0]      cfg_t,

    // The target cycle finishes; a reset cycle drops everything.
    input  wire                  finish,
    input  wire                  treset,
    // What the cycle takes: a read, a write formed of an AW and its W, and the
    // responses R and B give.
    input  wire                  ar_fire,
    input  wire [ID_W-1:0]       ar_id,
    input  wire [29:0]           ar_addr,
    input  wire                  aw_fire,
    input  wire                  new_write,
    input  wire [ID_W-1:0]       nw_id,
    input  wire [29:0]           nw_addr,
    input  wire                  r_fire,
    input  wire                  b_fire,

    // The port's state for the target cycle.
    output wire                  room,
    output wire                  slot_free,
    output wire [SW-1:0]         alloc_slot,
    output wire                  r_valid,
    output wire [ID_W-1:0]       r_id,
    output wire [SW-1:0]         r_slot,
    output wire                  b_valid,
    output wire [ID_W-1:0]       b_id,
    output wire                  no_reads,
    // The target cycle's DRAM clocks are done.
    output wire                  done,
    // The bank model is the port's timing, since the last target reset.
    output reg                   active,

    // Reads, writes, row hits, misses, conflicts, and the sum of read
    // latencies (target cycles from AR to R).
    output reg  [63:0]           stat_reads,
    output reg  [63:0]           stat_writes,
    output reg  [63:0]           stat_hits,
    output reg  [63:0]           stat_misses,
    output reg  [63:0]           stat_conflicts,
    output reg  [63:0]           stat_latency
);
    localparam H     = 1 << HLOG2;
    localparam HE    = H - 64;
    localparam WORDS = H / 64;
    localparam WW    = HLOG2 - 6;
    localparam NB    = BURSTS;
    localparam TAG_W = (SW > ID_W) ? SW : ID_W;
    localparam BKW   = 5;
    localparam IQW   = $clog2(INQ);
    localparam BQ    = 32;
    localparam BQW   = $clog2(BQ);

    // ---- Configuration and its derived constants ----
    wire [11:0] T [18];
    for (genvar i = 0; i < 18; ++i) begin : g_t
        assign T[i] = cfg_t[i*12 +: 12];
    end
    localparam I_BL = 0, I_CCDS = 1, I_CCDL = 2, I_CL = 3, I_CWL = 4, I_RCDRD = 5, I_RCDWR = 6,
               I_RP = 7, I_RAS = 8, I_RC = 9, I_RTP = 10, I_WR = 11, I_RRDS = 12, I_RRDL = 13,
               I_FAW = 14, I_WTRS = 15, I_WTRL = 16, I_RTWX = 17;

    wire [1:0] g_lpcs   = cfg_geom[1:0];
    wire [1:0] g_lbgs   = cfg_geom[3:2];
    wire [1:0] g_lbanks = cfg_geom[5:4];
    wire [3:0] g_lrow   = cfg_geom[9:6];
    wire [1:0] g_lvisit = cfg_geom[11:10];
    wire       g_act2   = cfg_geom[12];

    function automatic signed [15:0] smax(input signed [15:0] a, input signed [15:0] b);
        smax = (a > b) ? a : b;
    endfunction

    reg signed [15:0] k_ccds, k_ccdl, k_rtw, k_wtrs, k_wtrl, k_reach_r, k_reach_w, k_reach_a;
    reg signed [15:0] k_wrpre, k_dep, k_ccdl1, k_rcdrd, k_rcdwr, k_rp, k_ras, k_rc, k_rtp;
    reg signed [15:0] k_rrds, k_rrdl;
    reg [16:0]        k_twoclk;
    reg [2:0]         k_bb;
    always @(posedge clk) begin
        k_ccds    <= 16'(T[I_CCDS]);
        k_ccdl    <= 16'(T[I_CCDL]);
        k_rtw     <= smax(16'(T[I_CCDS]), 16'(T[I_CL]) + 16'(T[I_BL]) + 16'(T[I_RTWX]) - 16'(T[I_CWL]));
        k_wtrs    <= 16'(T[I_CWL]) + 16'(T[I_BL]) + 16'(T[I_WTRS]);
        k_wtrl    <= 16'(T[I_CWL]) + 16'(T[I_BL]) + 16'(T[I_WTRL]);
        k_reach_r <= smax(smax(k_ccds, k_ccdl), k_rtw) - 16'sd1;
        k_reach_w <= smax(smax(k_ccds, k_ccdl), smax(k_wtrs, k_wtrl)) - 16'sd1;
        k_reach_a <= smax(smax(smax(16'(T[I_RRDS]), 16'(T[I_RRDL])), 16'(T[I_FAW])) - 16'sd1,
                          g_act2 ? 16'sd1 : 16'sd0);
        k_wrpre   <= 16'(T[I_CWL]) + 16'(T[I_BL]) + 16'(T[I_WR]);
        k_dep     <= 16'(T[I_CL]) + 16'(T[I_BL]);
        k_ccdl1   <= smax(16'(T[I_CCDL]), 16'sd1);
        k_rcdrd   <= 16'(T[I_RCDRD]);
        k_rcdwr   <= 16'(T[I_RCDWR]);
        k_rp      <= 16'(T[I_RP]);
        k_ras     <= 16'(T[I_RAS]);
        k_rc      <= 16'(T[I_RC]);
        k_rtp     <= 16'(T[I_RTP]);
        k_rrds    <= 16'(T[I_RRDS]);
        k_rrdl    <= 16'(T[I_RRDL]);
        k_twoclk  <= {cfg_core_clock, 1'b0};
        k_bb      <= 3'(g_lpcs) + 3'(g_lbgs) + 3'(g_lbanks);
    end

    // The bank and row of a channel address (sim/common/dram_bank_model.h's
    // decode): a row visit, then the bank bits, the row above the column, and
    // the bank XORed with the row folded to its width.
    function automatic [BKW+ROW_W-1:0] decode(input [29:5] a);
        reg [24:0]      tx;
        reg [24:0]      b;
        reg [ROW_W-1:0] r;
        reg [24:0]      mask;
        tx = a[29:5];
        mask = (25'd1 << k_bb) - 25'd1;
        b = (tx >> g_lvisit) & mask;
        r = ROW_W'(tx >> (32'(k_bb) + 32'(g_lrow)));
        for (integer k = 0; k < ROW_W; ++k) begin
            if (k_bb != 0 && 32'(k) * 32'(k_bb) < ROW_W)
                b = b ^ ((25'(r) >> (32'(k) * 32'(k_bb))) & mask);
        end
        decode = {BKW'(b), r};
    endfunction

    // ---- The port: slots, responses, requests in flight ----
    reg [SLOTS-1:0]  slot_busy;
    reg [ID_W-1:0]   slot_id [SLOTS];
    reg [SW:0]       slots_used;
    reg [15:0]       inflight;
    reg              cur_r_valid, cur_b_valid;
    reg [SW-1:0]     cur_r_slot;
    reg [ID_W-1:0]   cur_r_id, cur_b_id;
    reg [31:0]       tcyc;

    // Reads departed, in order, with the target cycle they are ready in; the
    // first `rf_ready` are ready in the next cycle.
    reg [SW-1:0]     rf_slot [SLOTS];
    reg [31:0]       rf_cyc  [SLOTS];
    reg [SW-1:0]     rf_head, rf_tail;
    reg [SW:0]       rf_count, rf_ready;
    // Writes the controller has buffered, to acknowledge in order.
    localparam AQW = $clog2(WSLOTS);
    reg [ID_W-1:0]   af_id [WSLOTS];
    reg [AQW-1:0]    af_head, af_tail;
    reg [AQW:0]      af_count;
    // What this target cycle's DRAM clocks made ready by the next one.
    reg [SW:0]       new_ready;
    reg [AQW:0]      new_acks;

    assign room       = (cfg_queue == 16'd0) | (inflight < cfg_queue);
    assign slot_free  = (32'(slots_used) < SLOTS);
    assign r_valid    = cur_r_valid;
    assign r_id       = cur_r_id;
    assign r_slot     = cur_r_slot;
    assign b_valid    = cur_b_valid;
    assign b_id       = cur_b_id;
    assign no_reads   = (slots_used == 0);

    // The lowest free slot once R has taken its read.
    wire [SLOTS-1:0] free_slots = ~slot_busy | ((r_fire ? SLOTS'(1) : SLOTS'(0)) << cur_r_slot);
    reg  [SW-1:0]    lowest_free;
    always @(*) begin
        lowest_free = '0;
        for (integer s = SLOTS - 1; s >= 0; --s)
            if (free_slots[s])
                lowest_free = SW'(s);
    end
    assign alloc_slot = lowest_free;

    // This cycle's requests, staged for the channel: a write, then a read.
    reg              st_w_valid, st_r_valid;
    reg [ID_W-1:0]   st_w_id;
    reg [SW-1:0]     st_r_slot;
    reg [29:0]       st_w_addr, st_r_addr;

    // ---- The channel ----
    // Requests in the order they formed; the controller's read and write
    // buffers.
    reg              iq_w   [INQ];
    reg [TAG_W-1:0]  iq_tag [INQ];
    reg [29:0]       iq_addr[INQ];
    reg [IQW-1:0]    iq_head, iq_tail;
    reg [IQW:0]      iq_count;
    reg [TAG_W-1:0]  rb_tag [BQ], wb_tag [BQ];
    reg [BKW-1:0]    rb_bank[BQ], wb_bank[BQ];
    reg [ROW_W-1:0]  rb_row [BQ], wb_row [BQ];
    reg [BQW-1:0]    rb_head, rb_tail, wb_head, wb_tail;
    reg [BQW:0]      rb_count, wb_count;

    reg [15:0]       d;              // the DRAM clock, modulo
    reg [31:0]       credit;
    reg [15:0]       ingress_ready;
    reg [7:0]        rq_cnt, wq_cnt; // reserved bursts whose column has not issued
    reg              wmode;
    reg [15:0]       frontier, batch_floor, batch_last, read_floor;

    // The bank table. A fresh bank has no row open and no time that matters.
    reg [31:0]       bt_open, bt_fresh;
    reg [ROW_W-1:0]  bt_row  [32];
    reg [15:0]       bt_act  [32];
    reg [15:0]       bt_colr [32];
    reg [15:0]       bt_prer [32];
    reg [4:0]        scrub_b;

    wire [15:0]      e     = d + 16'd1;              // the first clock a reservation may take
    wire [HLOG2-1:0] base  = HLOG2'(e);              // its place on the ring
    function automatic signed [15:0] rel(input [15:0] t);
        rel = $signed(t - e);
    endfunction

    // ---- Calendars ----
    // The command bus is two calendars, row and column commands, so a
    // reservation marks both at once.
    localparam B_CMDR = 0, B_ROW = 1, B_KR = 2, B_KW = 3, B_DEPV = 4, B_FRP = 5, B_FWP = 6,
               B_FRG = 7, B_FWG = 8, B_FAP = 9, B_FAG = 10, B_CMDC = 11, NBM = 12;

    reg  [WW-1:0]    s_word;         // the word the search reads
    wire [WW-1:0]    s_word1 = s_word + WW'(1);
    reg              op_w;
    reg [TAG_W-1:0]  op_tag;
    reg [BKW-1:0]    op_bank;
    reg [ROW_W-1:0]  op_row;
    reg              op_pc;
    reg [2:0]        op_pg;

    wire [WW-1:0]    end_word = WW'(base >> 6);
    reg              bm_clear;
    reg  [WW-1:0]    bm_clear_word;
    // A target reset (or the port leaving the bank model) clears the
    // calendars at the cycle's end, before the next cycle's first clock.
    wire             bm_clear_all = finish & (treset | ~active);
    reg  [NBM-1:0]   wr_en;
    reg  [WW-1:0]    wr_word [NBM];
    reg  [63:0]      wr_mask [NBM];
    wire [63:0]      rd_cmdr, rd_cmdc, rd_row, rd_kr, rd_kw, rd_depv;
    wire [63:0]      rd_cmd = rd_cmdr | rd_cmdc;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0]      rd_row1;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [63:0]      rd_frp, rd_fwp, rd_frg, rd_fwg, rd_fap, rd_fag;

    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_cmdr (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(s_word), .rd_data(rd_cmdr),
        .we(wr_en[B_CMDR]), .wr_inst(1'b0), .wr_word(wr_word[B_CMDR]), .wr_mask(wr_mask[B_CMDR]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_cmdc (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(s_word), .rd_data(rd_cmdc),
        .we(wr_en[B_CMDC]), .wr_inst(1'b0), .wr_word(wr_word[B_CMDC]), .wr_mask(wr_mask[B_CMDC]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(2)) m_row (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(2'b00), .rd_word({s_word1, s_word}), .rd_data({rd_row1, rd_row}),
        .we(wr_en[B_ROW]), .wr_inst(1'b0), .wr_word(wr_word[B_ROW]), .wr_mask(wr_mask[B_ROW]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_kr (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(end_word), .rd_data(rd_kr),
        .we(wr_en[B_KR]), .wr_inst(1'b0), .wr_word(wr_word[B_KR]), .wr_mask(wr_mask[B_KR]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_kw (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(end_word), .rd_data(rd_kw),
        .we(wr_en[B_KW]), .wr_inst(1'b0), .wr_word(wr_word[B_KW]), .wr_mask(wr_mask[B_KW]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_depv (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(end_word), .rd_data(rd_depv),
        .we(wr_en[B_DEPV]), .wr_inst(1'b0), .wr_word(wr_word[B_DEPV]), .wr_mask(wr_mask[B_DEPV]));
    foldsim_mem_bmap #(.INST(2), .WORDS(WORDS), .NREAD(1)) m_frp (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pc), .rd_word(s_word), .rd_data(rd_frp),
        .we(wr_en[B_FRP]), .wr_inst(op_pc), .wr_word(wr_word[B_FRP]), .wr_mask(wr_mask[B_FRP]));
    foldsim_mem_bmap #(.INST(2), .WORDS(WORDS), .NREAD(1)) m_fwp (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pc), .rd_word(s_word), .rd_data(rd_fwp),
        .we(wr_en[B_FWP]), .wr_inst(op_pc), .wr_word(wr_word[B_FWP]), .wr_mask(wr_mask[B_FWP]));
    foldsim_mem_bmap #(.INST(8), .WORDS(WORDS), .NREAD(1)) m_frg (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pg), .rd_word(s_word), .rd_data(rd_frg),
        .we(wr_en[B_FRG]), .wr_inst(op_pg), .wr_word(wr_word[B_FRG]), .wr_mask(wr_mask[B_FRG]));
    foldsim_mem_bmap #(.INST(8), .WORDS(WORDS), .NREAD(1)) m_fwg (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pg), .rd_word(s_word), .rd_data(rd_fwg),
        .we(wr_en[B_FWG]), .wr_inst(op_pg), .wr_word(wr_word[B_FWG]), .wr_mask(wr_mask[B_FWG]));
    foldsim_mem_bmap #(.INST(2), .WORDS(WORDS), .NREAD(1)) m_fap (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pc), .rd_word(s_word), .rd_data(rd_fap),
        .we(wr_en[B_FAP]), .wr_inst(op_pc), .wr_word(wr_word[B_FAP]), .wr_mask(wr_mask[B_FAP]));
    foldsim_mem_bmap #(.INST(8), .WORDS(WORDS), .NREAD(1)) m_fag (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pg), .rd_word(s_word), .rd_data(rd_fag),
        .we(wr_en[B_FAG]), .wr_inst(op_pg), .wr_word(wr_word[B_FAG]), .wr_mask(wr_mask[B_FAG]));

    // The read each departure completes: a slot per clock of the ring.
    reg [SW-1:0]     dep_slot [H];

    // ---- The engine ----
    localparam S_IDLE = 4'd0, S_PUSH = 4'd1, S_TICK = 4'd3, S_BANK = 4'd4,
               S_SETUP = 4'd5, S_SRCH = 4'd6, S_CHECK = 4'd7, S_COMMIT = 4'd8, S_END = 4'd9,
               S_WALK = 4'd10;
    localparam K_PRE = 2'd0, K_ACT = 2'd1, K_C0 = 2'd2, K_C1 = 2'd3;
    reg [3:0]        state;

    assign done = (state == S_IDLE);

    // The operation: the request being reserved, and its commands, as clocks
    // relative to e.
    reg              op_hit, op_conf;
    reg signed [15:0] r_act, r_colr, r_prer, r_floor;
    reg signed [15:0] c_pre, c_act, c_col0, c_col1;
    reg [1:0]        kind;

    // The search: the next word to read (as the ring position and the clocks
    // left to the horizon) and the one read last cycle, to decide on.
    reg [HLOG2-1:0]  s_pos;
    reg signed [15:0] s_left;
    reg              s_fetch;
    reg              sb_valid, b_last;
    reg [63:0]       b_cand;
    reg [WW-1:0]     b_word;

    function automatic [63:0] span_mask(input [5:0] off, input signed [15:0] left);
        reg [6:0] top;
        top = (left >= 16'sd64 - 16'(off)) ? 7'd64 : 7'(7'(off) + 7'(left));
        span_mask = (~64'd0 << off) & ((top == 7'd64) ? ~64'd0 : ~(~64'd0 << top));
    endfunction

    wire [5:0]  s_off   = s_pos[5:0];
    wire [63:0] s_mask  = span_mask(s_off, s_left);
    wire signed [15:0] s_take = (s_left >= 16'sd64 - 16'(s_off)) ? 16'sd64 - 16'(s_off) : s_left;
    wire [63:0] f_col   = rd_cmd | (op_w ? (rd_fwp | rd_fwg) : (rd_frp | rd_frg));
    wire [63:0] f_act   = rd_cmd | rd_row | (g_act2 ? {rd_row1[0], rd_row[63:1]} : 64'd0) | rd_fap | rd_fag;
    wire [63:0] f_pre   = rd_cmd | rd_row;
    wire [63:0] forb    = (kind == K_PRE) ? f_pre : ((kind == K_ACT) ? f_act : f_col);

    reg  [5:0]  b_ffs;
    always @(*) begin
        b_ffs = '0;
        for (integer j = 63; j >= 0; --j)
            if (b_cand[j])
                b_ffs = 6'(j);
    end
    wire [HLOG2-1:0]  b_pos = HLOG2'({b_word, b_ffs});
    wire signed [15:0] b_rel = 16'(HLOG2'(b_pos - base));

    // ---- Commit: per calendar, up to two ranges of clocks to mark ----
    reg              cm_v    [NBM][2];
    reg signed [15:0] cm_lo  [NBM][2];
    reg signed [15:0] cm_hi  [NBM][2];
    reg [HLOG2-1:0]  cm_pos  [NBM];
    reg signed [15:0] cm_left[NBM];
    reg              cm_busy [NBM];
    reg              cm_next [NBM];   // the second range is still to mark
    reg              cm_start;

    reg signed [15:0] ck_colr, ck_prer, ck_lastcol;
    reg              cm_any;
    always @(*) begin
        cm_any = 1'b0;
        for (integer m = 0; m < NBM; ++m)
            cm_any = cm_any | cm_busy[m];
    end

    // The read data the cycle's departure completes, and the clock's columns.
    wire        end_depv = rd_depv[base[5:0]];
    wire        end_kr   = rd_kr[base[5:0]];
    wire        end_kw   = rd_kw[base[5:0]];

    // Ingress head, decoded, and whether it enters its buffer this tick.
    wire               ih_w    = iq_w[iq_head];
    wire [TAG_W-1:0]   ih_tag  = iq_tag[iq_head];
    wire [BKW+ROW_W-1:0] ih_dec = decode(iq_addr[iq_head][29:5]);
    wire [8:0]         wb_after = 9'(wq_cnt) + 9'(NB) * (9'(wb_count) + 9'd1);
    wire [8:0]         rb_after = 9'(rq_cnt) + 9'(NB) * (9'(rb_count) + 9'd1);
    wire               ih_time = ($signed(d - ingress_ready) >= 0);
    wire               ih_room = ih_w ? (wb_after <= 9'(cfg_wq)) : (rb_after <= 9'(cfg_rq));
    wire               move    = (iq_count != 0) & ih_time & ih_room;
    wire               mv_w    = move & ih_w;
    wire               mv_r    = move & ~ih_w;
    // The buffers' bursts after the move, and the mode they put the channel in.
    wire [8:0]         wn = 9'(wq_cnt) + 9'(NB) * (9'(wb_count) + 9'(mv_w));
    wire [8:0]         rn = 9'(rq_cnt) + 9'(NB) * (9'(rb_count) + 9'(mv_r));
    wire               to_write = ~wmode & ((wn >= 9'(cfg_wr_high)) | (rn == 0));
    wire               to_read  = wmode & (wn < 9'(cfg_wr_low)) & (rn != 0);
    wire               wmode_n  = to_write | (wmode & ~to_read);
    wire [BQW:0]       sel_count = wmode_n ? (wb_count + (BQW+1)'(mv_w))
                                           : (rb_count + (BQW+1)'(mv_r));

    // The end of DRAM clock d: clock d+1's departing read and the columns it
    // issues leave the controller; then the next clock.
    task automatic end_clock();
        if (end_depv) begin
            rf_slot[rf_tail] <= dep_slot[base];
            rf_cyc[rf_tail]  <= tcyc + 32'd1 + 32'(cfg_latency);
            rf_tail          <= rf_tail + 1'b1;
            rf_count         <= rf_count + 1'b1;
        end
        rq_cnt <= rq_cnt - (end_kr ? 8'd1 : 8'd0);
        wq_cnt <= wq_cnt - (end_kw ? 8'd1 : 8'd0);
        // The ring's word ends with this clock: it is all past.
        if (base[5:0] == 6'd63) begin
            bm_clear      <= 1'b1;
            bm_clear_word <= end_word;
        end
        // Times older than any constraint is long move up, so none wraps.
        if (~bt_fresh[scrub_b]) begin
            if ($signed(bt_act[scrub_b] - (e + 16'd1)) < -16'sd16384)
                bt_act[scrub_b] <= e + 16'd1 - 16'd16384;
            if ($signed(bt_colr[scrub_b] - (e + 16'd1)) < -16'sd16384)
                bt_colr[scrub_b] <= e + 16'd1 - 16'd16384;
            if ($signed(bt_prer[scrub_b] - (e + 16'd1)) < -16'sd16384)
                bt_prer[scrub_b] <= e + 16'd1 - 16'd16384;
        end
        scrub_b <= scrub_b + 1'b1;
        if ($signed(frontier - e) < 0)      frontier      <= e;
        if ($signed(batch_last - e) < 0)    batch_last    <= e;
        if ($signed(batch_floor - e) < 0)   batch_floor   <= e;
        if ($signed(read_floor - e) < 0)    read_floor    <= e;
        if ($signed(ingress_ready - e) < 0) ingress_ready <= e;
        d     <= d + 16'd1;
    endtask

    always @(posedge clk) begin
        bm_clear     <= 1'b0;
        cm_start     <= 1'b0;

        if (reset) begin
            state          <= S_IDLE;
            active         <= 1'b0;
            stat_reads     <= '0;
            stat_writes    <= '0;
            stat_hits      <= '0;
            stat_misses    <= '0;
            stat_conflicts <= '0;
            stat_latency   <= '0;
        end
        if (reset || finish) begin
            // ---- The target cycle ends: the port's state for the next ----
            if (finish && treset)
                active <= enable;
            if (reset || treset || ~active) begin
                slot_busy   <= '0;
                slots_used  <= '0;
                inflight    <= '0;
                cur_r_valid <= 1'b0;
                cur_b_valid <= 1'b0;
                tcyc        <= '0;
                rf_head     <= '0;
                rf_tail     <= '0;
                rf_count    <= '0;
                rf_ready    <= '0;
                af_head     <= '0;
                af_tail     <= '0;
                af_count    <= '0;
                st_w_valid  <= 1'b0;
                st_r_valid  <= 1'b0;
                iq_head     <= '0;
                iq_tail     <= '0;
                iq_count    <= '0;
                rb_head     <= '0;
                rb_tail     <= '0;
                rb_count    <= '0;
                wb_head     <= '0;
                wb_tail     <= '0;
                wb_count    <= '0;
                d           <= '0;
                credit      <= 32'(cfg_data_rate);
                ingress_ready <= '0;
                rq_cnt      <= '0;
                wq_cnt      <= '0;
                wmode       <= 1'b0;
                frontier    <= '0;
                batch_floor <= '0;
                batch_last  <= '0;
                read_floor  <= '0;
                bt_open     <= '0;
                bt_fresh    <= '1;
                scrub_b     <= '0;
            end else begin
                tcyc <= tcyc + 32'd1;
                // R and B, then what is ready next cycle.
                begin : release_
                    reg r_load, b_load;
                    r_load = (~cur_r_valid | r_fire) & (rf_ready != 0);
                    b_load = (~cur_b_valid | b_fire) & (af_count != 0);
                    if (r_load) begin
                        cur_r_valid <= 1'b1;
                        cur_r_slot  <= rf_slot[rf_head];
                        cur_r_id    <= slot_id[rf_slot[rf_head]];
                        rf_head     <= rf_head + 1'b1;
                        rf_count    <= rf_count - 1'b1;
                        rf_ready    <= rf_ready - 1'b1;
                    end else if (r_fire) begin
                        cur_r_valid <= 1'b0;
                    end
                    if (b_load) begin
                        cur_b_valid <= 1'b1;
                        cur_b_id    <= af_id[af_head];
                        af_head     <= af_head + 1'b1;
                        af_count    <= af_count - 1'b1;
                    end else if (b_fire) begin
                        cur_b_valid <= 1'b0;
                    end
                    if (r_fire)
                        slot_busy[cur_r_slot] <= 1'b0;
                    if (ar_fire) begin
                        slot_busy[alloc_slot] <= 1'b1;
                        slot_id[alloc_slot]   <= ar_id;
                    end
                    slots_used <= slots_used + (SW+1)'(ar_fire) - (SW+1)'(r_fire);
                    inflight <= inflight + 16'(ar_fire) + 16'(aw_fire) - 16'(new_ready) - 16'(new_acks);
                    if (ar_fire)
                        stat_latency <= stat_latency - 64'(tcyc);
                    if (r_fire)
                        stat_latency <= stat_latency + 64'(tcyc) - (ar_fire ? 64'(tcyc) : 64'd0);
                end
                st_w_valid <= new_write;
                st_w_id    <= nw_id;
                st_w_addr  <= nw_addr;
                st_r_valid <= ar_fire;
                st_r_slot  <= alloc_slot;
                st_r_addr  <= ar_addr;
                credit     <= credit + 32'(cfg_data_rate);
            end
            if (~reset)
                state <= ~(treset ? enable : active) ? S_IDLE
                       : ((~treset & active & (new_write | ar_fire)) ? S_PUSH : S_TICK);
            new_ready <= '0;
            new_acks  <= '0;
        end else begin
            case (state)
            S_PUSH: begin
                // A cycle's write reaches the channel before its read.
                if (st_w_valid) begin
                    iq_w[iq_tail]    <= 1'b1;
                    iq_tag[iq_tail]  <= TAG_W'(st_w_id);
                    iq_addr[iq_tail] <= st_w_addr;
                    iq_tail          <= iq_tail + 1'b1;
                    iq_count         <= iq_count + 1'b1;
                    st_w_valid       <= 1'b0;
                end else begin
                    if (st_r_valid) begin
                        iq_w[iq_tail]    <= 1'b0;
                        iq_tag[iq_tail]  <= TAG_W'(st_r_slot);
                        iq_addr[iq_tail] <= st_r_addr;
                        iq_tail          <= iq_tail + 1'b1;
                        iq_count         <= iq_count + 1'b1;
                        st_r_valid       <= 1'b0;
                    end
                    state <= S_TICK;
                end
            end
            S_TICK: begin
                if (credit >= 32'(k_twoclk)) begin
                    // One DRAM clock d: the ingress head enters its buffer if
                    // it may, the channel picks reads or writes, and the head
                    // of that buffer is reserved if it can be.
                    credit <= credit - 32'(k_twoclk);
                    // With nothing to reserve the clock ends now; what the
                    // tick sets below wins over the clock's end.
                    if (sel_count == 0)
                        end_clock();
                    if (move) begin
                        iq_head  <= iq_head + 1'b1;
                        iq_count <= iq_count - 1'b1;
                        ingress_ready <= d + 16'(NB);
                        if (ih_w) begin
                            wb_tag[wb_tail]  <= ih_tag;
                            wb_bank[wb_tail] <= ih_dec[BKW+ROW_W-1:ROW_W];
                            wb_row[wb_tail]  <= ih_dec[ROW_W-1:0];
                            wb_tail          <= wb_tail + 1'b1;
                            af_id[af_tail]   <= ID_W'(ih_tag);
                            af_tail          <= af_tail + 1'b1;
                            af_count         <= af_count + 1'b1;
                            new_acks         <= new_acks + 1'b1;
                        end else begin
                            rb_tag[rb_tail]  <= ih_tag;
                            rb_bank[rb_tail] <= ih_dec[BKW+ROW_W-1:ROW_W];
                            rb_row[rb_tail]  <= ih_dec[ROW_W-1:0];
                            rb_tail          <= rb_tail + 1'b1;
                        end
                    end
                    if (move & ih_w)
                        wb_count <= wb_count + 1'b1;
                    if (move & ~ih_w)
                        rb_count <= rb_count + 1'b1;
                    wmode <= wmode_n;
                    if (to_write)
                        batch_floor <= ($signed(frontier + 16'd1 - e) > 0) ? frontier + 16'd1 : e;
                    if (to_read)
                        read_floor <= batch_last + 16'd1;
                    state <= (sel_count != 0) ? S_BANK : S_TICK;
                end else begin
                    state <= S_WALK;
                end
            end
            S_BANK: begin
                // The request at the head of the buffer the channel serves,
                // against its bank.
                begin : bank_
                    reg [BKW-1:0] b;
                    reg [ROW_W-1:0] row;
                    reg w;
                    w   = wmode;
                    b   = w ? wb_bank[wb_head] : rb_bank[rb_head];
                    row = w ? wb_row[wb_head] : rb_row[rb_head];
                    op_w    <= w;
                    op_tag  <= w ? wb_tag[wb_head] : rb_tag[rb_head];
                    op_bank <= b;
                    op_row  <= row;
                    op_pc   <= (g_lpcs != 0) ? b[0] : 1'b0;
                    op_pg   <= 3'((32'(b) & ((32'd1 << (32'(g_lpcs) + 32'(g_lbgs))) - 32'd1)));
                    op_hit  <= bt_open[b] & (bt_row[b] == row);
                    op_conf <= bt_open[b] & (bt_row[b] != row);
                    r_act   <= bt_fresh[b] ? -16'sd16384 : rel(bt_act[b]);
                    r_colr  <= bt_fresh[b] ? -16'sd1 : rel(bt_colr[b]);
                    r_prer  <= bt_fresh[b] ? -16'sd1 : rel(bt_prer[b]);
                    r_floor <= rel(w ? batch_floor : read_floor);
                    kind    <= (bt_open[b] & (bt_row[b] == row)) ? K_C0
                             : (bt_open[b] ? K_PRE : K_ACT);
                end
                state <= S_SETUP;
            end
            S_SETUP: begin
                // The search's lower bound, the first clock it may place at.
                begin : setup_
                    reg signed [15:0] lb;
                    case (kind)
                    K_PRE: lb = smax(16'sd0, r_prer);
                    K_ACT: lb = smax(op_conf ? c_pre + k_rp : 16'sd0, r_act + k_rc);
                    K_C0:  lb = smax(smax(16'sd0, r_floor),
                                     op_hit ? r_colr : c_act + (op_w ? k_rcdwr : k_rcdrd));
                    default: lb = c_col0 + k_ccdl1;
                    endcase
                    if (lb >= 16'(HE)) begin
                        state <= S_END;
                    end else begin
                        s_pos   <= HLOG2'(base + HLOG2'(lb));
                        s_left  <= 16'(HE) - lb;
                        s_word  <= WW'(HLOG2'(base + HLOG2'(lb)) >> 6);
                        s_fetch <= 1'b1;
                        sb_valid <= 1'b0;
                        state   <= S_SRCH;
                    end
                end
            end
            S_SRCH: begin
                // A word a cycle: the word read now, masked to the clocks
                // left; and last cycle's word, whose first free clock wins.
                if (s_fetch) begin
                    b_cand  <= ~forb & s_mask;
                    b_word  <= s_word;
                    sb_valid <= 1'b1;
                    b_last  <= (s_left - s_take == 0);
                    s_pos   <= HLOG2'(s_pos + HLOG2'(s_take));
                    s_word  <= WW'(HLOG2'(s_pos + HLOG2'(s_take)) >> 6);
                    s_left  <= s_left - s_take;
                    s_fetch <= (s_left - s_take != 0);
                end
                if (sb_valid) begin
                    if (b_cand != 0) begin
                        case (kind)
                        K_PRE: begin c_pre <= b_rel; kind <= K_ACT; state <= S_SETUP; end
                        K_ACT: begin c_act <= b_rel; kind <= K_C0; state <= S_SETUP; end
                        K_C0:  begin
                            c_col0 <= b_rel;
                            if (NB == 2) begin
                                kind  <= K_C1;
                                state <= S_SETUP;
                            end else begin
                                state <= S_CHECK;
                            end
                        end
                        default: begin c_col1 <= b_rel; state <= S_CHECK; end
                        endcase
                    end else if (b_last) begin
                        state <= S_END;
                    end
                end
            end
            S_CHECK: begin
                // The reservation must reach no further than the horizon.
                begin : check_
                    reg signed [15:0] lastcol, last, colr, prer;
                    reg act;
                    act     = ~op_hit;
                    lastcol = (NB == 2) ? c_col1 : c_col0;
                    last    = lastcol + (op_w ? k_reach_w : k_reach_r);
                    if (~op_w)
                        last = smax(last, lastcol + k_dep);
                    if (act)
                        last = smax(last, c_act + k_reach_a);
                    colr = op_hit ? r_colr : c_act + (op_w ? k_rcdwr : k_rcdrd);
                    prer = op_hit ? r_prer : c_act + k_ras;
                    ck_lastcol <= lastcol;
                    ck_colr    <= smax(colr, lastcol + 16'sd1);
                    ck_prer    <= smax(prer, lastcol + (op_w ? k_wrpre : k_rtp));
                    state      <= (last < 16'(HE)) ? S_COMMIT : S_END;
                    cm_start   <= (last < 16'(HE));
                end
            end
            S_COMMIT: begin
                if (cm_start) begin
                    // The bank, the controller, the read's departure.
                    if (~op_hit) begin
                        bt_open[op_bank] <= 1'b1;
                        bt_row[op_bank]  <= op_row;
                        bt_act[op_bank]  <= e + c_act;
                    end
                    bt_fresh[op_bank] <= 1'b0;
                    bt_colr[op_bank]  <= e + ck_colr;
                    bt_prer[op_bank]  <= e + ck_prer;
                    if (op_w) begin
                        wq_cnt     <= wq_cnt + 8'(NB);
                        wb_head    <= wb_head + 1'b1;
                        wb_count   <= wb_count - 1'b1;
                        batch_last <= ($signed(e + ck_lastcol - batch_last) > 0) ? e + ck_lastcol : batch_last;
                        stat_writes <= stat_writes + 64'd1;
                    end else begin
                        rq_cnt     <= rq_cnt + 8'(NB);
                        rb_head    <= rb_head + 1'b1;
                        rb_count   <= rb_count - 1'b1;
                        dep_slot[HLOG2'(base + HLOG2'(ck_lastcol + k_dep))] <= SW'(op_tag);
                        stat_reads <= stat_reads + 64'd1;
                    end
                    frontier <= ($signed(e + ck_lastcol - frontier) > 0) ? e + ck_lastcol : frontier;
                    if (op_hit)
                        stat_hits <= stat_hits + 64'd1;
                    else if (op_conf)
                        stat_conflicts <= stat_conflicts + 64'd1;
                    else
                        stat_misses <= stat_misses + 64'd1;
                end else if (!cm_any) begin
                    state <= S_END;
                end
            end
            S_END: begin
                end_clock();
                state <= S_TICK;
            end
            S_WALK: begin
                // Reads that are ready in the next target cycle.
                if (rf_ready != rf_count
                 && $signed(rf_cyc[SW'(rf_head + SW'(rf_ready))] - (tcyc + 32'd1)) <= 0) begin
                    rf_ready  <= rf_ready + 1'b1;
                    new_ready <= new_ready + 1'b1;
                end else begin
                    state <= S_IDLE;
                end
            end
            default:;
            endcase
        end
    end

    // ---- Commit: ranges of clocks marked in each calendar, a word a cycle ----
    function automatic signed [15:0] clamp0(input signed [15:0] v);
        clamp0 = (v < 0) ? 16'sd0 : v;
    endfunction

    // The ranges each calendar marks, from the reservation; an empty one is
    // dropped.
    always @(*) begin
        for (integer m = 0; m < NBM; ++m) begin
            for (integer k = 0; k < 2; ++k) begin
                cm_v[m][k]  = 1'b0;
                cm_lo[m][k] = 16'sd0;
                cm_hi[m][k] = 16'sd0;
            end
        end
        // Row commands: PRE, ACT; the row bus for PRE, and ACT's one or two
        // clocks. Column commands, and their directions for the controller's
        // queues.
        cm_v[B_CMDR][0] = op_conf;  cm_lo[B_CMDR][0] = c_pre;  cm_hi[B_CMDR][0] = c_pre;
        cm_v[B_CMDR][1] = ~op_hit;  cm_lo[B_CMDR][1] = c_act;  cm_hi[B_CMDR][1] = c_act;
        cm_v[B_ROW][0]  = op_conf;  cm_lo[B_ROW][0]  = c_pre;  cm_hi[B_ROW][0]  = c_pre;
        cm_v[B_ROW][1]  = ~op_hit;  cm_lo[B_ROW][1]  = c_act;
        cm_hi[B_ROW][1] = c_act + (g_act2 ? 16'sd1 : 16'sd0);
        cm_v[B_CMDC][0] = 1'b1;     cm_lo[B_CMDC][0] = c_col0; cm_hi[B_CMDC][0] = c_col0;
        cm_v[B_CMDC][1] = (NB == 2); cm_lo[B_CMDC][1] = c_col1; cm_hi[B_CMDC][1] = c_col1;
        cm_v[op_w ? B_KW : B_KR][0] = 1'b1;
        cm_lo[op_w ? B_KW : B_KR][0] = c_col0; cm_hi[op_w ? B_KW : B_KR][0] = c_col0;
        cm_v[op_w ? B_KW : B_KR][1] = (NB == 2);
        cm_lo[op_w ? B_KW : B_KR][1] = c_col1; cm_hi[op_w ? B_KW : B_KR][1] = c_col1;
        // The read's departure.
        cm_v[B_DEPV][0] = ~op_w;
        cm_lo[B_DEPV][0] = ck_lastcol + k_dep; cm_hi[B_DEPV][0] = ck_lastcol + k_dep;
        // What each column forbids to later columns of its pseudo-channel and
        // bank group, by direction.
        for (integer k = 0; k < NB; ++k) begin
            reg signed [15:0] c;
            c = (k == 0) ? c_col0 : c_col1;
            if (~op_w) begin
                cm_v[B_FRP][k] = 1'b1; cm_lo[B_FRP][k] = clamp0(c - k_ccds + 16'sd1); cm_hi[B_FRP][k] = c + k_ccds - 16'sd1;
                cm_v[B_FWP][k] = 1'b1; cm_lo[B_FWP][k] = clamp0(c - k_wtrs + 16'sd1); cm_hi[B_FWP][k] = c + k_rtw - 16'sd1;
                cm_v[B_FRG][k] = 1'b1; cm_lo[B_FRG][k] = clamp0(c - k_ccdl + 16'sd1); cm_hi[B_FRG][k] = c + k_ccdl - 16'sd1;
                cm_v[B_FWG][k] = 1'b1; cm_lo[B_FWG][k] = clamp0(c - k_wtrl + 16'sd1); cm_hi[B_FWG][k] = c;
            end else begin
                cm_v[B_FWP][k] = 1'b1; cm_lo[B_FWP][k] = clamp0(c - k_ccds + 16'sd1); cm_hi[B_FWP][k] = c + k_ccds - 16'sd1;
                cm_v[B_FRP][k] = 1'b1; cm_lo[B_FRP][k] = clamp0(c - k_rtw + 16'sd1);  cm_hi[B_FRP][k] = c + k_wtrs - 16'sd1;
                cm_v[B_FWG][k] = 1'b1; cm_lo[B_FWG][k] = clamp0(c - k_ccdl + 16'sd1); cm_hi[B_FWG][k] = c + k_ccdl - 16'sd1;
                cm_v[B_FRG][k] = 1'b1; cm_lo[B_FRG][k] = c;                            cm_hi[B_FRG][k] = c + k_wtrl - 16'sd1;
            end
        end
        // What the ACT forbids to later ACTs: tRRD within the pseudo-channel,
        // tRRD_L within the bank group.
        cm_v[B_FAP][0] = ~op_hit;
        cm_lo[B_FAP][0] = clamp0(c_act - k_rrds + 16'sd1); cm_hi[B_FAP][0] = c_act + k_rrds - 16'sd1;
        cm_v[B_FAG][0] = ~op_hit;
        cm_lo[B_FAG][0] = clamp0(c_act - k_rrdl + 16'sd1); cm_hi[B_FAG][0] = c_act + k_rrdl - 16'sd1;
        for (integer m = 0; m < NBM; ++m)
            for (integer k = 0; k < 2; ++k)
                if (cm_hi[m][k] < cm_lo[m][k])
                    cm_v[m][k] = 1'b0;
    end

    // A word a cycle per calendar; a range's last word starts the next.
    always @(posedge clk) begin
        wr_en <= '0;
        for (integer m = 0; m < NBM; ++m) begin
            if (reset) begin
                cm_busy[m] <= 1'b0;
            end else if (cm_start) begin
                cm_busy[m] <= cm_v[m][0] | cm_v[m][1];
                cm_next[m] <= cm_v[m][0] & cm_v[m][1];
                cm_pos[m]  <= HLOG2'(base + HLOG2'(cm_v[m][0] ? cm_lo[m][0] : cm_lo[m][1]));
                cm_left[m] <= cm_v[m][0] ? cm_hi[m][0] - cm_lo[m][0] + 16'sd1
                                         : cm_hi[m][1] - cm_lo[m][1] + 16'sd1;
            end else if (cm_busy[m]) begin
                reg signed [15:0] take;
                reg [5:0] off;
                off  = cm_pos[m][5:0];
                take = (cm_left[m] >= 16'sd64 - 16'(off)) ? 16'sd64 - 16'(off) : cm_left[m];
                wr_en[m]   <= 1'b1;
                wr_word[m] <= WW'(cm_pos[m] >> 6);
                wr_mask[m] <= span_mask(off, cm_left[m]);
                if (cm_left[m] == take) begin
                    cm_busy[m] <= cm_next[m];
                    cm_next[m] <= 1'b0;
                    cm_pos[m]  <= HLOG2'(base + HLOG2'(cm_lo[m][1]));
                    cm_left[m] <= cm_hi[m][1] - cm_lo[m][1] + 16'sd1;
                end else begin
                    cm_pos[m]  <= HLOG2'(cm_pos[m] + HLOG2'(take));
                    cm_left[m] <= cm_left[m] - take;
                end
            end
        end
    end

endmodule
