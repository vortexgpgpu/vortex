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
//
// Refresh, when on, takes a DRAM clock's turn to reserve every `cfg_refi`
// clocks: a precharge and a REF, placed as a precharge is, on the next bank in
// turn or (all-bank) on every bank. For all-bank refresh the channel keeps the
// latest time a precharge and an activate allow it, over the banks opened
// since the last one, instead of reading the bank table.
module foldsim_mem_bank #(
    parameter ID_W   = 4,
    parameter SLOTS  = 32,
    parameter WSLOTS = 16,
    parameter HLOG2  = 9,
    parameter BURSTS = 2,
    parameter ROW_W  = 20,
    parameter INQ    = 64,
    // R's fire of a cycle comes with the next cycle's requests: the port's
    // state for the next cycle is then offered for either fire of this one.
    parameter SPEC   = 0,
    // The reads the port may hold (SLOTS is the slots kept, more with SPEC 2:
    // a slot R gave back is known two cycles later).
    parameter LSLOTS = SLOTS,
    parameter SW     = $clog2(SLOTS)
) (
    input  wire                  clk,
    input  wire                  reset,

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
    // Refresh: [0] on, [1] all banks at once; clocks between refreshes, and
    // clocks a refreshed bank may not activate.
    input  wire [1:0]            cfg_refresh,
    input  wire [15:0]           cfg_refi,
    input  wire [15:0]           cfg_rfc,

    // The target cycle finishes; a reset cycle drops everything.
    input  wire                  finish,
    input  wire                  treset,
    // What the cycle takes: a read, a write formed of an AW and its W, and the
    // responses R and B give.
    input  wire                  ar_fire,
    input  wire [ID_W-1:0]       ar_id,
    input  wire [29:5]           ar_addr,
    input  wire                  aw_fire,
    input  wire                  new_write,
    input  wire [ID_W-1:0]       nw_id,
    input  wire [29:5]           nw_addr,
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
    // With SPEC, the outputs above are the next cycle's if R does not fire
    // in this one, and these if it does.
    output wire                  slot_free1,
    output wire                  r_valid1,
    output wire [ID_W-1:0]       r_id1,
    output wire [SW-1:0]         r_slot1,
    // SPEC: R fired in the cycle before (its ready, for the R offered then).
    output wire                  fire_prev,
    // SPEC 2: R's ready of a cycle comes two cycles later. AR's fire this cycle
    // for either fire of R the cycle before, and R's ready of the cycle before
    // as soon as it is in, for an AR whose fire depends on it. The next
    // cycle's port for each fire of R in the cycle before and this one
    // ({before, this}): R, the room (by the one before) and the slots free.
    // A read taken (at the cycle's end, or once its AR resolves), its slot.
    input  wire [1:0]            ar_fire_c,
    input  wire                  rr_peek_valid,
    input  wire                  rr_peek,
    output wire [3:0]            c_valid,
    output wire [4*SW-1:0]       c_slot,
    // The ready queue's first four slots, the next cycle's candidates by
    // R's fire.
    output wire [4*SW-1:0]       c_slot_n,
    output wire [4*ID_W-1:0]     c_id,
    output wire [1:0]            c_room,
    output wire [3:0]            c_sf,
    output wire                  ar_take,
    output wire [SW-1:0]         ar_take_slot,
    output wire                  no_reads,
    // The target cycle's DRAM clocks are done.
    output wire                  done,

    // Reads, writes, row hits, misses, conflicts, the sum of read latencies
    // (target cycles from AR to R), refreshes, and target cycles the port
    // could take no read.
    output reg  [63:0]           stat_reads,
    output reg  [63:0]           stat_writes,
    output reg  [63:0]           stat_hits,
    output reg  [63:0]           stat_misses,
    output reg  [63:0]           stat_conflicts,
    output reg  [63:0]           stat_latency,
    output reg  [63:0]           stat_refreshes,
    output reg  [63:0]           stat_full
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

    // The model has its settings: it starts at the first target reset, after
    // the runtime has written them.
    reg active;

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
    reg signed [15:0] k_rrds, k_rrdl, k_rfc;
    reg [16:0]        k_twoclk;
    reg [2:0]         k_bb;
    reg [4:0]         k_rowsh;
    reg [24:0]        k_bmask;
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
        k_rfc     <= $signed(cfg_rfc);
        k_twoclk  <= {cfg_core_clock, 1'b0};
        k_bb      <= 3'(g_lpcs) + 3'(g_lbgs) + 3'(g_lbanks);
        k_rowsh   <= 5'(g_lpcs) + 5'(g_lbgs) + 5'(g_lbanks) + 5'(g_lrow);
        k_bmask   <= (25'd1 << (3'(g_lpcs) + 3'(g_lbgs) + 3'(g_lbanks))) - 25'd1;
    end

    // The bank and row of a channel address (sim/common/dram_bank_model.h's
    // decode): a row visit, then the bank bits, the row above the column, and
    // the bank XORed with the row folded to its width.
    // The row folded onto a bank field of each width a geometry may have, so
    // the bank's width only selects among them.
    function automatic [5:0] fold(input [ROW_W-1:0] r, input integer bb);
        fold = '0;
        for (integer k = 0; k < ROW_W; k += bb)
            fold = fold ^ 6'((32'(r) >> k) & ((32'd1 << bb) - 32'd1));
    endfunction

    function automatic [BKW+ROW_W-1:0] decode(input [29:5] a);
        reg [24:0]      tx;
        reg [24:0]      b;
        reg [ROW_W-1:0] r;
        reg [5:0]       f;
        tx = a[29:5];
        b = (tx >> g_lvisit) & k_bmask;
        r = ROW_W'(tx >> k_rowsh);
        case (k_bb)
        3'd1: f = fold(r, 1);
        3'd2: f = fold(r, 2);
        3'd3: f = fold(r, 3);
        3'd4: f = fold(r, 4);
        3'd5: f = fold(r, 5);
        default: f = fold(r, 6);
        endcase
        decode = {BKW'(b ^ 25'(f)), r};
    endfunction

    // ---- The port: slots, responses, requests in flight ----
    reg [SLOTS-1:0]  slot_busy;
    reg [ID_W-1:0]   slot_id [SLOTS];
    reg [SW:0]       slots_used;
    reg [15:0]       inflight;
    reg              cur_r_valid, cur_b_valid;
    reg [SW-1:0]     cur_r_slot;
    reg [ID_W-1:0]   cur_r_id, cur_b_id;
    // SPEC: the second candidate R and whether each takes the ready queue's
    // head; the R the target was offered in the cycle whose fire comes
    // next; the slots free for either fire.
    reg              cr1_valid, cr_pop0, cr_pop1;
    reg [SW-1:0]     cr1_slot;
    reg [ID_W-1:0]   cr1_id;
    reg              pr_valid;
    reg [SW-1:0]     pr_slot;
    reg              sf0, sf1;
    // SPEC 2: the next cycle's candidates; an AR whose fire waits for R's
    // ready of the cycle before; the port as the target saw it last cycle,
    // counted once that cycle's fire of R is known.
    reg              cv [4];
    reg [SW-1:0]     cs [4];
    reg [ID_W-1:0]   ci [4];
    reg              csf [4];
    reg              croom [2];
    reg [1:0]        ar_pend_f;
    reg [ID_W-1:0]   ar_pend_id;
    reg [BKW-1:0]    ar_pend_bank;
    reg [ROW_W-1:0]  ar_pend_row;
    reg              fk_valid, fk_room, fk_sf0, fk_sf1;
    reg [31:0]       tcyc, tcyc1;    // the target cycle, and the next
    // SPEC 2: the ready queue's head, set as each cycle starts: whether it is
    // due by R's ready of two cycles before, and its slot.
    reg              hd_due;
    reg [SW-1:0]     hd_slot;
    // Requests in flight, less the reads and writes this cycle made ready.
    reg [15:0]       ifl_eff;

    // Reads departed, in order, with the target cycle they are ready in; the
    // first `rf_ready` are ready in the next cycle. A departure enters the
    // host cycle after its DRAM clock ends.
    // Ready cycles are kept modulo 2^24: an entry not yet counted ready is at
    // most LATENCY + 1 cycles ahead.
    reg [SW-1:0]     rf_slot [SLOTS];
    reg [23:0]       rf_cyc  [SLOTS];
    reg [SW-1:0]     rf_head, rf_tail, rf_scan;   // rf_scan = rf_head + rf_ready
    reg [SW:0]       rf_count, rf_ready;
    reg              rfp_valid;
    reg [SW-1:0]     rfp_slot;
    reg [23:0]       rfp_cyc;
    // Writes the controller has buffered, to acknowledge in order.
    localparam AQW = $clog2(WSLOTS);
    reg [ID_W-1:0]   af_id [WSLOTS];
    reg [AQW-1:0]    af_head, af_tail;
    reg [AQW:0]      af_count;
    // What this target cycle's DRAM clocks made ready by the next one.
    reg [SW:0]       new_ready;
    reg [AQW:0]      new_acks;
    // The last cycle's reads taken by AR and by R, for the latency sum.
    reg              lat_ar, lat_r;
    reg [31:0]       lat_t, lat_tr;

    assign room       = (cfg_queue == 16'd0) | (inflight < cfg_queue);
    assign slot_free  = (SPEC != 0) ? sf0 : (32'(slots_used) < LSLOTS);
    assign slot_free1 = (SPEC != 0) ? sf1 : slot_free;
    assign r_valid1   = (SPEC != 0) ? cr1_valid : cur_r_valid;
    assign r_id1      = (SPEC != 0) ? cr1_id : cur_r_id;
    assign r_slot1    = (SPEC != 0) ? cr1_slot : cur_r_slot;
    assign r_valid    = cur_r_valid;
    assign r_id       = cur_r_id;
    assign r_slot     = cur_r_slot;
    assign b_valid    = cur_b_valid;
    assign b_id       = cur_b_id;
    assign no_reads   = (slots_used == 0);

    // The lowest free slot once R has taken its read.
    // SPEC: r_fire is R's ready of the cycle before, for the R offered then.
    // SPEC 2: R's ready of two cycles before, for the ready queue's head, the
    // R offered then if it was ready by then.
    wire             fire_p     = (SPEC == 2) ? (r_fire & hd_due)
                                : (SPEC != 0) ? (pr_valid & r_fire) : r_fire;
    wire [SW-1:0]    fire_slot  = (SPEC == 2) ? hd_slot : (SPEC != 0) ? pr_slot : cur_r_slot;
    assign fire_prev = fire_p;
    // The lowest slot not held, as of the host cycle before: slots change
    // hands as a target cycle ends, which the next one cannot do in the host
    // cycle after, and as a late AR resolves, which waits that host cycle.
    reg  [SW-1:0]    lowest_idle;
    reg              idle_any;
    always @(posedge clk) begin : lowest_idle_
        reg [SW-1:0] l;
        l = '0;
        for (integer s = SLOTS - 1; s >= 0; --s)
            if (~slot_busy[s])
                l = SW'(s);
        lowest_idle <= l;
        idle_any    <= ~(&slot_busy);
    end
    // The lowest free slot once R has taken its read, which frees a held one;
    // the slots as one-hot, decoded ahead of R's fire.
    wire             fire_low   = ~idle_any | (fire_slot < lowest_idle);
    assign alloc_slot = (fire_p & fire_low) ? fire_slot : lowest_idle;
    wire [SLOTS-1:0] fire_oh    = SLOTS'(1) << fire_slot;
    wire [SLOTS-1:0] idle_oh    = SLOTS'(1) << lowest_idle;
    wire [SLOTS-1:0] alloc_oh   = (fire_p & fire_low) ? fire_oh : idle_oh;
    // SPEC 2: the head as the next cycle starts, whether R fires or not: due
    // by R's ready of two cycles before that cycle, this one's less one.
    wire [SW-1:0]    rf_head1   = rf_head + 1'b1;
    // Read only as a cycle ends, a host cycle at least after the head, the
    // cycle or the queue last moved; an entry entering then is not due.
    reg              hn_due0, hn_due1;
    always @(posedge clk) begin
        hn_due0 <= (rf_count != 0) & ($signed(rf_cyc[rf_head] - 24'(tcyc - 32'd1)) <= 0);
        hn_due1 <= (rf_count > 1) & ($signed(rf_cyc[rf_head1] - 24'(tcyc - 32'd1)) <= 0);
    end
    // SPEC 2: the queue's first four slots and their ids, and the slots and
    // room each count of reads and writes would leave, read ahead of the R
    // fire that chooses among them.
    wire [SW-1:0]    rq_s [4];
    wire [ID_W-1:0]  rq_i [4];
    for (genvar k = 0; k < 4; ++k) begin : g_rq
        assign rq_s[k] = rf_slot[SW'(rf_head + SW'(k))];
        assign rq_i[k] = slot_id[rq_s[k]];
    end
    wire [4:0]       su_lt;          // slots_used + j < LSLOTS, j = -3 .. 1
    for (genvar j = 0; j < 5; ++j) begin : g_su
        assign su_lt[j] = ($signed(32'(slots_used) + j - 3) < $signed(32'(LSLOTS)));
    end
    wire [15:0]      ifl_m1 = ifl_eff - 16'd1;
    wire [15:0]      ifl_m2 = ifl_eff - 16'd2;
    wire [2:0]       if_lt;          // ifl_eff + j < cfg_queue, j = 0 .. 2
    for (genvar j = 0; j < 3; ++j) begin : g_if
        assign if_lt[j] = ((ifl_eff + 16'(j)) < cfg_queue);
    end

    // This cycle's requests, staged for the channel: a write, then a read.
    reg              st_w_valid, st_r_valid;
    reg [ID_W-1:0]   st_w_id;
    reg [SW-1:0]     st_r_slot;
    // Decoded when staged, so the clock engine's cycle carries no decode.
    reg [BKW-1:0]    st_w_bank, st_r_bank;
    reg [ROW_W-1:0]  st_w_row, st_r_row;

    // ---- The channel ----
    // Requests in the order they formed, decoded; the controller's read and
    // write buffers.
    reg              iq_w   [INQ];
    reg [TAG_W-1:0]  iq_tag [INQ];
    reg [BKW-1:0]    iq_bank[INQ];
    reg [ROW_W-1:0]  iq_row [INQ];
    reg [IQW-1:0]    iq_head, iq_tail;
    reg [IQW:0]      iq_count;
    reg [TAG_W-1:0]  rb_tag [BQ], wb_tag [BQ];
    reg [BKW-1:0]    rb_bank[BQ], wb_bank[BQ];
    reg [ROW_W-1:0]  rb_row [BQ], wb_row [BQ];
    reg [BQW-1:0]    rb_head, rb_tail, wb_head, wb_tail;
    reg [BQW:0]      rb_count, wb_count;
    // The clock each entry entered its buffer at, and the entries the
    // reservations' clock has reached.
    reg [15:0]       rb_stamp [BQ], wb_stamp [BQ];
    reg [BQW:0]      rb_vis, wb_vis;

    // Two engines share the channel. The clock engine runs each target
    // cycle's DRAM clocks with it: requests enter the buffers, writes are
    // acknowledged and reads depart. The reservations' engine follows it,
    // clock by clock, placing the buffers' heads; it may lag the clock engine
    // by less than tCL + tBL, which no departure can tell from no lag. A
    // request enters its buffer without waiting for the reservations only
    // when room is certain.
    reg [15:0]       d;              // the reservations' DRAM clock, modulo
    reg [15:0]       da;             // the clock engine's
    reg [7:0]        rq_cnt, wq_cnt; // reserved bursts whose column has not issued
    reg              wmode;
    reg [15:0]       frontier, batch_floor, batch_last, read_floor;
    // Refresh: when the next is due, and the bank it is for. All-bank: a bank
    // was opened since the last; the latest precharge and activate + tRC any
    // bank allows; and no bank activates before ab_floor.
    reg [15:0]       ref_due;
    reg [4:0]        ref_bank;
    reg              ab_open;
    reg [15:0]       ab_prer, ab_actrc, ab_floor;

    // The bank table, written at one port. A fresh bank has no row open and
    // no time that matters.
    reg [31:0]       bt_open, bt_fresh;
    reg [ROW_W-1:0]  bt_row  [32];
    reg [15:0]       bt_act  [32];
    reg [15:0]       bt_colr [32];
    reg [15:0]       bt_prer [32];
    reg [4:0]        scrub_b;

    reg  [15:0]      e;                              // the first clock a reservation may take: d + 1
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
    // The word the search reads next: its first as it starts, else the one
    // after this; read a host cycle ahead (no calendar it reads is written
    // while it runs).
    wire [WW-1:0]    s_rword;
    wire [WW-1:0]    s_rword1 = s_rword + WW'(1);
    reg              op_w;
    reg              op_ref;         // the operation is a refresh
    /* verilator lint_off UNUSEDSIGNAL */
    reg [TAG_W-1:0]  op_tag;     // a slot index may be narrower than an id
    /* verilator lint_on UNUSEDSIGNAL */
    reg [BKW-1:0]    op_bank;
    reg [ROW_W-1:0]  op_row;
    reg              op_pc;
    reg [2:0]        op_pg;

    wire [WW-1:0]    end_word = WW'(base >> 6);
    wire [WW-1:0]    end_word1 = end_word + WW'(1);
    reg  [15:0]      ea;                             // da + 1
    wire [HLOG2-1:0] base_a  = HLOG2'(ea);
    wire [WW-1:0]    end_word_a = WW'(base_a >> 6);
    wire [HLOG2-1:0] base_a1 = base_a + 1'b1;
    wire [WW-1:0]    end_word_a1 = WW'(base_a1 >> 6);
    reg              bm_clear;
    reg  [WW-1:0]    bm_clear_word;
    // A target reset (or the port leaving the bank model) clears the
    // calendars in the host cycle after it; that cycle takes no DRAM clock.
    reg              bm_clear_all;
    reg  [NBM-1:0]   wr_en;
    reg  [WW-1:0]    wr_word [NBM];
    reg  [63:0]      wr_mask [NBM];
    wire [63:0]      rd_cmdr, rd_cmdc, rd_row, rd_kr, rd_kw, rd_depv, rd_depv1;
    wire [63:0]      rd_kr1, rd_kw1;
    wire [63:0]      rd_cmd = rd_cmdr | rd_cmdc;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0]      rd_row1;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [63:0]      rd_frp, rd_fwp, rd_frg, rd_fwg, rd_fap, rd_fag;

    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_cmdr (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(s_rword), .rd_data(rd_cmdr),
        .we(wr_en[B_CMDR]), .wr_inst(1'b0), .wr_word(wr_word[B_CMDR]), .wr_mask(wr_mask[B_CMDR]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(1)) m_cmdc (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(1'b0), .rd_word(s_rword), .rd_data(rd_cmdc),
        .we(wr_en[B_CMDC]), .wr_inst(1'b0), .wr_word(wr_word[B_CMDC]), .wr_mask(wr_mask[B_CMDC]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(2)) m_row (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(2'b00), .rd_word({s_rword1, s_rword}), .rd_data({rd_row1, rd_row}),
        .we(wr_en[B_ROW]), .wr_inst(1'b0), .wr_word(wr_word[B_ROW]), .wr_mask(wr_mask[B_ROW]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(2)) m_kr (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(2'b00), .rd_word({end_word1, end_word}), .rd_data({rd_kr1, rd_kr}),
        .we(wr_en[B_KR]), .wr_inst(1'b0), .wr_word(wr_word[B_KR]), .wr_mask(wr_mask[B_KR]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(2)) m_kw (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(2'b00), .rd_word({end_word1, end_word}), .rd_data({rd_kw1, rd_kw}),
        .we(wr_en[B_KW]), .wr_inst(1'b0), .wr_word(wr_word[B_KW]), .wr_mask(wr_mask[B_KW]));
    foldsim_mem_bmap #(.INST(1), .WORDS(WORDS), .NREAD(2)) m_depv (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(2'b00), .rd_word({end_word_a1, end_word_a}), .rd_data({rd_depv1, rd_depv}),
        .we(wr_en[B_DEPV]), .wr_inst(1'b0), .wr_word(wr_word[B_DEPV]), .wr_mask(wr_mask[B_DEPV]));
    foldsim_mem_bmap #(.INST(2), .WORDS(WORDS), .NREAD(1)) m_frp (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pc), .rd_word(s_rword), .rd_data(rd_frp),
        .we(wr_en[B_FRP]), .wr_inst(op_pc), .wr_word(wr_word[B_FRP]), .wr_mask(wr_mask[B_FRP]));
    foldsim_mem_bmap #(.INST(2), .WORDS(WORDS), .NREAD(1)) m_fwp (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pc), .rd_word(s_rword), .rd_data(rd_fwp),
        .we(wr_en[B_FWP]), .wr_inst(op_pc), .wr_word(wr_word[B_FWP]), .wr_mask(wr_mask[B_FWP]));
    foldsim_mem_bmap #(.INST(8), .WORDS(WORDS), .NREAD(1)) m_frg (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pg), .rd_word(s_rword), .rd_data(rd_frg),
        .we(wr_en[B_FRG]), .wr_inst(op_pg), .wr_word(wr_word[B_FRG]), .wr_mask(wr_mask[B_FRG]));
    foldsim_mem_bmap #(.INST(8), .WORDS(WORDS), .NREAD(1)) m_fwg (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pg), .rd_word(s_rword), .rd_data(rd_fwg),
        .we(wr_en[B_FWG]), .wr_inst(op_pg), .wr_word(wr_word[B_FWG]), .wr_mask(wr_mask[B_FWG]));
    foldsim_mem_bmap #(.INST(2), .WORDS(WORDS), .NREAD(1)) m_fap (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pc), .rd_word(s_rword), .rd_data(rd_fap),
        .we(wr_en[B_FAP]), .wr_inst(op_pc), .wr_word(wr_word[B_FAP]), .wr_mask(wr_mask[B_FAP]));
    foldsim_mem_bmap #(.INST(8), .WORDS(WORDS), .NREAD(1)) m_fag (
        .clk, .reset, .clear_all(bm_clear_all), .clear(bm_clear), .clear_word(bm_clear_word),
        .rd_inst(op_pg), .rd_word(s_rword), .rd_data(rd_fag),
        .we(wr_en[B_FAG]), .wr_inst(op_pg), .wr_word(wr_word[B_FAG]), .wr_mask(wr_mask[B_FAG]));

    // The read each departure completes: a slot per clock of the ring.
    reg [SW-1:0]     dep_slot [H];
    // The read departing in the clock engine's next clock (ea), and whether
    // the ready queue's first unwalked entries are ready by the next target
    // cycle: kept as the clocks, the walk and the cycles move.
    reg              f_dep;
    reg [SW-1:0]     f_dslot;
    reg              dv0, dv1;

    // ---- The engine ----
    localparam S_IDLE = 4'd0, S_TICK = 4'd2, S_START = 4'd3,
               S_BANK2 = 4'd5, S_SETUP = 4'd6, S_C1 = 4'd7, S_SRCH = 4'd8,
               S_CHECK2 = 4'd10, S_COMMIT = 4'd11, S_END = 4'd12,
               S_RBANK = 4'd14, S_RBANK2 = 4'd15;
    localparam K_PRE = 3'd0, K_ACT = 3'd1, K_C0 = 3'd2, K_C1 = 3'd3, K_REF = 3'd4;
    reg [3:0]        state;
    localparam A_IDLE = 3'd0, A_PUSH = 3'd1, A_TICK = 3'd2, A_END = 3'd3, A_WALK = 3'd4,
               A_FAST = 3'd5, A_RES = 3'd6;
    reg [2:0]        a_state;

    // A cycle finishes once the clock engine is done with it; the
    // reservations may still follow, except into a target reset, which drops
    // everything.
    assign done = (a_state == A_IDLE) & (~treset | ((state == S_IDLE) & caught));

    // SPEC 2: an AR whose fire does not depend on R's fire the cycle before
    // is taken at its cycle's end.
    wire             sp2_now    = (ar_fire_c[0] == ar_fire_c[1]);
    wire             sp2_take   = sp2_now & ar_fire_c[0];
    wire             ar_now     = (SPEC == 2) ? sp2_take : ar_fire;
    wire             res_f1     = rr_peek & hd_due;
    // A late AR waits the host cycle the slots take to settle after the end.
    reg              ended;
    always @(posedge clk)
        ended <= finish;
    wire             res_go     = (a_state == A_RES) & rr_peek_valid & ~ended;
    wire             res_take   = res_go & ar_pend_f[res_f1];
    for (genvar x = 0; x < 4; ++x) begin : g_cand
        assign c_valid[x]               = cv[x];
        assign c_slot[x*SW +: SW]       = cs[x];
        assign c_id[x*ID_W +: ID_W]     = ci[x];
        assign c_sf[x]                  = csf[x];
        assign c_slot_n[x*SW +: SW]     = rq_s[x];
    end
    assign c_room       = {croom[1], croom[0]};
    assign ar_take      = (SPEC == 2) & ((finish & ~treset & active & sp2_take) | res_take);
    assign ar_take_slot = res_go ? lowest_idle : alloc_slot;

    // The operation: the request being reserved, and its commands, as clocks
    // relative to e.
    reg              op_hit, op_conf;
    reg signed [15:0] r_actrc, r_colr, r_prer, r_floor;   // r_actrc: the ACT time + tRC
    reg signed [15:0] c_pre, c_act, c_col0, c_col1, c_ref;
    reg [2:0]        kind;

    // The search: the word it reads, the clocks of the word before its first
    // (relative to e, modulo the ring) and the clocks of it to search. Its
    // last word is the horizon's, the clocks of which up to the horizon are
    // searched (both kept from e). The free clocks of a first column's word
    // past tCCD_L, shifted down by it, and its place, for the second.
    reg  [HLOG2-1:0]  s_wrel;
    reg  [63:0]       s_mask;
    reg  [WW-1:0]     s_lw;
    reg  [63:0]       s_hmask;
    reg  [63:0]       c1_m;
    reg signed [15:0] c1_base;        // c_col0 less its place, plus tCCD_L
    always @(posedge clk) begin : horizon_
        reg [HLOG2-1:0] last;
        last    = base + HLOG2'(HE - 1);
        s_lw    <= WW'(last >> 6);
        s_hmask <= ~(~64'd1 << last[5:0]);
    end

    // The clocks from `off` on, `left` of them at most (left >= 0).
    function automatic [63:0] span_mask(input [5:0] off, input signed [15:0] left);
        reg [6:0] top;
        top = 7'(off) + 7'(left[5:0]);
        span_mask = (~64'd0 << off)
                  & (((left[15:6] != 0) | top[6]) ? ~64'd0 : ~(~64'd0 << top[5:0]));
    endfunction

    function automatic [63:0] from_bit(input [5:0] p);
        from_bit = ~64'd0 << p;
    endfunction

    wire [63:0] f_col   = rd_cmd | (op_w ? (rd_fwp | rd_fwg) : (rd_frp | rd_frg));
    wire [63:0] f_act   = rd_cmd | rd_row | (g_act2 ? {rd_row1[0], rd_row[63:1]} : 64'd0) | rd_fap | rd_fag;
    wire [63:0] f_pre   = rd_cmd | rd_row;
    reg  [63:0] forb;
    always @(posedge clk)
        forb <= ((kind == K_PRE) | (kind == K_REF)) ? f_pre : ((kind == K_ACT) ? f_act : f_col);

    function automatic [5:0] ffs64(input [63:0] x);
        ffs64 = '0;
        for (integer j = 63; j >= 0; --j)
            if (x[j])
                ffs64 = 6'(j);
    endfunction
    wire [63:0]       s_cand = ~forb & s_mask;
    wire [5:0]        s_ffs  = ffs64(s_cand);
    wire signed [15:0] s_rel = 16'(HLOG2'(s_wrel + HLOG2'(s_ffs)));
    // The word before the first column's, relative to e: the word holding e
    // starts before it.
    wire signed [15:0] s_wrel_s = (32'(s_wrel) > HE) ? 16'(s_wrel) - 16'(H) : 16'(s_wrel);
    // The clocks of the word from its first free one on.
    function automatic [63:0] from_first(input [63:0] x);
        from_first = x;
        for (integer k = 1; k < 64; k = k * 2)
            from_first = from_first | (from_first << k);
    endfunction

    // The bounds a command's search starts from that do not depend on the
    // clock found for the one before, ahead of it; then less the time from
    // that command, for the test of which bound wins.
    reg signed [15:0] r_abf;          // rel(ab_floor)
    reg signed [15:0] t_act, t_c0;    // smax(r_actrc, r_abf), smax(0, r_floor)
    reg signed [15:0] t_act_m, t_ref_m, t_c0_m;
    wire signed [15:0] k_rcd = op_w ? k_rcdwr : k_rcdrd;
    always @(posedge clk) begin : bounds_
        r_abf   <= rel(ab_floor);
        t_act   <= smax(r_actrc, r_abf);
        t_c0    <= smax(16'sd0, r_floor);
        t_act_m <= t_act - k_rp;
        t_ref_m <= r_actrc - k_rp;
        t_c0_m  <= t_c0 - k_rcd;
    end
    // The search's lower bound for a command, as the greatest of up to three
    // terms (0 among them, or not), and its place on the ring.
    reg signed [15:0] s_lb;
    reg [HLOG2-1:0]   s_lp;
    always @(*) begin : lower_
        reg signed [15:0] a, b;
        reg               z, ab;
        z = 1'b1;
        case (kind)
        K_PRE: begin a = r_prer;  b = r_prer; end
        K_ACT: begin
            if (~op_conf) begin a = r_actrc; b = r_abf; end
            else begin a = c_pre + k_rp; b = t_act; z = 1'b0; end
        end
        K_REF: begin
            if (~op_conf) begin a = r_actrc; b = r_actrc; end
            else begin a = c_pre + k_rp; b = r_actrc; z = 1'b0; end
        end
        K_C0: begin
            if (op_hit) begin a = r_floor; b = r_colr; end
            else begin a = c_act + k_rcd; b = t_c0; z = 1'b0; end
        end
        default: begin a = c_col0 + k_ccdl1; b = a; z = 1'b0; end
        endcase
        // Which of a and b is greater, from the clock found and the bound
        // less the time from it where the clock found is one of them.
        case (kind)
        K_ACT:   ab = ~op_conf ? (r_actrc > r_abf) : (c_pre > t_act_m);
        K_REF:   ab = ~op_conf | (c_pre > t_ref_m);
        K_C0:    ab = op_hit ? (r_floor > r_colr) : ~(t_c0_m > c_act);
        default: ab = 1'b1;
        endcase
        if (ab) begin
            s_lb = (z & (a <= 0)) ? 16'sd0 : a;
            s_lp = (z & (a <= 0)) ? base : base + HLOG2'(a);
        end else begin
            s_lb = (z & (b <= 0)) ? 16'sd0 : b;
            s_lp = (z & (b <= 0)) ? base : base + HLOG2'(b);
        end
    end
    reg signed [15:0] s_lbr;
    reg [HLOG2-1:0]   s_lpr;
    assign s_rword = (state == S_START) ? WW'(s_lpr >> 6) : s_word1;

    // ---- Commit: per calendar, up to two ranges of clocks to mark ----
    // (from the reservation, registered as the horizon is checked)
    reg              rx_v    [NBM][2];
    reg              rx_cl   [NBM][2];
    reg signed [15:0] rx_c   [NBM][2];
    reg signed [15:0] rx_lo  [NBM][2];
    reg signed [15:0] rx_hi  [NBM][2];
    reg              cm_v    [NBM][2];
    // Each range's first clock on the ring, the word of its last, and its
    // length, from the reservation's clocks.
    reg [HLOG2-1:0]  cm_p    [NBM][2];
    reg [WW-1:0]     cm_x    [NBM][2];
    reg signed [15:0] cm_len [NBM][2];
    reg [HLOG2-1:0]  cm_pos  [NBM];
    reg signed [15:0] cm_left[NBM];
    reg              cm_busy [NBM];
    reg              cm_next [NBM];   // the second range is still to mark
    reg              cm_start;

    // How far the reservation reaches, and the bank's new times.
    reg signed [15:0] ck_colr, ck_prer, ck_lastcol, ck_tcol, ck_tdep, ck_tact;
    always @(*) begin : check_
        reg signed [15:0] colr, prer;
        ck_lastcol = (NB == 2) ? c_col1 : c_col0;
        colr = op_hit ? r_colr : c_act + (op_w ? k_rcdwr : k_rcdrd);
        prer = op_hit ? r_prer : c_act + k_ras;
        ck_colr = smax(colr, ck_lastcol + 16'sd1);
        ck_prer = smax(prer, ck_lastcol + (op_w ? k_wrpre : k_rtp));
        ck_tcol = ck_lastcol + (op_w ? k_reach_w : k_reach_r);
        ck_tdep = op_w ? -16'sd16384 : ck_lastcol + k_dep;
        ck_tact = op_hit ? -16'sd16384 : c_act + k_reach_a;
        // A refresh's commands are within the horizon by its search.
        if (op_ref) begin
            ck_tcol = -16'sd16384;
            ck_tdep = -16'sd16384;
            ck_tact = -16'sd16384;
        end
    end
    reg              cm_any;
    always @(*) begin
        cm_any = 1'b0;
        for (integer m = 0; m < NBM; ++m)
            cm_any = cm_any | cm_busy[m];
    end

    // Clock d+1, read when its DRAM clock starts: the read departing in it and
    // the columns issued in it -- and the columns the clock's own reservation
    // places in it.
    reg              ed_depv, ed_kr, ed_kw;
    reg [SW-1:0]     ed_slot;

    // Ingress head, and whether it enters its buffer this tick: its
    // direction kept as the queue moves.
    reg                ih_w;
    // The bursts each buffer would hold with one more entry, kept as the
    // engines change them: a reservation moves bursts from its buffer to the
    // controller's count, an entry adds its bursts, a clock's end its column.
    reg  [8:0]         wb_after, rb_after;
    // The clocks the reservations lag the clock engine, da - d, and what the
    // engines test of it, kept as either engine ends a clock.
    reg  [15:0]        lag;
    reg                a_wait_lag, caught, lag_ge1, lag_ge2, lag_ge4;
    // The queues plus the lag's columns (one a clock, at most 255).
    reg  [8:0]         lim_r, lim_w;
    // Bursts reserved and not yet issued count columns only up to the
    // reservations' clock: past it, room is an upper bound, exact once they
    // have caught up (a reservation leaves a buffer's bursts unchanged).
    // Columns already reserved in clocks d+4 .. da have issued by clock da:
    // read over the two calendar words from clock d+1 and counted three host
    // cycles late, they are still in the counts however far d moved up since.
    // Each clock issues at most one column, so bursts past the queue by more
    // than the clocks the reservations lag fill it for certain.
    reg  [8:0]         ok_r, ok_w;
    reg  [127:0]       okm_r, okm_w;           // the columns in the window
    reg  [4:0]         okc_r [8], okc_w [8];   // and their count, by 16 clocks
    function automatic [4:0] ones16(input [15:0] x);
        ones16 = '0;
        for (integer i = 0; i < 16; ++i)
            ones16 = ones16 + 5'(x[i]);
    endfunction
    always @(posedge clk) begin : col_window_
        reg [127:0] m;
        reg [6:0]   lo;
        reg [15:0]  hi;
        lo = 7'(base[5:0]) + 7'd3;
        hi = 16'(base[5:0]) + lag - 16'd1;
        for (integer i = 0; i < 128; ++i)
            m[i] = (7'(i) >= lo) & lag_ge4 & (16'(i) <= hi);
        okm_r <= {rd_kr1, rd_kr} & m;
        okm_w <= {rd_kw1, rd_kw} & m;
        for (integer g = 0; g < 8; ++g) begin
            okc_r[g] <= ones16(okm_r[16*g +: 16]);
            okc_w[g] <= ones16(okm_w[16*g +: 16]);
        end
        ok_r  <= 9'(cfg_rq) + 9'(okc_r[0]) + 9'(okc_r[1]) + 9'(okc_r[2]) + 9'(okc_r[3])
                            + 9'(okc_r[4]) + 9'(okc_r[5]) + 9'(okc_r[6]) + 9'(okc_r[7]);
        ok_w  <= 9'(cfg_wq) + 9'(okc_w[0]) + 9'(okc_w[1]) + 9'(okc_w[2]) + 9'(okc_w[3])
                            + 9'(okc_w[4]) + 9'(okc_w[5]) + 9'(okc_w[6]) + 9'(okc_w[7]);
        if (reset | bm_clear_all | ~active) begin
            okm_r <= '0;
            okm_w <= '0;
            for (integer g = 0; g < 8; ++g) begin
                okc_r[g] <= '0;
                okc_w[g] <= '0;
            end
            ok_r  <= 9'(cfg_rq);
            ok_w  <= 9'(cfg_wq);
        end
    end
    function automatic room_ok(input w);
        room_ok = w ? (wb_after <= ok_w) : (rb_after <= ok_r);
    endfunction
    function automatic room_none(input w);
        room_none = w ? (wb_after > lim_w) : (rb_after > lim_r);
    endfunction
    // da - ingress_ready: the head enters no earlier than ingress_ready.
    reg  [15:0]        ig;
    wire               ih_time = ($signed(ig) >= 0);
    wire               ih_room = room_ok(ih_w);
    wire               move    = (iq_count != 0) & ih_time & ih_room;
    wire               a_wait_room = (iq_count != 0) & ih_time & ~ih_room & ~caught & ~room_none(ih_w);

    // The fast path: the cycle's requests join the ingress queue behind its
    // head and its one DRAM clock runs, all in the host cycle after the
    // target cycle's end, whenever no wait would have held the clock.
    wire               fv_q    = (iq_count != 0);
    wire               fv_any  = fv_q | st_w_valid | st_r_valid;
    wire               fh_w    = fv_q ? ih_w : st_w_valid;
    wire [TAG_W-1:0]   fh_tag  = fv_q ? iq_tag[iq_head] : (st_w_valid ? TAG_W'(st_w_id) : TAG_W'(st_r_slot));
    wire [BKW-1:0]     fh_bank = fv_q ? iq_bank[iq_head] : (st_w_valid ? st_w_bank : st_r_bank);
    wire [ROW_W-1:0]   fh_row  = fv_q ? iq_row[iq_head] : (st_w_valid ? st_w_row : st_r_row);
    wire               fh_room = room_ok(fh_w);
    wire               f_move  = fv_any & ih_time & fh_room;
    wire               f_wait_room = fv_any & ih_time & ~fh_room & ~caught & ~room_none(fh_w);
    // The DRAM clocks' credit less one clock's, and less two.
    reg  signed [33:0] cr1, cr2;
    wire               f_clock = ~cr1[33] & cr2[33] & ~a_wait_lag & ~f_wait_room;
    // The clock's departing read, and the reads ready by the next target
    // cycle: the ready queue's first unwalked entries, then that departure
    // (ready with it when the latency is none).
    // (f_dep, f_dslot: the read departing in clock ea, kept below.)
    reg  [23:0]        f_dcyc;           // tcyc1 + cfg_latency
    wire               lat0    = (cfg_latency == 16'd0);
    wire [SW:0]        f_unw   = rf_count - rf_ready;
    function automatic due(input [23:0] cyc);
        due = ($signed(cyc - tcyc1[23:0]) <= 0);
    endfunction
    wire               f_w1    = (f_unw != 0) ? dv0 : (f_dep & lat0);
    wire               f_w2    = (f_unw > 1) ? dv1
                               : ((f_unw == 1) & f_dep & lat0);
    wire               ref_now  = cfg_refresh[0] & (cfg_refi != 16'd0) & ($signed(e - ref_due) >= 0);

    // The bank table's port: a reservation's bank, or the scrub's.
    // The scrub reads a bank in one host cycle and writes it in the next.
    reg              sc_pend;
    reg [4:0]        sc_b;
    reg [15:0]       sc_act, sc_colr, sc_prer;
    reg [ROW_W-1:0]  sc_row;
    // The reservation's, absolute; its column and precharge times relative to
    // e as checked, made absolute as it commits.
    reg [15:0]       cw_act;
    reg signed [15:0] rk_colr, rk_prer;
    reg [ROW_W-1:0]  cw_row;
    wire [15:0]      cw_colr = op_ref ? bt_colr[op_bank] : e + rk_colr;
    wire [15:0]      cw_prer = op_ref ? bt_prer[op_bank] : e + rk_prer;
    wire             bt_commit = (state == S_COMMIT) & cm_start & ~(op_ref & cfg_refresh[1]);
    wire [4:0]       bt_wa     = bt_commit ? op_bank : sc_b;
    function automatic [15:0] scrubbed(input [15:0] t);
        scrubbed = ($signed(t - (e + 16'd1)) < -16'sd16384) ? e + 16'd1 - 16'd16384 : t;
    endfunction
    always @(posedge clk) begin
        if (bt_commit | sc_pend) begin
            bt_row[bt_wa]  <= bt_commit ? cw_row  : sc_row;
            bt_act[bt_wa]  <= bt_commit ? cw_act  : sc_act;
            bt_colr[bt_wa] <= bt_commit ? cw_colr : sc_colr;
            bt_prer[bt_wa] <= bt_commit ? cw_prer : sc_prer;
        end
    end

    // The end of DRAM clock d: clock d+1's departing read and the columns it
    // issues leave the controller; then the next clock.
    // The reservations' clock d ends: the columns of clock d+1 leave the
    // controller.
    // The buffers' bursts as the reservations' clock sees them, the entry
    // reached in clock d counted, and the mode they put the channel in: a
    // clock with nothing to reserve and no refresh ends in S_TICK.
    // The entry each buffer reaches next (head + vis), kept as they move.
    // The tests of the bursts are made for either reach, ahead of it.
    // Whether the entry reached next entered in clock d (tk_*), its bursts
    // counted (tk_?n0: rq_cnt + NB * rb_vis), and the entries not yet
    // reached, kept as the clocks, the reach and the buffers move.
    reg  [BQW-1:0]   rb_reach, wb_reach;
    reg              tk_rb, tk_wb;
    reg  [8:0]       tk_rn0, tk_wn0;
    reg  [BQW:0]     rb_unr, wb_unr;
    // The column bits of clock e, kept likewise.
    reg              ek_r, ek_w;
    wire [BQW:0]     tk_rb_vis = rb_vis + (BQW+1)'(tk_rb);
    wire [BQW:0]     tk_wb_vis = wb_vis + (BQW+1)'(tk_wb);
    wire [8:0]       tk_wn1 = tk_wn0 + 9'(NB);
    wire             tk_whi = tk_wb ? (tk_wn1 >= 9'(cfg_wr_high)) : (tk_wn0 >= 9'(cfg_wr_high));
    wire             tk_wlo = tk_wb ? (tk_wn1 < 9'(cfg_wr_low)) : (tk_wn0 < 9'(cfg_wr_low));
    wire             tk_rnz = (tk_rn0 != 9'd0) | tk_rb;
    wire             tk_to_write = ~wmode & (tk_whi | ~tk_rnz);
    wire             tk_to_read  = wmode & tk_wlo & tk_rnz;
    wire             tk_wmode_n  = tk_to_write | (wmode & ~tk_to_read);
    wire             tk_sel_any  = tk_wmode_n ? ((wb_vis != 0) | tk_wb) : ((rb_vis != 0) | tk_rb);
    wire             tk_empty    = ~ref_now & ~tk_sel_any;
    // The floors, absolute and relative to e (e = d + 1).
    wire             tk_fr_up    = ($signed(frontier - d) > 0);
    wire [15:0]      tk_bfloor   = tk_fr_up ? frontier + 16'd1 : e;
    wire signed [15:0] tk_rbf    = tk_fr_up ? $signed(frontier - d) : 16'sd0;
    wire signed [15:0] tk_rbfl   = rel(batch_floor);
    wire signed [15:0] tk_rbl    = $signed(batch_last - d);
    wire signed [15:0] tk_rrf    = rel(read_floor);

    // As a clock ends, whether the next is already due (S_IDLE's test, a host
    // cycle early): the reservations then go straight on to it.
    wire next_clock = active & ~bm_clear_all & lag_ge2;

    // A search from the first clock the command may take, to the horizon.
    task automatic search(input signed [15:0] lb, input [HLOG2-1:0] pos);
        if (lb >= 16'(HE)) begin
            state <= S_END;
        end else begin
            s_word <= WW'(pos >> 6);
            s_wrel <= HLOG2'(lb) - HLOG2'(pos[5:0]);
            s_mask <= from_bit(pos[5:0]) & ((WW'(pos >> 6) == s_lw) ? s_hmask : ~64'd0);
            state  <= S_SRCH;
        end
    endtask

    task automatic end_clock();
        end_clock_cols(ed_kr, ed_kw);
    endtask

    // Clock d ends, with clock d+1's columns as given.
    task automatic end_clock_cols(input kr, input kw);
        rq_cnt <= rq_cnt - (kr ? 8'd1 : 8'd0);
        wq_cnt <= wq_cnt - (kw ? 8'd1 : 8'd0);
        // The ring's word ends with this clock: it is all past.
        if (base[5:0] == 6'd63) begin
            bm_clear      <= 1'b1;
            bm_clear_word <= end_word;
        end
        if ($signed(frontier - e) < 0)      frontier      <= e;
        if ($signed(batch_last - e) < 0)    batch_last    <= e;
        if ($signed(batch_floor - e) < 0)   batch_floor   <= e;
        if ($signed(read_floor - e) < 0)    read_floor    <= e;
        if ($signed(ab_prer - e) < 0)       ab_prer       <= e;
        if ($signed(ab_actrc - e) < 0)      ab_actrc      <= e;
        if ($signed(ab_floor - e) < 0)      ab_floor      <= e;
        d <= d + 16'd1;
        e <= e + 16'd1;
    endtask

    // The clock engine's clock da ends: clock da+1's departing read leaves
    // the controller.
    task automatic end_clock_a();
        rfp_valid <= ed_depv;
        rfp_slot  <= ed_slot;
        rfp_cyc   <= f_dcyc;
        ig        <= ($signed(ig) >= 0) ? 16'd0 : ig + 16'd1;
        da        <= da + 16'd1;
        ea        <= ea + 16'd1;
    endtask

    // A clock ends in either engine, and the columns of the reservations'.
    wire             da_inc = ((a_state == A_FAST) & f_clock) | (a_state == A_END);
    wire             d_inc  = ((state == S_TICK) & tk_empty) | ((state == S_COMMIT) & ~cm_start & ~cm_any)
                            | (state == S_END);
    wire             end_kr = (state == S_TICK) ? ek_r : ed_kr;
    wire             end_kw = (state == S_TICK) ? ek_w : ed_kw;
    wire             eng_clr = (reset | finish) & (reset | treset | ~active);
    // The lag's tests for it one less, the same, one more, and none.
    wire [15:0]      lag_c [4];
    assign lag_c[0] = lag - 16'd1;
    assign lag_c[1] = lag;
    assign lag_c[2] = lag + 16'd1;
    assign lag_c[3] = 16'd0;
    wire [3:0]       wl_c, cg_c, g1_c, g2_c, g3_c;
    wire [8:0]       lr_c [4], lw_c [4];
    for (genvar c = 0; c < 4; ++c) begin : g_lag
        wire [8:0] cols = ($signed(lag_c[c]) > 16'sd255) ? 9'd255 : 9'(lag_c[c]);
        assign wl_c[c] = ($signed(lag_c[c]) > k_dep - 16'sd1);
        assign cg_c[c] = (lag_c[c] == 16'd0);
        assign g1_c[c] = ($signed(16'd0 - lag_c[c]) < 0);
        assign g2_c[c] = ($signed(16'd1 - lag_c[c]) < 0);
        assign g3_c[c] = ($signed(lag_c[c]) >= 16'sd4);
        assign lr_c[c] = 9'(cfg_rq) + cols;
        assign lw_c[c] = 9'(cfg_wq) + cols;
    end
    always @(posedge clk) begin : lag_
        reg [1:0] c;
        c = eng_clr ? 2'd3 : ((da_inc & ~d_inc) ? 2'd2 : ((d_inc & ~da_inc) ? 2'd0 : 2'd1));
        lag        <= lag_c[c];
        a_wait_lag <= wl_c[c];
        caught     <= cg_c[c];
        lag_ge1    <= g1_c[c];
        lag_ge2    <= g2_c[c];
        lag_ge4    <= g3_c[c];
        lim_r      <= lr_c[c];
        lim_w      <= lw_c[c];
    end

    // The departure bit of a clock once this host cycle's calendar write and
    // clear are in.
    function automatic bit_after(input [3:0] m, input [63:0] rd, input [WW-1:0] w, input [5:0] b);
        if (bm_clear_all)
            bit_after = 1'b0;
        else if (wr_en[m] & (wr_word[m] == w))
            bit_after = rd[b] | wr_mask[m][b];
        else if (bm_clear & (bm_clear_word == w))
            bit_after = 1'b0;
        else
            bit_after = rd[b];
    endfunction
    // The clock engine moves an entry into a buffer.
    wire             fast_mv  = (a_state == A_FAST) & f_clock & f_move;
    wire             tick_mv  = (a_state == A_TICK) & ~cr1[33] & ~a_wait_lag & ~a_wait_room & move;
    wire             rb_add_w = (fast_mv & ~fh_w) | (tick_mv & ~ih_w);
    wire             wb_add_w = (fast_mv & fh_w) | (tick_mv & ih_w);
    wire             lag_is1  = lag_ge1 & ~lag_ge2;
    // The walk moves a step, the cycle ends, a departure enters the queue.
    wire             rf_adv = ((a_state == A_FAST) & f_clock & f_w1)
                            | ((a_state == A_WALK) & (rf_ready != rf_count) & dv0);
    wire             t_next = finish & ~reset & ~treset & active;
    wire             rf_wr  = rfp_valid | ((a_state == A_FAST) & f_clock & f_dep);
    wire [2:0]       sc_due, sc_aft;
    wire [1:0]       sc_due1;
    for (genvar k = 0; k < 3; ++k) begin : g_scan
        wire [SW-1:0] i = rf_scan + SW'(k);
        assign sc_due[k]  = due(rf_cyc[i]);
        if (k < 2) begin : g_due1
            assign sc_due1[k] = ($signed(rf_cyc[i] - (tcyc1[23:0] + 24'd1)) <= 0);
        end
        // A departure entered now is ready no sooner than its latency.
        assign sc_aft[k]  = (rf_wr & (rf_tail == i)) ? lat0 : sc_due[k];
    end
    always @(posedge clk) begin : ingress_head_
        reg [IQW-1:0] h;
        reg           p0v, p1v, took, pop;
        p0v  = st_w_valid | st_r_valid;
        p1v  = st_w_valid & st_r_valid;
        took = f_clock & f_move & ~fv_q;
        pop  = (fast_mv & fv_q) | tick_mv;
        h    = iq_head + IQW'(pop);
        if ((a_state == A_FAST) & p0v & (iq_tail == h))
            ih_w <= ~(took & p1v) & st_w_valid;
        else if ((a_state == A_FAST) & p1v & (IQW'(iq_tail + 1'b1) == h))
            ih_w <= 1'b0;
        else if ((a_state == A_PUSH) & p0v & (iq_tail == h))
            ih_w <= st_w_valid;
        else
            ih_w <= iq_w[h];
    end

    always @(posedge clk) begin : clock_ahead_
        // Clock ea moves up in A_FAST too, which ends in no state that
        // reads these in the host cycle after.
        f_dep   <= eng_clr ? 1'b0
                 : ((a_state == A_END) ? bit_after(B_DEPV, rd_depv1, end_word_a1, base_a1[5:0])
                                       : bit_after(B_DEPV, rd_depv, end_word_a, base_a[5:0]));
        f_dslot <= (a_state == A_END) ? dep_slot[base_a1] : dep_slot[base_a];
        // Clock e's column bits, and the reach, as the next host cycle starts.
        begin : reach_
            reg [HLOG2-1:0] b1;
            reg             rtk, wtk, sw, sr;
            reg [15:0]      rst, wst;
            b1   = base + 1'b1;
            ek_r <= eng_clr ? 1'b0
                  : (d_inc ? bit_after(B_KR, (b1[5:0] == 0) ? rd_kr1 : rd_kr, WW'(b1 >> 6), b1[5:0])
                           : bit_after(B_KR, rd_kr, end_word, base[5:0]));
            ek_w <= eng_clr ? 1'b0
                  : (d_inc ? bit_after(B_KW, (b1[5:0] == 0) ? rd_kw1 : rd_kw, WW'(b1 >> 6), b1[5:0])
                           : bit_after(B_KW, rd_kw, end_word, base[5:0]));
            rtk = (state == S_TICK) & tk_rb;
            wtk = (state == S_TICK) & tk_wb;
            // The entry the reach will be at entered in the clock d will be:
            // read, or entering now, with clock da.
            rst = rb_stamp[BQW'(rb_reach + BQW'(rtk))];
            wst = wb_stamp[BQW'(wb_reach + BQW'(wtk))];
            sr  = (rb_add_w & (rb_tail == BQW'(rb_reach + BQW'(rtk))))
                ? (d_inc ? lag_is1 : caught)
                : (d_inc ? (rst == e) : (rst == d));
            sw  = (wb_add_w & (wb_tail == BQW'(wb_reach + BQW'(wtk))))
                ? (d_inc ? lag_is1 : caught)
                : (d_inc ? (wst == e) : (wst == d));
            rb_unr <= eng_clr ? '0 : rb_unr + (BQW+1)'(rb_add_w) - (BQW+1)'(rtk);
            wb_unr <= eng_clr ? '0 : wb_unr + (BQW+1)'(wb_add_w) - (BQW+1)'(wtk);
            tk_rb  <= ~eng_clr & (rb_add_w | ((rb_unr - (BQW+1)'(rtk)) != 0)) & sr;
            tk_wb  <= ~eng_clr & (wb_add_w | ((wb_unr - (BQW+1)'(wtk)) != 0)) & sw;
            // A reservation moves bursts from its buffer to the controller's
            // count: only the reach and clocks' columns change these.
            tk_rn0 <= eng_clr ? 9'd0 : tk_rn0 + (rtk ? 9'(NB) : 9'd0) - ((d_inc & end_kr) ? 9'd1 : 9'd0);
            tk_wn0 <= eng_clr ? 9'd0 : tk_wn0 + (wtk ? 9'(NB) : 9'd0) - ((d_inc & end_kw) ? 9'd1 : 9'd0);
        end
        dv0     <= eng_clr ? 1'b0 : (t_next ? sc_due1[0] : (rf_adv ? sc_aft[1] : sc_aft[0]));
        dv1     <= eng_clr ? 1'b0 : (t_next ? sc_due1[1] : (rf_adv ? sc_aft[2] : sc_aft[1]));
    end

    always @(posedge clk) begin : engines
        // A buffer may gain an entry from the clock engine and lose one to a
        // reservation in the same host cycle.
        reg rb_add, rb_sub, wb_add, wb_sub;
        rb_add = 1'b0;
        rb_sub = 1'b0;
        wb_add = 1'b0;
        wb_sub = 1'b0;
        bm_clear     <= 1'b0;
        bm_clear_all <= 1'b0;
        cm_start     <= 1'b0;
        rfp_valid    <= 1'b0;
        sc_pend      <= 1'b0;

        if (rfp_valid) begin
            rf_slot[rf_tail] <= rfp_slot;
            rf_cyc[rf_tail]  <= rfp_cyc;
            rf_tail          <= rf_tail + 1'b1;
            rf_count         <= rf_count + 1'b1;
        end
        lat_ar <= 1'b0;
        lat_r  <= 1'b0;
        if (lat_ar | lat_r)
            stat_latency <= stat_latency + (lat_r ? 64'(lat_tr) : 64'd0) - (lat_ar ? 64'(lat_t) : 64'd0);

        // The engine runs while the target evaluates its cycle; the cycle
        // finishes only once the engine is idle.
        // ---- The clock engine ----
        case (a_state)
        A_RES: begin
            // The cycle's AR, once R's ready of the cycle before is in.
            if (res_go) begin
                if (res_take) begin
                    slot_busy[lowest_idle] <= 1'b1;
                    slot_id[lowest_idle]   <= ar_pend_id;
                    slots_used <= slots_used + 1'b1;
                    inflight   <= inflight + 16'd1;
                    ifl_eff    <= ifl_eff + 16'd1;
                    st_r_valid <= 1'b1;
                    st_r_slot  <= lowest_idle;
                    st_r_bank  <= ar_pend_bank;
                    st_r_row   <= ar_pend_row;
                    lat_ar     <= 1'b1;
                    lat_t      <= tcyc - 32'd1;
                end
                a_state <= A_FAST;
            end
        end
        A_FAST: begin
            begin : fast_
                // The staged requests in order, the write first; with the
                // clock, a head taken from them is not appended.
                reg              p0v, p1v, p0w;
                reg [TAG_W-1:0]  p0t;
                reg [BKW-1:0]    p0b;
                reg [ROW_W-1:0]  p0r;
                reg              took, app0;
                reg [IQW-1:0]    t;
                reg [IQW:0]      n;
                p0v  = st_w_valid | st_r_valid;
                p1v  = st_w_valid & st_r_valid;
                p0w  = st_w_valid;
                p0t  = st_w_valid ? TAG_W'(st_w_id) : TAG_W'(st_r_slot);
                p0b  = st_w_valid ? st_w_bank : st_r_bank;
                p0r  = st_w_valid ? st_w_row : st_r_row;
                took = f_clock & f_move & ~fv_q;
                // Both are written past the tail, the read over the write if
                // the write was taken: only the tail depends on the clock
                // (the queue never holds more than the requests in flight).
                if (p0v) begin
                    iq_w[iq_tail]    <= (took & p1v) ? 1'b0 : p0w;
                    iq_tag[iq_tail]  <= (took & p1v) ? TAG_W'(st_r_slot) : p0t;
                    iq_bank[iq_tail] <= (took & p1v) ? st_r_bank : p0b;
                    iq_row[iq_tail]  <= (took & p1v) ? st_r_row : p0r;
                end
                if (p1v) begin
                    iq_w[IQW'(iq_tail + 1'b1)]    <= 1'b0;
                    iq_tag[IQW'(iq_tail + 1'b1)]  <= TAG_W'(st_r_slot);
                    iq_bank[IQW'(iq_tail + 1'b1)] <= st_r_bank;
                    iq_row[IQW'(iq_tail + 1'b1)]  <= st_r_row;
                end
                app0 = p0v & ~took;
                t    = iq_tail + IQW'(app0) + IQW'(p1v);
                n    = iq_count + (IQW+1)'(app0) + (IQW+1)'(p1v);
                st_w_valid <= 1'b0;
                st_r_valid <= 1'b0;
                if (f_clock) begin
                    cr1     <= cr1 - 34'(k_twoclk);
                    cr2     <= cr2 - 34'(k_twoclk);
                    ifl_eff <= (f_move & fh_w & f_w1) ? ifl_m2 : ((f_move & fh_w) | f_w1) ? ifl_m1 : ifl_eff;
                    if (f_move) begin
                        if (fv_q) begin
                            iq_head <= iq_head + 1'b1;
                            n = n - 1'b1;
                        end
                        ig <= 16'd1 - 16'(NB);
                        if (fh_w) begin
                            wb_tag[wb_tail]   <= fh_tag;
                            wb_bank[wb_tail]  <= fh_bank;
                            wb_row[wb_tail]   <= fh_row;
                            wb_stamp[wb_tail] <= da;
                            wb_tail           <= wb_tail + 1'b1;
                            wb_add             = 1'b1;
                            af_id[af_tail]    <= ID_W'(fh_tag);
                            af_tail           <= af_tail + 1'b1;
                            af_count          <= af_count + 1'b1;
                            new_acks          <= new_acks + 1'b1;
                        end else begin
                            rb_tag[rb_tail]   <= fh_tag;
                            rb_bank[rb_tail]  <= fh_bank;
                            rb_row[rb_tail]   <= fh_row;
                            rb_stamp[rb_tail] <= da;
                            rb_tail           <= rb_tail + 1'b1;
                            rb_add             = 1'b1;
                        end
                    end else begin
                        ig <= ($signed(ig) >= 0) ? 16'd0 : ig + 16'd1;
                    end
                    // The clock ends: its departing read joins the ready
                    // queue, and the walk takes what is ready next cycle.
                    if (f_dep) begin
                        rf_slot[rf_tail] <= f_dslot;
                        rf_cyc[rf_tail]  <= f_dcyc;
                        rf_tail          <= rf_tail + 1'b1;
                        rf_count         <= rf_count + 1'b1;
                    end
                    da <= da + 16'd1;
                    ea <= ea + 16'd1;
                    if (f_w1) begin
                        rf_ready  <= rf_ready + 1'b1;
                        rf_scan   <= rf_scan + 1'b1;
                        new_ready <= new_ready + 1'b1;
                    end
                    a_state <= (f_w1 & f_w2) ? A_WALK : A_IDLE;
                end else begin
                    a_state <= A_TICK;
                end
                iq_tail  <= t;
                iq_count <= n;
            end
        end
        A_PUSH: begin
            // A cycle's write reaches the channel before its read.
            if (st_w_valid) begin
                iq_w[iq_tail]    <= 1'b1;
                iq_tag[iq_tail]  <= TAG_W'(st_w_id);
                iq_bank[iq_tail] <= st_w_bank;
                iq_row[iq_tail]  <= st_w_row;
                iq_tail          <= iq_tail + 1'b1;
                iq_count         <= iq_count + 1'b1;
                st_w_valid       <= 1'b0;
            end else begin
                if (st_r_valid) begin
                    iq_w[iq_tail]    <= 1'b0;
                    iq_tag[iq_tail]  <= TAG_W'(st_r_slot);
                    iq_bank[iq_tail] <= st_r_bank;
                    iq_row[iq_tail]  <= st_r_row;
                    iq_tail          <= iq_tail + 1'b1;
                    iq_count         <= iq_count + 1'b1;
                    st_r_valid       <= 1'b0;
                end
                a_state <= A_TICK;
            end
        end
        A_TICK: begin
            if (~cr1[33]) begin
                // One DRAM clock da, once the reservations are close enough
                // behind: the ingress head enters its buffer if it may.
                if (~a_wait_lag & ~a_wait_room) begin
                    cr1 <= cr1 - 34'(k_twoclk);
                    cr2 <= cr2 - 34'(k_twoclk);
                    if (move) begin
                        iq_head  <= iq_head + 1'b1;
                        iq_count <= iq_count - 1'b1;
                        ig       <= 16'd0 - 16'(NB);
                        if (ih_w) begin
                            wb_tag[wb_tail]   <= iq_tag[iq_head];
                            wb_bank[wb_tail]  <= iq_bank[iq_head];
                            wb_row[wb_tail]   <= iq_row[iq_head];
                            wb_stamp[wb_tail] <= da;
                            wb_tail           <= wb_tail + 1'b1;
                            wb_add             = 1'b1;
                            af_id[af_tail]    <= ID_W'(iq_tag[iq_head]);
                            af_tail           <= af_tail + 1'b1;
                            af_count          <= af_count + 1'b1;
                            new_acks          <= new_acks + 1'b1;
                            ifl_eff           <= ifl_eff - 16'd1;
                        end else begin
                            rb_tag[rb_tail]   <= iq_tag[iq_head];
                            rb_bank[rb_tail]  <= iq_bank[iq_head];
                            rb_row[rb_tail]   <= iq_row[iq_head];
                            rb_stamp[rb_tail] <= da;
                            rb_tail           <= rb_tail + 1'b1;
                            rb_add             = 1'b1;
                        end
                    end
                    ed_depv <= f_dep;
                    ed_slot <= f_dslot;
                    a_state <= A_END;
                end
            end else begin
                a_state <= A_WALK;
            end
        end
        A_END: begin
            end_clock_a();
            a_state <= A_TICK;
        end
        A_WALK: begin
            // Reads that are ready in the next target cycle.
            if (rf_ready != rf_count && dv0) begin
                rf_ready  <= rf_ready + 1'b1;
                rf_scan   <= rf_scan + 1'b1;
                new_ready <= new_ready + 1'b1;
                ifl_eff   <= ifl_eff - 16'd1;
            end else begin
                a_state <= A_IDLE;
            end
        end
        default:;
        endcase

        // ---- The reservations' engine ----
        case (state)
        S_IDLE: begin
            // The next clock, once the clock engine has had its ingress.
            if (active & ~bm_clear_all & lag_ge1)
                state <= S_TICK;
        end
        S_TICK: begin
            // Clock d: the entry that entered a buffer in it, if any, is
            // reached; clock d+1's columns are read; a bank's times move up
            // (the table's port is free).
            if (tk_rb) begin
                rb_vis   <= tk_rb_vis;
                rb_reach <= rb_reach + 1'b1;
            end
            if (tk_wb) begin
                wb_vis   <= tk_wb_vis;
                wb_reach <= wb_reach + 1'b1;
            end
            ed_kr   <= ek_r;
            ed_kw   <= ek_w;
            sc_pend <= ~bt_fresh[scrub_b];
            sc_b    <= scrub_b;
            sc_row  <= bt_row[scrub_b];
            sc_act  <= scrubbed(bt_act[scrub_b]);
            sc_colr <= scrubbed(bt_colr[scrub_b]);
            sc_prer <= scrubbed(bt_prer[scrub_b]);
            scrub_b <= scrub_b + 1'b1;
            if (tk_empty) begin
                // A clock with nothing to reserve.
                wmode <= tk_wmode_n;
                if (tk_to_write)
                    batch_floor <= tk_bfloor;
                if (tk_to_read)
                    read_floor <= batch_last + 16'd1;
                end_clock_cols(ek_r, ek_w);
                state <= next_clock ? S_TICK : S_IDLE;
            end else begin
                // The channel picks reads or writes, and the head of that
                // buffer is to be reserved (or the refresh).
                wmode <= tk_wmode_n;
                if (tk_to_write)
                    batch_floor <= tk_bfloor;
                if (tk_to_read)
                    read_floor <= batch_last + 16'd1;
                if (ref_now) begin
                    state <= S_RBANK;
                end else begin
                    op_ref  <= 1'b0;
                    op_w    <= tk_wmode_n;
                    op_tag  <= tk_wmode_n ? wb_tag[wb_head] : rb_tag[rb_head];
                    op_bank <= tk_wmode_n ? wb_bank[wb_head] : rb_bank[rb_head];
                    op_row  <= tk_wmode_n ? wb_row[wb_head] : rb_row[rb_head];
                    r_floor <= tk_wmode_n ? (tk_to_write ? tk_rbf : tk_rbfl) : (tk_to_read ? tk_rbl : tk_rrf);
                    state   <= S_BANK2;
                end
            end
        end
        S_BANK2: begin : bank2_
            // Against its bank; then the first command's bound.
            reg               open, hit;
            open = bt_open[op_bank];
            hit  = open & (bt_row[op_bank] == op_row);
            op_pc   <= (g_lpcs != 0) ? op_bank[0] : 1'b0;
            op_pg   <= 3'((32'(op_bank) & ((32'd1 << (32'(g_lpcs) + 32'(g_lbgs))) - 32'd1)));
            op_hit  <= hit;
            op_conf <= open & ~hit;
            r_actrc <= (bt_fresh[op_bank] ? -16'sd16384 : rel(bt_act[op_bank])) + k_rc;
            r_colr  <= bt_fresh[op_bank] ? -16'sd1 : rel(bt_colr[op_bank]);
            r_prer  <= bt_fresh[op_bank] ? -16'sd1 : rel(bt_prer[op_bank]);
            kind    <= hit ? K_C0 : (open ? K_PRE : K_ACT);
            state   <= S_SETUP;
        end
        S_SETUP: begin
            s_lbr <= s_lb;
            s_lpr <= s_lp;
            state <= S_START;
        end
        S_START: begin
            search(s_lbr, s_lpr);
        end
        S_SRCH: begin
            // A word a cycle, masked to the clocks left: its first free clock
            // wins.
            // The next word is read whatever this one holds: a search that
            // found its clock restarts from its own bound.
            s_word <= s_word1;
            s_wrel <= s_wrel + HLOG2'(64);
            s_mask <= (s_word1 == s_lw) ? s_hmask : ~64'd0;
            if (s_cand != 0) begin
                case (kind)
                K_PRE: begin c_pre <= s_rel; kind <= op_ref ? K_REF : K_ACT; state <= S_SETUP; end
                K_REF: begin c_ref <= s_rel; state <= S_CHECK2; end
                K_ACT: begin c_act <= s_rel; kind <= K_C0; state <= S_SETUP; end
                K_C0:  begin
                    c_col0  <= s_rel;
                    c1_m    <= (($signed(k_ccdl1) >= 16'sd64) ? 64'd0 : (s_cand >> k_ccdl1[5:0]))
                             & from_first(s_cand);
                    c1_base <= s_wrel_s + k_ccdl1;
                    state   <= (NB == 2) ? S_C1 : S_CHECK2;
                end
                default: begin c_col1 <= s_rel; state <= S_CHECK2; end
                endcase
            end else if (s_word == s_lw) begin
                state <= S_END;
            end
        end
        S_C1: begin
            // The second column in the first's word, or a search from it.
            if (c1_m != 0) begin
                c_col1 <= c1_base + 16'(ffs64(c1_m));
                state  <= S_CHECK2;
            end else begin
                kind  <= K_C1;
                state <= S_SETUP;
            end
        end
        S_CHECK2: begin
            // The reservation must reach no further than the horizon.
            for (integer m = 0; m < NBM; ++m) begin
                for (integer k = 0; k < 2; ++k) begin
                    reg clamped;
                    clamped      = rx_cl[m][k] & ($signed(rx_c[m][k] + rx_lo[m][k]) < 0);
                    cm_v[m][k]   <= rx_v[m][k] & (rx_hi[m][k] >= rx_lo[m][k])
                                  & (~rx_cl[m][k] | ($signed(rx_c[m][k] + rx_hi[m][k]) >= 0));
                    cm_p[m][k]   <= clamped ? base : base + HLOG2'(rx_c[m][k]) + HLOG2'(rx_lo[m][k]);
                    cm_x[m][k]   <= WW'(HLOG2'(base + HLOG2'(rx_c[m][k]) + HLOG2'(rx_hi[m][k])) >> 6);
                    cm_len[m][k] <= clamped ? rx_c[m][k] + rx_hi[m][k] + 16'sd1
                                            : rx_hi[m][k] - rx_lo[m][k] + 16'sd1;
                end
            end
            cw_row  <= (op_hit | op_ref) ? bt_row[op_bank] : op_row;
            cw_act  <= op_ref ? e + c_ref + k_rfc - k_rc : (op_hit ? bt_act[op_bank] : e + c_act);
            rk_colr <= ck_colr;
            rk_prer <= ck_prer;
            begin : check2_
                reg ok;
                ok = (ck_tcol < 16'(HE)) & (ck_tdep < 16'(HE)) & (ck_tact < 16'(HE));
                state    <= ok ? S_COMMIT : S_END;
                cm_start <= ok;
            end
        end
        S_COMMIT: begin
            if (cm_start & op_ref) begin
                // A refresh closes its bank (every bank), and the next is due.
                if (cfg_refresh[1]) begin
                    bt_open  <= '0;
                    ab_open  <= 1'b0;
                    ab_floor <= e + c_ref + k_rfc;
                    ab_actrc <= e + c_ref + k_rfc;
                end else begin
                    bt_open[op_bank]  <= 1'b0;
                    bt_fresh[op_bank] <= 1'b0;
                    ref_bank <= (ref_bank + 5'd1) & k_bmask[4:0];
                end
                ref_due        <= ref_due + cfg_refi;
                stat_refreshes <= stat_refreshes + 64'd1;
            end else if (cm_start) begin
                // The bank (its times through the table's port), the
                // controller, the read's departure, and a column in clock d+1
                // for the clock's end.
                if (~op_hit)
                    bt_open[op_bank] <= 1'b1;
                bt_fresh[op_bank] <= 1'b0;
                ab_open <= 1'b1;
                if ($signed(cw_prer - ab_prer) > 0)
                    ab_prer <= cw_prer;
                if (~op_hit && $signed(e + c_act + k_rc - ab_actrc) > 0)
                    ab_actrc <= e + c_act + k_rc;
                if (op_w) begin
                    wq_cnt     <= wq_cnt + 8'(NB);
                    wb_head    <= wb_head + 1'b1;
                    wb_vis     <= wb_vis - 1'b1;
                    wb_sub      = 1'b1;
                    batch_last <= ($signed(e + ck_lastcol - batch_last) > 0) ? e + ck_lastcol : batch_last;
                    stat_writes <= stat_writes + 64'd1;
                    if (c_col0 == 0) ed_kw <= 1'b1;
                end else begin
                    rq_cnt     <= rq_cnt + 8'(NB);
                    rb_head    <= rb_head + 1'b1;
                    rb_vis     <= rb_vis - 1'b1;
                    rb_sub      = 1'b1;
                    dep_slot[cm_p[B_DEPV][0]] <= SW'(op_tag);
                    stat_reads <= stat_reads + 64'd1;
                    if (c_col0 == 0) ed_kr <= 1'b1;
                end
                frontier <= ($signed(e + ck_lastcol - frontier) > 0) ? e + ck_lastcol : frontier;
                if (op_hit)
                    stat_hits <= stat_hits + 64'd1;
                else if (op_conf)
                    stat_conflicts <= stat_conflicts + 64'd1;
                else
                    stat_misses <= stat_misses + 64'd1;
            end else if (!cm_any) begin
                end_clock();
                state <= next_clock ? S_TICK : S_IDLE;
            end
        end
        S_END: begin
            end_clock();
            state <= next_clock ? S_TICK : S_IDLE;
        end
        S_RBANK: begin
            // A refresh: the bank in turn.
            op_ref  <= 1'b1;
            op_w    <= 1'b0;
            op_bank <= ref_bank;
            state   <= S_RBANK2;
        end
        S_RBANK2: begin
            // Precharged first if open, then REF: against the bank, or the
            // channel's latest times for all banks.
            op_hit <= 1'b0;
            if (cfg_refresh[1]) begin
                op_conf <= ab_open;
                r_prer  <= rel(ab_prer);
                r_actrc <= rel(ab_actrc);
                kind    <= ab_open ? K_PRE : K_REF;
            end else begin
                op_conf <= bt_open[op_bank];
                r_prer  <= bt_fresh[op_bank] ? -16'sd1 : rel(bt_prer[op_bank]);
                r_actrc <= (bt_fresh[op_bank] ? -16'sd16384 : rel(bt_act[op_bank])) + k_rc;
                kind    <= bt_open[op_bank] ? K_PRE : K_REF;
            end
            state <= S_SETUP;
        end
        default:;
        endcase

        rb_count <= rb_count + (BQW+1)'(rb_add) - (BQW+1)'(rb_sub);
        wb_count <= wb_count + (BQW+1)'(wb_add) - (BQW+1)'(wb_sub);
        // A reservation's bursts move from its buffer to the controller's
        // count: only entries and clocks' columns change the bursts held.
        rb_after <= rb_after + (rb_add ? 9'(NB) : 9'd0) - ((d_inc & end_kr) ? 9'd1 : 9'd0);
        wb_after <= wb_after + (wb_add ? 9'(NB) : 9'd0) - ((d_inc & end_kw) ? 9'd1 : 9'd0);

        if (reset || finish) begin
            // ---- The target cycle ends: the port's state for the next ----
            if (finish && treset)
                active <= 1'b1;
            if (reset || treset || ~active) begin
                slot_busy   <= '0;
                slots_used  <= '0;
                inflight    <= '0;
                ifl_eff     <= '0;
                hd_due      <= 1'b0;
                cur_r_valid <= 1'b0;
                cur_b_valid <= 1'b0;
                cr1_valid   <= 1'b0;
                for (integer x = 0; x < 4; ++x) begin
                    cv[x]  <= 1'b0;
                    csf[x] <= 1'b1;
                end
                croom[0]    <= 1'b1;
                croom[1]    <= 1'b1;
                fk_valid    <= 1'b0;
                cr_pop0     <= 1'b0;
                cr_pop1     <= 1'b0;
                pr_valid    <= 1'b0;
                sf0         <= 1'b1;
                sf1         <= 1'b1;
                tcyc        <= '0;
                tcyc1       <= 32'd1;
                f_dcyc      <= 24'(32'd1 + 32'(cfg_latency));
                rf_head     <= '0;
                rf_tail     <= '0;
                rf_count    <= '0;
                rf_ready    <= '0;
                rf_scan     <= '0;
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
                da          <= '0;
                e           <= 16'd1;
                ea          <= 16'd1;
                ig          <= '0;
                rb_after    <= 9'(NB);
                wb_after    <= 9'(NB);
                rb_vis      <= '0;
                wb_vis      <= '0;
                rb_reach    <= '0;
                wb_reach    <= '0;
                state       <= S_IDLE;
                cr1         <= 34'(cfg_data_rate) - 34'(k_twoclk);
                cr2         <= 34'(cfg_data_rate) - 34'(k_twoclk) - 34'(k_twoclk);
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
                bm_clear_all <= 1'b1;
                ref_due     <= cfg_refi;
                ref_bank    <= '0;
                ab_open     <= 1'b0;
                ab_prer     <= '0;
                ab_actrc    <= '0;
                ab_floor    <= '0;
            end else begin
                tcyc  <= tcyc1;
                tcyc1 <= tcyc1 + 32'd1;
                f_dcyc <= 24'(tcyc1 + 32'd1 + 32'(cfg_latency));
                // The port as the target saw it this cycle.
                if (SPEC == 2) begin
                    // Last cycle's, now that its fire of R is known.
                    if (fk_valid & ~(fk_room & (fire_p ? fk_sf1 : fk_sf0)))
                        stat_full <= stat_full + 64'd1;
                    fk_valid <= 1'b1;
                    fk_room  <= croom[fire_p];
                    fk_sf0   <= csf[{fire_p, 1'b0}];
                    fk_sf1   <= csf[{fire_p, 1'b1}];
                end else if (~(room & ((SPEC != 0) ? (fire_p ? sf1 : sf0) : slot_free))) begin
                    stat_full <= stat_full + 64'd1;
                end
                // R and B, then what is ready next cycle.
                if (SPEC == 2) begin : release_spec2_
                    // R fired two cycles before leaves the ready queue; then
                    // the next cycle's candidates, the n-th ready read for n
                    // fires in the cycle before and this one.
                    reg              pop;
                    reg [SW-1:0]     hd;
                    reg [SW:0]       rdy, used0;
                    reg              arf [2];
                    pop    = fire_p;
                    hd     = pop ? rf_head1 : rf_head;
                    rdy    = rf_ready - (pop ? (SW+1)'(1) : (SW+1)'(0));
                    used0  = slots_used - (pop ? (SW+1)'(1) : (SW+1)'(0));
                    arf[0] = sp2_now ? sp2_take : ar_fire_c[0];
                    arf[1] = sp2_now ? sp2_take : ar_fire_c[1];
                    for (integer x = 0; x < 4; ++x) begin
                        reg [1:0] n;
                        reg [2:0] j;
                        n = 2'(x >> 1) + 2'(x & 1);
                        j = 3'd3 - 3'(pop) + 3'(arf[x >> 1]) - 3'(x >> 1) - 3'(x & 1);
                        cv[x]  <= (32'(n) < 32'(rdy));
                        cs[x]  <= rq_s[2'(n) + 2'(pop)];
                        ci[x]  <= rq_i[2'(n) + 2'(pop)];
                        csf[x] <= su_lt[j];
                    end
                    for (integer p2 = 0; p2 < 2; ++p2)
                        croom[p2] <= (cfg_queue == 16'd0) | if_lt[2'(arf[p2]) + 2'(aw_fire)];
                    hd_due     <= pop ? hn_due1 : hn_due0;
                    hd_slot    <= pop ? rq_s[1] : rq_s[0];
                    rf_head    <= hd;
                    rf_count   <= rf_count - (pop ? (SW+1)'(1) : (SW+1)'(0));
                    rf_ready   <= rdy;
                    slots_used <= used0 + (sp2_take ? (SW+1)'(1) : (SW+1)'(0));
                    ar_pend_f  <= ar_fire_c;
                    ar_pend_id <= ar_id;
                    {ar_pend_bank, ar_pend_row} <= decode(ar_addr[29:5]);
                end else if (SPEC != 0) begin : release_spec_
                    // This cycle's R is the candidate the last cycle's fire
                    // chose; then the next cycle's candidates, from the ready
                    // queue after this one's take.
                    reg              rv, pop, p0, p1, v0, v1;
                    reg [SW-1:0]     rs, s0, s1, hd;
                    reg [ID_W-1:0]   ri, i0, i1;
                    reg [SW:0]       rdy, used;
                    rv  = fire_p ? cr1_valid : cur_r_valid;
                    rs  = fire_p ? cr1_slot  : cur_r_slot;
                    ri  = fire_p ? cr1_id    : cur_r_id;
                    pop = fire_p ? cr_pop1   : cr_pop0;
                    hd  = rf_head + (pop ? SW'(1) : SW'(0));
                    rdy = rf_ready - (pop ? (SW+1)'(1) : (SW+1)'(0));
                    p0  = ~rv & (rdy != 0);
                    v0  = rv | p0;
                    s0  = p0 ? rf_slot[hd] : rs;
                    i0  = p0 ? slot_id[rf_slot[hd]] : ri;
                    p1  = rv & (rdy != 0);
                    v1  = rv ? (rdy != 0) : v0;
                    s1  = rv ? rf_slot[hd] : s0;
                    i1  = rv ? slot_id[rf_slot[hd]] : i0;
                    if (~rv)
                        p1 = p0;
                    cur_r_valid <= v0;
                    cur_r_slot  <= s0;
                    cur_r_id    <= i0;
                    cr1_valid   <= v1;
                    cr1_slot    <= s1;
                    cr1_id      <= i1;
                    cr_pop0     <= p0;
                    cr_pop1     <= p1;
                    pr_valid    <= rv;
                    pr_slot     <= rs;
                    rf_head     <= hd;
                    rf_count    <= rf_count - (pop ? (SW+1)'(1) : (SW+1)'(0));
                    rf_ready    <= rdy;
                    used = slots_used + (SW+1)'(ar_fire) - (SW+1)'(fire_p);
                    slots_used  <= used;
                    sf0 <= (32'(used) < LSLOTS);
                    sf1 <= (32'(used) - 32'(rv) < LSLOTS);
                end else begin : release_
                    reg r_load;
                    r_load = (~cur_r_valid | r_fire) & (rf_ready != 0);
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
                    slots_used <= slots_used + (SW+1)'(ar_fire) - (SW+1)'(r_fire);
                end
                begin : release_b_
                    reg b_load;
                    b_load = (~cur_b_valid | b_fire) & (af_count != 0);
                    if (b_load) begin
                        cur_b_valid <= 1'b1;
                        cur_b_id    <= af_id[af_head];
                        af_head     <= af_head + 1'b1;
                        af_count    <= af_count - 1'b1;
                    end else if (b_fire) begin
                        cur_b_valid <= 1'b0;
                    end
                    for (integer i = 0; i < SLOTS; ++i) begin
                        if (fire_p & fire_oh[i])
                            slot_busy[i] <= 1'b0;
                        if (ar_now & alloc_oh[i]) begin
                            slot_busy[i] <= 1'b1;
                            slot_id[i]   <= ar_id;
                        end
                    end
                    inflight <= ifl_eff + 16'(ar_now) + 16'(aw_fire);
                    ifl_eff  <= ifl_eff + 16'(ar_now) + 16'(aw_fire);
                    lat_ar <= ar_now;
                    lat_r  <= fire_p;
                    lat_t  <= tcyc;
                    lat_tr <= tcyc - 32'(SPEC);
                end
                st_w_valid <= new_write;
                st_w_id    <= nw_id;
                {st_w_bank, st_w_row} <= decode(nw_addr[29:5]);
                st_r_valid <= ar_now;
                st_r_slot  <= alloc_slot;
                {st_r_bank, st_r_row} <= decode(ar_addr[29:5]);
                cr1        <= cr1 + 34'(cfg_data_rate);
                cr2        <= cr2 + 34'(cfg_data_rate);
            end
            // A clearing cycle's next host cycle clears the calendars, so the
            // first DRAM clock waits a host cycle (in A_PUSH).
            a_state <= ~(treset | active) ? A_IDLE
                     : ((treset | ~active) ? A_PUSH
                        : (((SPEC == 2) && ~sp2_now) ? A_RES : A_FAST));
            new_ready <= '0;
            new_acks  <= '0;
        end
        if (reset) begin
            state          <= S_IDLE;
            a_state        <= A_IDLE;
            active         <= 1'b0;
            stat_reads     <= '0;
            stat_writes    <= '0;
            stat_hits      <= '0;
            stat_misses    <= '0;
            stat_conflicts <= '0;
            stat_latency   <= '0;
            stat_refreshes <= '0;
            stat_full      <= '0;
            lat_ar         <= 1'b0;
            lat_r          <= 1'b0;
            rfp_valid      <= 1'b0;
            sc_pend        <= 1'b0;
        end
    end

    // ---- Commit: ranges of clocks marked in each calendar, a word a cycle ----

    // The ranges each calendar marks, from the reservation: a clock c of it,
    // and the range c + lo .. c + hi, its start no earlier than clock 0 if
    // clamped; an empty one is dropped.
    always @(*) begin
        for (integer m = 0; m < NBM; ++m) begin
            for (integer k = 0; k < 2; ++k) begin
                rx_v[m][k]  = 1'b0;
                rx_cl[m][k] = 1'b0;
                rx_c[m][k]  = 16'sd0;
                rx_lo[m][k] = 16'sd0;
                rx_hi[m][k] = 16'sd0;
            end
        end
        // Row commands: PRE, ACT; the row bus for PRE, and ACT's one or two
        // clocks. Column commands, and their directions for the controller's
        // queues.
        rx_v[B_CMDR][0] = op_conf;  rx_c[B_CMDR][0] = c_pre;
        rx_v[B_CMDR][1] = ~op_hit;  rx_c[B_CMDR][1] = c_act;
        rx_v[B_ROW][0]  = op_conf;  rx_c[B_ROW][0]  = c_pre;
        rx_v[B_ROW][1]  = ~op_hit;  rx_c[B_ROW][1]  = c_act;
        rx_hi[B_ROW][1] = g_act2 ? 16'sd1 : 16'sd0;
        rx_v[B_CMDC][0] = 1'b1;     rx_c[B_CMDC][0] = c_col0;
        rx_v[B_CMDC][1] = (NB == 2); rx_c[B_CMDC][1] = c_col1;
        rx_v[op_w ? B_KW : B_KR][0] = 1'b1;     rx_c[op_w ? B_KW : B_KR][0] = c_col0;
        rx_v[op_w ? B_KW : B_KR][1] = (NB == 2); rx_c[op_w ? B_KW : B_KR][1] = c_col1;
        // The read's departure.
        rx_v[B_DEPV][0] = ~op_w;
        rx_c[B_DEPV][0] = ck_lastcol; rx_lo[B_DEPV][0] = k_dep; rx_hi[B_DEPV][0] = k_dep;
        // What each column forbids to later columns of its pseudo-channel and
        // bank group, by direction.
        for (integer k = 0; k < NB; ++k) begin
            reg signed [15:0] c;
            c = (k == 0) ? c_col0 : c_col1;
            for (integer m = B_FRP; m <= B_FWG; ++m) begin
                rx_v[m][k]  = 1'b1;
                rx_cl[m][k] = 1'b1;
                rx_c[m][k]  = c;
            end
            if (~op_w) begin
                rx_lo[B_FRP][k] = 16'sd1 - k_ccds; rx_hi[B_FRP][k] = k_ccds - 16'sd1;
                rx_lo[B_FWP][k] = 16'sd1 - k_wtrs; rx_hi[B_FWP][k] = k_rtw - 16'sd1;
                rx_lo[B_FRG][k] = 16'sd1 - k_ccdl; rx_hi[B_FRG][k] = k_ccdl - 16'sd1;
                rx_lo[B_FWG][k] = 16'sd1 - k_wtrl; rx_hi[B_FWG][k] = 16'sd0;
            end else begin
                rx_lo[B_FWP][k] = 16'sd1 - k_ccds; rx_hi[B_FWP][k] = k_ccds - 16'sd1;
                rx_lo[B_FRP][k] = 16'sd1 - k_rtw;  rx_hi[B_FRP][k] = k_wtrs - 16'sd1;
                rx_lo[B_FWG][k] = 16'sd1 - k_ccdl; rx_hi[B_FWG][k] = k_ccdl - 16'sd1;
                rx_cl[B_FRG][k] = 1'b0;
                rx_lo[B_FRG][k] = 16'sd0;          rx_hi[B_FRG][k] = k_wtrl - 16'sd1;
            end
        end
        // What the ACT forbids to later ACTs: tRRD within the pseudo-channel,
        // tRRD_L within the bank group.
        rx_v[B_FAP][0] = ~op_hit; rx_cl[B_FAP][0] = 1'b1; rx_c[B_FAP][0] = c_act;
        rx_lo[B_FAP][0] = 16'sd1 - k_rrds; rx_hi[B_FAP][0] = k_rrds - 16'sd1;
        rx_v[B_FAG][0] = ~op_hit; rx_cl[B_FAG][0] = 1'b1; rx_c[B_FAG][0] = c_act;
        rx_lo[B_FAG][0] = 16'sd1 - k_rrdl; rx_hi[B_FAG][0] = k_rrdl - 16'sd1;
        // A refresh: its precharge, if any, and REF, on the row-command bus.
        if (op_ref) begin
            for (integer m = 0; m < NBM; ++m)
                for (integer k = 0; k < 2; ++k)
                    rx_v[m][k] = 1'b0;
            rx_v[B_CMDR][0] = op_conf; rx_c[B_CMDR][0] = c_pre;
            rx_v[B_CMDR][1] = 1'b1;    rx_c[B_CMDR][1] = c_ref;
            rx_v[B_ROW][0]  = op_conf; rx_c[B_ROW][0]  = c_pre;
            rx_v[B_ROW][1]  = 1'b1;    rx_c[B_ROW][1]  = c_ref;
            rx_hi[B_ROW][1] = 16'sd0;
        end
    end

    // A word a cycle per calendar, the first as the reservation commits; a
    // range's last word starts the next.
    function automatic signed [15:0] word_take(input [5:0] off, input signed [15:0] left);
        word_take = (left >= 16'sd64 - 16'(off)) ? 16'sd64 - 16'(off) : left;
    endfunction
    always @(posedge clk) begin
        wr_en <= '0;
        for (integer m = 0; m < NBM; ++m) begin
            reg [HLOG2-1:0]   pos;
            reg signed [15:0] left, take;
            reg [63:0]        mask;
            reg               act, nxt, ends;
            if (cm_start) begin : start_
                // Both ranges in one word are marked in one write. Each
                // range's first word, ahead of which one is first.
                reg [63:0]        sm [2];
                reg signed [15:0] tk [2];
                reg               one;
                for (integer k = 0; k < 2; ++k) begin
                    sm[k] = span_mask(cm_p[m][k][5:0], cm_len[m][k]);
                    tk[k] = word_take(cm_p[m][k][5:0], cm_len[m][k]);
                end
                one  = cm_v[m][0] & cm_v[m][1] & (cm_len[m][0] <= 16'sd64)
                     & (cm_len[m][1] <= 16'sd64) & (WW'(cm_p[m][0] >> 6) == cm_x[m][0])
                     & (WW'(cm_p[m][1] >> 6) == cm_x[m][1])
                     & (WW'(cm_p[m][0] >> 6) == WW'(cm_p[m][1] >> 6));
                act  = cm_v[m][0] | cm_v[m][1];
                nxt  = cm_v[m][0] & cm_v[m][1] & ~one;
                pos  = cm_v[m][0] ? cm_p[m][0] : cm_p[m][1];
                left = cm_v[m][0] ? cm_len[m][0] : cm_len[m][1];
                take = cm_v[m][0] ? tk[0] : tk[1];
                ends = cm_v[m][0] ? (cm_len[m][0] == tk[0]) : (cm_len[m][1] == tk[1]);
                mask = cm_v[m][0] ? (sm[0] | (one ? sm[1] : 64'd0)) : sm[1];
            end else begin
                act  = cm_busy[m];
                nxt  = cm_next[m];
                pos  = cm_pos[m];
                left = cm_left[m];
                take = word_take(cm_pos[m][5:0], cm_left[m]);
                ends = (cm_left[m] == take);
                mask = span_mask(cm_pos[m][5:0], cm_left[m]);
            end
            if (act) begin
                wr_en[m]   <= 1'b1;
                wr_word[m] <= WW'(pos >> 6);
                wr_mask[m] <= mask;
                if (ends) begin
                    cm_busy[m] <= nxt;
                    cm_next[m] <= 1'b0;
                    cm_pos[m]  <= cm_p[m][1];
                    cm_left[m] <= cm_len[m][1];
                end else begin
                    cm_busy[m] <= 1'b1;
                    cm_next[m] <= nxt;
                    cm_pos[m]  <= HLOG2'(pos + HLOG2'(take));
                    cm_left[m] <= left - take;
                end
            end else begin
                cm_busy[m] <= 1'b0;
            end
            if (reset)
                cm_busy[m] <= 1'b0;
        end
    end

endmodule
