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

// A folded simulator's virtual memory: one target RAM's rows for every
// instance, kept in device memory behind an AXI4 512-bit master.
//
// The simulator never waits for the controller: it issues a read or a write
// only when `free` allowed it, and takes a bubble while an instance's read is
// pending. The controller keeps the RAM's semantics and returns each read's
// row tagged with its instance, in any order across instances.
//
// Row `row` of instance `inst` lives at base + ((row * INSTANCES + inst) <<
// log2(ROW_BYTES)), ROW_BYTES being DW/8 rounded up to a power of two of at
// least 64: one burst of ceil(DW/512) beats, which never crosses 4 KiB.
//
// Requests flow in order through four stages:
//   X  the inputs, registered (two cycles more for INSTANCES not a power of
//      two: the multiply; a power of two is a shift); the host cache's tags
//      are read.
//   Y  the read's row against the cache, every buffered write, the write
//      taking a buffer slot this cycle and its own cycle's write: the newest
//      earlier copy of the row. The write hits or fills a cache line, or
//      goes around the cache.
//   Z  a write that goes around, or a dirty line the write evicts, takes a
//      write-buffer slot. The read's match: its slot.
//   R  a read whose newest copy is a whole row, in the cache or the buffer,
//      answers from it; any other read joins the read queue, waiting for a
//      partial write's B first.
// Writes leave the buffer in order on AW and W (one AXI id, so their B's
// come back in order) and free their slot on B. Reads leave the read queue in
// order on AR, each with its own id; R data may come back in any order and
// interleaved, and its last beat answers the read.
//
// The host cache (CACHE_ROWS > 0, up to a line per row) is direct-mapped and
// write-back, indexed by the low bits of row * INSTANCES + inst (consecutive
// instances, consecutive lines). A whole-row write allocates without a fetch,
// a read-first pair's too (its read still gets the old row, from the buffer
// or memory); a partial write hits or goes around. A read that misses goes to
// memory, and its answer fills the line, clean, unless the line holds a dirty
// row (a fill takes the place of a write at X, in a cycle without one; a
// write of its row getting there first drops it). A pair's read fills
// nothing: its write is newer.
// Such a pair leaves its new row in the cache while its read fetches the old
// one: until that read has its data, the line must not reach memory. Only a
// write's allocation (or a reset's sweep, after which no read answers) evicts
// a dirty line, and an instance with a read outstanding writes nothing. So
// where every line belongs to one instance (INSTANCES a power of two, at most
// CACHE_ROWS; or a line per row), the line is safe by construction. Elsewhere
// the pair pins its line (a few pins, else it goes around as below) until the
// read has its data, and a write whose line is pinned goes around the cache
// instead. A pair that goes around waits in the buffer for its read.
//
// Request-queue room (`free`): a read holds an entry from the cycle it is
// taken until it answers locally or goes out on AR; a write until it is in
// the cache, or until the B of the slot it (or the line it evicts) takes.
//
// Known-zero rows (KNOWN_ZERO): a bit per row of the image says whether a
// write took the row since the reset. While `zero` is high -- the host says
// memory holds zero in every row not written since the reset, as a runtime
// that cleared the image does -- a row not written is zero: a read of it is
// answered on chip (zeros, like a cache hit, filling no line: the bit already
// answers it), and a write of it is made whole, zeros where its mask is off,
// so it may fill a line and memory's copy of the row no longer matters. A
// read-first pair's read sees the row before its write. The bits are looked
// up as requests are taken, so they follow request order; the reset clears
// them, a word of each of KBANKS one-bit memories a cycle, within its drain.
// With `zero` low the controller behaves as without the bits.
//
// Reset takes no request, drops every read not yet on AR and answers none
// from before it; every write taken before it still goes to the cache or
// memory, and then every dirty line goes to memory and every line is
// invalidated: memory holds the RAM's image (which the host may then change). It finishes, with free back at
// QUEUE, once memory has answered every transaction: until then free stays 0.
//
// With TAGW tag bits (foldsim_vmem_group), rd_inst and wr_inst carry a member
// tag above the instance: the response returns it, the address ignores it (the
// group gives its members disjoint rows), and the simulator's guarantees hold
// per (tag, instance).
//
// A region move (`move`, the host placing the image elsewhere, as a new
// process does without a reset) ends the old region's use as a reset does,
// but the simulator keeps its state: free drops to 0 and `moving` rises;
// every request taken before it, or within the simulator's lead after it, is
// served and answered against the old region (no fill); then, once no request
// has come for QUIET cycles and none is in flight, the cache's sweep writes
// every dirty line back to the old base and invalidates every line, and the
// known-zero bits are cleared. Once memory has answered every transaction,
// `moving` falls, free returns to QUEUE and the new base takes effect. The
// base the controller uses is held from `move` (and from a reset) until then.
module foldsim_vmem #(
    // Instance-index bits: the global instance index over all copies.
    parameter IW             = 8,
    // Row-address bits.
    parameter AW             = 10,
    // Row data bits: a multiple of 8, up to 4096.
    parameter DW             = 512,
    // Mask bits: DW/8 byte enables, or 1 for a whole-row enable. A bit
    // enables DW/8/MW bytes.
    parameter MW             = DW / 8,
    // A same-cycle read of the row being written returns the old row (1) or
    // the new one (0).
    parameter RDW_READ_FIRST = 1,
    // Request-queue entries.
    parameter QUEUE          = 32,
    parameter AXI_ID_W       = 4,
    // Reads on AXI at once, at most 2^AXI_ID_W.
    parameter OUTSTANDING    = 16,
    // The instances the layout spaces rows by.
    parameter INSTANCES      = 1 << IW,
    // Host-cache rows, a power of two up to 2^(AW + clog2(INSTANCES)), a
    // line per row (0: direct).
    parameter CACHE_ROWS     = 0,
    // Known-zero rows: a bit per row of the image (0: none, `zero` unused).
    parameter KNOWN_ZERO     = 1,
    // Tag bits above the instance index in rd_inst, wr_inst and rsp_inst
    // (foldsim_vmem_group's member): carried to the response, never part of
    // the address. Requests of different tags may share lines, so read-first
    // pairs pin theirs wherever lines are not one per row.
    parameter TAGW           = 0
) (
    input  wire                        clk,
    input  wire                        reset,

    input  wire                        rd_valid,
    input  wire [IW+TAGW-1:0]          rd_inst,
    input  wire [AW-1:0]               rd_row,

    input  wire                        wr_valid,
    input  wire [IW+TAGW-1:0]          wr_inst,
    input  wire [AW-1:0]               wr_row,
    input  wire [DW-1:0]               wr_data,
    input  wire [MW-1:0]               wr_mask,

    output wire                        rsp_valid,
    output reg  [IW+TAGW-1:0]          rsp_inst,
    output reg  [DW-1:0]               rsp_data,

    output reg  [$clog2(QUEUE+1)-1:0]  free,

    input  wire [63:0]                 base,
    // Memory holds zero in every row not written since the reset.
    input  wire                        zero,
    // A cycle's pulse: the region moves; `base` may change from this cycle on.
    input  wire                        move,
    // The move is under way (from the cycle after `move`).
    output wire                        moving,

    output reg                         m_awvalid,
    input  wire                        m_awready,
    output reg  [63:0]                 m_awaddr,
    output wire [AXI_ID_W-1:0]         m_awid,
    output wire [7:0]                  m_awlen,
    output reg                         m_wvalid,
    input  wire                        m_wready,
    output reg  [511:0]                m_wdata,
    output reg  [63:0]                 m_wstrb,
    output reg                         m_wlast,
    input  wire                        m_bvalid,
    output wire                        m_bready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [AXI_ID_W-1:0]         m_bid,
    /* verilator lint_on UNUSEDSIGNAL */
    output reg                         m_arvalid,
    input  wire                        m_arready,
    output reg  [63:0]                 m_araddr,
    output reg  [AXI_ID_W-1:0]         m_arid,
    output wire [7:0]                  m_arlen,
    input  wire                        m_rvalid,
    output reg                         m_rready,
    input  wire [511:0]                m_rdata,
    input  wire [AXI_ID_W-1:0]         m_rid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire                        m_rlast
    /* verilator lint_on UNUSEDSIGNAL */
);
    localparam FW     = $clog2(QUEUE + 1);
    localparam IT     = IW + TAGW;
    localparam QW     = (QUEUE > 1) ? $clog2(QUEUE) : 1;
    localparam OW     = (OUTSTANDING > 1) ? $clog2(OUTSTANDING) : 1;
    localparam KB     = $clog2(INSTANCES);
    localparam POW2   = ((INSTANCES & (INSTANCES - 1)) == 0) ? 1 : 0;
    localparam LW     = AW + KB;
    localparam BYTES  = DW / 8;
    localparam RB     = ($clog2(BYTES) > 6) ? $clog2(BYTES) : 6;
    localparam BEATS  = (DW + 511) / 512;
    localparam BW     = (BEATS > 1) ? $clog2(BEATS) : 1;
    localparam GROUP  = BYTES / MW;
    // A row's offset in the image: its bits.
    localparam OB     = LW + RB;
    localparam CACHE  = (CACHE_ROWS > 0) ? 1 : 0;
    // A line's index and tag bits.
    localparam CI     = (CACHE_ROWS > 1) ? $clog2(CACHE_ROWS) : 1;
    localparam CT     = (LW > CI) ? LW - CI : 1;
    // Lines shared by instances (a line holds rows of one instance when
    // INSTANCES is a power of two, at most CACHE_ROWS, or when every row has
    // a line): read-first pairs pin theirs (PINS at once).
    localparam PIN    = (CACHE != 0) && (CACHE_ROWS < (1 << AW) * INSTANCES)
                     && (POW2 == 0 || CACHE_ROWS < INSTANCES || TAGW > 0);
    localparam PINS   = 8;
    localparam PW     = $clog2(PINS);
    localparam KZ     = (KNOWN_ZERO != 0) ? 1 : 0;
    // A move's flush starts once no request has come for this many cycles:
    // past the simulator's lead (L = 3) after free dropped to 0.
    localparam QUIET  = 15;

    // Where a local answer's row comes from.
    localparam SRC_WB = 2'd0;  // the write buffer
    localparam SRC_CA = 2'd1;  // the cache, by the read's own port
    localparam SRC_CB = 2'd2;  // the cache line the same-cycle write reads first
    localparam SRC_ZR = 2'd3;  // a known-zero row

`ifdef SIMULATION
    initial begin
        if ((DW % 8) != 0 || DW > 4096)
            $error("foldsim_vmem: DW must be a multiple of 8, at most 4096");
        if ((BYTES % MW) != 0)
            $error("foldsim_vmem: MW must divide DW/8");
        if (OUTSTANDING > (1 << AXI_ID_W))
            $error("foldsim_vmem: OUTSTANDING exceeds the AXI ids");
        if (INSTANCES > (1 << IW))
            $error("foldsim_vmem: INSTANCES exceeds the instance index");
        if (QUEUE < 4)
            $error("foldsim_vmem: QUEUE must be at least 4");
        if (CACHE_ROWS != 0 && (CACHE_ROWS < 2 || (CACHE_ROWS & (CACHE_ROWS - 1)) != 0 || CI > LW))
            $error("foldsim_vmem: CACHE_ROWS must be a power of two, at most 2^(AW + clog2(INSTANCES))");
    end
`endif

    function automatic [QW-1:0] qinc(input [QW-1:0] p);
        qinc = (p == QW'(QUEUE - 1)) ? '0 : p + 1'b1;
    endfunction

    function automatic [QW-1:0] onehot_index(input [QUEUE-1:0] v);
        onehot_index = '0;
        for (integer i = 0; i < QUEUE; ++i)
            if (v[i])
                onehot_index = onehot_index | QW'(i);
    endfunction

    function automatic [OW-1:0] first_zero(input [OUTSTANDING-1:0] v);
        first_zero = '0;
        for (integer i = OUTSTANDING - 1; i >= 0; --i)
            if (!v[i])
                first_zero = OW'(i);
    endfunction

    function automatic [BYTES-1:0] byte_enables(input [MW-1:0] mask);
        for (integer j = 0; j < BYTES; ++j)
            byte_enables[j] = mask[j / GROUP];
    endfunction

    // ---- State that outlives a reset: what the cache and memory have, or
    // are owed ----
    // A reset drains it rather than clearing it, so it starts at its
    // power-up value: 0, as the FPGA's flops and block RAMs (INIT 0) do.
    // Simulation starts it there too.
    reg              draining;
    reg              x_wr, y_wr, x_sw, y_sw, x_fl, y_fl, z_wa, z_ev;
    reg [QUEUE-1:0]  wb_valid;
    reg [QW-1:0]     wb_tail, wb_aw, wb_w, wb_head;
    // Writes in the buffer: all, before AW, before their last W beat, before B.
    reg [FW-1:0]     wb_cnt, n_aw, n_w, n_b;
    reg [BW-1:0]     w_beat;
    reg [OUTSTANDING-1:0] id_busy;
    reg              rc_valid;
    reg [63:0]       base_q;
`ifdef SIMULATION
    initial begin
        base_q    = '0;
        draining  = 1'b0;
        x_wr      = 1'b0;
        y_wr      = 1'b0;
        x_sw      = 1'b0;
        y_sw      = 1'b0;
        x_fl      = 1'b0;
        y_fl      = 1'b0;
        z_wa      = 1'b0;
        z_ev      = 1'b0;
        wb_valid  = '0;
        wb_tail   = '0;
        wb_aw     = '0;
        wb_w      = '0;
        wb_head   = '0;
        wb_cnt    = '0;
        n_aw      = '0;
        n_w       = '0;
        n_b       = '0;
        w_beat    = '0;
        id_busy   = '0;
        rc_valid  = 1'b0;
        m_awvalid = 1'b0;
        m_wvalid  = 1'b0;
        m_arvalid = 1'b0;
    end
`endif

    if (TAGW > 0) begin : g_tag
        // A write's tag only names its member, for the checks below.
        /* verilator lint_off UNUSEDSIGNAL */
        wire unused = |wr_inst[IT-1:IW];
        /* verilator lint_on UNUSEDSIGNAL */
    end

    wire accept = ~reset & ~draining;
    wire taken_rd = rd_valid & accept;
    wire taken_wr = wr_valid & accept;

    // ---- Intake: a row's index in the image, row * INSTANCES + inst ----
    // i_busy, i_rbusy: a write, a read, still in the multiply.
    wire              i_rd, i_wr, i_busy, i_rbusy;
    wire [IT-1:0]     i_rinst;
    wire [LW-1:0]     i_rlin, i_wlin;
    wire [DW-1:0]     i_wdata;
    wire [MW-1:0]     i_wmask;
    if (POW2 != 0) begin : g_lin_shift
        // inst < INSTANCES: its low KB bits.
        assign i_rd    = taken_rd;
        assign i_wr    = taken_wr;
        assign i_busy  = 1'b0;
        assign i_rbusy = 1'b0;
        assign i_rinst = rd_inst;
        assign i_rlin  = (LW'(rd_row) << KB) | LW'(rd_inst[IW-1:0] & IW'(INSTANCES - 1));
        assign i_wlin  = (LW'(wr_row) << KB) | LW'(wr_inst[IW-1:0] & IW'(INSTANCES - 1));
        assign i_wdata = wr_data;
        assign i_wmask = wr_mask;
    end else begin : g_lin_mul
        // The multiply in a DSP, its input and product registered: two more
        // cycles. A write taken still reaches the cache or memory through a
        // reset.
        reg          a_rd, a_wr, p_rd, p_wr;
        reg [AW-1:0] a_rrow, a_wrow;
        reg [IT-1:0] a_rinst, p_rinst;
        reg [IW-1:0] a_winst, p_winst;
        reg [LW-1:0] p_rprod, p_wprod;
        reg [DW-1:0] a_wdata, p_wdata;
        reg [MW-1:0] a_wmask, p_wmask;
`ifdef SIMULATION
        initial begin
            a_wr = 1'b0;
            p_wr = 1'b0;
            a_rd = 1'b0;
            p_rd = 1'b0;
        end
`endif
        always @(posedge clk) begin
            a_rd    <= taken_rd;
            a_wr    <= taken_wr;
            a_rrow  <= rd_row;
            a_wrow  <= wr_row;
            a_rinst <= rd_inst;
            a_winst <= wr_inst[IW-1:0];
            a_wdata <= wr_data;
            a_wmask <= wr_mask;
            p_rd    <= a_rd & ~reset;
            p_wr    <= a_wr;
            p_rprod <= LW'(a_rrow) * LW'(INSTANCES);
            p_wprod <= LW'(a_wrow) * LW'(INSTANCES);
            p_rinst <= a_rinst;
            p_winst <= a_winst;
            p_wdata <= a_wdata;
            p_wmask <= a_wmask;
        end
        assign i_rd    = p_rd;
        assign i_wr    = p_wr;
        assign i_busy  = a_wr | p_wr;
        assign i_rbusy = a_rd | p_rd;
        assign i_rinst = p_rinst;
        assign i_rlin  = p_rprod + LW'(p_rinst[IW-1:0]);
        assign i_wlin  = p_wprod + LW'(p_winst);
        assign i_wdata = p_wdata;
        assign i_wmask = p_wmask;
    end

    // The base in use (base_q): `base` but through a reset's drain and a
    // move, which still address the old region; taken again as either ends.
    // base + offset adds only the offset's bits; above them, base or base + 1
    // (kept registered: base is static while requests flow).
    reg [63-OB:0] base_hi1;
    wire          base_hold;
    always @(posedge clk) begin
        if (~base_hold)
            base_q <= base;
        base_hi1 <= base_q[63:OB] + 1'b1;
    end
    function automatic [63:0] row_addr(input [LW-1:0] lin);
        reg [OB:0] lo;
        lo = {1'b0, base_q[OB-1:0]} + {1'b0, lin, {RB{1'b0}}};
        row_addr = {lo[OB] ? base_hi1 : base_q[63:OB], lo[OB-1:0]};
    endfunction

    // ---- The sweep of a reset's drain: every line, in order ----
    reg              sw_done;
    reg [CI-1:0]     sw_idx;
    wire             sweep_go;

    // ---- A fill: a row memory answered, for its line ----
    // It takes the write's place at X in a cycle without one. A write of its
    // row reaching X first would be newer (only the read's instance writes
    // the row, after its answer): the fill is dropped then.
    wire             fl_v;
    wire [LW-1:0]    fl_lin;
    wire [DW-1:0]    fl_data;
    // A move (mv_busy) fills nothing either: its sweep empties the cache.
    reg              mv_busy;
    wire             fill_go = fl_v & ~i_wr & ~reset & ~draining & ~mv_busy;

    // ---- Known-zero rows: looked up as requests are taken ----
    // At X: the write's row, the read's row was zero before this cycle's
    // write (zero high, no write since the reset); the reset's clear is done.
    wire             kz_w, kz_r, kz_done;

    // ---- A move: its flush (the sweep and the clear) has begun, begins ----
    reg              mv_flush;
    wire             mv_go;
`ifdef SIMULATION
    initial begin
        mv_busy  = 1'b0;
        mv_flush = 1'b0;
    end
`endif

    // ---- X: the requests, in the image's terms; the tags are read ----
    reg              x_rd;
    reg [IT-1:0]     x_rinst;
    reg [LW-1:0]     x_rlin, x_wlin;
    reg [DW-1:0]     x_wdata;
    reg [MW-1:0]     x_wmask;
    reg [CI-1:0]     x_swidx;

    always @(posedge clk) begin
        x_rd    <= i_rd & ~reset;
        x_wr    <= i_wr;
        x_sw    <= sweep_go;
        x_fl    <= fill_go;
        x_swidx <= sw_idx;
        x_rinst <= i_rinst;
        x_rlin  <= i_rlin;
        x_wlin  <= fill_go ? fl_lin : i_wlin;
        x_wdata <= fill_go ? fl_data : i_wdata;
        x_wmask <= fill_go ? '1 : i_wmask;
    end

    // A write of a known-zero row, made whole: zeros where its mask is off.
    reg [DW-1:0] x_wzdata;
    always @(*) begin
        for (integer j = 0; j < BYTES; ++j)
            x_wzdata[j*8 +: 8] = (kz_w & ~x_wmask[j / GROUP]) ? 8'd0 : x_wdata[j*8 +: 8];
    end
    // The read's row is zero: a known-zero row, but a write-first pair's
    // (its own write is newer); a pair's read sees the row before its write.
    wire x_same = x_rd & x_wr & (x_rlin == x_wlin);
    wire x_zero = x_rd & (x_same ? ((RDW_READ_FIRST != 0) & kz_w) : kz_r);

    // ---- Y: the newest earlier copy of the read's row ----
    reg              y_rd, y_wfull, y_zero;
    reg [IT-1:0]     y_rinst;
    reg [LW-1:0]     y_rlin, y_wlin;
    reg [DW-1:0]     y_wdata;
    reg [MW-1:0]     y_wmask;
    reg [CI-1:0]     y_swidx;
    reg [63:0]       y_raddr;

    always @(posedge clk) begin
        y_rd    <= x_rd & ~reset;
        y_zero  <= x_zero & ~reset;
        y_wr    <= x_wr;
        y_sw    <= x_sw;
        y_fl    <= x_fl;
        y_swidx <= x_swidx;
        y_rinst <= x_rinst;
        y_rlin  <= x_rlin;
        y_wlin  <= x_wlin;
        y_wdata <= x_wzdata;
        y_wmask <= x_wmask | {MW{kz_w}};
        y_wfull <= (&x_wmask) | kz_w;
        y_raddr <= row_addr(x_rlin);
    end

    // The cache's view at Y, and its rows at Z and R. c_victim: the write's
    // line holds a dirty row.
    wire             c_rhit, c_whit, c_victim, c_same_idx;
    wire [CT-1:0]    c_vtag;
    wire [DW-1:0]    c_ca, c_cb;

    wire y_same_row = y_rd & y_wr & (y_rlin == y_wlin);
    wire rf_pair    = y_same_row & (RDW_READ_FIRST != 0);
    // Pins (PIN): the write's line is pinned; a pin is free, and which.
    wire          y_pinned, pin_room;
    wire [PW-1:0] pin_next;
    // A whole-row write fills a line, but over a pinned line, or a read-first
    // pair's with no pin free: those go around.
    wire w_alloc  = (CACHE != 0) & y_wr & ~c_whit & y_wfull
                  & ~(PIN & (y_pinned | (rf_pair & ~pin_room)));
    wire w_around = y_wr & ~c_whit & ~w_alloc;
    // A read-first pair filling a shared line pins it until its read has its
    // data.
    wire y_pin    = PIN & rf_pair & w_alloc;
    wire evict    = (w_alloc | y_sw) & c_victim;
    // A fill takes a line that holds no dirty row (no write-back, no pin).
    wire f_alloc  = (CACHE != 0) & y_fl & ~c_victim;
    wire [CI-1:0] y_widx = y_sw ? y_swidx : y_wlin[CI-1:0];
    wire [LW-1:0] ev_lin = LW'({c_vtag, y_widx});
    // The write is done with the cache: its room is free.
    wire y_wdone  = y_wr & ~w_around & ~evict;

    // The read's own write, around the cache: seen by it, or held until it
    // has its data.
    wire pair_wf  = y_same_row & w_around & (RDW_READ_FIRST == 0);
    wire pair_rf  = rf_pair & w_around;
    // In the cache: a hit, or a write-first pair's line.
    wire c_local  = c_rhit | (y_same_row & w_alloc & (RDW_READ_FIRST == 0));

    // The write buffer: writes in request order, until their B. A write is
    // live until a later write of its row takes a slot, so a row has at most
    // one live write: a read's match needs no priority.
    reg [LW-1:0]     wb_lin  [QUEUE];
    reg [QUEUE-1:0]  wb_live, wb_full, wb_blk;
    reg [DW-1:0]     wb_data [QUEUE];
    reg [MW-1:0]     wb_mask [QUEUE];
    reg [63:0]       wb_addr [QUEUE];

    reg              z_rd, z_local, z_pair, z_zhit, z_ub, z_merge, z_wfull, z_blk, z_pin, z_fill;
    reg [LW-1:0]     z_rlin;
    reg [PW-1:0]     z_pinid;
    reg [1:0]        z_src;
    reg [QUEUE-1:0]  z_hit;
    reg [QW-1:0]     z_zslot;
    reg [IT-1:0]     z_rinst;
    reg [63:0]       z_raddr;
    reg [LW-1:0]     z_alin;
    reg [DW-1:0]     z_wdata;
    reg [MW-1:0]     z_wmask;

    wire z_alloc = z_wa | z_ev;
    wire z_afull = z_ev | z_wfull;

    reg [QUEUE-1:0] rhit;
    always @(*) begin
        for (integer i = 0; i < QUEUE; ++i)
            rhit[i] = wb_valid[i] & wb_live[i] & (wb_lin[i] == y_rlin);
    end
    // The write taking a slot this cycle is newer than the buffer's.
    wire zhit   = z_alloc & (z_alin == y_rlin);
    wire wb_fwd = pair_wf ? y_wfull : (zhit ? z_afull : |(rhit & wb_full));
    wire y_local = (y_rd & (c_local | wb_fwd)) | y_zero;
    wire [1:0] y_src = y_zero ? SRC_ZR
                     : ~c_local ? SRC_WB
                     : ((c_same_idx & (c_whit | w_alloc | f_alloc)) ? SRC_CB : SRC_CA);
    // A read that goes to memory fills its line with the answer, but a
    // pair's: its write is newer.
    wire y_fill = (CACHE != 0) & ~y_same_row;

    always @(posedge clk) begin
        z_rd    <= y_rd & ~reset;
        z_local <= y_local & ~reset;
        z_src   <= y_src;
        z_merge <= y_same_row & (RDW_READ_FIRST == 0);
        z_hit   <= rhit;
        z_pair  <= pair_wf;
        z_zhit  <= zhit;
        z_zslot <= wb_tail;
        z_ub    <= pair_rf;
        z_pin   <= y_pin;
        z_pinid <= pin_next;
        z_fill  <= y_fill;
        z_rlin  <= y_rlin;
        z_rinst <= y_rinst;
        z_raddr <= y_raddr;
        z_wa    <= w_around;
        z_ev    <= evict;
        z_alin  <= evict ? ev_lin : y_wlin;
        z_wdata <= y_wdata;
        z_wmask <= y_wmask;
        z_wfull <= y_wfull;
        z_blk   <= pair_rf;
        // R's last beat answers a read the cycle after it: never in the cycle
        // a read answers locally.
        m_rready <= ~(y_local & ~reset);
    end

    // ---- Z: a write around the cache, or an evicted line, takes a slot ----
    // A slot valid at Y is not taken again before R ends (a write taken
    // needs the room its B gave back, and reaches Z three cycles later), so
    // its data is still the matched write's.
    wire [DW-1:0]    z_adata = z_ev ? c_cb : z_wdata;
    wire [MW-1:0]    z_amask = z_ev ? '1 : z_wmask;
    wire [63:0]      z_aaddr = row_addr(z_alin);
    wire [QW-1:0]    z_slot  = z_pair ? wb_tail : (z_zhit ? z_zslot : onehot_index(z_hit));
    wire             z_any   = z_pair | z_zhit | (|z_hit);

    reg [QUEUE-1:0] whit;
    always @(*) begin
        for (integer i = 0; i < QUEUE; ++i)
            whit[i] = wb_valid[i] & (wb_lin[i] == z_alin);
    end

    // A write-first pair's line: the old row under the write.
    reg [DW-1:0] z_pdata;
    always @(*) begin
        for (integer j = 0; j < BYTES; ++j)
            z_pdata[j*8 +: 8] = (z_merge & z_wmask[j / GROUP]) ? z_wdata[j*8 +: 8] : c_cb[j*8 +: 8];
    end

    reg              r_local, r_rd, r_part, r_ub, r_pin, r_fill;
    reg [LW-1:0]     r_rlin;
    reg [PW-1:0]     r_pinid;
    reg [1:0]        r_src;
    reg [QW-1:0]     r_slot, r_ubslot;
    reg [IT-1:0]     r_inst;
    reg [63:0]       r_addr;
    reg [DW-1:0]     r_pdata;

    always @(posedge clk) begin
        r_local  <= z_local & ~reset;
        r_rd     <= z_rd & ~z_local & ~reset;
        r_part   <= z_any;
        r_ub     <= z_ub;
        r_pin    <= z_pin;
        r_pinid  <= z_pinid;
        r_fill   <= z_fill;
        r_rlin   <= z_rlin;
        r_src    <= z_src;
        r_slot   <= z_slot;
        r_ubslot <= wb_tail;
        r_inst   <= z_rinst;
        r_addr   <= z_raddr;
        r_pdata  <= z_pdata;
    end

    // ---- R: answer locally, or queue for AR ----
    wire b_pop = m_bvalid;

    reg [63:0]       rq_addr  [QUEUE];
    reg [IT-1:0]     rq_inst  [QUEUE];
    reg              rq_ub    [QUEUE];
    reg [QW-1:0]     rq_ubslot [QUEUE];
    reg [QW-1:0]     rq_dslot [QUEUE];
    reg              rq_pin   [QUEUE];
    reg              rq_fill  [QUEUE];
    reg [LW-1:0]     rq_lin   [QUEUE];
    reg [PW-1:0]     rq_pinid [QUEUE];
    reg [QUEUE-1:0]  rq_dep;
    reg [QW-1:0]     rq_head, rq_tail;
    reg [FW-1:0]     rq_count;

    wire rq_push = r_rd;
    wire r_dep   = r_part & wb_valid[r_slot] & ~(b_pop & (wb_head == r_slot));

    // ---- AR: the read queue's head, with a free id ----
    wire          id_avail = ~(&id_busy);
    wire [OW-1:0] id_next  = first_zero(id_busy);
    wire          rq_issue = (rq_count != 0) & ~rq_dep[rq_head] & id_avail
                           & (~m_arvalid | m_arready) & ~reset;

    reg [IT-1:0]  tbl_inst  [OUTSTANDING];
    reg           tbl_ub    [OUTSTANDING];
    reg [QW-1:0]  tbl_ubslot [OUTSTANDING];
    reg           tbl_pin   [OUTSTANDING];
    reg           tbl_fill  [OUTSTANDING];
    reg [LW-1:0]  tbl_lin   [OUTSTANDING];
    reg [PW-1:0]  tbl_pinid [OUTSTANDING];
    reg [BW-1:0]  rcnt      [OUTSTANDING];

    always @(posedge clk) begin
        if (rq_push) begin
            rq_addr[rq_tail]   <= r_addr;
            rq_inst[rq_tail]   <= r_inst;
            rq_ub[rq_tail]     <= r_ub;
            rq_ubslot[rq_tail] <= r_ubslot;
            rq_dslot[rq_tail]  <= r_slot;
            rq_pin[rq_tail]    <= r_pin;
            rq_fill[rq_tail]   <= r_fill;
            rq_lin[rq_tail]    <= r_rlin;
            rq_pinid[rq_tail]  <= r_pinid;
        end
        // A queued read waits for its row's newest partial write: its B
        // frees the slot.
        for (integer i = 0; i < QUEUE; ++i)
            if (b_pop & (rq_dslot[i] == wb_head))
                rq_dep[i] <= 1'b0;
        if (rq_push)
            rq_dep[rq_tail] <= r_dep;
        if (reset) begin
            rq_head  <= '0;
            rq_tail  <= '0;
            rq_count <= '0;
        end else begin
            if (rq_push)
                rq_tail <= qinc(rq_tail);
            if (rq_issue)
                rq_head <= qinc(rq_head);
            rq_count <= rq_count + FW'(rq_push) - FW'(rq_issue);
        end
        if (rq_issue) begin
            m_araddr            <= rq_addr[rq_head];
            m_arid              <= AXI_ID_W'(id_next);
            tbl_inst[id_next]   <= rq_inst[rq_head];
            tbl_ub[id_next]     <= rq_ub[rq_head];
            tbl_ubslot[id_next] <= rq_ubslot[rq_head];
            tbl_pin[id_next]    <= rq_pin[rq_head];
            tbl_fill[id_next]   <= rq_fill[rq_head];
            tbl_lin[id_next]    <= rq_lin[rq_head];
            tbl_pinid[id_next]  <= rq_pinid[rq_head];
        end
        if (rq_issue)
            m_arvalid <= 1'b1;
        else if (m_arready)
            m_arvalid <= 1'b0;
    end
    assign m_arlen = 8'(BEATS - 1);

    // ---- R: beats by id, the last one answers ----
    reg [OW-1:0]  rc_id;
    reg [511:0]   rc_data;
    /* verilator lint_off UNUSEDSIGNAL */
    reg           rc_rlast;
    /* verilator lint_on UNUSEDSIGNAL */
    always @(posedge clk) begin
        rc_valid <= m_rvalid & m_rready;
        rc_id    <= OW'(m_rid);
        rc_data  <= m_rdata;
        rc_rlast <= m_rlast;
    end
    wire [BW-1:0] rc_beat = rcnt[rc_id];
    wire          rc_last = (rc_beat == BW'(BEATS - 1));
    wire          rc_done = rc_valid & rc_last;

    // The last beat's bits past DW are padding.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [BEATS*512-1:0] rc_row;
    /* verilator lint_on UNUSEDSIGNAL */
    if (BEATS > 1) begin : g_beats
        for (genvar k = 0; k < BEATS - 1; ++k) begin : g_bm
            reg [511:0] bm [OUTSTANDING];
            always @(posedge clk)
                if (rc_valid & (rc_beat == BW'(k)))
                    bm[rc_id] <= rc_data;
            assign rc_row[k*512 +: 512] = bm[rc_id];
        end
    end
    assign rc_row[(BEATS-1)*512 +: 512] = rc_data;

    // An id's beat count restarts with its read.
    always @(posedge clk) begin
        if (rc_valid)
            rcnt[rc_id] <= rc_last ? '0 : rc_beat + 1'b1;
        if (rq_issue)
            rcnt[id_next] <= '0;
    end

    // ---- Responses: local (R) or from memory ----
    // None is offered while reset is high, not even one made before it.
    reg rsp_q;
    assign rsp_valid = rsp_q & ~reset;
    always @(posedge clk) begin
        rsp_q <= (r_local | (rc_done & ~draining)) & ~reset;
        if (r_local) begin
            rsp_inst <= r_inst;
            case (r_src)
            SRC_CA:  rsp_data <= c_ca;
            SRC_CB:  rsp_data <= r_pdata;
            SRC_ZR:  rsp_data <= '0;
            default: rsp_data <= wb_data[r_slot];
            endcase
        end else begin
            rsp_inst <= tbl_inst[rc_id];
            rsp_data <= rc_row[DW-1:0];
        end
    end

    // The fill: the last answer from memory whose line may take it. Reset
    // and a move drop it, and no answer during the drain or the move fills.
    if (CACHE != 0) begin : g_fill
        reg          v;
        reg [LW-1:0] lin;
        reg [DW-1:0] data;
`ifdef SIMULATION
        initial v = 1'b0;
`endif
        always @(posedge clk) begin
            if (rc_done & tbl_fill[rc_id] & ~draining & ~reset & ~mv_busy & ~move) begin
                v    <= 1'b1;
                lin  <= tbl_lin[rc_id];
                data <= rc_row[DW-1:0];
            end else if (reset | move | fill_go | (i_wr & (i_wlin == lin))) begin
                v    <= 1'b0;
            end
        end
        assign fl_v    = v;
        assign fl_lin  = lin;
        assign fl_data = data;
    end else begin : g_no_fill
        /* verilator lint_off UNUSEDSIGNAL */
        wire unused = tbl_fill[rc_id] | (|tbl_lin[rc_id]);
        /* verilator lint_on UNUSEDSIGNAL */
        assign fl_v    = 1'b0;
        assign fl_lin  = '0;
        assign fl_data = '0;
    end

    // ---- AW and W: the buffer's writes in order ----
    wire aw_issue = (n_aw != 0) & (~wb_blk[wb_aw] | draining) & (~m_awvalid | m_awready);
    wire w_issue  = (n_w != 0) & (~m_wvalid | m_wready);
    wire w_end    = w_issue & (w_beat == BW'(BEATS - 1));

    wire [BEATS*512-1:0] w_row  = (BEATS*512)'(wb_data[wb_w]);
    wire [BEATS*64-1:0]  w_strb = (BEATS*64)'(byte_enables(wb_mask[wb_w]));

    always @(posedge clk) begin
        if (aw_issue) begin
            m_awaddr <= wb_addr[wb_aw];
            wb_aw    <= qinc(wb_aw);
        end
        if (aw_issue)
            m_awvalid <= 1'b1;
        else if (m_awready)
            m_awvalid <= 1'b0;

        if (w_issue) begin
            m_wdata <= w_row[32'(w_beat) * 512 +: 512];
            m_wstrb <= w_strb[32'(w_beat) * 64 +: 64];
            m_wlast <= (w_beat == BW'(BEATS - 1));
            w_beat  <= (w_beat == BW'(BEATS - 1)) ? '0 : w_beat + 1'b1;
        end
        if (w_end)
            wb_w <= qinc(wb_w);
        if (w_issue)
            m_wvalid <= 1'b1;
        else if (m_wready)
            m_wvalid <= 1'b0;

        wb_cnt <= wb_cnt + FW'(z_alloc) - FW'(b_pop);
        n_aw   <= n_aw + FW'(z_alloc) - FW'(aw_issue);
        n_w    <= n_w + FW'(aw_issue) - FW'(w_end);
        n_b    <= n_b + FW'(w_end) - FW'(b_pop);
    end
    assign m_awid   = '0;
    assign m_awlen  = 8'(BEATS - 1);
    assign m_bready = 1'b1;

    // ---- The buffer's slots: taken at Z, freed by B ----
    always @(posedge clk) begin
        if (z_alloc) begin
            wb_lin[wb_tail]  <= z_alin;
            wb_data[wb_tail] <= z_adata;
            wb_mask[wb_tail] <= z_amask;
            wb_addr[wb_tail] <= z_aaddr;
        end
        for (integer i = 0; i < QUEUE; ++i)
            if (z_alloc & whit[i])
                wb_live[i] <= 1'b0;
        if (z_alloc) begin
            wb_live[wb_tail] <= 1'b1;
            wb_full[wb_tail] <= z_afull;
            wb_tail          <= qinc(wb_tail);
        end
        // A write held for its own cycle's read goes once that read has its
        // data.
        if (r_local & r_ub)
            wb_blk[r_ubslot] <= 1'b0;
        if (rc_done & tbl_ub[rc_id])
            wb_blk[tbl_ubslot[rc_id]] <= 1'b0;
        if (z_alloc)
            wb_blk[wb_tail] <= z_wa & z_blk;

        if (b_pop)
            wb_head <= qinc(wb_head);
        begin : valid_
            reg [QUEUE-1:0] v;
            v = wb_valid;
            if (b_pop)
                v[wb_head] = 1'b0;
            if (z_alloc)
                v[wb_tail] = 1'b1;
            wb_valid <= v;
        end
    end

    // ---- Ids: busy from AR until the last beat of R ----
    always @(posedge clk) begin
        begin : ids_
            reg [OUTSTANDING-1:0] b;
            b = id_busy;
            if (rc_done)
                b[rc_id] = 1'b0;
            if (rq_issue)
                b[id_next] = 1'b1;
            id_busy <= b;
        end
    end

    // ---- Pins: a shared line held for its read-first pair's read ----
    // Released when the read has its data (one answer a cycle at most). A
    // reset drops them all: no read from before it answers.
    wire          pin_rel   = (r_local & r_pin) | (rc_done & tbl_pin[rc_id]);
    wire [PW-1:0] pin_relid = r_local ? r_pinid : tbl_pinid[rc_id];

    if (PIN) begin : g_pin
        reg [PINS-1:0] pin_v;
        reg [CI-1:0]   pin_idx [PINS];
        reg            pinned_q;
        wire [CI-1:0]  x_idx = x_wlin[CI-1:0];
        reg  [PINS-1:0] pm;
        always @(*) begin
            for (integer i = 0; i < PINS; ++i)
                pm[i] = pin_v[i] & (pin_idx[i] == x_idx);
        end
        always @(posedge clk) begin
            // A pair at Y may pin X's line this cycle: taken as pinned.
            pinned_q <= (|pm) | (rf_pair & y_wfull & (y_wlin[CI-1:0] == x_idx));
            if (y_pin)
                pin_idx[pin_next] <= y_wlin[CI-1:0];
            begin : v_
                reg [PINS-1:0] v;
                v = pin_v;
                if (pin_rel)
                    v[pin_relid] = 1'b0;
                if (y_pin)
                    v[pin_next] = 1'b1;
                pin_v <= reset ? '0 : v;
            end
        end
        reg [PW-1:0] nx;
        always @(*) begin
            nx = '0;
            for (integer i = PINS - 1; i >= 0; --i)
                if (!pin_v[i])
                    nx = PW'(i);
        end
`ifdef SIMULATION
        always @(posedge clk) begin
            for (integer i = 0; i < PINS; ++i)
                if (evict & pin_v[i] & (pin_idx[i] == y_widx))
                    $error("foldsim_vmem: pinned line %0d evicted", y_widx);
        end
`endif
        assign y_pinned = pinned_q;
        assign pin_room = ~(&pin_v);
        assign pin_next = nx;
    end else begin : g_no_pin
        /* verilator lint_off UNUSEDSIGNAL */
        wire unused = pin_rel | (|pin_relid);
        /* verilator lint_on UNUSEDSIGNAL */
        assign y_pinned = 1'b0;
        assign pin_room = 1'b1;
        assign pin_next = '0;
    end

    // ---- The host cache ----
    if (CACHE != 0) begin : g_cache
        // A line: valid, dirty (the write's lookup only), tag. Writes leave a
        // line dirty, fills clean. Two copies of the tags, one per lookup;
        // each is read at X, and the update of the cycle before (made at Y as
        // the read was made) is forwarded. Whether it is at X's lines, and its
        // tag X's, is known at X.
        localparam TE = CT + 1;
        localparam TW = CT + 2;
        reg [TE-1:0] tag_r [CACHE_ROWS];
        reg [TW-1:0] tag_w [CACHE_ROWS];
        reg [TE-1:0] tr_q;
        reg [TW-1:0] tw_q, tu_ent;
        reg          tu_v, selr_q, selw_q, bhr_q, bhw_q;
`ifdef SIMULATION
        initial begin
            for (integer i = 0; i < CACHE_ROWS; ++i) begin
                tag_r[i] = '0;
                tag_w[i] = '0;
            end
        end
`endif
        wire [CI-1:0] y_ridx = y_rlin[CI-1:0];
        wire [CI-1:0] x_widx = x_sw ? x_swidx : x_wlin[CI-1:0];
        wire          tu_we  = c_whit | w_alloc | f_alloc | y_sw;
        wire [TW-1:0] tu_new = y_sw ? '0 : {1'b1, y_wr, CT'(y_wlin >> CI)};

        always @(posedge clk) begin
            tr_q <= tag_r[x_rlin[CI-1:0]];
            tw_q <= tag_w[x_widx];
            if (tu_we) begin
                tag_r[y_widx] <= {tu_new[TW-1], tu_new[CT-1:0]};
                tag_w[y_widx] <= tu_new;
            end
            tu_v   <= tu_we;
            tu_ent <= tu_new;
            selr_q <= (y_wr | y_sw | y_fl) & (y_widx == x_rlin[CI-1:0]);
            selw_q <= (y_wr | y_sw | y_fl) & (y_widx == x_widx);
            bhr_q  <= ~y_sw & (CT'(y_wlin >> CI) == CT'(x_rlin >> CI));
            bhw_q  <= ~y_sw & (CT'(y_wlin >> CI) == CT'(x_wlin >> CI));
        end
        wire br = selr_q & tu_v;
        wire bw = selw_q & tu_v;

        assign c_rhit     = y_rd & (br ? bhr_q : (tr_q[TE-1] & (tr_q[CT-1:0] == CT'(y_rlin >> CI))));
        assign c_whit     = y_wr & (bw ? bhw_q : (tw_q[TW-1] & (tw_q[CT-1:0] == CT'(y_wlin >> CI))));
        assign c_victim   = bw ? tu_ent[TW-2] : tw_q[TW-2];
        assign c_vtag     = bw ? tu_ent[CT-1:0] : tw_q[CT-1:0];
        assign c_same_idx = (y_ridx == y_widx);

        // The rows: port A reads for the read (two cycles: an output
        // register); port B writes for the write, reading first the line it
        // replaces or the read of its cycle shares.
        reg [DW-1:0] cdata [CACHE_ROWS];
        reg [DW-1:0] ca1, ca2, cb1;
        // An allocation's mask is whole: the enables are the write's own,
        // apart from whether it writes the line at all.
        wire             cwe = c_whit | w_alloc | f_alloc;
        wire [BYTES-1:0] be  = byte_enables(y_wmask);
        always @(posedge clk) begin
            ca1 <= cdata[y_ridx];
            ca2 <= ca1;
        end
        always @(posedge clk) begin
            cb1 <= cdata[y_widx];
            if (cwe) begin
                for (integer j = 0; j < BYTES; ++j)
                    if (be[j])
                        cdata[y_widx][j*8 +: 8] <= y_wdata[j*8 +: 8];
            end
        end
        assign c_ca = ca2;
        assign c_cb = cb1;
    end else begin : g_direct
        assign c_rhit     = 1'b0;
        assign c_whit     = 1'b0;
        assign c_victim   = 1'b0;
        assign c_vtag     = '0;
        assign c_same_idx = 1'b0;
        assign c_ca       = '0;
        assign c_cb       = '0;
    end

    // ---- Known-zero rows: written since the reset ----
    // Row lin's bit is bit lin / KBANKS of bank lin % KBANKS. A write sets
    // its row's bit as it is taken, reading the bit first; a read reads its
    // row's bit as it is taken, so it sees every earlier write's, but not a
    // same-cycle one's (a pair's read takes its write's old bit). A later
    // write of its row comes only after its answer (guarantee 4). The reset
    // clears a word of every bank a cycle in its drain, which takes no
    // request; a write taken before it and still in the multiply then sets
    // nothing, as the reset forgets it anyway.
    if (KZ != 0) begin : g_kz
        localparam KROWS  = (1 << AW) * INSTANCES;
        localparam KBB    = (AW < 4) ? AW : 4;
        localparam KBANKS = 1 << KBB;
        localparam KDEPTH = (KROWS + KBANKS - 1) / KBANKS;
        localparam KAW    = (KDEPTH > 1) ? $clog2(KDEPTH) : 1;
        reg              zv, done, looked;
        reg [KAW-1:0]    ptr;
        wire             clear = (draining | mv_flush) & ~reset & ~done;
        wire             set   = i_wr & ~reset & ~draining;
        wire [KAW-1:0]   wa    = clear ? ptr : KAW'(i_wlin >> KBB);
        wire [KAW-1:0]   ra    = KAW'(i_rlin >> KBB);
        wire [KBANKS-1:0] wq, rq;
        for (genvar b = 0; b < KBANKS; ++b) begin : g_bank
            reg mem [KDEPTH];
            reg wo, ro;
            always @(posedge clk) begin
                wo <= mem[wa];
                ro <= mem[ra];
                if (clear | (set & (i_wlin[KBB-1:0] == KBB'(b))))
                    mem[wa] <= ~clear;
            end
            assign wq[b] = wo;
            assign rq[b] = ro;
        end
        always @(posedge clk) begin
            zv     <= zero & ~reset;
            looked <= set;
            if (reset | mv_go) begin
                ptr  <= '0;
                done <= 1'b0;
            end else if (clear) begin
                ptr <= ptr + 1'b1;
                if (ptr == KAW'(KDEPTH - 1))
                    done <= 1'b1;
            end
        end
        assign kz_w    = looked & zv & ~wq[x_wlin[KBB-1:0]];
        assign kz_r    = zv & ~rq[x_rlin[KBB-1:0]];
        assign kz_done = done;
    end else begin : g_no_kz
        /* verilator lint_off UNUSEDSIGNAL */
        wire unused = zero;
        /* verilator lint_on UNUSEDSIGNAL */
        assign kz_w    = 1'b0;
        assign kz_r    = 1'b0;
        assign kz_done = 1'b1;
    end

    // The sweep of a reset's drain or a move's flush: a line per cycle while
    // the buffer has room for what is in flight, once the writes taken before
    // it are in.
    assign sweep_go = (CACHE != 0) & (draining | mv_flush) & ~reset & ~sw_done & ~i_busy & ~x_wr & ~y_wr
                    & ~x_fl & ~y_fl & (wb_cnt <= FW'(QUEUE - 4));
    always @(posedge clk) begin
        if (reset | mv_go) begin
            sw_done <= (CACHE == 0);
            sw_idx  <= '0;
        end else if (sweep_go) begin
            sw_idx <= sw_idx + 1'b1;
            if (sw_idx == CI'(CACHE_ROWS - 1))
                sw_done <= 1'b1;
        end
    end

    // ---- Room, and the drain after a reset ----
    wire idle = sw_done & kz_done & ~i_busy & ~x_wr & ~y_wr & ~x_sw & ~y_sw & ~x_fl & ~y_fl & ~z_alloc
              & (wb_cnt == 0) & (id_busy == '0)
              & ~m_arvalid & ~m_awvalid & ~m_wvalid & ~rc_valid;

    // ---- A move: serve what was taken, then the flush ----
    // Nothing taken is in flight but writes in the buffer: no read anywhere,
    // no write still to look up its known-zero bit or reach the cache.
    wire quiet = ~i_busy & ~i_rbusy & ~x_rd & ~y_rd & ~z_rd & ~r_rd & ~r_local & (rq_count == '0)
               & (id_busy == '0) & ~rc_valid & ~x_wr & ~y_wr;
    reg  [3:0] mv_cnt;
    wire       mv_start = move & ~mv_busy & ~reset & ~draining;
    assign     mv_go    = mv_busy & ~mv_flush & quiet & (mv_cnt == 4'(QUIET));
    wire       mv_end   = mv_flush & idle;
    always @(posedge clk) begin
        if (reset) begin
            mv_busy  <= 1'b0;
            mv_flush <= 1'b0;
        end else begin
            if (mv_start)
                mv_busy <= 1'b1;
            else if (mv_end)
                mv_busy <= 1'b0;
            if (mv_go)
                mv_flush <= 1'b1;
            else if (mv_end)
                mv_flush <= 1'b0;
        end
        // Cycles since the last request (QUIET at most).
        if (mv_start | rd_valid | wr_valid)
            mv_cnt <= '0;
        else if (mv_cnt != 4'(QUIET))
            mv_cnt <= mv_cnt + 1'b1;
    end
    assign moving    = mv_busy;
    assign base_hold = reset | (draining & ~idle) | (mv_busy & ~mv_end) | move;

    // Room given back counts a cycle late: free only rises later. room counts
    // requests and their room until a move's flush too (free shows 0 from the
    // move); the flush, as the drain, ends with everything back.
    reg  [2:0]    rel_q;
    reg  [FW-1:0] room;
    // Outside reset and its drain, every request is taken.
    wire [3:0]    delta  = {1'b0, rel_q} - 4'(rd_valid) - 4'(wr_valid);
    wire [FW-1:0] room_n = reset ? '0
                         : (draining | mv_flush) ? (idle ? FW'(QUEUE) : '0)
                         : room + FW'($signed(delta));

    always @(posedge clk) begin
        rel_q <= (reset | draining | mv_flush) ? 3'd0
               : 3'(r_local) + 3'(rq_issue) + 3'(b_pop) + 3'(y_wdone);
        if (reset)
            draining <= 1'b1;
        else if (draining & idle)
            draining <= 1'b0;
        room <= room_n;
        if (mv_start | (mv_busy & ~mv_end))
            free <= '0;
        else
            free <= room_n;
    end

`ifdef SIMULATION
    // The simulator's guarantees, and the protocol memory must keep.
    reg sim_pending [1 << IT];
    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < (1 << IT); ++i)
                sim_pending[i] <= 1'b0;
        end else begin
            if ((rd_valid && 32'(rd_inst[IW-1:0]) >= INSTANCES) || (wr_valid && 32'(wr_inst[IW-1:0]) >= INSTANCES))
                $error("foldsim_vmem: an instance index past INSTANCES");
            if ((FW+1)'(rd_valid) + (FW+1)'(wr_valid) > (FW+1)'(room))
                $error("foldsim_vmem: a request without room (%0d)", room);
            if ((rd_valid | wr_valid) & mv_flush)
                $error("foldsim_vmem: a request in a move's flush, QUIET cycles after free dropped to 0");
            if (rd_valid & sim_pending[rd_inst])
                $error("foldsim_vmem: a second read of instance %0d outstanding", rd_inst);
            if (wr_valid & sim_pending[wr_inst])
                $error("foldsim_vmem: a write of instance %0d, whose read is outstanding", wr_inst);
            if (rsp_valid)
                sim_pending[rsp_inst] <= 1'b0;
            if (taken_rd)
                sim_pending[rd_inst] <= 1'b1;
        end
        if ((m_rvalid & m_rready) && (32'(m_rid) >= OUTSTANDING || !id_busy[OW'(m_rid)]))
            $error("foldsim_vmem: R for id %0d, which has no read", m_rid);
        if (rc_valid && (rc_rlast != rc_last))
            $error("foldsim_vmem: R's last beat out of place for id %0d", rc_id);
        if (m_bvalid & ((n_b == 0) | (m_bid != '0)))
            $error("foldsim_vmem: B with no write, or of id %0d", m_bid);
        if (r_local & rc_done)
            $error("foldsim_vmem: two responses in one cycle");
        if (z_alloc & (wb_cnt == FW'(QUEUE)) & ~b_pop)
            $error("foldsim_vmem: the write buffer overflows");
        if ((base_q & ((64'd1 << RB) - 1)) != 0)
            $error("foldsim_vmem: base not aligned to a row");
    end
`endif

endmodule
