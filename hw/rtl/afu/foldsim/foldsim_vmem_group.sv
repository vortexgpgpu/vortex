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

// N virtual memories of one class copy behind one foldsim_vmem controller.
//
// Each member keeps the port contract of a controller of its own: its read
// and write of a host cycle, its responses tagged with the instance, its own
// `free`. A copy evaluates one instance a host cycle but may access every
// member in it; the controller takes one read and one write a host cycle, so
// the members' requests wait in a FIFO each (a read and a write of a host
// cycle are one entry, and reach the controller together: their order is
// RDW_READ_FIRST's) and leave one entry a cycle, round robin, while the
// controller has room. A member's `free` is twice its free entries, so the
// simulator's room rule (free >= 2 (L + 1)) holds per FIFO, and bubbles take
// up the rest: a member whose FIFO fills sees no room.
//
// The members' rows are disjoint rows of the controller's image: member m's
// row r is the controller's row MROW[m] | r (the compiler places the members
// largest first, so the image of member m is the image a controller of its
// own would have, at a byte offset of the region). A read
// and a write of one member keep every guarantee of the controller, per
// (member, instance, row); the controller tags each request with its member
// (foldsim_vmem TAGW) and the answer goes to that member. The host cache and
// the known-zero bits are the controller's, shared by the members (rows of
// several members may share lines, so read-first pairs pin theirs).
//
// A member's row, data and mask come in the low bits of its ports (MAW, MDW
// and MMW bits). Its data is zero-extended to the controller's DW bits; with
// byte enables (MW = DW/8), its mask enables its own bytes as it says (every
// one with a whole-row enable, MMW = 1) and the padding above them (written
// as zeros, never read), so a member's whole-row write stays whole for the
// cache.
//
// Reset and moves keep the controller's contract for every member:
//  * reset: no member is answered from then on for a request before it; the
//    FIFOs' writes still reach the controller (their reads are dropped), and
//    only then does the controller's own reset start (and its drain, which
//    writes the cache back): free stays 0 until that drain ends.
//  * move: the base is held, free drops to 0, and the FIFOs drain into the
//    controller against the old region, answered as usual; once they are
//    empty and no member has issued for QUIET cycles (the simulator's lead
//    after free dropped), the controller's own move starts, and free returns
//    when it ends.
// Nothing else is reset: as the controller does, the group carries what it
// took through a reset, from a power-up state of 0.
module foldsim_vmem_group #(
    // Members.
    parameter N              = 2,
    // The members' instance bits and instances (one class copy).
    parameter IW             = 8,
    parameter INSTANCES      = 1 << IW,
    // The controller's row bits: every member's rows, GROW_m | row.
    parameter AW             = 10,
    // The controller's row data and mask bits (DW/8 byte enables, or 1).
    parameter DW             = 512,
    parameter MW             = DW / 8,
    // Each member's row bits, data bits (at most DW), mask bits (MDW/8 byte
    // enables, or 1) and first row in the controller's image (a multiple of
    // 2^MAW): its FIFO keeps only its own bits.
    parameter int MAW [N]    = '{default: AW},
    parameter int MDW [N]    = '{default: DW},
    parameter int MMW [N]    = '{default: MW},
    parameter int MROW [N]   = '{default: 0},
    parameter RDW_READ_FIRST = 1,
    parameter QUEUE          = 32,
    parameter AXI_ID_W       = 4,
    parameter OUTSTANDING    = 16,
    parameter CACHE_ROWS     = 0,
    parameter KNOWN_ZERO     = 1,
    // FIFO entries per member, a power of two of at least 4.
    parameter FIFO           = 8,
    localparam GF            = $clog2(2 * FIFO + 1)
) (
    input  wire                        clk,
    input  wire                        reset,

    input  wire                        rd_valid [N],
    input  wire [IW-1:0]               rd_inst  [N],
    input  wire [AW-1:0]               rd_row   [N],
    input  wire                        wr_valid [N],
    input  wire [IW-1:0]               wr_inst  [N],
    input  wire [AW-1:0]               wr_row   [N],
    input  wire [DW-1:0]               wr_data  [N],
    input  wire [MW-1:0]               wr_mask  [N],

    output wire                        rsp_valid [N],
    output wire [IW-1:0]               rsp_inst,
    output wire [DW-1:0]               rsp_data,

    output reg  [GF-1:0]               free [N],

    input  wire [63:0]                 base,
    input  wire                        zero,
    input  wire                        move,
    output wire                        moving,

    output wire                        m_awvalid,
    input  wire                        m_awready,
    output wire [63:0]                 m_awaddr,
    output wire [AXI_ID_W-1:0]         m_awid,
    output wire [7:0]                  m_awlen,
    output wire                        m_wvalid,
    input  wire                        m_wready,
    output wire [511:0]                m_wdata,
    output wire [63:0]                 m_wstrb,
    output wire                        m_wlast,
    input  wire                        m_bvalid,
    output wire                        m_bready,
    input  wire [AXI_ID_W-1:0]         m_bid,
    output wire                        m_arvalid,
    input  wire                        m_arready,
    output wire [63:0]                 m_araddr,
    output wire [AXI_ID_W-1:0]         m_arid,
    output wire [7:0]                  m_arlen,
    input  wire                        m_rvalid,
    output wire                        m_rready,
    input  wire [511:0]                m_rdata,
    input  wire [AXI_ID_W-1:0]         m_rid,
    input  wire                        m_rlast
);
    localparam TW    = (N > 1) ? $clog2(N) : 1;
    localparam FW    = $clog2(QUEUE + 1);
    localparam PW    = $clog2(FIFO);
    localparam CW    = $clog2(FIFO + 1);
    // As the controller's: past the simulator's lead after free dropped.
    localparam QUIET = 15;

`ifdef SIMULATION
    initial begin
        if (N < 2)
            $error("foldsim_vmem_group: N must be at least 2 (one memory takes foldsim_vmem)");
        if (FIFO < 4 || (FIFO & (FIFO - 1)) != 0)
            $error("foldsim_vmem_group: FIFO must be a power of two of at least 4");
        for (integer m = 0; m < N; ++m) begin
            if (MDW[m] < 1 || MDW[m] > DW || MAW[m] < 1 || MAW[m] > AW)
                $error("foldsim_vmem_group: member %0d's data or row bits out of range", m);
            if (MMW[m] != 1 && (MW == 1 || MMW[m] * 8 != MDW[m]))
                $error("foldsim_vmem_group: member %0d's mask must be 1 or MDW/8 byte enables", m);
            if ((MROW[m] % (1 << MAW[m])) != 0 || MROW[m] + (1 << MAW[m]) > (1 << AW))
                $error("foldsim_vmem_group: member %0d's rows must be aligned and in the image", m);
        end
    end
`endif

    // ---- The controller's side ----
    wire              c_rd, c_wr;
    wire [IW+TW-1:0]  c_inst;
    wire [AW-1:0]     c_rrow, c_wrow;
    wire [DW-1:0]     c_wdata;
    wire [MW-1:0]     c_wmask;
    wire              c_rsp_valid;
    wire [IW+TW-1:0]  c_rsp_inst;
    wire [FW-1:0]     c_free;
    wire              c_moving;
    reg               c_reset;
    wire              c_move;
    wire [63:0]       c_base;

    // ---- Reset and moves: state ----
    // rst_pend: a reset came, the controller has not been reset yet (the
    // FIFOs drain first); c_drain: the controller's reset drain; mv_pend: a
    // move came, the controller's move has not started yet.
    reg               rst_pend, c_drain, mv_pend, mv_tail;
    reg  [3:0]        mv_cnt;
    reg  [63:0]       base_q;
`ifdef SIMULATION
    initial begin
        rst_pend = 1'b0;
        c_drain  = 1'b0;
        mv_pend  = 1'b0;
        mv_tail  = 1'b0;
        c_reset  = 1'b0;
    end
`endif

    // ---- The members' FIFOs ----
    wire [N-1:0]      push, nonempty, many, grant_pop;
    wire [CW-1:0]     cnt_next [N];
    reg  [N-1:0]      grant;
    wire              room_ok = (c_free >= FW'(2));
    wire              drop_rd = reset | rst_pend;
    wire [IW-1:0]     h_inst  [N];
    wire              h_rd    [N], h_wr [N];
    wire [AW-1:0]     h_rrow  [N], h_wrow [N];
    wire [DW-1:0]     h_wdata [N];
    wire [MW-1:0]     h_wmask [N];

    for (genvar m = 0; m < N; ++m) begin : g_member
        localparam A  = MAW[m];
        localparam D  = MDW[m];
        localparam K  = MMW[m];
        localparam EW = 2 + IW + 2 * A + D + K;
        reg  [EW-1:0] mem [FIFO];
        reg  [PW-1:0] wp, rp;
        reg  [CW-1:0] cnt;
`ifdef SIMULATION
        initial begin
            wp  = '0;
            rp  = '0;
            cnt = '0;
        end
`endif
        // A read and a write of a host cycle are one instance's.
        wire [IW-1:0] inst = wr_valid[m] ? wr_inst[m] : rd_inst[m];
        assign push[m]      = rd_valid[m] | wr_valid[m];
        assign grant_pop[m] = grant[m] & room_ok & ~c_reset;
        always @(posedge clk) begin
            if (push[m]) begin
                mem[wp] <= {rd_valid[m], wr_valid[m], inst, rd_row[m][A-1:0], wr_row[m][A-1:0],
                            wr_data[m][D-1:0], wr_mask[m][K-1:0]};
                wp <= wp + 1'b1;
            end
            if (grant_pop[m])
                rp <= rp + 1'b1;
            cnt <= cnt_next[m];
        end
        assign cnt_next[m] = cnt + CW'(push[m]) - CW'(grant_pop[m]);
        assign nonempty[m] = (cnt != '0);
        assign many[m]     = (cnt > CW'(1));
        wire [EW-1:0] h = mem[rp];
        wire [A-1:0]  rrow, wrow;
        wire [K-1:0]  mask = h[K-1:0];
        assign {h_rd[m], h_wr[m], h_inst[m], rrow, wrow} = h[EW-1 -: 2 + IW + 2 * A];
        assign h_rrow[m]  = AW'(MROW[m]) | AW'(rrow);
        assign h_wrow[m]  = AW'(MROW[m]) | AW'(wrow);
        assign h_wdata[m] = DW'(h[K +: D]);
        if (MW == 1) begin : g_row_en
            assign h_wmask[m] = MW'(mask);
        end else if (K == 1) begin : g_whole
            assign h_wmask[m] = '1;
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused_mask = mask;
            /* verilator lint_on UNUSEDSIGNAL */
        end else begin : g_bytes
            assign h_wmask[m] = ~MW'(0) << K | MW'(mask);
        end
        /* verilator lint_off UNUSEDSIGNAL */
        wire unused = (|wr_data[m]) | (|rd_row[m]) | (|wr_row[m]) | (|wr_mask[m]);
        /* verilator lint_on UNUSEDSIGNAL */

`ifdef SIMULATION
        always @(posedge clk) begin
            if (push[m] & (cnt == CW'(FIFO)) & ~grant_pop[m])
                $error("foldsim_vmem_group: member %0d's FIFO overflows", m);
            if (grant[m] & ~nonempty[m])
                $error("foldsim_vmem_group: member %0d granted while empty", m);
        end
`endif
    end

    // Room: twice the free entries, 0 while the group or the controller is
    // reset, draining or moving (and from the cycle a reset or a move comes).
    wire block = reset | rst_pend | c_reset | c_drain | move | mv_pend | c_moving;
    always @(posedge clk) begin
        for (integer m = 0; m < N; ++m)
            free[m] <= block ? '0 : GF'(2 * (FIFO - 32'(cnt_next[m])));
    end

    // ---- The grant, a cycle ahead: round robin among the members with an
    // entry next cycle, from the member after the last one granted ----
    // A member wants the grant with two entries or more, with one not
    // granted now (granted, it leaves now, but for want of room: then the
    // next cycle's grant passes it over), or with an entry coming. Each bit
    // is a wide OR of registers and the member's request: no count is added.
    reg  [N-1:0] after_q;
    wire [N-1:0] want, left;
    for (genvar m = 0; m < N; ++m) begin : g_want
        assign want[m] = many[m] | (nonempty[m] & ~grant[m]) | push[m];
        assign left[m] = (cnt_next[m] != '0);
    end
    // The lowest set bit of x, and the bits above it.
    function automatic [N-1:0] lowest(input [N-1:0] x);
        for (integer i = 0; i < N; ++i)
            lowest[i] = x[i] & ~(|(x & N'((N'(1) << i) - 1'b1)));
    endfunction
    function automatic [N-1:0] above(input [N-1:0] x);
        for (integer i = 0; i < N; ++i)
            above[i] = |(x & N'((N'(1) << i) - 1'b1));
    endfunction
    wire [N-1:0] hi     = want & after_q;
    wire         any_hi = |hi;
    wire [N-1:0] pick   = any_hi ? lowest(hi) : lowest(want);
    wire [N-1:0] next   = any_hi ? above(hi) : above(want);
`ifdef SIMULATION
    initial begin
        grant   = '0;
        after_q = '0;
    end
`endif
    always @(posedge clk) begin
        grant <= pick;
        if (|want)
            after_q <= next;
    end

    // ---- The granted entry, to the controller ----
    reg              s_rd, s_wr;
    reg [TW-1:0]     s_tag;
    reg [IW-1:0]     s_inst;
    reg [AW-1:0]     s_rrow, s_wrow;
    reg [DW-1:0]     s_wdata;
    reg [MW-1:0]     s_wmask;
    always @(*) begin
        s_rd    = 1'b0;
        s_wr    = 1'b0;
        s_tag   = '0;
        s_inst  = '0;
        s_rrow  = '0;
        s_wrow  = '0;
        s_wdata = '0;
        s_wmask = '0;
        for (integer m = 0; m < N; ++m) begin
            if (grant[m]) begin
                s_rd    = s_rd    | h_rd[m];
                s_wr    = s_wr    | h_wr[m];
                s_tag   = s_tag   | TW'(m);
                s_inst  = s_inst  | h_inst[m];
                s_rrow  = s_rrow  | h_rrow[m];
                s_wrow  = s_wrow  | h_wrow[m];
                s_wdata = s_wdata | h_wdata[m];
                s_wmask = s_wmask | h_wmask[m];
            end
        end
    end
    wire go = room_ok & ~c_reset;
    assign c_rd    = s_rd & go & ~drop_rd;
    assign c_wr    = s_wr & go;
    assign c_inst  = {s_tag, s_inst};
    assign c_rrow  = s_rrow;
    assign c_wrow  = s_wrow;
    assign c_wdata = s_wdata;
    assign c_wmask = s_wmask;

    // ---- Reset: drain the FIFOs' writes, then reset the controller ----
    wire drained_next = (left == '0);
    always @(posedge clk) begin
        c_reset  <= (reset | rst_pend) & drained_next;
        rst_pend <= reset | (rst_pend & ~c_reset);
        if (c_reset)
            c_drain <= 1'b1;
        else if (c_free == FW'(QUEUE))
            c_drain <= 1'b0;
    end

    // ---- Moves: drain the FIFOs, then move the controller ----
    wire empty  = (nonempty == '0);
    wire mv_go  = mv_pend & empty & (mv_cnt == 4'(QUIET)) & ~c_moving;
    always @(posedge clk) begin
        if (reset | rst_pend | c_reset)
            mv_pend <= 1'b0;
        else if (mv_go)
            mv_pend <= 1'b0;
        else if (move & ~c_drain)
            mv_pend <= 1'b1;
        if (move | (|push))
            mv_cnt <= '0;
        else if (mv_cnt != 4'(QUIET))
            mv_cnt <= mv_cnt + 1'b1;
    end
    // A move during the controller's drain goes straight on: the drain
    // already ends the old region's use, and the controller takes the new
    // base after it.
    assign c_move = mv_go | (move & c_drain);
    // Until free is back: a cycle past the controller's move.
    always @(posedge clk)
        mv_tail <= c_moving;
    assign moving = mv_pend | c_move | c_moving | mv_tail;

    // The base the controller sees: held from a reset or a move until the
    // controller's own reset or move holds it.
    wire hold = reset | rst_pend | move | mv_pend;
    always @(posedge clk)
        if (~hold)
            base_q <= base;
    assign c_base = hold ? base_q : base;

    // ---- Responses: to the member of the tag ----
    wire [TW-1:0] rsp_tag = c_rsp_inst[IW+TW-1:IW];
    for (genvar m = 0; m < N; ++m) begin : g_rsp
        assign rsp_valid[m] = c_rsp_valid & (rsp_tag == TW'(m)) & ~(reset | rst_pend);
    end
    assign rsp_inst = c_rsp_inst[IW-1:0];

    foldsim_vmem #(
        .IW             (IW),
        .AW             (AW),
        .DW             (DW),
        .MW             (MW),
        .RDW_READ_FIRST (RDW_READ_FIRST),
        .QUEUE          (QUEUE),
        .AXI_ID_W       (AXI_ID_W),
        .OUTSTANDING    (OUTSTANDING),
        .INSTANCES      (INSTANCES),
        .CACHE_ROWS     (CACHE_ROWS),
        .KNOWN_ZERO     (KNOWN_ZERO),
        .TAGW           (TW)
    ) vmem (
        .clk       (clk),
        .reset     (c_reset),
        .rd_valid  (c_rd),
        .rd_inst   (c_inst),
        .rd_row    (c_rrow),
        .wr_valid  (c_wr),
        .wr_inst   (c_inst),
        .wr_row    (c_wrow),
        .wr_data   (c_wdata),
        .wr_mask   (c_wmask),
        .rsp_valid (c_rsp_valid),
        .rsp_inst  (c_rsp_inst),
        .rsp_data  (rsp_data),
        .free      (c_free),
        .base      (c_base),
        .zero      (zero),
        .move      (c_move),
        .moving    (c_moving),
        .m_awvalid (m_awvalid),
        .m_awready (m_awready),
        .m_awaddr  (m_awaddr),
        .m_awid    (m_awid),
        .m_awlen   (m_awlen),
        .m_wvalid  (m_wvalid),
        .m_wready  (m_wready),
        .m_wdata   (m_wdata),
        .m_wstrb   (m_wstrb),
        .m_wlast   (m_wlast),
        .m_bvalid  (m_bvalid),
        .m_bready  (m_bready),
        .m_bid     (m_bid),
        .m_arvalid (m_arvalid),
        .m_arready (m_arready),
        .m_araddr  (m_araddr),
        .m_arid    (m_arid),
        .m_arlen   (m_arlen),
        .m_rvalid  (m_rvalid),
        .m_rready  (m_rready),
        .m_rdata   (m_rdata),
        .m_rid     (m_rid),
        .m_rlast   (m_rlast)
    );

endmodule
