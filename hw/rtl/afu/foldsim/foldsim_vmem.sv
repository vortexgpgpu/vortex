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
//   S0  the inputs, registered (two cycles more for INSTANCES not a power of
//       two: the multiply); a write is taken into the write buffer.
//   S1  the read's row is compared with every write in the buffer (and the
//       write of its own cycle): the newest earlier write of the row, if any.
//   S2  that write's slot.
//   S3  a read whose newest earlier write is a whole row answers from the
//       buffer; any other read joins the read queue, waiting for a partial
//       write's B first.
// Writes leave the buffer in order on AW and W (one AXI id, so their B's
// come back in order) and free their slot on B. Reads leave the read queue in
// order on AR, each with its own id; R data may come back in any order and
// interleaved, and its last beat answers the read.
//
// Request-queue room (`free`): a read holds an entry from the cycle it is
// taken until it answers from the buffer or goes out on AR; a write from the
// cycle it is taken until its B.
//
// Reset takes no request, drops every read not yet on AR and answers none
// from before it; every write taken before it still goes to memory. It
// finishes, with free back at QUEUE, once memory has answered every
// transaction: until then free stays 0.
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
    // Host-cache rows (0: direct).
    parameter CACHE_ROWS     = 0
) (
    input  wire                        clk,
    input  wire                        reset,

    input  wire                        rd_valid,
    input  wire [IW-1:0]               rd_inst,
    input  wire [AW-1:0]               rd_row,

    input  wire                        wr_valid,
    input  wire [IW-1:0]               wr_inst,
    input  wire [AW-1:0]               wr_row,
    input  wire [DW-1:0]               wr_data,
    input  wire [MW-1:0]               wr_mask,

    output wire                        rsp_valid,
    output reg  [IW-1:0]               rsp_inst,
    output reg  [DW-1:0]               rsp_data,

    output reg  [$clog2(QUEUE+1)-1:0]  free,

    input  wire [63:0]                 base,

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
    localparam QW     = (QUEUE > 1) ? $clog2(QUEUE) : 1;
    localparam OW     = (OUTSTANDING > 1) ? $clog2(OUTSTANDING) : 1;
    localparam LW     = AW + $clog2(INSTANCES);
    localparam BYTES  = DW / 8;
    localparam RB     = ($clog2(BYTES) > 6) ? $clog2(BYTES) : 6;
    localparam BEATS  = (DW + 511) / 512;
    localparam BW     = (BEATS > 1) ? $clog2(BEATS) : 1;
    localparam GROUP  = BYTES / MW;
    // A row's offset in the image: its bits.
    localparam OB     = LW + RB;

`ifdef SIMULATION
    initial begin
        if (CACHE_ROWS != 0)
            $error("foldsim_vmem: CACHE_ROWS must be 0");
        if ((DW % 8) != 0 || DW > 4096)
            $error("foldsim_vmem: DW must be a multiple of 8, at most 4096");
        if ((BYTES % MW) != 0)
            $error("foldsim_vmem: MW must divide DW/8");
        if (OUTSTANDING > (1 << AXI_ID_W))
            $error("foldsim_vmem: OUTSTANDING exceeds the AXI ids");
        if (INSTANCES > (1 << IW))
            $error("foldsim_vmem: INSTANCES exceeds the instance index");
        if (QUEUE < 2)
            $error("foldsim_vmem: QUEUE must be at least 2");
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

    // ---- State that outlives a reset: what memory has, or is owed ----
    // A reset drains it rather than clearing it, so it starts at its
    // power-up value: 0, as the FPGA's flops (INIT 0) do. Simulation starts
    // it there too.
    reg              draining;
    reg              s0_wr;
    reg [QUEUE-1:0]  wb_valid;
    reg [QW-1:0]     wb_tail, wb_aw, wb_w, wb_head;
    // Writes in the buffer: before AW, before their last W beat, before B.
    reg [FW-1:0]     n_aw, n_w, n_b;
    reg [BW-1:0]     w_beat;
    reg [OUTSTANDING-1:0] id_busy;
    reg              rc_valid;
`ifdef SIMULATION
    initial begin
        draining  = 1'b0;
        s0_wr     = 1'b0;
        wb_valid  = '0;
        wb_tail   = '0;
        wb_aw     = '0;
        wb_w      = '0;
        wb_head   = '0;
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

    wire accept = ~reset & ~draining;
    wire taken_rd = rd_valid & accept;
    wire taken_wr = wr_valid & accept;

    // ---- Intake: a row's index in the image, row * INSTANCES + inst ----
    wire              i_rd, i_wr, i_busy;
    wire [IW-1:0]     i_rinst;
    wire [LW-1:0]     i_rlin, i_wlin;
    wire [DW-1:0]     i_wdata;
    wire [MW-1:0]     i_wmask;
    if (INSTANCES == (1 << IW)) begin : g_lin_shift
        assign i_rd    = taken_rd;
        assign i_wr    = taken_wr;
        assign i_busy  = 1'b0;
        assign i_rinst = rd_inst;
        assign i_rlin  = LW'({rd_row, rd_inst});
        assign i_wlin  = LW'({wr_row, wr_inst});
        assign i_wdata = wr_data;
        assign i_wmask = wr_mask;
    end else begin : g_lin_mul
        // The multiply in a DSP, its input and product registered: two more
        // cycles. A write taken still reaches the buffer through a reset.
        reg          a_rd, a_wr, p_rd, p_wr;
        reg [AW-1:0] a_rrow, a_wrow;
        reg [IW-1:0] a_rinst, a_winst, p_rinst, p_winst;
        reg [LW-1:0] p_rprod, p_wprod;
        reg [DW-1:0] a_wdata, p_wdata;
        reg [MW-1:0] a_wmask, p_wmask;
`ifdef SIMULATION
        initial begin
            a_wr = 1'b0;
            p_wr = 1'b0;
        end
`endif
        always @(posedge clk) begin
            a_rd    <= taken_rd;
            a_wr    <= taken_wr;
            a_rrow  <= rd_row;
            a_wrow  <= wr_row;
            a_rinst <= rd_inst;
            a_winst <= wr_inst;
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
        assign i_rinst = p_rinst;
        assign i_rlin  = p_rprod + LW'(p_rinst);
        assign i_wlin  = p_wprod + LW'(p_winst);
        assign i_wdata = p_wdata;
        assign i_wmask = p_wmask;
    end

    // ---- S0: the requests, in the image's terms ----
    reg              s0_rd;
    reg [IW-1:0]     s0_rinst;
    reg [LW-1:0]     s0_rlin, s0_wlin;
    reg [DW-1:0]     s0_wdata;
    reg [MW-1:0]     s0_wmask;

    always @(posedge clk) begin
        s0_rd    <= i_rd & ~reset;
        s0_wr    <= i_wr;
        s0_rinst <= i_rinst;
        s0_rlin  <= i_rlin;
        s0_wlin  <= i_wlin;
        s0_wdata <= i_wdata;
        s0_wmask <= i_wmask;
    end

    // ---- The write buffer: writes in request order, until their B ----
    // A write is live until a later write of its row arrives, so a row has at
    // most one live write: a read's match needs no priority.
    reg [LW-1:0]     wb_lin  [QUEUE];
    reg [QUEUE-1:0]  wb_live, wb_full, wb_blk;
    reg [DW-1:0]     wb_data [QUEUE];
    reg [MW-1:0]     wb_mask [QUEUE];
    reg [63:0]       wb_addr [QUEUE];

    // base + offset adds only the offset's bits; above them, base or base + 1
    // (kept registered: base is static while requests flow).
    reg [63-OB:0] base_hi1;
    always @(posedge clk)
        base_hi1 <= base[63:OB] + 1'b1;
    function automatic [63:0] row_addr(input [LW-1:0] lin);
        reg [OB:0] lo;
        lo = {1'b0, base[OB-1:0]} + {1'b0, lin, {RB{1'b0}}};
        row_addr = {lo[OB] ? base_hi1 : base[63:OB], lo[OB-1:0]};
    endfunction

    wire [63:0] s0_raddr = row_addr(s0_rlin);
    wire [63:0] s0_waddr = row_addr(s0_wlin);
    wire        s0_wfull = &s0_wmask;
    wire        s0_same  = (s0_rlin == s0_wlin);
    // The write of the read's own cycle: seen by it, or held until it is read.
    wire        pair_wf  = s0_rd & s0_wr & s0_same & (RDW_READ_FIRST == 0);
    wire        pair_rf  = s0_rd & s0_wr & s0_same & (RDW_READ_FIRST != 0);

    // ---- S1: the read against the buffer ----
    reg [QUEUE-1:0] rhit, whit;
    always @(*) begin
        for (integer i = 0; i < QUEUE; ++i) begin
            rhit[i] = wb_valid[i] & wb_live[i] & (wb_lin[i] == s0_rlin);
            whit[i] = wb_valid[i] & (wb_lin[i] == s0_wlin);
        end
    end
    wire s1_fwd = s0_rd & (pair_wf ? s0_wfull : |(rhit & wb_full));

    reg              s2_rd, s2_fwd, s2_pair, s2_ub;
    reg [QUEUE-1:0]  s2_hit;
    reg [QW-1:0]     s2_tail;
    reg [IW-1:0]     s2_inst;
    reg [63:0]       s2_addr;

    always @(posedge clk) begin
        s2_rd   <= s0_rd & ~reset;
        s2_fwd  <= s1_fwd & ~reset;
        s2_hit  <= rhit;
        s2_pair <= pair_wf;
        s2_ub   <= pair_rf;
        s2_tail <= wb_tail;
        s2_inst <= s0_rinst;
        s2_addr <= s0_raddr;
        // R's last beat answers a read the cycle after it: never in the cycle
        // a forwarded read does.
        m_rready <= ~(s1_fwd & ~reset);
    end

    // ---- S2: the matching write's slot ----
    wire [QW-1:0] s2_slot = s2_pair ? s2_tail : onehot_index(s2_hit);
    wire          s2_any  = s2_pair | (|s2_hit);

    reg              s3_fwd, s3_rd, s3_part, s3_ub;
    reg [QW-1:0]     s3_slot, s3_ubslot;
    reg [IW-1:0]     s3_inst;
    reg [63:0]       s3_addr;

    always @(posedge clk) begin
        s3_fwd    <= s2_rd & s2_fwd & ~reset;
        s3_rd     <= s2_rd & ~s2_fwd & ~reset;
        s3_part   <= s2_any;
        s3_ub     <= s2_ub;
        s3_slot   <= s2_slot;
        s3_ubslot <= s2_tail;
        s3_inst   <= s2_inst;
        s3_addr   <= s2_addr;
    end

    // ---- S3: answer from the buffer, or queue for AR ----
    // A slot valid in S1 is not taken again before S3 ends (a write taken
    // needs the room its B gave back, two cycles before it reaches the
    // buffer), so its data is still the matched write's.
    wire b_pop = m_bvalid;

    reg [63:0]       rq_addr  [QUEUE];
    reg [IW-1:0]     rq_inst  [QUEUE];
    reg              rq_ub    [QUEUE];
    reg [QW-1:0]     rq_ubslot [QUEUE];
    reg [QW-1:0]     rq_dslot [QUEUE];
    reg [QUEUE-1:0]  rq_dep;
    reg [QW-1:0]     rq_head, rq_tail;
    reg [FW-1:0]     rq_count;

    wire rq_push = s3_rd;
    wire s3_dep  = s3_part & wb_valid[s3_slot] & ~(b_pop & (wb_head == s3_slot));

    // ---- AR: the read queue's head, with a free id ----
    wire          id_avail = ~(&id_busy);
    wire [OW-1:0] id_next  = first_zero(id_busy);
    wire          rq_issue = (rq_count != 0) & ~rq_dep[rq_head] & id_avail
                           & (~m_arvalid | m_arready) & ~reset;

    reg [IW-1:0]  tbl_inst  [OUTSTANDING];
    reg           tbl_ub    [OUTSTANDING];
    reg [QW-1:0]  tbl_ubslot [OUTSTANDING];
    reg [BW-1:0]  rcnt      [OUTSTANDING];

    always @(posedge clk) begin
        if (rq_push) begin
            rq_addr[rq_tail]   <= s3_addr;
            rq_inst[rq_tail]   <= s3_inst;
            rq_ub[rq_tail]     <= s3_ub;
            rq_ubslot[rq_tail] <= s3_ubslot;
            rq_dslot[rq_tail]  <= s3_slot;
        end
        // A queued read waits for its row's newest partial write: its B
        // frees the slot.
        for (integer i = 0; i < QUEUE; ++i)
            if (b_pop & (rq_dslot[i] == wb_head))
                rq_dep[i] <= 1'b0;
        if (rq_push)
            rq_dep[rq_tail] <= s3_dep;
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

    // ---- Responses: from the buffer (S3) or from R ----
    // None is offered while reset is high, not even one made before it.
    reg rsp_q;
    assign rsp_valid = rsp_q & ~reset;
    always @(posedge clk) begin
        rsp_q <= (s3_fwd | (rc_done & ~draining)) & ~reset;
        if (s3_fwd) begin
            rsp_inst <= s3_inst;
            rsp_data <= wb_data[s3_slot];
        end else begin
            rsp_inst <= tbl_inst[rc_id];
            rsp_data <= rc_row[DW-1:0];
        end
    end

    // ---- AW and W: the buffer's writes in order ----
    wire aw_issue = (n_aw != 0) & (~wb_blk[wb_aw] | draining) & (~m_awvalid | m_awready);
    wire w_issue  = (n_w != 0) & (~m_wvalid | m_wready);
    wire w_end    = w_issue & (w_beat == BW'(BEATS - 1));

    wire [BEATS*512-1:0] w_row = (BEATS*512)'(wb_data[wb_w]);
    reg  [BEATS*64-1:0]  w_strb;
    always @(*) begin
        w_strb = '0;
        for (integer j = 0; j < BYTES; ++j)
            w_strb[j] = wb_mask[wb_w][j / GROUP];
    end

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

        n_aw <= n_aw + FW'(s0_wr) - FW'(aw_issue);
        n_w  <= n_w + FW'(aw_issue) - FW'(w_end);
        n_b  <= n_b + FW'(w_end) - FW'(b_pop);
    end
    assign m_awid   = '0;
    assign m_awlen  = 8'(BEATS - 1);
    assign m_bready = 1'b1;

    // ---- The buffer's slots: taken in S0's write, freed by B ----
    always @(posedge clk) begin
        if (s0_wr) begin
            wb_lin[wb_tail]  <= s0_wlin;
            wb_data[wb_tail] <= s0_wdata;
            wb_mask[wb_tail] <= s0_wmask;
            wb_addr[wb_tail] <= s0_waddr;
        end
        for (integer i = 0; i < QUEUE; ++i)
            if (s0_wr & whit[i])
                wb_live[i] <= 1'b0;
        if (s0_wr) begin
            wb_live[wb_tail] <= 1'b1;
            wb_full[wb_tail] <= s0_wfull;
            wb_tail          <= qinc(wb_tail);
        end
        // A write held for its own cycle's read goes once that read has its
        // data.
        if (s3_fwd & s3_ub)
            wb_blk[s3_ubslot] <= 1'b0;
        if (rc_done & tbl_ub[rc_id])
            wb_blk[tbl_ubslot[rc_id]] <= 1'b0;
        if (s0_wr)
            wb_blk[wb_tail] <= pair_rf;

        if (b_pop)
            wb_head <= qinc(wb_head);
        begin : valid_
            reg [QUEUE-1:0] v;
            v = wb_valid;
            if (b_pop)
                v[wb_head] = 1'b0;
            if (s0_wr)
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

    // ---- Room, and the drain after a reset ----
    wire idle = (n_aw == 0) & (n_w == 0) & (n_b == 0) & ~i_busy & ~s0_wr & (id_busy == '0)
              & ~m_arvalid & ~m_awvalid & ~m_wvalid & ~rc_valid;
    // Room given back counts a cycle late: free only rises later.
    reg  [1:0] rel_q;
    // Outside reset and its drain, every request is taken.
    wire [2:0] delta = {1'b0, rel_q} - 3'(rd_valid) - 3'(wr_valid);

    always @(posedge clk) begin
        rel_q <= (reset | draining) ? 2'd0 : 2'(s3_fwd) + 2'(rq_issue) + 2'(b_pop);
        if (reset) begin
            draining <= 1'b1;
            free     <= '0;
        end else if (draining) begin
            if (idle) begin
                draining <= 1'b0;
                free     <= FW'(QUEUE);
            end
        end else begin
            free <= free + FW'($signed(delta));
        end
    end

`ifdef SIMULATION
    // The simulator's guarantees, and the protocol memory must keep.
    reg sim_pending [INSTANCES];
    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < INSTANCES; ++i)
                sim_pending[i] <= 1'b0;
        end else begin
            if ((FW+1)'(rd_valid) + (FW+1)'(wr_valid) > (FW+1)'(free))
                $error("foldsim_vmem: a request without room (free %0d)", free);
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
        if (s3_fwd & rc_done)
            $error("foldsim_vmem: two responses in one cycle");
        if ((base & ((64'd1 << RB) - 1)) != 0)
            $error("foldsim_vmem: base not aligned to a row");
    end
`endif

endmodule
