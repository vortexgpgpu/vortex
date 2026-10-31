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

// N AXI4 masters (foldsim_vmem controllers) on one device-memory master, for
// a kernel with more virtual memories than masters.
//
// The ids pass through unchanged: AXI returns the responses of one id in the
// order of its requests, so a FIFO per id of the controllers that issued them
// says where each R burst and each B goes, whichever controller's id it is.
// Several controllers may use one id at once; their transactions then simply
// complete in order. W follows AW's order (AXI4 has no write interleaving): a
// FIFO of the controllers whose AW went out gives W's source, and a burst's
// last beat moves to the next. A controller's AW is taken once there is room
// for its B and its W turn, and an AR once there is room for its R.
//
// AR, AW and W leave from registers (a controller's request is taken as the
// register loads, from the controller granted a cycle ahead, round robin
// among those offering one); R and B arrive in a
// two-entry buffer each, so every ready towards the master is a register too.
// W is offered as soon as its AW is on the master (not after AWREADY), as
// AXI asks.
//
// Nothing here is reset: as the controllers do, it carries every transaction
// on the master through a kernel reset to its end, from a power-up state of 0.
module foldsim_vmem_share #(
    // The controllers.
    parameter N      = 2,
    // The AXI id width, the controllers' and the master's.
    parameter ID_W   = 4,
    // Reads of one id in flight across the controllers.
    parameter RDEPTH = N,
    // Writes in flight (AW out, B not back), in all and of one id.
    parameter WDEPTH = 32 * N
) (
    input  wire                  clk,

    input  wire                  s_awvalid [N],
    output wire                  s_awready [N],
    input  wire [63:0]           s_awaddr  [N],
    input  wire [ID_W-1:0]       s_awid    [N],
    input  wire [7:0]            s_awlen   [N],
    input  wire                  s_wvalid  [N],
    output wire                  s_wready  [N],
    input  wire [511:0]          s_wdata   [N],
    input  wire [63:0]           s_wstrb   [N],
    input  wire                  s_wlast   [N],
    output wire                  s_bvalid  [N],
    input  wire                  s_bready  [N],
    output wire [ID_W-1:0]       s_bid     [N],
    input  wire                  s_arvalid [N],
    output wire                  s_arready [N],
    input  wire [63:0]           s_araddr  [N],
    input  wire [ID_W-1:0]       s_arid    [N],
    input  wire [7:0]            s_arlen   [N],
    output wire                  s_rvalid  [N],
    input  wire                  s_rready  [N],
    output wire [511:0]          s_rdata   [N],
    output wire [ID_W-1:0]       s_rid     [N],
    output wire                  s_rlast   [N],

    output reg                   m_awvalid,
    input  wire                  m_awready,
    output reg  [63:0]           m_awaddr,
    output reg  [ID_W-1:0]       m_awid,
    output reg  [7:0]            m_awlen,
    output reg                   m_wvalid,
    input  wire                  m_wready,
    output reg  [511:0]          m_wdata,
    output reg  [63:0]           m_wstrb,
    output reg                   m_wlast,
    input  wire                  m_bvalid,
    output reg                   m_bready,
    input  wire [ID_W-1:0]       m_bid,
    output reg                   m_arvalid,
    input  wire                  m_arready,
    output reg  [63:0]           m_araddr,
    output reg  [ID_W-1:0]       m_arid,
    output reg  [7:0]            m_arlen,
    input  wire                  m_rvalid,
    output reg                   m_rready,
    input  wire [511:0]          m_rdata,
    input  wire [ID_W-1:0]       m_rid,
    input  wire                  m_rlast
);
    localparam SW  = (N > 1) ? $clog2(N) : 1;
    localparam NID = 1 << ID_W;
    localparam RPW = (RDEPTH > 1) ? $clog2(RDEPTH) : 1;
    localparam RCW = $clog2(RDEPTH + 1);
    localparam WPW = (WDEPTH > 1) ? $clog2(WDEPTH) : 1;
    localparam WCW = $clog2(WDEPTH + 1);

    // The lowest bit `v` holds, one-hot; its index.
    function automatic [N-1:0] first_one(input [N-1:0] v);
        first_one = v & (~v + 1'b1);
    endfunction
    function automatic [SW-1:0] index_of(input [N-1:0] oh);
        index_of = '0;
        for (integer i = 0; i < N; ++i)
            if (oh[i])
                index_of = index_of | SW'(i);
    endfunction

    // ---- Where R bursts go: per id, the controllers whose ARs went out ----
    reg [SW-1:0]  rf_mem  [NID][RDEPTH];
    reg [RPW-1:0] rf_head [NID];
    reg [RPW-1:0] rf_tail [NID];
    reg [RCW-1:0] rf_cnt  [NID];

    // ---- Where B's go, per id; and W's sources, in AW's order ----
    reg [SW-1:0]  bf_mem  [NID][WDEPTH];
    reg [WPW-1:0] bf_head [NID];
    reg [WPW-1:0] bf_tail [NID];
    reg [WCW-1:0] bf_cnt  [NID];
    reg [SW-1:0]  wo_mem  [WDEPTH];
    reg [WPW-1:0] wo_head, wo_tail;
    reg [WCW-1:0] wo_cnt;

    // Each request channel's grant, chosen a cycle ahead: the controller it
    // takes from (one-hot, and its index), and the controllers after it.
    reg [N-1:0]   ar_goh, ar_above, aw_goh, aw_above;
    reg [SW-1:0]  ar_g, aw_g;

    // R and B buffers: two entries each.
    reg [SW-1:0]   rq_dst  [2];
    reg [ID_W-1:0] rq_id   [2];
    reg [511:0]    rq_data [2];
    reg            rq_last [2];
    reg            rq_h, rq_t;
    reg [1:0]      rq_cnt;
    reg [SW-1:0]   bq_dst  [2];
    reg [ID_W-1:0] bq_id   [2];
    reg            bq_h, bq_t;
    reg [1:0]      bq_cnt;

    // Their power-up value is 0, as the FPGA's flops have it (the controllers'
    // too); simulation starts there.
`ifdef SIMULATION
    initial begin
        for (integer i = 0; i < NID; ++i) begin
            rf_head[i] = '0;
            rf_tail[i] = '0;
            rf_cnt[i]  = '0;
            bf_head[i] = '0;
            bf_tail[i] = '0;
            bf_cnt[i]  = '0;
        end
        wo_head   = '0;
        wo_tail   = '0;
        wo_cnt    = '0;
        ar_goh    = '0;
        ar_above  = '0;
        ar_g      = '0;
        aw_goh    = '0;
        aw_above  = '0;
        aw_g      = '0;
        rq_h      = 1'b0;
        rq_t      = 1'b0;
        rq_cnt    = '0;
        bq_h      = 1'b0;
        bq_t      = 1'b0;
        bq_cnt    = '0;
        m_awvalid = 1'b0;
        m_wvalid  = 1'b0;
        m_arvalid = 1'b0;
        m_rready  = 1'b0;
        m_bready  = 1'b0;
    end
`endif

    // ---- AR ----
    wire [N-1:0] ar_want;
    for (genvar i = 0; i < N; ++i) begin : g_ar_want
        assign ar_want[i] = s_arvalid[i] & (rf_cnt[s_arid[i]] != RCW'(RDEPTH));
    end
    wire          ar_load = ~m_arvalid | m_arready;
    wire [N-1:0]  ar_hit  = ar_goh & ar_want;
    wire          ar_take = ar_load & (|ar_hit);
    for (genvar i = 0; i < N; ++i) begin : g_ar_ready
        assign s_arready[i] = ar_load & ar_hit[i];
    end
    // The granted controller's request, by its one-hot grant.
    reg [63:0]     ar_addr_g;
    reg [ID_W-1:0] ar_id_g;
    reg [7:0]      ar_len_g;
    always @(*) begin
        ar_addr_g = '0;
        ar_id_g   = '0;
        ar_len_g  = '0;
        for (integer i = 0; i < N; ++i)
            if (ar_goh[i]) begin
                ar_addr_g = ar_addr_g | s_araddr[i];
                ar_id_g   = ar_id_g | s_arid[i];
                ar_len_g  = ar_len_g | s_arlen[i];
            end
    end
    // The grant moves on after a take, or from a controller that offers
    // nothing: to the next one after it that does, round robin.
    wire [N-1:0] ar_later = ar_want & ar_above;
    wire [N-1:0] ar_next  = (|ar_later) ? first_one(ar_later) : first_one(ar_want);
    wire         ar_move  = (ar_take | ~(|ar_hit)) & (|ar_next);
    always @(posedge clk) begin
        if (ar_take) begin
            m_araddr <= ar_addr_g;
            m_arid   <= ar_id_g;
            m_arlen  <= ar_len_g;
        end
        if (ar_take)
            m_arvalid <= 1'b1;
        else if (m_arready)
            m_arvalid <= 1'b0;
        if (ar_move) begin
            ar_goh   <= ar_next;
            ar_above <= ~((ar_next << 1) - 1'b1);
            ar_g     <= index_of(ar_next);
        end
    end

    // ---- AW, and W's turn ----
    wire wo_room = (wo_cnt != WCW'(WDEPTH));
    wire [N-1:0] aw_want;
    for (genvar i = 0; i < N; ++i) begin : g_aw_want
        assign aw_want[i] = s_awvalid[i] & (bf_cnt[s_awid[i]] != WCW'(WDEPTH)) & wo_room;
    end
    wire          aw_load = ~m_awvalid | m_awready;
    wire [N-1:0]  aw_hit  = aw_goh & aw_want;
    wire          aw_take = aw_load & (|aw_hit);
    for (genvar i = 0; i < N; ++i) begin : g_aw_ready
        assign s_awready[i] = aw_load & aw_hit[i];
    end
    reg [63:0]     aw_addr_g;
    reg [ID_W-1:0] aw_id_g;
    reg [7:0]      aw_len_g;
    always @(*) begin
        aw_addr_g = '0;
        aw_id_g   = '0;
        aw_len_g  = '0;
        for (integer i = 0; i < N; ++i)
            if (aw_goh[i]) begin
                aw_addr_g = aw_addr_g | s_awaddr[i];
                aw_id_g   = aw_id_g | s_awid[i];
                aw_len_g  = aw_len_g | s_awlen[i];
            end
    end
    wire [N-1:0] aw_later = aw_want & aw_above;
    wire [N-1:0] aw_next  = (|aw_later) ? first_one(aw_later) : first_one(aw_want);
    wire         aw_move  = (aw_take | ~(|aw_hit)) & (|aw_next);
    always @(posedge clk) begin
        if (aw_take) begin
            m_awaddr <= aw_addr_g;
            m_awid   <= aw_id_g;
            m_awlen  <= aw_len_g;
        end
        if (aw_take)
            m_awvalid <= 1'b1;
        else if (m_awready)
            m_awvalid <= 1'b0;
        if (aw_move) begin
            aw_goh   <= aw_next;
            aw_above <= ~((aw_next << 1) - 1'b1);
            aw_g     <= index_of(aw_next);
        end
    end

    // ---- W: the burst of the oldest AW whose W is not done ----
    wire [SW-1:0] w_src  = wo_mem[wo_head];
    wire          w_load = ~m_wvalid | m_wready;
    wire          w_turn = (wo_cnt != 0);
    wire          w_take = w_load & w_turn & s_wvalid[w_src];
    wire          w_end  = w_take & s_wlast[w_src];
    for (genvar i = 0; i < N; ++i) begin : g_w_ready
        assign s_wready[i] = w_load & w_turn & (w_src == SW'(i));
    end
    always @(posedge clk) begin
        if (w_take) begin
            m_wdata <= s_wdata[w_src];
            m_wstrb <= s_wstrb[w_src];
            m_wlast <= s_wlast[w_src];
        end
        if (w_take)
            m_wvalid <= 1'b1;
        else if (m_wready)
            m_wvalid <= 1'b0;
        if (aw_take) begin
            wo_mem[wo_tail] <= aw_g;
            wo_tail <= (32'(wo_tail) == WDEPTH - 1) ? '0 : wo_tail + 1'b1;
        end
        if (w_end)
            wo_head <= (32'(wo_head) == WDEPTH - 1) ? '0 : wo_head + 1'b1;
        wo_cnt <= wo_cnt + WCW'(aw_take) - WCW'(w_end);
    end

    // ---- The route FIFOs: pushed as a request goes out, popped as its
    // response comes in (R's last beat, B) ----
    wire r_in  = m_rvalid & m_rready;
    wire r_pop = r_in & m_rlast;
    wire b_in  = m_bvalid & m_bready;
    always @(posedge clk) begin
        for (integer i = 0; i < NID; ++i) begin
            if (ar_take && ar_id_g == ID_W'(i)) begin
                rf_mem[i][rf_tail[i]] <= ar_g;
                rf_tail[i] <= (32'(rf_tail[i]) == RDEPTH - 1) ? '0 : rf_tail[i] + 1'b1;
            end
            if (r_pop && m_rid == ID_W'(i))
                rf_head[i] <= (32'(rf_head[i]) == RDEPTH - 1) ? '0 : rf_head[i] + 1'b1;
            rf_cnt[i] <= rf_cnt[i] + RCW'(ar_take && ar_id_g == ID_W'(i))
                                   - RCW'(r_pop && m_rid == ID_W'(i));
            if (aw_take && aw_id_g == ID_W'(i)) begin
                bf_mem[i][bf_tail[i]] <= aw_g;
                bf_tail[i] <= (32'(bf_tail[i]) == WDEPTH - 1) ? '0 : bf_tail[i] + 1'b1;
            end
            if (b_in && m_bid == ID_W'(i))
                bf_head[i] <= (32'(bf_head[i]) == WDEPTH - 1) ? '0 : bf_head[i] + 1'b1;
            bf_cnt[i] <= bf_cnt[i] + WCW'(aw_take && aw_id_g == ID_W'(i))
                                   - WCW'(b_in && m_bid == ID_W'(i));
        end
    end

    // ---- R: into the buffer with its controller, out to that controller ----
    wire [SW-1:0] r_dst  = rf_mem[m_rid][rf_head[m_rid]];
    wire [SW-1:0] rq_to  = rq_dst[rq_h];
    wire          rq_any = (rq_cnt != 0);
    wire          r_out  = rq_any & s_rready[rq_to];
    for (genvar i = 0; i < N; ++i) begin : g_r
        assign s_rvalid[i] = rq_any & (rq_to == SW'(i));
        assign s_rdata[i]  = rq_data[rq_h];
        assign s_rid[i]    = rq_id[rq_h];
        assign s_rlast[i]  = rq_last[rq_h];
    end
    always @(posedge clk) begin
        if (r_in) begin
            rq_dst[rq_t]  <= r_dst;
            rq_id[rq_t]   <= m_rid;
            rq_data[rq_t] <= m_rdata;
            rq_last[rq_t] <= m_rlast;
            rq_t <= ~rq_t;
        end
        if (r_out)
            rq_h <= ~rq_h;
        rq_cnt   <= rq_cnt + 2'(r_in) - 2'(r_out);
        m_rready <= (rq_cnt + 2'(r_in) - 2'(r_out)) != 2'd2;
    end

    // ---- B: likewise ----
    wire [SW-1:0] b_dst  = bf_mem[m_bid][bf_head[m_bid]];
    wire [SW-1:0] bq_to  = bq_dst[bq_h];
    wire          bq_any = (bq_cnt != 0);
    wire          b_out  = bq_any & s_bready[bq_to];
    for (genvar i = 0; i < N; ++i) begin : g_b
        assign s_bvalid[i] = bq_any & (bq_to == SW'(i));
        assign s_bid[i]    = bq_id[bq_h];
    end
    always @(posedge clk) begin
        if (b_in) begin
            bq_dst[bq_t] <= b_dst;
            bq_id[bq_t]  <= m_bid;
            bq_t <= ~bq_t;
        end
        if (b_out)
            bq_h <= ~bq_h;
        bq_cnt   <= bq_cnt + 2'(b_in) - 2'(b_out);
        m_bready <= (bq_cnt + 2'(b_in) - 2'(b_out)) != 2'd2;
    end

`ifdef SIMULATION
    always @(posedge clk) begin
        if (r_in && rf_cnt[m_rid] == 0)
            $error("foldsim_vmem_share: R for id %0d, which has no read", m_rid);
        if (b_in && bf_cnt[m_bid] == 0)
            $error("foldsim_vmem_share: B for id %0d, which has no write", m_bid);
        if (w_take && 32'(w_src) >= N)
            $error("foldsim_vmem_share: W from controller %0d", w_src);
    end
`endif

endmodule
