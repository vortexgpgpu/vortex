// The folded memory ports (foldsim_mem_fold: one endpoint threaded PORTS ways, one AXI master) against their reference
// (foldsim_mem_par: PORTS unfolded endpoints, a master each), on the same REQ records: every port's RSP records must be
// identical, record by record (the same target behaviour; host timing may differ). Each port's requests are its own
// (reads and single-beat writes in the port's interleaved blocks: CH_SHIFT 5), drawn from a hash of (port, record),
// so both see the same sequence whatever order they take them in. Device memory: an AXI model per master, answering in
// order after 1 to +lat host cycles. Plusargs: +n=<records a port> +lat=<n> +mode=random|one|all|rows
// (one: port 3 alone, back to back; all: every port every cycle; rows: few rows, many row hits; random: mixed).
// Prints PASS or the first difference.
`include "foldsim_mem_rec.vh"

module axi_mem #(parameter DATA_W = 512) (
    input  wire clk, input wire reset, input int lat,
    input  wire awvalid, output wire awready, input wire [63:0] awaddr,
    input  wire wvalid, output wire wready, input wire [DATA_W-1:0] wdata, input wire [DATA_W/8-1:0] wstrb,
    output reg bvalid, input wire bready,
    input  wire arvalid, output wire arready, input wire [63:0] araddr,
    output reg rvalid, input wire rready, output reg [DATA_W-1:0] rdata);
  bit [DATA_W-1:0] mem [longint];
  longint aq [$]; int ad [$]; int bd [$]; longint cyc = 0;
  assign arready = 1'b1;
  // A write: its AW and its W, taken together.
  assign awready = awvalid && wvalid;
  assign wready  = awvalid && wvalid;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (reset) begin aq.delete(); ad.delete(); bd.delete(); rvalid <= 0; bvalid <= 0; end
    else begin
      if (arvalid) begin aq.push_back(araddr >> 6); ad.push_back(int'(cyc) + 1 + ($urandom % lat)); end
      if (awvalid && wvalid) begin
        bit [DATA_W-1:0] old = mem.exists(awaddr >> 6) ? mem[awaddr >> 6] : '0;
        for (int b = 0; b < DATA_W / 8; b++) if (wstrb[b]) old[b*8 +: 8] = wdata[b*8 +: 8];
        mem[awaddr >> 6] = old;
        bd.push_back(int'(cyc) + 1 + ($urandom % lat));
      end
      if (rvalid && rready) rvalid <= 0;
      if ((!rvalid || rready) && aq.size() != 0 && ad[0] <= int'(cyc)) begin
        rvalid <= 1; rdata <= mem.exists(aq[0]) ? mem[aq[0]] : '0; void'(aq.pop_front()); void'(ad.pop_front());
      end
      if (bvalid && bready) bvalid <= 0;
      if ((!bvalid || bready) && bd.size() != 0 && bd[0] <= int'(cyc)) begin bvalid <= 1; void'(bd.pop_front()); end
    end
  end
endmodule

module tb;
  localparam PORTS = 32, ID_W = 9, ADDR_W = 64, DATA_W = 512, SPEC = 2, NC = 4, NW = 4, CW = 5;
  localparam REQ_W    = 6 + 2 * ADDR_W + 2 * ID_W + 16 + DATA_W + DATA_W / 8;
  localparam RSP_W = 2 * NW + 2 * NC + NC * DATA_W + NC + NC * ID_W + NC * 2 + 3 + ID_W;
  // The DRAM model's configuration: the foldsim_mem unit test's bank setups (HBM2, data rate 800, core clock 400,
  // queues 32 / 32, watermarks 26 / 7), from dram_bank_model.h (portfold_gen/cfgdump).
  localparam [18*12-1:0] CFG_T = {12'd2, 12'd2, 12'd2, 12'd6, 12'd2, 12'd2, 12'd5, 12'd2, 12'd19, 12'd14, 12'd6,
                                  12'd6, 12'd6, 12'd1, 12'd6, 12'd4, 12'd2, 12'd2};
  localparam [12:0] CFG_GEOM = 13'd6505;

  logic clk = 0, reset = 1;
  always #5 clk = ~clk;
  int n = 200, lat = 20, mode = 0; string ms;
  initial begin
    if (!$value$plusargs("n=%d", n)) n = 200;
    if (!$value$plusargs("lat=%d", lat)) lat = 20;
    if ($value$plusargs("mode=%s", ms)) mode = (ms == "one") ? 1 : (ms == "all") ? 2 : (ms == "rows") ? 3 : 0;
  end

  // Port p's record k, the same for both.
  function automatic logic [31:0] h(input int p, input int k, input int salt);
    logic [31:0] v = 32'(p * 7919 + k * 104729 + salt * 1299709) * 32'h9e3779b9;
    v ^= v >> 15; v *= 32'h2c1b3c6d; v ^= v >> 12; v *= 32'h297a2d39; v ^= v >> 15;
    return v;
  endfunction
  function automatic logic [REQ_W-1:0] rec(input int p, input int k);
    logic [REQ_W-1:0] r = '0;
    int o = 0;
    bit act = (mode == 1) ? (p == 3) : (mode == 2) ? 1'b1 : (h(p, k, 9) % 100 < 70);
    bit rd = act && (mode == 2 ? 1'b1 : (h(p, k, 1) % 100 < 45));
    bit wr = act && (mode == 2 ? 1'b1 : (h(p, k, 2) % 100 < 30));
    logic [63:0] blk = (mode == 3) ? 64'(h(p, k, 3) % 8) : 64'(h(p, k, 3) % 4096);
    logic [63:0] ra = (((blk << 5) | 64'(p)) << 6), wa = ((((64'(h(p, k, 4) % 4096)) << 5) | 64'(p)) << 6);
    if (mode == 3) wa = (((64'(h(p, k, 4) % 8) << 5) | 64'(p)) << 6);
    r[o +: 1] = wr; o += 1;                          // awvalid
    r[o +: ADDR_W] = wa; o += ADDR_W;                // awaddr
    r[o +: ID_W] = ID_W'(h(p, k, 5) % 4); o += ID_W; // awid
    o += 8;                                          // awlen 0
    r[o +: 1] = wr; o += 1;                          // wvalid
    for (int w = 0; w < DATA_W / 32; w++) r[o + w * 32 +: 32] = h(p, k, 100 + w);
    o += DATA_W;                                     // wdata
    r[o +: DATA_W / 8] = {(DATA_W / 8){1'b1}}; o += DATA_W / 8;  // wstrb
    r[o +: 1] = 1'b1; o += 1;                        // wlast
    r[o +: 1] = rd; o += 1;                          // arvalid
    r[o +: ADDR_W] = ra; o += ADDR_W;                // araddr
    r[o +: ID_W] = ID_W'(h(p, k, 6) % 4); o += ID_W; // arid
    o += 8;                                          // arlen 0
    r[o +: 1] = (mode == 2) ? 1'b1 : (h(p, k, 7) % 100 < 85); o += 1;  // rready
    r[o +: 1] = (mode == 2) ? 1'b1 : (h(p, k, 8) % 100 < 85); o += 1;  // bready
    return r;
  endfunction

`define SIDE(S) \
  logic S``_req_valid, S``_req_ready, S``_req_treset, S``_rsp_valid, S``_rsp_ready; \
  logic [CW-1:0] S``_req_chan, S``_rsp_chan; logic [REQ_W-1:0] S``_req_bits; logic [RSP_W-1:0] S``_rsp_bits; \
  int S``_k [PORTS]; int S``_turn = 0; \
  always_comb begin \
    S``_req_valid = 0; S``_req_chan = '0; S``_req_bits = '0; S``_req_treset = 0; \
    for (int j = 0; j < PORTS; j++) begin \
      automatic int p = (S``_turn + j) % PORTS; \
      if (!S``_req_valid && S``_k[p] < n) begin \
        S``_req_valid = 1; S``_req_chan = CW'(p); S``_req_bits = rec(p, S``_k[p]); S``_req_treset = (S``_k[p] < 2); \
      end \
    end \
  end \
  always @(posedge clk) if (reset) begin for (int p = 0; p < PORTS; p++) S``_k[p] <= 0; end \
    else if (S``_req_valid && S``_req_ready) begin S``_k[S``_req_chan] <= S``_k[S``_req_chan] + 1; S``_turn <= (int'(S``_req_chan) + 1) % PORTS; end \
  assign S``_rsp_ready = 1'b1;
  `SIDE(p)
  `SIDE(f)

  // The reference: PORTS endpoints, a master and memory each.
  wire [PORTS-1:0] pm_awvalid, pm_awready, pm_wvalid, pm_wready, pm_wlast, pm_bvalid, pm_bready, pm_arvalid,
                   pm_arready, pm_rvalid, pm_rready;
  wire [PORTS*64-1:0] pm_awaddr, pm_araddr; wire [PORTS*8-1:0] pm_awlen, pm_arlen;
  wire [PORTS*DATA_W-1:0] pm_wdata, pm_rdata; wire [PORTS*DATA_W/8-1:0] pm_wstrb;
  wire p_idle, p_error, p_stall, f_idle, f_error, f_stall;
  wire [PORTS*8*64-1:0] p_stats, f_stats;
`define CFG .base(64'h0), .cfg_data_rate(16'd800), .cfg_core_clock(16'd400), .cfg_latency(16'd0), \
    .cfg_queue(16'd0), .cfg_rq(8'd32), .cfg_wq(8'd32), .cfg_wr_high(8'd26), .cfg_wr_low(8'd7), \
    .cfg_geom(CFG_GEOM), .cfg_t(CFG_T), .cfg_refresh(2'd0), .cfg_refi(16'd0), .cfg_rfc(16'd0)
  foldsim_mem_par #(.PORTS(PORTS), .ID_W(ID_W), .DATA_W(DATA_W), .SPEC(SPEC), .CH_SHIFT(5)) par (
    .clk, .reset, `CFG, .stats(p_stats), .idle(p_idle), .error(p_error), .stall(p_stall),
    .req_valid(p_req_valid), .req_ready(p_req_ready), .req_chan(p_req_chan), .req_treset(p_req_treset),
    .req_bits(p_req_bits), .rsp_valid(p_rsp_valid), .rsp_ready(p_rsp_ready), .rsp_chan(p_rsp_chan),
    .rsp_bits(p_rsp_bits),
    .m_awvalid(pm_awvalid), .m_awready(pm_awready), .m_awaddr(pm_awaddr), .m_awlen(pm_awlen),
    .m_wvalid(pm_wvalid), .m_wready(pm_wready), .m_wdata(pm_wdata), .m_wstrb(pm_wstrb), .m_wlast(pm_wlast),
    .m_bvalid(pm_bvalid), .m_bready(pm_bready), .m_arvalid(pm_arvalid), .m_arready(pm_arready),
    .m_araddr(pm_araddr), .m_arlen(pm_arlen), .m_rvalid(pm_rvalid), .m_rready(pm_rready), .m_rdata(pm_rdata));
  for (genvar p = 0; p < PORTS; ++p) begin : g_pm
    axi_mem #(DATA_W) m(.clk, .reset, .lat,
      .awvalid(pm_awvalid[p]), .awready(pm_awready[p]), .awaddr(pm_awaddr[p*64 +: 64]),
      .wvalid(pm_wvalid[p]), .wready(pm_wready[p]), .wdata(pm_wdata[p*DATA_W +: DATA_W]),
      .wstrb(pm_wstrb[p*DATA_W/8 +: DATA_W/8]), .bvalid(pm_bvalid[p]), .bready(pm_bready[p]),
      .arvalid(pm_arvalid[p]), .arready(pm_arready[p]), .araddr(pm_araddr[p*64 +: 64]),
      .rvalid(pm_rvalid[p]), .rready(pm_rready[p]), .rdata(pm_rdata[p*DATA_W +: DATA_W]));
  end

  // The folded ports: one threaded endpoint, one master.
  wire fm_awvalid, fm_awready, fm_wvalid, fm_wready, fm_wlast, fm_bvalid, fm_bready, fm_arvalid, fm_arready,
       fm_rvalid, fm_rready;
  wire [63:0] fm_awaddr, fm_araddr; wire [7:0] fm_awlen, fm_arlen;
  wire [DATA_W-1:0] fm_wdata, fm_rdata; wire [DATA_W/8-1:0] fm_wstrb;
  foldsim_mem_fold #(.PORTS(PORTS), .ID_W(ID_W), .DATA_W(DATA_W), .SPEC(SPEC)) fold (
    .clk, .reset, `CFG, .stats(f_stats), .idle(f_idle), .error(f_error), .stall(f_stall),
    .req_valid(f_req_valid), .req_ready(f_req_ready), .req_chan(f_req_chan), .req_treset(f_req_treset),
    .req_bits(f_req_bits), .rsp_valid(f_rsp_valid), .rsp_ready(f_rsp_ready), .rsp_chan(f_rsp_chan),
    .rsp_bits(f_rsp_bits),
    .m_awvalid(fm_awvalid), .m_awready(fm_awready), .m_awaddr(fm_awaddr), .m_awlen(fm_awlen),
    .m_wvalid(fm_wvalid), .m_wready(fm_wready), .m_wdata(fm_wdata), .m_wstrb(fm_wstrb), .m_wlast(fm_wlast),
    .m_bvalid(fm_bvalid), .m_bready(fm_bready), .m_arvalid(fm_arvalid), .m_arready(fm_arready),
    .m_araddr(fm_araddr), .m_arlen(fm_arlen), .m_rvalid(fm_rvalid), .m_rready(fm_rready), .m_rdata(fm_rdata));
  axi_mem #(DATA_W) fm(.clk, .reset, .lat,
    .awvalid(fm_awvalid), .awready(fm_awready), .awaddr(fm_awaddr), .wvalid(fm_wvalid), .wready(fm_wready),
    .wdata(fm_wdata), .wstrb(fm_wstrb), .bvalid(fm_bvalid), .bready(fm_bready), .arvalid(fm_arvalid),
    .arready(fm_arready), .araddr(fm_araddr), .rvalid(fm_rvalid), .rready(fm_rready), .rdata(fm_rdata));

  always @(posedge clk) if ($test$plusargs("dbg") && cyc > 4 && cyc < 140)
    $display("%0d t %0d fin %0b qv %b qr %b sv %b sr %b", cyc, fold.t, fold.fin, fold.q_valid, fold.q_ready, fold.s_valid, fold.s_ready);
  // A record as the target sees it (AXI): R's data and id of copy x only with its rvalid, B's id only with bvalid.
  // The rest comes from the bank model's slot data and id registers, never reset (random here, per endpoint).
  // RSP field offsets (PORTFOLD_IFACE.md 1a): rvalid, rdata, rid, bvalid, bid.
  localparam RV_LO = 3 * NW, RD_LO = RV_LO + NC, RI_LO = RD_LO + NC * DATA_W + NC, BV_LO = RI_LO + NC * ID_W + NC * 2,
             BI_LO = BV_LO + 3;
  function automatic logic [RSP_W-1:0] seen(input logic [RSP_W-1:0] r);
    for (int x = 0; x < NC; x++)
      if (!r[RV_LO + x]) begin
        r[RD_LO + x * DATA_W +: DATA_W] = '0;
        r[RI_LO + x * ID_W +: ID_W] = '0;
      end
    if (!r[BV_LO]) r[BI_LO +: ID_W] = '0;
    return r;
  endfunction
  // Compare each port's RSP records in order.
  logic [RSP_W-1:0] pq [PORTS][$], fq [PORTS][$];
  int got [PORTS]; bit bad = 0; longint cyc = 0; int done_p = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (!reset) begin
      if (p_rsp_valid) pq[p_rsp_chan].push_back(p_rsp_bits);
      if (f_rsp_valid) fq[f_rsp_chan].push_back(f_rsp_bits);
      for (int p = 0; p < PORTS; p++)
        while (pq[p].size() != 0 && fq[p].size() != 0) begin
          if (!bad && seen(pq[p][0]) !== seen(fq[p][0])) begin
            logic [RSP_W-1:0] d = seen(pq[p][0]) ^ seen(fq[p][0]);
            int lo = 0;
            for (int f = 0; f < `FMR_RSP_FIELDS; f++) begin
              if (|((d >> lo) << (RSP_W - `FMR_RSP_W(f)))) $display("  field %0d (bit %0d) differs", f, lo);
              lo += `FMR_RSP_W(f);
            end
            $display("MISMATCH port %0d record %0d (host cycle %0d): folded %h.. reference %h..", p, got[p], cyc,
                     fq[p][0][63:0], pq[p][0][63:0]);
            bad = 1;
          end
          void'(pq[p].pop_front()); void'(fq[p].pop_front()); got[p]++;
        end
    end
  end
  initial begin
    for (int p = 0; p < PORTS; p++) got[p] = 0;
    repeat (4) @(posedge clk); reset = 0;
    forever begin
      @(posedge clk);
      done_p = 0;
      for (int p = 0; p < PORTS; p++) if (got[p] >= n - 2) done_p++;
      if (bad) begin $display("FAIL"); $finish; end
      if (done_p == PORTS) begin
        $display("PASS %0d records x %0d ports in %0d host cycles (errors: reference %0d, folded %0d)", n - 2, PORTS,
                 cyc, p_error, f_error);
        $finish;
      end
      if (cyc > 64'(n) * 64'd4000) begin
        $display("TIMEOUT at host cycle %0d (n %0d): records compared per port min %0d; REQ taken p %0d f %0d; RSP queued p %0d f %0d",
                 cyc, n, got.min()[0], p_k.sum(), f_k.sum(), pq[0].size(), fq[0].size());
        $display("  folded: ar %0d aw %0d w %0d rtags %0d btags %0d", fold.ar_n, fold.aw_n, fold.w_n, fold.rt_n, fold.bt_n);
        for (int p = 0; p < PORTS; p++)
          $display("  port %0d: k %0d got %0d q_full %b q_taken %b s_got %b r_full %b b_cnt %0d", p, f_k[p], got[p],
                   fold.q_full[p], fold.q_taken[p], fold.s_got[p], fold.r_full[p], fold.b_cnt[p]);
        $finish;
      end
    end
  end
endmodule
