// Replays a simulator's side of the folded memory ports, recorded at foldsim_mem_ser's per-port channels in a gate
// run (pfgate gate copies: ser_trace.txt, fold_trace.txt), into foldsim_mem_ser + foldsim_mem_fold, cycle for cycle:
// the channels' valid and the simulator's RSP readies as recorded, each REQ field's token bits in the recorded order.
// Prints each port's REQ / RSP record counts at the end and every RSP record's port in order (RSPLOG), to compare
// with the recording's (`R` lines). Plusargs: +trace=<ser_trace.txt> +cfg=<fold_trace.txt> +cycles=<n>.
`include "foldsim_mem_rec.vh"

module tb;
  localparam PORTS = 32, ID_W = 9, ADDR_W = 64, DATA_W = 512, SPEC = 2, NC = 4, NW = 4, CW = 5;
  localparam REQ_W = 6 + 2 * ADDR_W + 2 * ID_W + 16 + DATA_W + DATA_W / 8;
  localparam RSP_W = 2 * NW + 2 * NC + NC * DATA_W + NC + NC * ID_W + NC * 2 + 3 + ID_W;
  localparam QF = `FMR_REQ_FIELDS + 1, SF = `FMR_RSP_FIELDS;

  logic clk = 0;
  always #5 clk = ~clk;
  longint cyc = 0, ncyc = 2000000; int off = 0;
  string tpath, cpath;

  function automatic int qlo(input int f); int o = 0; for (int j = 0; j < f; j++) o += `FMR_REQ_W(j); return o; endfunction

  // The recording, as events by cycle.
  typedef struct { longint c; byte kind; int p; int f; logic [511:0] v; } ev_t;
  ev_t ev [$];
  logic [63:0] base = 0; logic [15:0] rate = 0, core = 0, lat = 0, queue = 0, refi = 0, rfc = 0;
  logic [7:0] rq = 0, wq = 0, hi = 0, lo = 0; logic [12:0] geom = 0; logic [18*12-1:0] tt = 0; logic [1:0] refresh = 0;
  // The settings as they changed in the recording ("c <cycle>" then "C ..." in the fold trace).
  typedef struct { longint c; logic [63:0] base; logic [15:0] rate, core, lat, queue, refi, rfc; logic [7:0] rq, wq, hi, lo;
                   logic [12:0] geom; logic [18*12-1:0] tt; logic [1:0] refresh; } cfg_t;
  cfg_t cf [$]; int ci = 0;
  initial begin
    int fd, n; string kind; longint c; int p, f; logic [511:0] v; logic [15:0] sr;
    if (!$value$plusargs("trace=%s", tpath)) $fatal(1, "+trace");
    if (!$value$plusargs("cfg=%s", cpath)) $fatal(1, "+cfg");
    void'($value$plusargs("cycles=%d", ncyc));
    void'($value$plusargs("off=%d", off));
    fd = $fopen(cpath, "r");
    begin
      string line; longint cc = 0;
      while ($fgets(line, fd)) begin
        cfg_t k;
        if ($sscanf(line, "c %d", c) == 1) cc = c;
        else if ($sscanf(line, "C base %h rate %d core %d lat %d queue %d rq %d wq %d hi %d lo %d geom %d t %h refresh %d refi %d rfc %d",
                         k.base, k.rate, k.core, k.lat, k.queue, k.rq, k.wq, k.hi, k.lo, k.geom, k.tt, k.refresh, k.refi, k.rfc) == 14) begin
          k.c = cc; cf.push_back(k);
        end
      end
    end
    $fclose(fd);
    $display("replay: %0d settings", cf.size());
    fd = $fopen(tpath, "r");
    while (!$feof(fd)) begin
      ev_t e;
      if ($fscanf(fd, "%s", kind) != 1) break;
      case (kind)
        "X": begin n = $fscanf(fd, "%d", c); e.c = c; e.kind = "X"; end
        "V": begin n = $fscanf(fd, "%d %d %h %h", c, p, v, sr); e.c = c; e.kind = "V"; e.p = p; e.v = v; e.f = int'(sr); end
        "F": begin n = $fscanf(fd, "%d %d %d %h", c, p, f, v); e.c = c; e.kind = "F"; e.p = p; e.f = f; e.v = v; end
        "T": begin n = $fscanf(fd, "%d %d %d", c, p, f); e.c = c; e.kind = "T"; e.p = p; e.f = f; end
        "G": begin n = $fscanf(fd, "%d %d %d", c, p, f); e.kind = "-"; end  // the simulator took an RSP field
        default: begin n = $fscanf(fd, "%d %d", c, p); e.kind = "-"; end  // Q / R: the streams, not replayed
      endcase
      if (e.kind != "-") ev.push_back(e);
    end
    $fclose(fd);
    $display("replay: %0d events", ev.size());
  end

  // The simulator side as recorded.
  logic reset = 1;
  logic [PORTS*QF-1:0] sq_valid, sq_ready; logic [PORTS*REQ_W-1:0] sq_bits; logic [PORTS-1:0] sq_treset;
  logic [PORTS*SF-1:0] ss_valid, ss_ready; logic [PORTS*RSP_W-1:0] ss_bits;
  logic [QF-1:0] qv [PORTS]; logic [SF-1:0] sr [PORTS];
  logic [511:0] fq [PORTS][QF][$]; logic tq [PORTS][$];
  int ei = 0;
  initial for (int p = 0; p < PORTS; p++) begin qv[p] = '0; sr[p] = '0; end
  // Events of cycle `cyc` take effect for that cycle (the recording samples at the edge ending it).
  always @(negedge clk) begin
    reset = 0;
    while (ci < cf.size() && cf[ci].c <= cyc + off) begin
      base = cf[ci].base; rate = cf[ci].rate; core = cf[ci].core; lat = cf[ci].lat; queue = cf[ci].queue; rq = cf[ci].rq;
      wq = cf[ci].wq; hi = cf[ci].hi; lo = cf[ci].lo; geom = cf[ci].geom; tt = cf[ci].tt; refresh = cf[ci].refresh;
      refi = cf[ci].refi; rfc = cf[ci].rfc; ci++;
    end
    while (ei < ev.size() && ev[ei].c <= cyc + off) begin
      case (ev[ei].kind)
        "X": reset = 1;
        "V": begin qv[ev[ei].p] = QF'(ev[ei].v); sr[ev[ei].p] = SF'(ev[ei].f); end
        "F": fq[ev[ei].p][ev[ei].f].push_back(ev[ei].v);
        "T": tq[ev[ei].p].push_back(ev[ei].f[0]);
      endcase
      ei++;
    end
    drive();
  end
  // Driven procedurally after each negedge's events and each posedge's pops (an always_comb over queues may not
  // re-evaluate when a queue changes).
  task automatic drive();
    for (int p = 0; p < PORTS; p++) begin
      sq_valid[p*QF +: QF] = qv[p];
      for (int f = 0; f < QF - 1; f++) begin
        automatic logic [511:0] b = (fq[p][f].size() != 0) ? fq[p][f][0] : '0;
        for (int i = 0; i < `FMR_REQ_W(f); i++) sq_bits[p*REQ_W + qlo(f) + i] = b[i];
      end
      sq_treset[p] = (tq[p].size() != 0) ? tq[p][0] : 1'b0;
      ss_ready[p*SF +: SF] = sr[p];
    end
  endtask
  // A token taken leaves its queue (the recording has one F / T line per token taken).
  always @(posedge clk) begin
    cyc <= cyc + 1;
    for (int p = 0; p < PORTS; p++) begin
      for (int f = 0; f < QF - 1; f++)
        if (sq_valid[p*QF+f] && sq_ready[p*QF+f]) begin
          if (fq[p][f].size() == 0) $display("REPLAY %0d: port %0d field %0d taken with no recorded token", cyc, p, f);
          else void'(fq[p][f].pop_front());
        end
      if (sq_valid[p*QF+QF-1] && sq_ready[p*QF+QF-1] && tq[p].size() != 0) void'(tq[p].pop_front());
    end
  end

  // The device under test.
  logic req_valid, req_ready, req_treset, rsp_valid, rsp_ready;
  logic [CW-1:0] req_chan, rsp_chan; logic [REQ_W-1:0] req_bits; logic [RSP_W-1:0] rsp_bits;
  foldsim_mem_ser #(.PORTS(PORTS), .ID_W(ID_W), .DATA_W(DATA_W), .SPEC(SPEC)) ser (
    .clk, .reset, .q_valid(sq_valid), .q_ready(sq_ready), .q_bits(sq_bits), .q_treset(sq_treset),
    .s_valid(ss_valid), .s_ready(ss_ready), .s_bits(ss_bits),
    .req_valid, .req_ready, .req_chan, .req_treset, .req_bits, .rsp_valid, .rsp_ready, .rsp_chan, .rsp_bits);
  wire m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready, m_arvalid, m_arready, m_rvalid, m_rready;
  wire [63:0] m_awaddr, m_araddr; wire [7:0] m_awlen, m_arlen; wire [DATA_W-1:0] m_wdata, m_rdata; wire [DATA_W/8-1:0] m_wstrb;
  wire [PORTS*8*64-1:0] stats; wire idle, error, stall;
  foldsim_mem_fold #(.PORTS(PORTS), .ID_W(ID_W), .DATA_W(DATA_W), .SPEC(SPEC)) fold (
    .clk, .reset, .base, .cfg_data_rate(rate), .cfg_core_clock(core), .cfg_latency(lat), .cfg_queue(queue),
    .cfg_rq(rq), .cfg_wq(wq), .cfg_wr_high(hi), .cfg_wr_low(lo), .cfg_geom(geom), .cfg_t(tt), .cfg_refresh(refresh),
    .cfg_refi(refi), .cfg_rfc(rfc), .stats, .idle, .error, .stall,
    .req_valid, .req_ready, .req_chan, .req_treset, .req_bits, .rsp_valid, .rsp_ready, .rsp_chan, .rsp_bits,
    .m_awvalid, .m_awready, .m_awaddr, .m_awlen, .m_wvalid, .m_wready, .m_wdata, .m_wstrb, .m_wlast,
    .m_bvalid, .m_bready, .m_arvalid, .m_arready, .m_araddr, .m_arlen, .m_rvalid, .m_rready, .m_rdata);
  // Device memory: none was touched before the recorded stall; an AXI slave that answers in order, at once.
  assign m_arready = 1'b1; assign m_awready = m_awvalid & m_wvalid; assign m_wready = m_awvalid & m_wvalid;
  logic [7:0] rpend = 0, bpend = 0;
  assign m_rvalid = (rpend != 0); assign m_rdata = '0; assign m_bvalid = (bpend != 0);
  always @(posedge clk) begin
    rpend <= rpend + 8'(m_arvalid) - 8'(m_rvalid && m_rready);
    bpend <= bpend + 8'(m_awvalid && m_wvalid) - 8'(m_bvalid && m_bready);
  end

  int nreq [PORTS], nrsp [PORTS];
  initial for (int p = 0; p < PORTS; p++) begin nreq[p] = 0; nrsp[p] = 0; end
  always @(posedge clk) begin
    if (req_valid && req_ready) begin nreq[req_chan]++; $display("REQLOG %0d %0d", cyc, req_chan); end
    if (rsp_valid && rsp_ready) begin nrsp[rsp_chan]++; $display("RSPLOG %0d %0d", cyc, rsp_chan); end
    if (fold.ar_take) $display("AXILOG AR %0d %0d %h", cyc, fold.t, fold.t_araddr);
    if (fold.aw_take) $display("AXILOG AW %0d %0d %h", cyc, fold.t, fold.t_awaddr);
    if (fold.r_in) $display("AXILOG RI %0d %0d", cyc, fold.r_port);
    if (fold.b_in) $display("AXILOG BI %0d %0d", cyc, fold.b_port);
  if ((fold.t == 12 || fold.t == 13) && fold.fin) $display("ILOG I %0d t %0d bm.a_state %h bm.state %h bm.caught %h bm.active %h g_rlag_cnt %h t_treset2Fgot %h t_awvalid2Fgot %h t_arvalid2Fgot %h pending_aw %h pending_w %h rd_out %h wr_out %h bm.ended %h clk_enabled %h bm.iq_count %h bm.slots_used %h", cyc, fold.t, fold.threaded.bank_model2Fa_state_ext.Memory[fold.t], fold.threaded.bank_model2Fstate_ext.Memory[fold.t], fold.threaded.bank_model2Fcaught_ext.Memory[fold.t], fold.threaded.bank_model2Factive_ext.Memory[fold.t], fold.threaded.g_rlag_cnt_ext.Memory[fold.t], fold.threaded.t_treset2Fgot_ext.Memory[fold.t], fold.threaded.t_awvalid2Fgot_ext.Memory[fold.t], fold.threaded.t_arvalid2Fgot_ext.Memory[fold.t], fold.threaded.pending_aw_ext.Memory[fold.t], fold.threaded.pending_w_ext.Memory[fold.t], fold.threaded.rd_out_ext.Memory[fold.t], fold.threaded.wr_out_ext.Memory[fold.t], fold.threaded.bank_model2Fended_ext.Memory[fold.t], fold.threaded.clk_enabled_ext.Memory[fold.t], fold.threaded.bank_model2Fiq_count_ext.Memory[fold.t], fold.threaded.bank_model2Fslots_used_ext.Memory[fold.t]);
    if ((fold.t == 12 || fold.t == 13) && fold.fin)
      $display("ELOG E %0d t %0d rst %b qfull %b qtaken %h qv %h qr %h sgot %h sv %h sr %h", cyc, fold.t, fold.t_rst[fold.t],
               fold.q_full[fold.t], fold.q_taken[fold.t], fold.q_valid, fold.q_ready, fold.s_got[fold.t], fold.s_valid, fold.s_ready);
    if (cyc == ncyc) begin
      for (int p = 0; p < PORTS; p++) $display("REPLAY port %0d req %0d rsp %0d", p, nreq[p], nrsp[p]);
      $finish;
    end
  end
endmodule
