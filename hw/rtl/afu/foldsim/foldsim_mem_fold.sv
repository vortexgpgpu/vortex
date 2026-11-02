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

// The folded memory ports (PORTFOLD_IFACE.md): the REQ / RSP record streams
// of foldsim_mem_par, served by ONE memory endpoint threaded PORTS ways
// (foldsim_mem_threaded_foldsim_mem: foldsim_mem through Foldsim's FAME-5
// pass, hw/syn/foldsim/portfold_gen.sh, from a frozen snapshot) and one AXI4
// master. The threaded endpoint runs thread `t` a host cycle at a time,
// in turn; at its turn this module is that port's surroundings, exactly as
// foldsim_mem_par's are for an endpoint of its own: its REQ record's tokens
// (offered until taken), its RSP record's fields (taken once each), and its
// AXI master (requests into the shared master's queues, its R and B from a
// buffer of its own). Every channel of the thread is valid / ready at its
// turn, so each turn is one of the port's endpoint clocks (the first, after
// the module's reset, with no edge; the second in the endpoint's reset).
//
// Exact by construction: a port's endpoint state lives in its thread and its
// surroundings in its own buffers; what the endpoint offers the target is its
// DRAM model's (foldsim_mem), and the shared master's back-pressure and
// latency only delay the host cycle a port's target cycle finishes on.
// The shared master: one AXI id, so R and B in order; a FIFO of port tags
// for each returns them to the port that asked, and a request is taken only
// with room for its tag. A port's W waits in a buffer of its own until its AW
// is taken, so W beats follow AWs across ports.
`include "foldsim_mem_rec.vh"

module foldsim_mem_fold #(
    parameter PORTS    = 32,
    parameter ID_W     = 9,
    parameter ADDR_W   = 64,
    parameter DATA_W   = 512,
    // The bank model's slots and ring, and the token ends: the threaded
    // snapshot is made with foldsim_mem's defaults (portfold_gen.sh).
    parameter SLOTS    = 32,
    parameter HLOG2    = 9,
    parameter STAGED   = 0,
    parameter SPEC     = 2,
    parameter NC       = (SPEC == 2) ? 4 : ((SPEC != 0) ? 2 : 1),
    parameter NW       = (SPEC == 2) ? 4 : 1,
    parameter CW       = (PORTS > 1) ? $clog2(PORTS) : 1,
    parameter REQ_W    = 6 + 2 * ADDR_W + 2 * ID_W + 16 + DATA_W + DATA_W / 8,
    parameter RSP_W    = 2 * NW + 2 * NC + NC * DATA_W + NC + NC * ID_W + NC * 2 + 3 + ID_W,
    // Requests the shared master may hold, and reads / writes in flight on it
    // (queued or in memory: their port tags).
    parameter QDEPTH   = 64
) (
    input  wire                   clk,
    input  wire                   reset,
    input  wire [63:0]            base,
    input  wire [15:0]            cfg_data_rate,
    input  wire [15:0]            cfg_core_clock,
    input  wire [15:0]            cfg_latency,
    input  wire [15:0]            cfg_queue,
    input  wire [7:0]             cfg_rq,
    input  wire [7:0]             cfg_wq,
    input  wire [7:0]             cfg_wr_high,
    input  wire [7:0]             cfg_wr_low,
    input  wire [12:0]            cfg_geom,
    input  wire [18*12-1:0]       cfg_t,
    input  wire [1:0]             cfg_refresh,
    input  wire [15:0]            cfg_refi,
    input  wire [15:0]            cfg_rfc,
    output wire [PORTS*8*64-1:0]  stats,
    output wire                   idle,
    output wire                   error,
    output wire                   stall,

    input  wire                   req_valid,
    output wire                   req_ready,
    input  wire [CW-1:0]          req_chan,
    input  wire                   req_treset,
    input  wire [REQ_W-1:0]       req_bits,

    output reg                    rsp_valid,
    input  wire                   rsp_ready,
    output reg  [CW-1:0]          rsp_chan,
    output reg  [RSP_W-1:0]       rsp_bits,

    // The one AXI4 master to device memory.
    output wire                   m_awvalid,
    input  wire                   m_awready,
    output wire [63:0]            m_awaddr,
    output wire [7:0]             m_awlen,
    output wire                   m_wvalid,
    input  wire                   m_wready,
    output wire [DATA_W-1:0]      m_wdata,
    output wire [DATA_W/8-1:0]    m_wstrb,
    output wire                   m_wlast,
    input  wire                   m_bvalid,
    output wire                   m_bready,
    output wire                   m_arvalid,
    input  wire                   m_arready,
    output wire [63:0]            m_araddr,
    output wire [7:0]             m_arlen,
    input  wire                   m_rvalid,
    output wire                   m_rready,
    input  wire [DATA_W-1:0]      m_rdata
);
    localparam QF = `FMR_REQ_FIELDS + 1;
    localparam SF = `FMR_RSP_FIELDS;
    localparam QW = $clog2(QDEPTH);

    function automatic integer req_off(input integer i);
        integer j, o;
        o = 0;
        for (j = 0; j < i; j = j + 1) o = o + `FMR_REQ_W(j);
        return o;
    endfunction
    function automatic integer rsp_off(input integer i);
        integer j, o;
        o = 0;
        for (j = 0; j < i; j = j + 1) o = o + `FMR_RSP_W(j);
        return o;
    endfunction
    // The simulator's own field table (foldsim_sim.vh with folded ports:
    // FOLDSIM_PORTS_REQ_W, FOLDSIM_PORTS_RSP_W, FOLDSIM_PORTS) must be this one.
    initial begin
        if (PORTS < 2)
            $error("foldsim_mem_fold: PORTS %0d; a threaded endpoint has 2 threads or more", PORTS);
        if (SLOTS != 32 || HLOG2 != 9 || STAGED != 0)
            $error("foldsim_mem_fold: the threaded endpoint is made with SLOTS 32, HLOG2 9, STAGED 0");
`ifdef FOLDSIM_PORTS_REQ_W
        if (`FOLDSIM_PORTS_REQ_W != REQ_W)
            $error("foldsim_mem_fold: the simulator's REQ record is %0d bits, not %0d", `FOLDSIM_PORTS_REQ_W, REQ_W);
`endif
`ifdef FOLDSIM_PORTS_RSP_W
        if (`FOLDSIM_PORTS_RSP_W != RSP_W)
            $error("foldsim_mem_fold: the simulator's RSP record is %0d bits, not %0d", `FOLDSIM_PORTS_RSP_W, RSP_W);
`endif
`ifdef FOLDSIM_PORTS
        if (`FOLDSIM_PORTS != PORTS)
            $error("foldsim_mem_fold: the simulator has %0d ports, not %0d", `FOLDSIM_PORTS, PORTS);
`endif
`ifdef FOLDSIM_PORTS_REQ_WDATA_LO
        if (`FOLDSIM_PORTS_REQ_WDATA_LO != req_off(5) || `FOLDSIM_PORTS_REQ_BREADY_LO != req_off(13))
            $error("foldsim_mem_fold: the simulator's REQ field offsets differ");
`endif
`ifdef FOLDSIM_PORTS_RSP_RDATA_LO
        if (`FOLDSIM_PORTS_RSP_RDATA_LO != rsp_off(4) || `FOLDSIM_PORTS_RSP_BID_LO != rsp_off(10))
            $error("foldsim_mem_fold: the simulator's RSP field offsets differ");
`endif
        if (req_off(`FMR_REQ_FIELDS) != REQ_W)
            $error("foldsim_mem_fold: REQ_W %0d is not the fields' %0d", REQ_W, req_off(`FMR_REQ_FIELDS));
        if (rsp_off(`FMR_RSP_FIELDS) != RSP_W)
            $error("foldsim_mem_fold: RSP_W %0d is not the fields' %0d", RSP_W, rsp_off(`FMR_RSP_FIELDS));
    end

    // ---- The threaded endpoint: the current thread's channels ----
    // The endpoint threaded PORTS ways (its snapshot made for this module's
    // PORTS, ID_W, DATA_W, CH_SHIFT and SPEC): the current thread.
    wire [CW-1:0] t;
    // The current thread finishes its endpoint clock this host cycle: its
    // surroundings change only then.
    wire fin;
    // The endpoint's own reset is an input of its threads like any other
    // (foldsim-fame5's hostReset only presets the registers that have one):
    // each thread takes its first endpoint clock in reset, as an unfolded
    // endpoint takes this module's reset, and exchanges nothing on it.
    reg  [PORTS-1:0] t_rst;
    wire run = fin & ~t_rst[t];

    // Each port's REQ record and the tokens taken of it.
    reg  [PORTS-1:0]  q_full;
    reg  [QF-1:0]     q_taken [PORTS];
    reg  [REQ_W-1:0]  q_bits  [PORTS];
    reg  [PORTS-1:0]  q_treset;
    assign req_ready = !q_full[req_chan];
    wire [REQ_W-1:0]  qb = q_bits[t];
    wire [QF-1:0]     q_valid = {QF{q_full[t] & ~t_rst[t]}} & ~q_taken[t];
    wire [QF-1:0]     q_ready;

    // Each port's RSP record, field by field.
    reg  [SF-1:0]     s_got [PORTS];
    reg  [RSP_W-1:0]  s_bits [PORTS];
    wire [SF-1:0]     s_valid;
    wire [SF-1:0]     s_ready = ~s_got[t] & {SF{~t_rst[t]}};
    wire [RSP_W-1:0]  s_in;

    // Each port's AXI surroundings: its read data, its write acks.
    reg  [PORTS-1:0]   r_full;
    reg  [DATA_W-1:0]  r_data [PORTS];
    reg  [5:0]         b_cnt  [PORTS];
    wire               t_awvalid, t_wvalid, t_arvalid, t_rready, t_bready, t_wlast;
    wire [63:0]        t_awaddr, t_araddr;
    wire [7:0]         t_awlen, t_arlen;
    wire [DATA_W-1:0]  t_wdata;
    wire [DATA_W/8-1:0] t_wstrb;
    // The shared master's queues: AR (port, address, length), AW, W; and the
    // ports of the reads and writes it has in flight.
    // A read or write is taken only with room for its port tag: it is in
    // flight until its R or B returns.
    reg  [QW:0]        ar_n, aw_n, w_n, rt_n, bt_n;
    wire               t_arready = (rt_n < (QW+1)'(QDEPTH));
    // The shared master's W beats follow its AWs (AXI4), and an endpoint may
    // offer a write's W before its AW (which waits for an earlier write of
    // its block): each port's W waits in a buffer of its own, and its AW is
    // taken only with it, both queued together.
    reg  [PORTS-1:0]   wb_full;
    reg  [DATA_W+DATA_W/8:0] wb_data [PORTS];
    wire               t_wready  = ~wb_full[t];
    wire               t_awready = wb_full[t] & (bt_n < (QW+1)'(QDEPTH)) & (w_n < (QW+1)'(QDEPTH));
    wire               ar_take = run & t_arvalid & t_arready;
    wire               aw_take = run & t_awvalid & t_awready;
    wire               w_take  = run & t_wvalid & t_wready;  // into the port's buffer
    wire               r_take  = run & r_full[t] & t_rready;
    wire               b_take  = run & (b_cnt[t] != 0) & t_bready;

    wire [8*64-1:0]    t_stats;
    wire               t_idle, t_error, t_stall;
    reg  [PORTS-1:0]   p_idle, p_error, p_stall;
    assign idle  = &p_idle;
    assign error = |p_error;
    assign stall = |p_stall;
    reg  [8*64-1:0]    p_stats [PORTS];
    for (genvar p = 0; p < PORTS; ++p) begin : g_stats
        assign stats[p*8*64 +: 8*64] = p_stats[p];
    end

`define FMF_Q(i) qb[req_off(i) +: `FMR_REQ_W(i)]
    // Every channel is valid and ready at every turn, so each turn finishes
    // and the channels' own handshakes are unused.
    /* verilator lint_off PINCONNECTEMPTY */
    foldsim_mem_threaded_foldsim_mem threaded (
        .hostClock (clk),
        .hostReset (reset),
        .threadIdx (t),
        .finish    (fin),
        .reset_sink_valid (1'b1), .reset_sink_bits (t_rst[t]), .reset_sink_ready (),
        .base_sink_valid (1'b1), .base_sink_bits (base), .base_sink_ready (),
        .cfg_data_rate_sink_valid (1'b1), .cfg_data_rate_sink_bits (cfg_data_rate), .cfg_data_rate_sink_ready (),
        .cfg_core_clock_sink_valid (1'b1), .cfg_core_clock_sink_bits (cfg_core_clock), .cfg_core_clock_sink_ready (),
        .cfg_latency_sink_valid (1'b1), .cfg_latency_sink_bits (cfg_latency), .cfg_latency_sink_ready (),
        .cfg_queue_sink_valid (1'b1), .cfg_queue_sink_bits (cfg_queue), .cfg_queue_sink_ready (),
        .cfg_rq_sink_valid (1'b1), .cfg_rq_sink_bits (cfg_rq), .cfg_rq_sink_ready (),
        .cfg_wq_sink_valid (1'b1), .cfg_wq_sink_bits (cfg_wq), .cfg_wq_sink_ready (),
        .cfg_wr_high_sink_valid (1'b1), .cfg_wr_high_sink_bits (cfg_wr_high), .cfg_wr_high_sink_ready (),
        .cfg_wr_low_sink_valid (1'b1), .cfg_wr_low_sink_bits (cfg_wr_low), .cfg_wr_low_sink_ready (),
        .cfg_geom_sink_valid (1'b1), .cfg_geom_sink_bits (cfg_geom), .cfg_geom_sink_ready (),
        .cfg_t_sink_valid (1'b1), .cfg_t_sink_bits (cfg_t), .cfg_t_sink_ready (),
        .cfg_refresh_sink_valid (1'b1), .cfg_refresh_sink_bits (cfg_refresh), .cfg_refresh_sink_ready (),
        .cfg_refi_sink_valid (1'b1), .cfg_refi_sink_bits (cfg_refi), .cfg_refi_sink_ready (),
        .cfg_rfc_sink_valid (1'b1), .cfg_rfc_sink_bits (cfg_rfc), .cfg_rfc_sink_ready (),
        // The port's REQ tokens: the wires the simulator's channels would drive.
        .treset_valid_sink_valid (1'b1), .treset_valid_sink_bits (q_valid[14]), .treset_valid_sink_ready (),
        .treset_bits_sink_valid (1'b1), .treset_bits_sink_bits (q_treset[t]), .treset_bits_sink_ready (),
        .awvalid_valid_sink_valid (1'b1), .awvalid_valid_sink_bits (q_valid[0]), .awvalid_valid_sink_ready (),
        .awvalid_bits_sink_valid (1'b1), .awvalid_bits_sink_bits (`FMF_Q(0)), .awvalid_bits_sink_ready (),
        .awaddr_valid_sink_valid (1'b1), .awaddr_valid_sink_bits (q_valid[1]), .awaddr_valid_sink_ready (),
        .awaddr_bits_sink_valid (1'b1), .awaddr_bits_sink_bits (`FMF_Q(1)), .awaddr_bits_sink_ready (),
        .awid_valid_sink_valid (1'b1), .awid_valid_sink_bits (q_valid[2]), .awid_valid_sink_ready (),
        .awid_bits_sink_valid (1'b1), .awid_bits_sink_bits (`FMF_Q(2)), .awid_bits_sink_ready (),
        .awlen_valid_sink_valid (1'b1), .awlen_valid_sink_bits (q_valid[3]), .awlen_valid_sink_ready (),
        .awlen_bits_sink_valid (1'b1), .awlen_bits_sink_bits (`FMF_Q(3)), .awlen_bits_sink_ready (),
        .wvalid_valid_sink_valid (1'b1), .wvalid_valid_sink_bits (q_valid[4]), .wvalid_valid_sink_ready (),
        .wvalid_bits_sink_valid (1'b1), .wvalid_bits_sink_bits (`FMF_Q(4)), .wvalid_bits_sink_ready (),
        .wdata_valid_sink_valid (1'b1), .wdata_valid_sink_bits (q_valid[5]), .wdata_valid_sink_ready (),
        .wdata_bits_sink_valid (1'b1), .wdata_bits_sink_bits (`FMF_Q(5)), .wdata_bits_sink_ready (),
        .wstrb_valid_sink_valid (1'b1), .wstrb_valid_sink_bits (q_valid[6]), .wstrb_valid_sink_ready (),
        .wstrb_bits_sink_valid (1'b1), .wstrb_bits_sink_bits (`FMF_Q(6)), .wstrb_bits_sink_ready (),
        .wlast_valid_sink_valid (1'b1), .wlast_valid_sink_bits (q_valid[7]), .wlast_valid_sink_ready (),
        .wlast_bits_sink_valid (1'b1), .wlast_bits_sink_bits (`FMF_Q(7)), .wlast_bits_sink_ready (),
        .arvalid_valid_sink_valid (1'b1), .arvalid_valid_sink_bits (q_valid[8]), .arvalid_valid_sink_ready (),
        .arvalid_bits_sink_valid (1'b1), .arvalid_bits_sink_bits (`FMF_Q(8)), .arvalid_bits_sink_ready (),
        .araddr_valid_sink_valid (1'b1), .araddr_valid_sink_bits (q_valid[9]), .araddr_valid_sink_ready (),
        .araddr_bits_sink_valid (1'b1), .araddr_bits_sink_bits (`FMF_Q(9)), .araddr_bits_sink_ready (),
        .arid_valid_sink_valid (1'b1), .arid_valid_sink_bits (q_valid[10]), .arid_valid_sink_ready (),
        .arid_bits_sink_valid (1'b1), .arid_bits_sink_bits (`FMF_Q(10)), .arid_bits_sink_ready (),
        .arlen_valid_sink_valid (1'b1), .arlen_valid_sink_bits (q_valid[11]), .arlen_valid_sink_ready (),
        .arlen_bits_sink_valid (1'b1), .arlen_bits_sink_bits (`FMF_Q(11)), .arlen_bits_sink_ready (),
        .rready_valid_sink_valid (1'b1), .rready_valid_sink_bits (q_valid[12]), .rready_valid_sink_ready (),
        .rready_bits_sink_valid (1'b1), .rready_bits_sink_bits (`FMF_Q(12)), .rready_bits_sink_ready (),
        .bready_valid_sink_valid (1'b1), .bready_valid_sink_bits (q_valid[13]), .bready_valid_sink_ready (),
        .bready_bits_sink_valid (1'b1), .bready_bits_sink_bits (`FMF_Q(13)), .bready_bits_sink_ready (),
        // The port's RSP fields' readies.
        .awready_ready_sink_valid (1'b1), .awready_ready_sink_bits (s_ready[0]), .awready_ready_sink_ready (),
        .wready_ready_sink_valid (1'b1), .wready_ready_sink_bits (s_ready[1]), .wready_ready_sink_ready (),
        .arready_ready_sink_valid (1'b1), .arready_ready_sink_bits (s_ready[2]), .arready_ready_sink_ready (),
        .rvalid_ready_sink_valid (1'b1), .rvalid_ready_sink_bits (s_ready[3]), .rvalid_ready_sink_ready (),
        .rdata_ready_sink_valid (1'b1), .rdata_ready_sink_bits (s_ready[4]), .rdata_ready_sink_ready (),
        .rlast_ready_sink_valid (1'b1), .rlast_ready_sink_bits (s_ready[5]), .rlast_ready_sink_ready (),
        .rid_ready_sink_valid (1'b1), .rid_ready_sink_bits (s_ready[6]), .rid_ready_sink_ready (),
        .rresp_ready_sink_valid (1'b1), .rresp_ready_sink_bits (s_ready[7]), .rresp_ready_sink_ready (),
        .bvalid_ready_sink_valid (1'b1), .bvalid_ready_sink_bits (s_ready[8]), .bvalid_ready_sink_ready (),
        .bresp_ready_sink_valid (1'b1), .bresp_ready_sink_bits (s_ready[9]), .bresp_ready_sink_ready (),
        .bid_ready_sink_valid (1'b1), .bid_ready_sink_bits (s_ready[10]), .bid_ready_sink_ready (),
        // The port's AXI master's inputs.
        .m_awready_sink_valid (1'b1), .m_awready_sink_bits (t_awready), .m_awready_sink_ready (),
        .m_wready_sink_valid (1'b1), .m_wready_sink_bits (t_wready), .m_wready_sink_ready (),
        .m_bvalid_sink_valid (1'b1), .m_bvalid_sink_bits (b_cnt[t] != 0), .m_bvalid_sink_ready (),
        .m_arready_sink_valid (1'b1), .m_arready_sink_bits (t_arready), .m_arready_sink_ready (),
        .m_rvalid_sink_valid (1'b1), .m_rvalid_sink_bits (r_full[t]), .m_rvalid_sink_ready (),
        .m_rdata_sink_valid (1'b1), .m_rdata_sink_bits (r_data[t]), .m_rdata_sink_ready (),
        // Its outputs, every one taken at its turn.
        .stats_source_ready (1'b1), .stats_source_valid (), .stats_source_bits (t_stats),
        .idle_source_ready (1'b1), .idle_source_valid (), .idle_source_bits (t_idle),
        .error_source_ready (1'b1), .error_source_valid (), .error_source_bits (t_error),
        .stall_source_ready (1'b1), .stall_source_valid (), .stall_source_bits (t_stall),
        .treset_ready_source_ready (1'b1), .treset_ready_source_valid (), .treset_ready_source_bits (q_ready[14]),
        .awvalid_ready_source_ready (1'b1), .awvalid_ready_source_valid (), .awvalid_ready_source_bits (q_ready[0]),
        .awaddr_ready_source_ready (1'b1), .awaddr_ready_source_valid (), .awaddr_ready_source_bits (q_ready[1]),
        .awid_ready_source_ready (1'b1), .awid_ready_source_valid (), .awid_ready_source_bits (q_ready[2]),
        .awlen_ready_source_ready (1'b1), .awlen_ready_source_valid (), .awlen_ready_source_bits (q_ready[3]),
        .wvalid_ready_source_ready (1'b1), .wvalid_ready_source_valid (), .wvalid_ready_source_bits (q_ready[4]),
        .wdata_ready_source_ready (1'b1), .wdata_ready_source_valid (), .wdata_ready_source_bits (q_ready[5]),
        .wstrb_ready_source_ready (1'b1), .wstrb_ready_source_valid (), .wstrb_ready_source_bits (q_ready[6]),
        .wlast_ready_source_ready (1'b1), .wlast_ready_source_valid (), .wlast_ready_source_bits (q_ready[7]),
        .arvalid_ready_source_ready (1'b1), .arvalid_ready_source_valid (), .arvalid_ready_source_bits (q_ready[8]),
        .araddr_ready_source_ready (1'b1), .araddr_ready_source_valid (), .araddr_ready_source_bits (q_ready[9]),
        .arid_ready_source_ready (1'b1), .arid_ready_source_valid (), .arid_ready_source_bits (q_ready[10]),
        .arlen_ready_source_ready (1'b1), .arlen_ready_source_valid (), .arlen_ready_source_bits (q_ready[11]),
        .rready_ready_source_ready (1'b1), .rready_ready_source_valid (), .rready_ready_source_bits (q_ready[12]),
        .bready_ready_source_ready (1'b1), .bready_ready_source_valid (), .bready_ready_source_bits (q_ready[13]),
        .awready_valid_source_ready (1'b1), .awready_valid_source_valid (), .awready_valid_source_bits (s_valid[0]),
        .awready_bits_source_ready (1'b1), .awready_bits_source_valid (), .awready_bits_source_bits (s_in[rsp_off(0) +: `FMR_RSP_W(0)]),
        .wready_valid_source_ready (1'b1), .wready_valid_source_valid (), .wready_valid_source_bits (s_valid[1]),
        .wready_bits_source_ready (1'b1), .wready_bits_source_valid (), .wready_bits_source_bits (s_in[rsp_off(1) +: `FMR_RSP_W(1)]),
        .arready_valid_source_ready (1'b1), .arready_valid_source_valid (), .arready_valid_source_bits (s_valid[2]),
        .arready_bits_source_ready (1'b1), .arready_bits_source_valid (), .arready_bits_source_bits (s_in[rsp_off(2) +: `FMR_RSP_W(2)]),
        .rvalid_valid_source_ready (1'b1), .rvalid_valid_source_valid (), .rvalid_valid_source_bits (s_valid[3]),
        .rvalid_bits_source_ready (1'b1), .rvalid_bits_source_valid (), .rvalid_bits_source_bits (s_in[rsp_off(3) +: `FMR_RSP_W(3)]),
        .rdata_valid_source_ready (1'b1), .rdata_valid_source_valid (), .rdata_valid_source_bits (s_valid[4]),
        .rdata_bits_source_ready (1'b1), .rdata_bits_source_valid (), .rdata_bits_source_bits (s_in[rsp_off(4) +: `FMR_RSP_W(4)]),
        .rlast_valid_source_ready (1'b1), .rlast_valid_source_valid (), .rlast_valid_source_bits (s_valid[5]),
        .rlast_bits_source_ready (1'b1), .rlast_bits_source_valid (), .rlast_bits_source_bits (s_in[rsp_off(5) +: `FMR_RSP_W(5)]),
        .rid_valid_source_ready (1'b1), .rid_valid_source_valid (), .rid_valid_source_bits (s_valid[6]),
        .rid_bits_source_ready (1'b1), .rid_bits_source_valid (), .rid_bits_source_bits (s_in[rsp_off(6) +: `FMR_RSP_W(6)]),
        .rresp_valid_source_ready (1'b1), .rresp_valid_source_valid (), .rresp_valid_source_bits (s_valid[7]),
        .rresp_bits_source_ready (1'b1), .rresp_bits_source_valid (), .rresp_bits_source_bits (s_in[rsp_off(7) +: `FMR_RSP_W(7)]),
        .bvalid_valid_source_ready (1'b1), .bvalid_valid_source_valid (), .bvalid_valid_source_bits (s_valid[8]),
        .bvalid_bits_source_ready (1'b1), .bvalid_bits_source_valid (), .bvalid_bits_source_bits (s_in[rsp_off(8) +: `FMR_RSP_W(8)]),
        .bresp_valid_source_ready (1'b1), .bresp_valid_source_valid (), .bresp_valid_source_bits (s_valid[9]),
        .bresp_bits_source_ready (1'b1), .bresp_bits_source_valid (), .bresp_bits_source_bits (s_in[rsp_off(9) +: `FMR_RSP_W(9)]),
        .bid_valid_source_ready (1'b1), .bid_valid_source_valid (), .bid_valid_source_bits (s_valid[10]),
        .bid_bits_source_ready (1'b1), .bid_bits_source_valid (), .bid_bits_source_bits (s_in[rsp_off(10) +: `FMR_RSP_W(10)]),
        .m_awvalid_source_ready (1'b1), .m_awvalid_source_valid (), .m_awvalid_source_bits (t_awvalid),
        .m_awaddr_source_ready (1'b1), .m_awaddr_source_valid (), .m_awaddr_source_bits (t_awaddr),
        .m_awlen_source_ready (1'b1), .m_awlen_source_valid (), .m_awlen_source_bits (t_awlen),
        .m_wvalid_source_ready (1'b1), .m_wvalid_source_valid (), .m_wvalid_source_bits (t_wvalid),
        .m_wdata_source_ready (1'b1), .m_wdata_source_valid (), .m_wdata_source_bits (t_wdata),
        .m_wstrb_source_ready (1'b1), .m_wstrb_source_valid (), .m_wstrb_source_bits (t_wstrb),
        .m_wlast_source_ready (1'b1), .m_wlast_source_valid (), .m_wlast_source_bits (t_wlast),
        .m_bready_source_ready (1'b1), .m_bready_source_valid (), .m_bready_source_bits (t_bready),
        .m_arvalid_source_ready (1'b1), .m_arvalid_source_valid (), .m_arvalid_source_bits (t_arvalid),
        .m_araddr_source_ready (1'b1), .m_araddr_source_valid (), .m_araddr_source_bits (t_araddr),
        .m_arlen_source_ready (1'b1), .m_arlen_source_valid (), .m_arlen_source_bits (t_arlen),
        .m_rready_source_ready (1'b1), .m_rready_source_valid (), .m_rready_source_bits (t_rready)
    );
    /* verilator lint_on PINCONNECTEMPTY */
`undef FMF_Q

    // ---- The shared master's queues ----
    reg [CW+64+8-1:0]           arq [QDEPTH];
    reg [CW+64+8-1:0]           awq [QDEPTH];
    reg [DATA_W+DATA_W/8:0]     wq  [QDEPTH];
    reg [CW-1:0]                rtag [QDEPTH];
    reg [CW-1:0]                btag [QDEPTH];
    reg [QW-1:0] ar_h, ar_t, aw_h, aw_t, w_h, w_t, rt_h, rt_t, bt_h, bt_t;
    assign m_arvalid = (ar_n != 0);
    assign {m_araddr, m_arlen} = arq[ar_h][63+8:0];
    assign m_awvalid = (aw_n != 0);
    assign {m_awaddr, m_awlen} = awq[aw_h][63+8:0];
    assign m_wvalid  = (w_n != 0);
    assign {m_wlast, m_wstrb, m_wdata} = wq[w_h];
    // R and B return to their port's buffer, which must be free.
    wire [CW-1:0] r_port = rtag[rt_h];
    wire [CW-1:0] b_port = btag[bt_h];
    assign m_rready = !r_full[r_port] || (r_take && t == r_port);
    assign m_bready = (b_cnt[b_port] != 6'd63);
    wire r_in = m_rvalid & m_rready;
    wire b_in = m_bvalid & m_bready;

    // The RSP stream: a pointer visits the ports one a host cycle, as the
    // threads take turns, and sends the record of the port it is on when
    // that record is complete.
    reg  [CW-1:0]    sp;
    wire [PORTS-1:0] s_done;
    wire             s_send = s_done[sp] & (!rsp_valid || rsp_ready);

    wire [QF-1:0] q_fire = {QF{run}} & q_valid & q_ready;
    wire [SF-1:0] s_fire = {SF{run}} & s_valid & s_ready;
    // The RSP bits the current thread's fired fields write.
    wire [RSP_W-1:0] s_mask;
    for (genvar f = 0; f < SF; ++f) begin : g_smask
        assign s_mask[rsp_off(f) +: `FMR_RSP_W(f)] = {`FMR_RSP_W(f){s_fire[f]}};
    end

    // Each port's surroundings; the current thread's change at its turn.
    for (genvar p = 0; p < PORTS; ++p) begin : g_port
        wire cur = (t == CW'(p));
        wire req_in = req_valid && req_ready && (req_chan == CW'(p));
        assign s_done[p] = &s_got[p];
        wire s_take = s_send && (sp == CW'(p));
        always @(posedge clk) begin
            if (reset) begin
                t_rst[p]   <= 1'b1;
                q_full[p]  <= 1'b0;
                q_taken[p] <= '0;
                s_got[p]   <= '0;
                r_full[p]  <= 1'b0;
                b_cnt[p]   <= '0;
                p_idle[p]  <= 1'b1;
                p_error[p] <= 1'b0;
                p_stall[p] <= 1'b0;
                wb_full[p] <= 1'b0;
            end else begin
                if (cur && fin)
                    t_rst[p] <= 1'b0;
                // Its REQ record: in when the buffer is free (req_ready),
                // out once the endpoint has taken every token of it.
                if (req_in) begin
                    q_full[p]  <= 1'b1;
                    q_taken[p] <= '0;
                end else if (cur && run && q_full[p]) begin
                    if (&(q_taken[p] | q_fire)) begin
                        q_full[p]  <= 1'b0;
                        q_taken[p] <= '0;
                    end else begin
                        q_taken[p] <= q_taken[p] | q_fire;
                    end
                end
                // Its RSP record: fields in at its turns, out on the stream.
                if (s_take)
                    s_got[p] <= '0;
                else if (cur)
                    s_got[p] <= s_got[p] | s_fire;
                if (cur && run) begin
                    p_idle[p]  <= t_idle;
                    p_error[p] <= t_error;
                    p_stall[p] <= t_stall;
                end
                // Its read data buffer and its write acks.
                if (r_in && (r_port == CW'(p)))
                    r_full[p] <= 1'b1;
                else if (cur && r_take)
                    r_full[p] <= 1'b0;
                b_cnt[p] <= b_cnt[p] + ((b_in && b_port == CW'(p)) ? 6'd1 : 6'd0)
                                     - ((cur && b_take) ? 6'd1 : 6'd0);
                if (cur && w_take)
                    wb_full[p] <= 1'b1;
                else if (cur && aw_take)
                    wb_full[p] <= 1'b0;
            end
            if (req_in) begin
                q_bits[p]   <= req_bits;
                q_treset[p] <= req_treset;
            end
            if (cur)
                s_bits[p] <= (s_bits[p] & ~s_mask) | (s_in & s_mask);
            if (cur && run)
                p_stats[p] <= t_stats;
            if (r_in && (r_port == CW'(p)))
                r_data[p] <= m_rdata;
            if (cur && w_take)
                wb_data[p] <= {t_wlast, t_wstrb, t_wdata};
        end
    end

    // The RSP stream and the shared master.
    always @(posedge clk) begin
        if (reset) begin
            rsp_valid <= 1'b0;
            sp <= '0;
            ar_n <= '0; aw_n <= '0; w_n <= '0; rt_n <= '0; bt_n <= '0;
            ar_h <= '0; ar_t <= '0; aw_h <= '0; aw_t <= '0; w_h <= '0; w_t <= '0;
            rt_h <= '0; rt_t <= '0; bt_h <= '0; bt_t <= '0;
        end else begin
            if (rsp_valid && rsp_ready)
                rsp_valid <= 1'b0;
            if (s_send)
                rsp_valid <= 1'b1;
            sp <= (sp == CW'(PORTS - 1)) ? '0 : sp + 1'b1;
            if (ar_take) begin
                ar_t <= ar_t + 1'b1;
                rt_t <= rt_t + 1'b1;
            end
            if (aw_take) begin
                aw_t <= aw_t + 1'b1;
                bt_t <= bt_t + 1'b1;
                w_t  <= w_t + 1'b1;
            end
            if (m_arvalid && m_arready) ar_h <= ar_h + 1'b1;
            if (m_awvalid && m_awready) aw_h <= aw_h + 1'b1;
            if (m_wvalid && m_wready)   w_h  <= w_h + 1'b1;
            if (r_in) rt_h <= rt_h + 1'b1;
            if (b_in) bt_h <= bt_h + 1'b1;
            ar_n <= ar_n + (QW+1)'(ar_take) - (QW+1)'(m_arvalid && m_arready);
            aw_n <= aw_n + (QW+1)'(aw_take) - (QW+1)'(m_awvalid && m_awready);
            w_n  <= w_n  + (QW+1)'(aw_take) - (QW+1)'(m_wvalid && m_wready);
            rt_n <= rt_n + (QW+1)'(ar_take) - (QW+1)'(r_in);
            bt_n <= bt_n + (QW+1)'(aw_take) - (QW+1)'(b_in);
        end
        if (s_send) begin
            rsp_chan <= sp;
            rsp_bits <= s_bits[sp];
        end
        if (ar_take) begin
            arq[ar_t]  <= {t, t_araddr, t_arlen};
            rtag[rt_t] <= t;
        end
        if (aw_take) begin
            awq[aw_t]  <= {t, t_awaddr, t_awlen};
            btag[bt_t] <= t;
            wq[w_t]    <= wb_data[t];
        end
    end
endmodule
