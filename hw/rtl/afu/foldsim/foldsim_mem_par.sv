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

// The folded memory ports' boundary (PORTFOLD_IFACE.md: a REQ and an RSP
// stream of port-tagged records, one a port and target cycle each way) in
// front of PORTS unfolded foldsim_mem endpoints: the reference the folded
// engine is tested against, and what a simulator top that emits the records
// is gated with before the folded engine exists.
//
// A REQ record of port p waits in p's buffer until p's endpoint has taken
// every one of its tokens (each field a token, as the simulator's channels
// give them; the record's treset is the cycle's reset token); the stream
// stalls while the record at its head is for a port whose buffer is full.
// The endpoint's output tokens of a cycle fill p's response record, field by
// field; a complete record leaves on RSP, the ports in turn, and only then
// does p's endpoint offer its next cycle's tokens.
`include "foldsim_mem_rec.vh"

module foldsim_mem_par #(
    parameter PORTS    = 32,
    parameter ID_W     = 4,
    parameter ADDR_W   = 64,
    parameter DATA_W   = 512,
    parameter DEPTH    = 16,
    parameter SLOTS    = 32,
    parameter HLOG2    = 9,
    parameter STAGED   = 0,
    parameter CH_SHIFT = 0,
    parameter SPEC     = 0,
    parameter NC       = (SPEC == 2) ? 4 : ((SPEC != 0) ? 2 : 1),
    parameter NW       = (SPEC == 2) ? 4 : 1,
    parameter CW       = (PORTS > 1) ? $clog2(PORTS) : 1,
    parameter REQ_W    = 6 + 2 * ADDR_W + 2 * ID_W + 16 + DATA_W + DATA_W / 8,
    parameter RSP_W    = 2 * NW + 2 * NC + NC * DATA_W + NC + NC * ID_W + NC * 2 + 3 + ID_W
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

    // Each endpoint's AXI4 master to device memory, port p at [p].
    output wire [PORTS-1:0]           m_awvalid,
    input  wire [PORTS-1:0]           m_awready,
    output wire [PORTS*64-1:0]        m_awaddr,
    output wire [PORTS*8-1:0]         m_awlen,
    output wire [PORTS-1:0]           m_wvalid,
    input  wire [PORTS-1:0]           m_wready,
    output wire [PORTS*DATA_W-1:0]    m_wdata,
    output wire [PORTS*DATA_W/8-1:0]  m_wstrb,
    output wire [PORTS-1:0]           m_wlast,
    input  wire [PORTS-1:0]           m_bvalid,
    output wire [PORTS-1:0]           m_bready,
    output wire [PORTS-1:0]           m_arvalid,
    input  wire [PORTS-1:0]           m_arready,
    output wire [PORTS*64-1:0]        m_araddr,
    output wire [PORTS*8-1:0]         m_arlen,
    input  wire [PORTS-1:0]           m_rvalid,
    output wire [PORTS-1:0]           m_rready,
    input  wire [PORTS*DATA_W-1:0]    m_rdata
);
    localparam QF = `FMR_REQ_FIELDS + 1;   // the fields and the reset token
    localparam SF = `FMR_RSP_FIELDS;

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
`ifdef FOLDSIM_PORTS_REQ_W
        if (`FOLDSIM_PORTS_REQ_W != REQ_W)
            $error("foldsim_mem_par: the simulator's REQ record is %0d bits, not %0d", `FOLDSIM_PORTS_REQ_W, REQ_W);
`endif
`ifdef FOLDSIM_PORTS_RSP_W
        if (`FOLDSIM_PORTS_RSP_W != RSP_W)
            $error("foldsim_mem_par: the simulator's RSP record is %0d bits, not %0d", `FOLDSIM_PORTS_RSP_W, RSP_W);
`endif
`ifdef FOLDSIM_PORTS
        if (`FOLDSIM_PORTS != PORTS)
            $error("foldsim_mem_par: the simulator has %0d ports, not %0d", `FOLDSIM_PORTS, PORTS);
`endif
`ifdef FOLDSIM_PORTS_REQ_WDATA_LO
        if (`FOLDSIM_PORTS_REQ_WDATA_LO != req_off(5) || `FOLDSIM_PORTS_REQ_BREADY_LO != req_off(13))
            $error("foldsim_mem_par: the simulator's REQ field offsets differ");
`endif
`ifdef FOLDSIM_PORTS_RSP_RDATA_LO
        if (`FOLDSIM_PORTS_RSP_RDATA_LO != rsp_off(4) || `FOLDSIM_PORTS_RSP_BID_LO != rsp_off(10))
            $error("foldsim_mem_par: the simulator's RSP field offsets differ");
`endif
        if (req_off(`FMR_REQ_FIELDS) != REQ_W)
            $error("foldsim_mem_par: REQ_W %0d is not the fields' %0d", REQ_W, req_off(`FMR_REQ_FIELDS));
        if (rsp_off(`FMR_RSP_FIELDS) != RSP_W)
            $error("foldsim_mem_par: RSP_W %0d is not the fields' %0d", RSP_W, rsp_off(`FMR_RSP_FIELDS));
    end

    wire [PORTS-1:0] p_idle, p_error, p_stall, q_free, s_done;
    assign idle  = &p_idle;
    assign error = |p_error;
    assign stall = |p_stall;
    assign req_ready = q_free[req_chan];

    // The response stream: the complete records in turn (a register stage).
    wire [RSP_W-1:0] s_rec [PORTS];
    reg  [CW-1:0]    turn;
    reg  [PORTS-1:0] s_take;
    always @(*) begin
        s_take = '0;
        if (!rsp_valid || rsp_ready)
            for (integer k = PORTS - 1; k >= 0; k = k - 1)
                if (s_done[(int'(turn) + k) % PORTS]) begin
                    s_take = '0;
                    s_take[(int'(turn) + k) % PORTS] = 1'b1;
                end
    end
    always @(posedge clk) begin
        if (reset) begin
            rsp_valid <= 1'b0;
            turn      <= '0;
        end else begin
            if (rsp_valid && rsp_ready)
                rsp_valid <= 1'b0;
            for (integer k = 0; k < PORTS; k = k + 1)
                if (s_take[k]) begin
                    rsp_valid <= 1'b1;
                    rsp_chan  <= CW'(k);
                    rsp_bits  <= s_rec[k];
                    turn      <= CW'((k + 1) % PORTS);
                end
        end
    end

    for (genvar p = 0; p < PORTS; ++p) begin : g_port
        // The port's REQ record and the tokens of it its endpoint has taken.
        reg              q_full;
        reg [REQ_W-1:0]  q_bits;
        reg              q_treset;
        reg [QF-1:0]     q_taken;
        wire [QF-1:0]    q_valid = {QF{q_full}} & ~q_taken;
        wire [QF-1:0]    q_ready;
        wire [QF-1:0]    q_fire = q_valid & q_ready;
        wire             q_load = req_valid && q_free[p] && (int'(req_chan) == p);
        assign q_free[p] = !q_full;
        always @(posedge clk) begin
            if (reset) begin
                q_full  <= 1'b0;
                q_taken <= '0;
            end else if (q_load) begin
                q_full   <= 1'b1;
                q_bits   <= req_bits;
                q_treset <= req_treset;
                q_taken  <= '0;
            end else if (q_full) begin
                if (&(q_taken | q_fire)) begin
                    q_full  <= 1'b0;
                    q_taken <= '0;
                end else begin
                    q_taken <= q_taken | q_fire;
                end
            end
        end
`define FMR_Q(i) q_bits[req_off(i) +: `FMR_REQ_W(i)]

        // The port's RSP record, field by field.
        reg  [SF-1:0]    s_got;
        reg  [RSP_W-1:0] s_bits;
        wire [SF-1:0]    s_valid;
        wire [SF-1:0]    s_ready = ~s_got;
        wire [SF-1:0]    s_fire = s_valid & s_ready;
        wire [RSP_W-1:0] s_in;
        assign s_done[p] = &s_got;
        assign s_rec[p]  = s_bits;
        always @(posedge clk) begin
            if (reset) begin
                s_got <= '0;
            end else if (s_take[p]) begin
                s_got <= '0;
            end else begin
                s_got <= s_got | s_fire;
            end
            for (integer f = 0; f < SF; f = f + 1)
                if (s_fire[f])
                    for (integer b = rsp_off(f); b < rsp_off(f + 1); b = b + 1)
                        s_bits[b] <= s_in[b];
        end

        foldsim_mem #(
            .ID_W     (ID_W),
            .ADDR_W   (ADDR_W),
            .DATA_W   (DATA_W),
            .DEPTH    (DEPTH),
            .SLOTS    (SLOTS),
            .HLOG2    (HLOG2),
            .STAGED   (STAGED),
            .CH_SHIFT (CH_SHIFT),
            .SPEC     (SPEC)
        ) mem (
            .clk            (clk),
            .reset          (reset),
            .base           (base),
            .cfg_data_rate  (cfg_data_rate),
            .cfg_core_clock (cfg_core_clock),
            .cfg_latency    (cfg_latency),
            .cfg_queue      (cfg_queue),
            .cfg_rq         (cfg_rq),
            .cfg_wq         (cfg_wq),
            .cfg_wr_high    (cfg_wr_high),
            .cfg_wr_low     (cfg_wr_low),
            .cfg_geom       (cfg_geom),
            .cfg_t          (cfg_t),
            .cfg_refresh    (cfg_refresh),
            .cfg_refi       (cfg_refi),
            .cfg_rfc        (cfg_rfc),
            .stats          (stats[p*8*64 +: 8*64]),
            // The reference's endpoints' telemetry is not read.
            /* verilator lint_off PINCONNECTEMPTY */
            .axi_stats      (),
            .dbg            (),
            /* verilator lint_on PINCONNECTEMPTY */
            .idle           (p_idle[p]),
            .error          (p_error[p]),
            .stall          (p_stall[p]),
            .treset_valid   (q_valid[14]), .treset_ready (q_ready[14]), .treset_bits (q_treset),
            .awvalid_valid  (q_valid[0]),  .awvalid_ready (q_ready[0]),  .awvalid_bits (`FMR_Q(0)),
            .awaddr_valid   (q_valid[1]),  .awaddr_ready  (q_ready[1]),  .awaddr_bits  (`FMR_Q(1)),
            .awid_valid     (q_valid[2]),  .awid_ready    (q_ready[2]),  .awid_bits    (`FMR_Q(2)),
            .awlen_valid    (q_valid[3]),  .awlen_ready   (q_ready[3]),  .awlen_bits   (`FMR_Q(3)),
            .wvalid_valid   (q_valid[4]),  .wvalid_ready  (q_ready[4]),  .wvalid_bits  (`FMR_Q(4)),
            .wdata_valid    (q_valid[5]),  .wdata_ready   (q_ready[5]),  .wdata_bits   (`FMR_Q(5)),
            .wstrb_valid    (q_valid[6]),  .wstrb_ready   (q_ready[6]),  .wstrb_bits   (`FMR_Q(6)),
            .wlast_valid    (q_valid[7]),  .wlast_ready   (q_ready[7]),  .wlast_bits   (`FMR_Q(7)),
            .arvalid_valid  (q_valid[8]),  .arvalid_ready (q_ready[8]),  .arvalid_bits (`FMR_Q(8)),
            .araddr_valid   (q_valid[9]),  .araddr_ready  (q_ready[9]),  .araddr_bits  (`FMR_Q(9)),
            .arid_valid     (q_valid[10]), .arid_ready    (q_ready[10]), .arid_bits    (`FMR_Q(10)),
            .arlen_valid    (q_valid[11]), .arlen_ready   (q_ready[11]), .arlen_bits   (`FMR_Q(11)),
            .rready_valid   (q_valid[12]), .rready_ready  (q_ready[12]), .rready_bits  (`FMR_Q(12)),
            .bready_valid   (q_valid[13]), .bready_ready  (q_ready[13]), .bready_bits  (`FMR_Q(13)),
            .awready_valid  (s_valid[0]),  .awready_ready (s_ready[0]),  .awready_bits (s_in[rsp_off(0) +: `FMR_RSP_W(0)]),
            .wready_valid   (s_valid[1]),  .wready_ready  (s_ready[1]),  .wready_bits  (s_in[rsp_off(1) +: `FMR_RSP_W(1)]),
            .arready_valid  (s_valid[2]),  .arready_ready (s_ready[2]),  .arready_bits (s_in[rsp_off(2) +: `FMR_RSP_W(2)]),
            .rvalid_valid   (s_valid[3]),  .rvalid_ready  (s_ready[3]),  .rvalid_bits  (s_in[rsp_off(3) +: `FMR_RSP_W(3)]),
            .rdata_valid    (s_valid[4]),  .rdata_ready   (s_ready[4]),  .rdata_bits   (s_in[rsp_off(4) +: `FMR_RSP_W(4)]),
            .rlast_valid    (s_valid[5]),  .rlast_ready   (s_ready[5]),  .rlast_bits   (s_in[rsp_off(5) +: `FMR_RSP_W(5)]),
            .rid_valid      (s_valid[6]),  .rid_ready     (s_ready[6]),  .rid_bits     (s_in[rsp_off(6) +: `FMR_RSP_W(6)]),
            .rresp_valid    (s_valid[7]),  .rresp_ready   (s_ready[7]),  .rresp_bits   (s_in[rsp_off(7) +: `FMR_RSP_W(7)]),
            .bvalid_valid   (s_valid[8]),  .bvalid_ready  (s_ready[8]),  .bvalid_bits  (s_in[rsp_off(8) +: `FMR_RSP_W(8)]),
            .bresp_valid    (s_valid[9]),  .bresp_ready   (s_ready[9]),  .bresp_bits   (s_in[rsp_off(9) +: `FMR_RSP_W(9)]),
            .bid_valid      (s_valid[10]), .bid_ready     (s_ready[10]), .bid_bits     (s_in[rsp_off(10) +: `FMR_RSP_W(10)]),
            .m_awvalid (m_awvalid[p]),
            .m_awready (m_awready[p]),
            .m_awaddr  (m_awaddr[p*64 +: 64]),
            .m_awlen   (m_awlen[p*8 +: 8]),
            .m_wvalid  (m_wvalid[p]),
            .m_wready  (m_wready[p]),
            .m_wdata   (m_wdata[p*DATA_W +: DATA_W]),
            .m_wstrb   (m_wstrb[p*DATA_W/8 +: DATA_W/8]),
            .m_wlast   (m_wlast[p]),
            .m_bvalid  (m_bvalid[p]),
            .m_bready  (m_bready[p]),
            .m_arvalid (m_arvalid[p]),
            .m_arready (m_arready[p]),
            .m_araddr  (m_araddr[p*64 +: 64]),
            .m_arlen   (m_arlen[p*8 +: 8]),
            .m_rvalid  (m_rvalid[p]),
            .m_rready  (m_rready[p]),
            .m_rdata   (m_rdata[p*DATA_W +: DATA_W])
        );
`undef FMR_Q
    end
endmodule
