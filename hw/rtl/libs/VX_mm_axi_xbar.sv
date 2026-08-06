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

`include "VX_platform.vh"

// ============================================================================
// VX_mm_axi_xbar — full N-master to M-slave AXI4 crossbar (reduced view), flat
// packed ports. The AXI analog of VX_stream_xbar: each master picks its target
// slave through `s_awsel` / `s_arsel` (the caller's address decode, like
// VX_stream_xbar's `sel_in`), and one VX_mm_axi_arb per slave arbitrates the
// masters routed to it — exactly as VX_stream_xbar builds one VX_stream_arb
// per output. Read and write fabrics are independent, so a master may target
// different slaves for AW and AR simultaneously.
//
// Response routing: each slave's VX_mm_axi_arb returns B/R to its current owning
// master; across the M per-slave arbs the per-master responses are mutually
// exclusive (a master has at most one outstanding write and one read), so they
// combine by a simple gated OR back to the master port.
//
// Ordering assumption (matches the single-outstanding VX_mm_axi_arb building
// block): each master keeps at most one outstanding write (issues its next AW
// only after WLAST/B of the previous) and one outstanding read. `s_awsel` /
// `s_arsel` must stay stable for the life of the addressed transaction. Under
// this contract a master's W beats are unambiguously owned by the single slave
// its current AW targeted.
//
// OUT_REG (0/1): registered AXI stage on every slave port (SLR-safe boundary).
// `collisions` counts cycles where two or more masters contend for the same
// slave (write or read), mirroring VX_stream_xbar's perf counter.
// ============================================================================

`TRACING_OFF
module VX_mm_axi_xbar #(
    parameter NUM_INPUTS    = 4,
    parameter NUM_OUTPUTS   = 4,
    parameter ADDR_WIDTH    = 64,
    parameter DATA_WIDTH    = 512,
    parameter ID_WIDTH      = 32,
    parameter `STRING ARBITER = "R",
    parameter STICKY        = 0,
    parameter MULTI_OUT     = 0,   // ID-routed multi-outstanding; only valid for NUM_OUTPUTS==1
    parameter OUT_REG       = 0,
    parameter STRB_WIDTH    = DATA_WIDTH/8,
    parameter SEL_WIDTH     = `LOG2UP(NUM_OUTPUTS),
    parameter PERF_CTR_BITS = `CLOG2(NUM_INPUTS+1)
) (
    input  wire clk,
    input  wire reset,

    // ---- Upstream masters (slave-side), packed ----
    input  wire [NUM_INPUTS-1:0]                  s_awvalid,
    output wire [NUM_INPUTS-1:0]                  s_awready,
    input  wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_awaddr,
    input  wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_awid,
    input  wire [NUM_INPUTS-1:0][7:0]             s_awlen,
    input  wire [NUM_INPUTS-1:0][SEL_WIDTH-1:0]   s_awsel,

    input  wire [NUM_INPUTS-1:0]                  s_wvalid,
    output wire [NUM_INPUTS-1:0]                  s_wready,
    input  wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_wdata,
    input  wire [NUM_INPUTS-1:0][STRB_WIDTH-1:0]  s_wstrb,
    input  wire [NUM_INPUTS-1:0]                  s_wlast,

    output wire [NUM_INPUTS-1:0]                  s_bvalid,
    input  wire [NUM_INPUTS-1:0]                  s_bready,
    output wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_bid,
    output wire [NUM_INPUTS-1:0][1:0]             s_bresp,

    input  wire [NUM_INPUTS-1:0]                  s_arvalid,
    output wire [NUM_INPUTS-1:0]                  s_arready,
    input  wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_araddr,
    input  wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_arid,
    input  wire [NUM_INPUTS-1:0][7:0]             s_arlen,
    input  wire [NUM_INPUTS-1:0][SEL_WIDTH-1:0]   s_arsel,

    output wire [NUM_INPUTS-1:0]                  s_rvalid,
    input  wire [NUM_INPUTS-1:0]                  s_rready,
    output wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_rdata,
    output wire [NUM_INPUTS-1:0]                  s_rlast,
    output wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_rid,
    output wire [NUM_INPUTS-1:0][1:0]             s_rresp,

    // ---- Downstream slaves (master-side), packed ----
    output wire [NUM_OUTPUTS-1:0]                 m_awvalid,
    input  wire [NUM_OUTPUTS-1:0]                 m_awready,
    output wire [NUM_OUTPUTS-1:0][ADDR_WIDTH-1:0] m_awaddr,
    output wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_awid,
    output wire [NUM_OUTPUTS-1:0][7:0]            m_awlen,

    output wire [NUM_OUTPUTS-1:0]                 m_wvalid,
    input  wire [NUM_OUTPUTS-1:0]                 m_wready,
    output wire [NUM_OUTPUTS-1:0][DATA_WIDTH-1:0] m_wdata,
    output wire [NUM_OUTPUTS-1:0][STRB_WIDTH-1:0] m_wstrb,
    output wire [NUM_OUTPUTS-1:0]                 m_wlast,

    input  wire [NUM_OUTPUTS-1:0]                 m_bvalid,
    output wire [NUM_OUTPUTS-1:0]                 m_bready,
    input  wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_bid,
    input  wire [NUM_OUTPUTS-1:0][1:0]            m_bresp,

    output wire [NUM_OUTPUTS-1:0]                 m_arvalid,
    input  wire [NUM_OUTPUTS-1:0]                 m_arready,
    output wire [NUM_OUTPUTS-1:0][ADDR_WIDTH-1:0] m_araddr,
    output wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_arid,
    output wire [NUM_OUTPUTS-1:0][7:0]            m_arlen,

    input  wire [NUM_OUTPUTS-1:0]                 m_rvalid,
    output wire [NUM_OUTPUTS-1:0]                 m_rready,
    input  wire [NUM_OUTPUTS-1:0][DATA_WIDTH-1:0] m_rdata,
    input  wire [NUM_OUTPUTS-1:0]                 m_rlast,
    input  wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_rid,
    input  wire [NUM_OUTPUTS-1:0][1:0]            m_rresp,

    output wire [PERF_CTR_BITS-1:0]               collisions
);
    if (NUM_OUTPUTS == 1) begin : g_single_output

        // Degenerate crossbar == plain N->1 arbiter; sel is unused.
        `UNUSED_VAR (s_awsel)
        `UNUSED_VAR (s_arsel)

        VX_mm_axi_arb #(
            .NUM_INPUTS (NUM_INPUTS),
            .ADDR_WIDTH (ADDR_WIDTH),
            .DATA_WIDTH (DATA_WIDTH),
            .ID_WIDTH   (ID_WIDTH),
            .ARBITER    (ARBITER),
            .STICKY     (STICKY),
            .MULTI_OUT  (MULTI_OUT),
            .OUT_REG    (OUT_REG),
            .STRB_WIDTH (STRB_WIDTH)
        ) arb (
            .clk (clk), .reset (reset),
            .s_awvalid (s_awvalid), .s_awready (s_awready),
            .s_awaddr  (s_awaddr),  .s_awid (s_awid), .s_awlen (s_awlen),
            .s_wvalid  (s_wvalid),  .s_wready (s_wready),
            .s_wdata   (s_wdata),   .s_wstrb (s_wstrb), .s_wlast (s_wlast),
            .s_bvalid  (s_bvalid),  .s_bready (s_bready),
            .s_bid     (s_bid),     .s_bresp (s_bresp),
            .s_arvalid (s_arvalid), .s_arready (s_arready),
            .s_araddr  (s_araddr),  .s_arid (s_arid), .s_arlen (s_arlen),
            .s_rvalid  (s_rvalid),  .s_rready (s_rready),
            .s_rdata   (s_rdata),   .s_rlast (s_rlast),
            .s_rid     (s_rid),     .s_rresp (s_rresp),
            .m_awvalid (m_awvalid), .m_awready (m_awready),
            .m_awaddr  (m_awaddr),  .m_awid (m_awid), .m_awlen (m_awlen),
            .m_wvalid  (m_wvalid),  .m_wready (m_wready),
            .m_wdata   (m_wdata),   .m_wstrb (m_wstrb), .m_wlast (m_wlast),
            .m_bvalid  (m_bvalid),  .m_bready (m_bready),
            .m_bid     (m_bid),     .m_bresp (m_bresp),
            .m_arvalid (m_arvalid), .m_arready (m_arready),
            .m_araddr  (m_araddr),  .m_arid (m_arid), .m_arlen (m_arlen),
            .m_rvalid  (m_rvalid),  .m_rready (m_rready),
            .m_rdata   (m_rdata),   .m_rlast (m_rlast),
            .m_rid     (m_rid),     .m_rresp (m_rresp)
        );

    end else begin : g_multi_output

        `STATIC_ASSERT((MULTI_OUT == 0),
            ("VX_mm_axi_xbar: MULTI_OUT (ID-routed) is only supported for NUM_OUTPUTS==1"))

        // Per-slave arbiter outputs, indexed [slave][master].
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0]                 aw_ready_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0]                 w_ready_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0]                 b_valid_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0][ID_WIDTH-1:0]   b_id_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0][1:0]            b_resp_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0]                 ar_ready_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0]                 r_valid_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0][DATA_WIDTH-1:0] r_data_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0]                 r_last_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0][ID_WIDTH-1:0]   r_id_a;
        wire [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0][1:0]            r_resp_a;

        for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin : g_slave_arb

            // Route only the masters that select this slave into its arbiter.
            wire [NUM_INPUTS-1:0] awvalid_j, arvalid_j;
            for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_mask
                assign awvalid_j[i] = s_awvalid[i] && (s_awsel[i] == SEL_WIDTH'(j));
                assign arvalid_j[i] = s_arvalid[i] && (s_arsel[i] == SEL_WIDTH'(j));
            end

            VX_mm_axi_arb #(
                .NUM_INPUTS (NUM_INPUTS),
                .ADDR_WIDTH (ADDR_WIDTH),
                .DATA_WIDTH (DATA_WIDTH),
                .ID_WIDTH   (ID_WIDTH),
                .ARBITER    (ARBITER),
                .STICKY     (STICKY),
                .OUT_REG    (OUT_REG),
                .STRB_WIDTH (STRB_WIDTH)
            ) arb (
                .clk (clk), .reset (reset),
                // AW/AR valids masked to this slave; data/W broadcast (arb picks owner).
                .s_awvalid (awvalid_j),  .s_awready (aw_ready_a[j]),
                .s_awaddr  (s_awaddr),   .s_awid (s_awid), .s_awlen (s_awlen),
                .s_wvalid  (s_wvalid),   .s_wready (w_ready_a[j]),
                .s_wdata   (s_wdata),    .s_wstrb (s_wstrb), .s_wlast (s_wlast),
                .s_bvalid  (b_valid_a[j]), .s_bready (s_bready),
                .s_bid     (b_id_a[j]),  .s_bresp (b_resp_a[j]),
                .s_arvalid (arvalid_j),  .s_arready (ar_ready_a[j]),
                .s_araddr  (s_araddr),   .s_arid (s_arid), .s_arlen (s_arlen),
                .s_rvalid  (r_valid_a[j]), .s_rready (s_rready),
                .s_rdata   (r_data_a[j]), .s_rlast (r_last_a[j]),
                .s_rid     (r_id_a[j]),  .s_rresp (r_resp_a[j]),
                .m_awvalid (m_awvalid[j]), .m_awready (m_awready[j]),
                .m_awaddr  (m_awaddr[j]),  .m_awid (m_awid[j]), .m_awlen (m_awlen[j]),
                .m_wvalid  (m_wvalid[j]),  .m_wready (m_wready[j]),
                .m_wdata   (m_wdata[j]),   .m_wstrb (m_wstrb[j]), .m_wlast (m_wlast[j]),
                .m_bvalid  (m_bvalid[j]),  .m_bready (m_bready[j]),
                .m_bid     (m_bid[j]),     .m_bresp (m_bresp[j]),
                .m_arvalid (m_arvalid[j]), .m_arready (m_arready[j]),
                .m_araddr  (m_araddr[j]),  .m_arid (m_arid[j]), .m_arlen (m_arlen[j]),
                .m_rvalid  (m_rvalid[j]),  .m_rready (m_rready[j]),
                .m_rdata   (m_rdata[j]),   .m_rlast (m_rlast[j]),
                .m_rid     (m_rid[j]),     .m_rresp (m_rresp[j])
            );
        end

        // ---- Combine per-slave results back to each master (mutually exclusive) ----
        logic [NUM_INPUTS-1:0]                  awready_c, wready_c, arready_c;
        logic [NUM_INPUTS-1:0]                  bvalid_c, rvalid_c, rlast_c;
        logic [NUM_INPUTS-1:0][ID_WIDTH-1:0]    bid_c, rid_c;
        logic [NUM_INPUTS-1:0][1:0]             bresp_c, rresp_c;
        logic [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  rdata_c;

        always @(*) begin
            awready_c = '0; wready_c = '0; arready_c = '0;
            bvalid_c  = '0; rvalid_c = '0; rlast_c = '0;
            bid_c = '0; rid_c = '0; bresp_c = '0; rresp_c = '0; rdata_c = '0;
            for (integer i = 0; i < NUM_INPUTS; ++i) begin
                for (integer j = 0; j < NUM_OUTPUTS; ++j) begin
                    awready_c[i] = awready_c[i] | aw_ready_a[j][i];
                    wready_c[i]  = wready_c[i]  | w_ready_a[j][i];
                    arready_c[i] = arready_c[i] | ar_ready_a[j][i];
                    if (b_valid_a[j][i]) begin
                        bvalid_c[i] = 1'b1;
                        bid_c[i]    = b_id_a[j][i];
                        bresp_c[i]  = b_resp_a[j][i];
                    end
                    if (r_valid_a[j][i]) begin
                        rvalid_c[i] = 1'b1;
                        rdata_c[i]  = r_data_a[j][i];
                        rlast_c[i]  = r_last_a[j][i];
                        rid_c[i]    = r_id_a[j][i];
                        rresp_c[i]  = r_resp_a[j][i];
                    end
                end
            end
        end

        assign s_awready = awready_c;
        assign s_wready  = wready_c;
        assign s_arready = arready_c;
        assign s_bvalid  = bvalid_c;
        assign s_bid     = bid_c;
        assign s_bresp   = bresp_c;
        assign s_rvalid  = rvalid_c;
        assign s_rdata   = rdata_c;
        assign s_rlast   = rlast_c;
        assign s_rid     = rid_c;
        assign s_rresp   = rresp_c;
    end

    // ---- Collision perf counter (cycles with >1 master contending a slave) ----
    // Inlined popcount + 1-cycle buffer so this libs IP depends on VX_platform
    // only (no VX_define macros such as POP_COUNT/BUFFER).
    reg [NUM_INPUTS-1:0]    collision_mask, collision_mask_r;
    reg [PERF_CTR_BITS-1:0] collision_count;
    reg [PERF_CTR_BITS-1:0] collisions_r;

    always @(*) begin
        collision_mask = '0;
        for (integer i = 0; i < NUM_INPUTS; ++i) begin
            for (integer k = i + 1; k < NUM_INPUTS; ++k) begin
                // write-address contention
                collision_mask[i] |= s_awvalid[i] && s_awvalid[k]
                                  && (s_awsel[i] == s_awsel[k]);
                // read-address contention
                collision_mask[i] |= s_arvalid[i] && s_arvalid[k]
                                  && (s_arsel[i] == s_arsel[k]);
            end
        end
    end

    always @(*) begin
        collision_count = '0;
        for (integer i = 0; i < NUM_INPUTS; ++i) begin
            collision_count = collision_count + PERF_CTR_BITS'(collision_mask_r[i]);
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            collision_mask_r <= '0;
            collisions_r     <= '0;
        end else begin
            collision_mask_r <= collision_mask;
            collisions_r     <= collisions_r + collision_count;
        end
    end

    assign collisions = collisions_r;

endmodule
`TRACING_ON
