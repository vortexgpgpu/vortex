// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_platform.vh"

// ============================================================================
// VX_axi_arb2 — Strict 2-master to 1-slave AXI4 arbiter.
//
// Mirrors the reduced AXI4 view used at the AFU memory-bank boundary:
//   AW: valid/ready/addr/id/len
//   W : valid/ready/data/strb/last
//   B : valid/ready/id/resp
//   AR: valid/ready/addr/id/len
//   R : valid/ready/data/last/id/resp
//
// Master 0 = Vortex (high priority); Master 1 = CP.
// Per-channel arbitration is single-outstanding per source — once a request
// is accepted on AW or AR, that channel is held to the same source until the
// corresponding response (B or R-last) completes. The other source stalls.
// W follows the granted AW source until WLAST. R is routed back to the
// source that owns the current AR. This is sufficient for the v1 CP, which
// issues short, isolated bursts when Vortex is idle.
// ============================================================================

`TRACING_OFF
module VX_axi_arb2 #(
    parameter ADDR_W = 64,
    parameter DATA_W = 512,
    parameter ID_W   = 32
) (
    input wire clk,
    input wire reset,

    // ---- Master 0 (Vortex bank-0) ----
    input  wire              s0_awvalid,
    output wire              s0_awready,
    input  wire [ADDR_W-1:0] s0_awaddr,
    input  wire [ID_W-1:0]   s0_awid,
    input  wire [7:0]        s0_awlen,

    input  wire              s0_wvalid,
    output wire              s0_wready,
    input  wire [DATA_W-1:0] s0_wdata,
    input  wire [DATA_W/8-1:0] s0_wstrb,
    input  wire              s0_wlast,

    output wire              s0_bvalid,
    input  wire              s0_bready,
    output wire [ID_W-1:0]   s0_bid,
    output wire [1:0]        s0_bresp,

    input  wire              s0_arvalid,
    output wire              s0_arready,
    input  wire [ADDR_W-1:0] s0_araddr,
    input  wire [ID_W-1:0]   s0_arid,
    input  wire [7:0]        s0_arlen,

    output wire              s0_rvalid,
    input  wire              s0_rready,
    output wire [DATA_W-1:0] s0_rdata,
    output wire              s0_rlast,
    output wire [ID_W-1:0]   s0_rid,
    output wire [1:0]        s0_rresp,

    // ---- Master 1 (CP) ----
    input  wire              s1_awvalid,
    output wire              s1_awready,
    input  wire [ADDR_W-1:0] s1_awaddr,
    input  wire [ID_W-1:0]   s1_awid,
    input  wire [7:0]        s1_awlen,

    input  wire              s1_wvalid,
    output wire              s1_wready,
    input  wire [DATA_W-1:0] s1_wdata,
    input  wire [DATA_W/8-1:0] s1_wstrb,
    input  wire              s1_wlast,

    output wire              s1_bvalid,
    input  wire              s1_bready,
    output wire [ID_W-1:0]   s1_bid,
    output wire [1:0]        s1_bresp,

    input  wire              s1_arvalid,
    output wire              s1_arready,
    input  wire [ADDR_W-1:0] s1_araddr,
    input  wire [ID_W-1:0]   s1_arid,
    input  wire [7:0]        s1_arlen,

    output wire              s1_rvalid,
    input  wire              s1_rready,
    output wire [DATA_W-1:0] s1_rdata,
    output wire              s1_rlast,
    output wire [ID_W-1:0]   s1_rid,
    output wire [1:0]        s1_rresp,

    // ---- Slave (downstream memory bank) ----
    output wire              m_awvalid,
    input  wire              m_awready,
    output wire [ADDR_W-1:0] m_awaddr,
    output wire [ID_W-1:0]   m_awid,
    output wire [7:0]        m_awlen,

    output wire              m_wvalid,
    input  wire              m_wready,
    output wire [DATA_W-1:0] m_wdata,
    output wire [DATA_W/8-1:0] m_wstrb,
    output wire              m_wlast,

    input  wire              m_bvalid,
    output wire              m_bready,
    input  wire [ID_W-1:0]   m_bid,
    input  wire [1:0]        m_bresp,

    output wire              m_arvalid,
    input  wire              m_arready,
    output wire [ADDR_W-1:0] m_araddr,
    output wire [ID_W-1:0]   m_arid,
    output wire [7:0]        m_arlen,

    input  wire              m_rvalid,
    output wire              m_rready,
    input  wire [DATA_W-1:0] m_rdata,
    input  wire              m_rlast,
    input  wire [ID_W-1:0]   m_rid,
    input  wire [1:0]        m_rresp
);

    // ---- AW arbitration with sticky write owner ----
    // owner_w_valid = a write transaction is in flight; owner_w = which source.
    // We treat AW+W+B as one atomic unit: AW is admitted, W flows to the
    // same source until WLAST, then we wait for B before releasing.
    reg owner_w_valid;
    reg owner_w;          // 0 = s0, 1 = s1
    reg w_in_progress;    // true between AW accept and WLAST

    wire aw_pick_s1 = !s0_awvalid && s1_awvalid;
    wire aw_fire   = m_awvalid && m_awready;
    wire w_last_fire = m_wvalid && m_wready && m_wlast;
    wire b_fire    = m_bvalid && m_bready;

    always @(posedge clk) begin
        if (reset) begin
            owner_w_valid <= 1'b0;
            owner_w       <= 1'b0;
            w_in_progress <= 1'b0;
        end else begin
            if (aw_fire && !owner_w_valid) begin
                owner_w_valid <= 1'b1;
                owner_w       <= aw_pick_s1;
                w_in_progress <= 1'b1;
            end
            if (w_in_progress && w_last_fire) begin
                w_in_progress <= 1'b0;
            end
            if (b_fire) begin
                owner_w_valid <= 1'b0;
            end
        end
    end

    // AW: if no owner, prefer s0 over s1. If owner, block both.
    assign m_awvalid = owner_w_valid ? 1'b0 :
                       (s0_awvalid ? s0_awvalid : s1_awvalid);
    assign m_awaddr  = aw_pick_s1 ? s1_awaddr : s0_awaddr;
    assign m_awid    = aw_pick_s1 ? s1_awid   : s0_awid;
    assign m_awlen   = aw_pick_s1 ? s1_awlen  : s0_awlen;
    assign s0_awready = !owner_w_valid && s0_awvalid && m_awready;
    assign s1_awready = !owner_w_valid && aw_pick_s1 && m_awready;

    // W: flow only from the current owner during w_in_progress.
    assign m_wvalid = w_in_progress && (owner_w ? s1_wvalid : s0_wvalid);
    assign m_wdata  = owner_w ? s1_wdata : s0_wdata;
    assign m_wstrb  = owner_w ? s1_wstrb : s0_wstrb;
    assign m_wlast  = owner_w ? s1_wlast : s0_wlast;
    assign s0_wready = w_in_progress && !owner_w && m_wready;
    assign s1_wready = w_in_progress &&  owner_w && m_wready;

    // B: route to owner.
    assign s0_bvalid = !owner_w && m_bvalid && owner_w_valid;
    assign s1_bvalid =  owner_w && m_bvalid && owner_w_valid;
    assign s0_bid    = m_bid;
    assign s1_bid    = m_bid;
    assign s0_bresp  = m_bresp;
    assign s1_bresp  = m_bresp;
    assign m_bready  = owner_w ? s1_bready : s0_bready;

    // ---- AR arbitration with sticky read owner ----
    reg owner_r_valid;
    reg owner_r;          // 0 = s0, 1 = s1

    wire ar_pick_s1 = !s0_arvalid && s1_arvalid;
    wire ar_fire    = m_arvalid && m_arready;
    wire r_last_fire = m_rvalid && m_rready && m_rlast;

    always @(posedge clk) begin
        if (reset) begin
            owner_r_valid <= 1'b0;
            owner_r       <= 1'b0;
        end else begin
            if (ar_fire && !owner_r_valid) begin
                owner_r_valid <= 1'b1;
                owner_r       <= ar_pick_s1;
            end
            if (r_last_fire) begin
                owner_r_valid <= 1'b0;
            end
        end
    end

    assign m_arvalid = owner_r_valid ? 1'b0 :
                       (s0_arvalid ? s0_arvalid : s1_arvalid);
    assign m_araddr  = ar_pick_s1 ? s1_araddr : s0_araddr;
    assign m_arid    = ar_pick_s1 ? s1_arid   : s0_arid;
    assign m_arlen   = ar_pick_s1 ? s1_arlen  : s0_arlen;
    assign s0_arready = !owner_r_valid && s0_arvalid && m_arready;
    assign s1_arready = !owner_r_valid && ar_pick_s1 && m_arready;

    // R: route to owner.
    assign s0_rvalid = !owner_r && m_rvalid && owner_r_valid;
    assign s1_rvalid =  owner_r && m_rvalid && owner_r_valid;
    assign s0_rdata  = m_rdata;
    assign s1_rdata  = m_rdata;
    assign s0_rlast  = m_rlast;
    assign s1_rlast  = m_rlast;
    assign s0_rid    = m_rid;
    assign s1_rid    = m_rid;
    assign s0_rresp  = m_rresp;
    assign s1_rresp  = m_rresp;
    assign m_rready  = owner_r ? s1_rready : s0_rready;

endmodule
`TRACING_ON
