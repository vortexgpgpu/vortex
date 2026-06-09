// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_axi_to_membus — bridges VX_cp_axi_m_if (AXI4 master) to a
// VX_mem_bus_if master. Used on the OPAE AFU where a CP AXI master must
// join the request/response-style fabric (local memory + CCI-P) — Vortex's
// memory port format is request/response, not AXI4.
//
// Burst-capable: an N-beat AXI INCR burst is expanded into N sequential
// per-cache-line mem_bus requests (one outstanding at a time — no tag
// reordering needed). Read beats are streamed back on R with RLAST on the
// final beat; a write burst emits a single B after its last beat.
//
// Tag encoding: the AXI ID (ID_W bits) is placed in the low bits of the
// mem_bus tag; the response routes it back untouched.
// ============================================================================

`TRACING_OFF
module VX_cp_axi_to_membus
  import VX_gpu_pkg::*;
#(
    parameter int ADDR_W   = 64,        // CP byte address width
    parameter int DATA_W   = 512,
    parameter int ID_W     = 6,
    parameter int MEM_ADDR_W = ADDR_W - $clog2(DATA_W/8) // CL address (output)
)(
    input wire clk,
    input wire reset,

    VX_cp_axi_m_if.slave axi_s,

    // VX_mem_bus_if master-side signals (flattened — caller wires the
    // interface fields).
    output wire                       mem_req_valid,
    output wire                       mem_req_rw,
    output wire [MEM_ADDR_W-1:0]      mem_req_addr,
    output wire [DATA_W-1:0]          mem_req_data,
    output wire [DATA_W/8-1:0]        mem_req_byteen,
    output wire [ID_W-1:0]            mem_req_tag,
    input  wire                       mem_req_ready,

    input  wire                       mem_rsp_valid,
    input  wire [DATA_W-1:0]          mem_rsp_data,
    input  wire [ID_W-1:0]            mem_rsp_tag,
    output wire                       mem_rsp_ready
);

    localparam int CL_SHIFT = $clog2(DATA_W / 8);
    localparam int CL_BYTES = DATA_W / 8;

    // ---- Write side (AW + N×W → N mem_req with rw=1, single B back) ----
    typedef enum logic [1:0] {
        WR_IDLE,
        WR_ISSUE,    // AW + current W beat in hand; drive mem_req
        WR_RESP      // last beat issued; wait for host to take B
    } wr_state_e;
    wr_state_e           wr_state;
    logic [ID_W-1:0]     wr_id;
    logic [ADDR_W-1:0]   wr_addr;     // current beat byte address
    logic [7:0]          wr_beats;    // beats remaining - 1 (awlen countdown)
    `UNUSED_VAR (wr_addr[CL_SHIFT-1:0])

    wire wr_last = (wr_beats == 8'd0);

    always_ff @(posedge clk) begin
        if (reset) begin
            wr_state <= WR_IDLE;
            wr_id    <= '0;
            wr_addr  <= '0;
            wr_beats <= '0;
        end else begin
            case (wr_state)
                WR_IDLE: begin
                    if (axi_s.awvalid) begin
                        wr_id    <= axi_s.awid;
                        wr_addr  <= axi_s.awaddr;
                        wr_beats <= axi_s.awlen;
                        wr_state <= WR_ISSUE;
                    end
                end
                WR_ISSUE: begin
                    // Drive mem_req from the current W beat; advance on accept.
                    if (axi_s.wvalid && mem_req_ready) begin
                        if (wr_last) begin
                            wr_state <= WR_RESP;
                        end else begin
                            wr_addr  <= wr_addr + ADDR_W'(CL_BYTES);
                            wr_beats <= wr_beats - 8'd1;
                        end
                    end
                end
                WR_RESP: begin
                    if (axi_s.bready) wr_state <= WR_IDLE;
                end
                default: wr_state <= WR_IDLE;
            endcase
        end
    end

    // AW accepted once, at the start of the burst.
    assign axi_s.awready = (wr_state == WR_IDLE) && axi_s.awvalid;
    // Each W beat is consumed when its mem_req is accepted.
    assign axi_s.wready  = (wr_state == WR_ISSUE) && mem_req_ready;
    assign axi_s.bvalid  = (wr_state == WR_RESP);
    assign axi_s.bid     = wr_id;
    assign axi_s.bresp   = 2'b00;
    `UNUSED_VAR (axi_s.awsize)
    `UNUSED_VAR (axi_s.awburst)
    `UNUSED_VAR (axi_s.wlast)

    // ---- Read side (AR → N mem_req with rw=0, N R beats with RLAST) ----
    typedef enum logic [1:0] {
        RD_IDLE,
        RD_ISSUE,     // drive mem_req for the current beat
        RD_WAIT_RSP,  // wait for the mem_bus response
        RD_RESP       // present the R beat
    } rd_state_e;
    rd_state_e         rd_state;
    logic [ID_W-1:0]   rd_id;
    logic [ADDR_W-1:0] rd_addr;     // current beat byte address
    logic [7:0]        rd_beats;    // beats remaining - 1 (arlen countdown)
    logic [DATA_W-1:0] rd_data;
    `UNUSED_VAR (rd_addr[CL_SHIFT-1:0])

    wire rd_last = (rd_beats == 8'd0);

    always_ff @(posedge clk) begin
        if (reset) begin
            rd_state <= RD_IDLE;
            rd_id    <= '0;
            rd_addr  <= '0;
            rd_beats <= '0;
            rd_data  <= '0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    if (axi_s.arvalid) begin
                        rd_id    <= axi_s.arid;
                        rd_addr  <= axi_s.araddr;
                        rd_beats <= axi_s.arlen;
                        rd_state <= RD_ISSUE;
                    end
                end
                RD_ISSUE: begin
                    // Advance only when the read actually wins the mem bus
                    // (writes have priority in the mem_req mux below).
                    if (!issue_wr && mem_req_ready) rd_state <= RD_WAIT_RSP;
                end
                RD_WAIT_RSP: begin
                    if (mem_rsp_valid) begin
                        rd_data  <= mem_rsp_data;
                        rd_state <= RD_RESP;
                    end
                end
                RD_RESP: begin
                    if (axi_s.rready) begin
                        if (rd_last) begin
                            rd_state <= RD_IDLE;
                        end else begin
                            rd_addr  <= rd_addr + ADDR_W'(CL_BYTES);
                            rd_beats <= rd_beats - 8'd1;
                            rd_state <= RD_ISSUE;
                        end
                    end
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    assign axi_s.arready = (rd_state == RD_IDLE);
    assign axi_s.rvalid  = (rd_state == RD_RESP);
    assign axi_s.rdata   = rd_data;
    assign axi_s.rid     = rd_id;
    assign axi_s.rlast   = rd_last;
    assign axi_s.rresp   = 2'b00;
    `UNUSED_VAR (axi_s.arsize)
    `UNUSED_VAR (axi_s.arburst)

    // ---- mem_req mux: writes win when both pending. ----
    wire issue_wr = (wr_state == WR_ISSUE) && axi_s.wvalid;
    wire issue_rd = (rd_state == RD_ISSUE);

    assign mem_req_valid  = issue_wr || issue_rd;
    assign mem_req_rw     = issue_wr;
    assign mem_req_addr   = issue_wr ? wr_addr[ADDR_W-1:CL_SHIFT]
                                     : rd_addr[ADDR_W-1:CL_SHIFT];
    assign mem_req_data   = axi_s.wdata;
    assign mem_req_byteen = issue_wr ? axi_s.wstrb : {(DATA_W/8){1'b1}};
    assign mem_req_tag    = issue_wr ? wr_id : rd_id;

    // ---- Response ready ----
    assign mem_rsp_ready  = (rd_state == RD_WAIT_RSP);
    `UNUSED_VAR (mem_rsp_tag)

endmodule
`TRACING_ON
