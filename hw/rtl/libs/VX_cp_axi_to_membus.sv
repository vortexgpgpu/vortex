// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_platform.vh"

// ============================================================================
// VX_cp_axi_to_membus — bridges VX_cp_axi_m_if (AXI4 master) to a
// VX_mem_bus_if master. Used on the OPAE AFU where the CP's axi_m needs
// to join the request/response-style fabric that already feeds local
// memory (Vortex's memory port format is request/response, not AXI4).
//
// v1 supports single-beat bursts only (awlen=arlen=0): this matches the
// CP's actual issue pattern (fetch = single 64 B read; completion =
// single 8 B write; DMA = single beat per command in the current engine).
// Multi-beat is documented as future work.
//
// Tag encoding: AXI ID (ID_W bits) is placed in the low bits of the
// VX_mem_bus_if tag's `value` field; the response routes it back
// untouched. UUID is tied to 0 (CP traffic has no Vortex UUID concept).
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
    // interface fields). Using flattened ports keeps this lib module
    // independent of VX_mem_bus_if's exact field layout.
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

    // ---- Write side (AW + W → mem_req with rw=1, B back) ----
    typedef enum logic [1:0] {
        WR_IDLE,
        WR_ISSUE,    // both AW + W in hand; drive mem_req
        WR_RESP      // wait for host to take B
    } wr_state_e;
    wr_state_e         wr_state;
    logic [ID_W-1:0]   wr_id;
    logic [ADDR_W-1:0] wr_addr;
    logic [DATA_W-1:0] wr_data;
    logic [DATA_W/8-1:0] wr_strb;
    // Low CL_SHIFT bits of wr_addr are the byte offset within a CL —
    // discarded when forming mem_req_addr (CL-addressed).
    `UNUSED_VAR (wr_addr[CL_SHIFT-1:0])

    always_ff @(posedge clk) begin
        if (reset) begin
            wr_state <= WR_IDLE;
            wr_id    <= '0;
            wr_addr  <= '0;
            wr_data  <= '0;
            wr_strb  <= '0;
        end else begin
            case (wr_state)
                WR_IDLE: begin
                    // Capture AW and W when both are present.
                    if (axi_s.awvalid && axi_s.wvalid) begin
                        wr_id    <= axi_s.awid;
                        wr_addr  <= axi_s.awaddr;
                        wr_data  <= axi_s.wdata;
                        wr_strb  <= axi_s.wstrb;
                        wr_state <= WR_ISSUE;
                    end
                end
                WR_ISSUE: begin
                    if (mem_req_ready) wr_state <= WR_RESP;
                end
                WR_RESP: begin
                    if (axi_s.bready) wr_state <= WR_IDLE;
                end
                default: wr_state <= WR_IDLE;
            endcase
        end
    end

    // Accept AW + W together (in the same cycle they both become valid).
    assign axi_s.awready = (wr_state == WR_IDLE) && axi_s.awvalid && axi_s.wvalid;
    assign axi_s.wready  = (wr_state == WR_IDLE) && axi_s.awvalid && axi_s.wvalid;
    assign axi_s.bvalid  = (wr_state == WR_RESP);
    assign axi_s.bid     = wr_id;
    assign axi_s.bresp   = 2'b00;
    `UNUSED_VAR (axi_s.awlen)
    `UNUSED_VAR (axi_s.awsize)
    `UNUSED_VAR (axi_s.awburst)
    `UNUSED_VAR (axi_s.wlast)

    // ---- Read side (AR → mem_req with rw=0, R back with rlast=1) ----
    typedef enum logic [1:0] {
        RD_IDLE,
        RD_ISSUE,
        RD_WAIT_RSP,
        RD_RESP
    } rd_state_e;
    rd_state_e         rd_state;
    logic [ID_W-1:0]   rd_id;
    logic [ADDR_W-1:0] rd_addr;
    logic [DATA_W-1:0] rd_data;
    `UNUSED_VAR (rd_addr[CL_SHIFT-1:0])

    always_ff @(posedge clk) begin
        if (reset) begin
            rd_state <= RD_IDLE;
            rd_id    <= '0;
            rd_addr  <= '0;
            rd_data  <= '0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    if (axi_s.arvalid) begin
                        rd_id    <= axi_s.arid;
                        rd_addr  <= axi_s.araddr;
                        rd_state <= RD_ISSUE;
                    end
                end
                RD_ISSUE: begin
                    if (mem_req_ready) rd_state <= RD_WAIT_RSP;
                end
                RD_WAIT_RSP: begin
                    if (mem_rsp_valid) begin
                        rd_data  <= mem_rsp_data;
                        rd_state <= RD_RESP;
                    end
                end
                RD_RESP: begin
                    if (axi_s.rready) rd_state <= RD_IDLE;
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    assign axi_s.arready = (rd_state == RD_IDLE);
    assign axi_s.rvalid  = (rd_state == RD_RESP);
    assign axi_s.rdata   = rd_data;
    assign axi_s.rid     = rd_id;
    assign axi_s.rlast   = 1'b1;
    assign axi_s.rresp   = 2'b00;
    `UNUSED_VAR (axi_s.arlen)
    `UNUSED_VAR (axi_s.arsize)
    `UNUSED_VAR (axi_s.arburst)

    // ---- mem_req mux: writes win when both pending (CP fetch + completion
    // don't actually contend in practice, but pick a deterministic policy) ----
    wire issue_wr = (wr_state == WR_ISSUE);
    wire issue_rd = (rd_state == RD_ISSUE);

    assign mem_req_valid  = issue_wr || issue_rd;
    assign mem_req_rw     = issue_wr;
    assign mem_req_addr   = issue_wr ? wr_addr[ADDR_W-1:CL_SHIFT]
                                     : rd_addr[ADDR_W-1:CL_SHIFT];
    assign mem_req_data   = wr_data;
    assign mem_req_byteen = issue_wr ? wr_strb : {(DATA_W/8){1'b1}};
    assign mem_req_tag    = issue_wr ? wr_id : rd_id;

    // ---- Response ready ----
    assign mem_rsp_ready  = (rd_state == RD_WAIT_RSP);
    `UNUSED_VAR (mem_rsp_tag)

endmodule
`TRACING_ON
