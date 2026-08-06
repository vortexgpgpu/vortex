// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_platform.vh"

// ============================================================================
// VX_mem_from_axi — bridges an AXI4 slave port (flat, interface-free) to a
// Vortex request/response memory master. The flat core behind VX_membus_from_axi
// and the inverse of VX_mem_to_axi: it lets any AXI master drive the Vortex
// request/response mem fabric, since that fabric is not AXI4.
//
// Burst-capable: an N-beat AXI INCR burst is expanded into N sequential
// per-cache-line mem requests (one outstanding at a time — no tag reordering).
// Read beats stream back on R with RLAST on the final beat; a write burst
// emits a single B after its last beat. The AXI ID is carried in the low bits
// of the mem tag and routed back untouched.
// ============================================================================

`TRACING_OFF
module VX_mem_from_axi #(
    parameter ADDR_W     = 64,        // AXI byte address width
    parameter DATA_W     = 512,
    parameter ID_W       = 6,
    parameter MEM_ADDR_W = ADDR_W - `CLOG2(DATA_W/8) // cache-line address (output)
)(
    input wire clk,
    input wire reset,

    // ---- AXI4 slave (flat, reduced view) ----
    input  wire                  s_awvalid,
    output wire                  s_awready,
    input  wire [ADDR_W-1:0]     s_awaddr,
    input  wire [ID_W-1:0]       s_awid,
    input  wire [7:0]            s_awlen,

    input  wire                  s_wvalid,
    output wire                  s_wready,
    input  wire [DATA_W-1:0]     s_wdata,
    input  wire [DATA_W/8-1:0]   s_wstrb,
    input  wire                  s_wlast,

    output wire                  s_bvalid,
    input  wire                  s_bready,
    output wire [ID_W-1:0]       s_bid,
    output wire [1:0]            s_bresp,

    input  wire                  s_arvalid,
    output wire                  s_arready,
    input  wire [ADDR_W-1:0]     s_araddr,
    input  wire [ID_W-1:0]       s_arid,
    input  wire [7:0]            s_arlen,

    output wire                  s_rvalid,
    input  wire                  s_rready,
    output wire [DATA_W-1:0]     s_rdata,
    output wire [ID_W-1:0]       s_rid,
    output wire                  s_rlast,
    output wire [1:0]            s_rresp,

    // ---- Vortex request/response memory master ----
    output wire                  mem_req_valid,
    output wire                  mem_req_rw,
    output wire [MEM_ADDR_W-1:0] mem_req_addr,
    output wire [DATA_W-1:0]     mem_req_data,
    output wire [DATA_W/8-1:0]   mem_req_byteen,
    output wire [ID_W-1:0]       mem_req_tag,
    input  wire                  mem_req_ready,

    input  wire                  mem_rsp_valid,
    input  wire [DATA_W-1:0]     mem_rsp_data,
    input  wire [ID_W-1:0]       mem_rsp_tag,
    output wire                  mem_rsp_ready
);
    localparam CL_SHIFT = `CLOG2(DATA_W / 8);
    localparam CL_BYTES = DATA_W / 8;

    // ---- Write side (AW + N×W → N mem_req with rw=1, single B back) ----
    typedef enum logic [1:0] { WR_IDLE, WR_ISSUE, WR_RESP } wr_state_e;
    wr_state_e         wr_state;
    logic [ID_W-1:0]   wr_id;
    logic [ADDR_W-1:0] wr_addr;
    logic [7:0]        wr_beats;
    `UNUSED_VAR (wr_addr[CL_SHIFT-1:0])

    wire wr_last = (wr_beats == 8'd0);

    always @(posedge clk) begin
        if (reset) begin
            wr_state <= WR_IDLE;
            wr_id    <= '0;
            wr_addr  <= '0;
            wr_beats <= '0;
        end else begin
            case (wr_state)
                WR_IDLE: if (s_awvalid) begin
                    wr_id    <= s_awid;
                    wr_addr  <= s_awaddr;
                    wr_beats <= s_awlen;
                    wr_state <= WR_ISSUE;
                end
                WR_ISSUE: if (s_wvalid && mem_req_ready) begin
                    if (wr_last) wr_state <= WR_RESP;
                    else begin
                        wr_addr  <= wr_addr + ADDR_W'(CL_BYTES);
                        wr_beats <= wr_beats - 8'd1;
                    end
                end
                WR_RESP: if (s_bready) wr_state <= WR_IDLE;
                default: wr_state <= WR_IDLE;
            endcase
        end
    end

    assign s_awready = (wr_state == WR_IDLE) && s_awvalid;
    assign s_wready  = (wr_state == WR_ISSUE) && mem_req_ready;
    assign s_bvalid  = (wr_state == WR_RESP);
    assign s_bid     = wr_id;
    assign s_bresp   = 2'b00;

    // ---- Read side (AR → N mem_req with rw=0, N R beats with RLAST) ----
    typedef enum logic [1:0] { RD_IDLE, RD_ISSUE, RD_WAIT_RSP, RD_RESP } rd_state_e;
    rd_state_e         rd_state;
    logic [ID_W-1:0]   rd_id;
    logic [ADDR_W-1:0] rd_addr;
    logic [7:0]        rd_beats;
    logic [DATA_W-1:0] rd_data;
    `UNUSED_VAR (rd_addr[CL_SHIFT-1:0])

    wire rd_last = (rd_beats == 8'd0);

    always @(posedge clk) begin
        if (reset) begin
            rd_state <= RD_IDLE;
            rd_id    <= '0;
            rd_addr  <= '0;
            rd_beats <= '0;
            rd_data  <= '0;
        end else begin
            case (rd_state)
                RD_IDLE: if (s_arvalid) begin
                    rd_id    <= s_arid;
                    rd_addr  <= s_araddr;
                    rd_beats <= s_arlen;
                    rd_state <= RD_ISSUE;
                end
                RD_ISSUE: if (!issue_wr && mem_req_ready) rd_state <= RD_WAIT_RSP;
                RD_WAIT_RSP: if (mem_rsp_valid) begin
                    rd_data  <= mem_rsp_data;
                    rd_state <= RD_RESP;
                end
                RD_RESP: if (s_rready) begin
                    if (rd_last) rd_state <= RD_IDLE;
                    else begin
                        rd_addr  <= rd_addr + ADDR_W'(CL_BYTES);
                        rd_beats <= rd_beats - 8'd1;
                        rd_state <= RD_ISSUE;
                    end
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    assign s_arready = (rd_state == RD_IDLE);
    assign s_rvalid  = (rd_state == RD_RESP);
    assign s_rdata   = rd_data;
    assign s_rid     = rd_id;
    assign s_rlast   = rd_last;
    assign s_rresp   = 2'b00;

    // ---- mem_req mux: writes win when both pending ----
    wire issue_wr = (wr_state == WR_ISSUE) && s_wvalid;
    wire issue_rd = (rd_state == RD_ISSUE);

    assign mem_req_valid  = issue_wr || issue_rd;
    assign mem_req_rw     = issue_wr;
    assign mem_req_addr   = issue_wr ? wr_addr[ADDR_W-1:CL_SHIFT]
                                     : rd_addr[ADDR_W-1:CL_SHIFT];
    assign mem_req_data   = s_wdata;
    assign mem_req_byteen = issue_wr ? s_wstrb : {(DATA_W/8){1'b1}};
    assign mem_req_tag    = issue_wr ? wr_id : rd_id;

    assign mem_rsp_ready  = (rd_state == RD_WAIT_RSP);
    `UNUSED_VAR (mem_rsp_tag)
    `UNUSED_VAR (s_wlast)

endmodule
`TRACING_ON
