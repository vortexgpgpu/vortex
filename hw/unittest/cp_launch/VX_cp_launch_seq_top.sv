// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// VX_cp_launch_seq_top — verilator-friendly wrapper around VX_cp_launch_seq:
// the command arrives packed, and the AXI read master is flattened.
module VX_cp_launch_seq_top
    import VX_cp_pkg::*;
#(
    parameter int ADDR_W = 64,
    parameter int DATA_W = 512,
    parameter int ID_W   = VX_CP_AXI_TID_WIDTH_C
) (
    input  wire                          clk,
    input  wire                          reset,

    input  wire                          grant,
    input  wire [$bits(cmd_t)-1:0]       cmd_packed,
    output wire                          done,
    output wire                          skip,
    output wire [15:0]                   skip_n,
    output wire                          fault,

    output wire                          dcr_hold,
    input  wire                          dcr_proxy_idle,
    output wire                          dcr_req_valid,
    output wire [`VX_DCR_ADDR_BITS-1:0]  dcr_req_addr,
    output wire [`VX_DCR_DATA_BITS-1:0]  dcr_req_data,
    input  wire                          dcr_req_ready,

    output wire                          start,
    input  wire                          gpu_busy,

    output wire                          arvalid,
    input  wire                          arready,
    output wire [ADDR_W-1:0]             araddr,
    output wire [ID_W-1:0]               arid,
    output wire [7:0]                    arlen,
    output wire [2:0]                    arsize,
    output wire [1:0]                    arburst,
    input  wire                          rvalid,
    output wire                          rready,
    input  wire [DATA_W-1:0]             rdata,
    input  wire [ID_W-1:0]               rid,
    input  wire                          rlast,
    input  wire [1:0]                    rresp,
    output wire                          awvalid,
    output wire                          wvalid,
    output wire                          bready
);
    VX_mem_axi_if #(
        .ADDR_W (ADDR_W),
        .DATA_W (DATA_W),
        .ID_W   (ID_W)
    ) axi ();

    VX_cp_launch_seq #(
        .ADDR_W (ADDR_W),
        .DATA_W (DATA_W),
        .ID_W   (ID_W)
    ) u_dut (
        .clk            (clk),
        .reset          (reset),
        .grant          (grant),
        .cmd            (cmd_t'(cmd_packed)),
        .done           (done),
        .skip           (skip),
        .skip_n         (skip_n),
        .fault          (fault),
        .dcr_hold       (dcr_hold),
        .dcr_proxy_idle (dcr_proxy_idle),
        .dcr_req_valid  (dcr_req_valid),
        .dcr_req_addr   (dcr_req_addr),
        .dcr_req_data   (dcr_req_data),
        .dcr_req_ready  (dcr_req_ready),
        .start          (start),
        .gpu_busy       (gpu_busy),
        .axi_m          (axi)
    );

    assign arvalid     = axi.arvalid;
    assign axi.arready = arready;
    assign araddr      = axi.araddr;
    assign arid        = axi.arid;
    assign arlen       = axi.arlen;
    assign arsize      = axi.arsize;
    assign arburst     = axi.arburst;
    assign axi.rvalid  = rvalid;
    assign rready      = axi.rready;
    assign axi.rdata   = rdata;
    assign axi.rid     = rid;
    assign axi.rlast   = rlast;
    assign axi.rresp   = rresp;
    assign awvalid     = axi.awvalid;
    assign wvalid      = axi.wvalid;
    assign bready      = axi.bready;
    assign axi.awready = 1'b0;
    assign axi.wready  = 1'b0;
    assign axi.bvalid  = 1'b0;
    assign axi.bid     = '0;
    assign axi.bresp   = '0;

    `UNUSED_VAR (axi.awaddr)
    `UNUSED_VAR (axi.awid)
    `UNUSED_VAR (axi.awlen)
    `UNUSED_VAR (axi.awsize)
    `UNUSED_VAR (axi.awburst)
    `UNUSED_VAR (axi.wdata)
    `UNUSED_VAR (axi.wstrb)
    `UNUSED_VAR (axi.wlast)

endmodule : VX_cp_launch_seq_top
