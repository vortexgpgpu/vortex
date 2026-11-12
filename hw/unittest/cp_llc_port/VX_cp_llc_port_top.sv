// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// Test wrapper around VX_cp_llc_port: the last-level cache side of the
// port's VX_mem_bus_if is flattened so the C++ harness can play the cache.
module VX_cp_llc_port_top import VX_gpu_pkg::*; #(
    parameter DATA_SIZE      = 64,
    parameter TAG_WIDTH      = 4,
    parameter SLOT_TAG_WIDTH = UUID_WIDTH + 1 + TAG_WIDTH,
    parameter ADDR_WIDTH     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)
) (
    input  wire                         clk,
    input  wire                         reset,

    input  wire                         req_valid,
    input  wire                         req_rw,
    input  wire                         req_noalloc,
    input  wire [ADDR_WIDTH-1:0]        req_addr,
    input  wire [DATA_SIZE*8-1:0]       req_data,
    input  wire [DATA_SIZE-1:0]         req_byteen,
    input  wire [TAG_WIDTH-1:0]         req_tag,
    output wire                         req_ready,

    output wire                         rsp_valid,
    output wire [DATA_SIZE*8-1:0]       rsp_data,
    output wire [TAG_WIDTH-1:0]         rsp_tag,
    input  wire                         rsp_ready,

    input  wire                         start_in,
    output wire                         start_out,
    output wire                         busy,

    output wire                         mem_req_valid,
    output wire                         mem_req_rw,
    output wire [ADDR_WIDTH-1:0]        mem_req_addr,
    output wire [DATA_SIZE*8-1:0]       mem_req_data,
    output wire [DATA_SIZE-1:0]         mem_req_byteen,
    output wire                         mem_req_flush,
    output wire                         mem_req_io,
    output wire                         mem_req_noalloc,
    output wire [SLOT_TAG_WIDTH-1:0]    mem_req_tag,
    input  wire                         mem_req_ready,

    input  wire                         mem_rsp_valid,
    input  wire [DATA_SIZE*8-1:0]       mem_rsp_data,
    input  wire [SLOT_TAG_WIDTH-1:0]    mem_rsp_tag,
    output wire                         mem_rsp_ready
);
    VX_mem_bus_if #(
        .DATA_SIZE  (DATA_SIZE),
        .TAG_WIDTH  (SLOT_TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) mem_bus_if ();

    VX_cp_llc_port #(
        .DATA_SIZE      (DATA_SIZE),
        .SLOT_TAG_WIDTH (SLOT_TAG_WIDTH),
        .TAG_WIDTH      (TAG_WIDTH),
        .ADDR_WIDTH     (ADDR_WIDTH)
    ) dut (
        .clk         (clk),
        .reset       (reset),
        .req_valid   (req_valid),
        .req_rw      (req_rw),
        .req_noalloc (req_noalloc),
        .req_addr    (req_addr),
        .req_data    (req_data),
        .req_byteen  (req_byteen),
        .req_tag     (req_tag),
        .req_ready   (req_ready),
        .rsp_valid   (rsp_valid),
        .rsp_data    (rsp_data),
        .rsp_tag     (rsp_tag),
        .rsp_ready   (rsp_ready),
        .start_in    (start_in),
        .start_out   (start_out),
        .busy        (busy),
        .mem_bus_if  (mem_bus_if)
    );

    assign mem_req_valid  = mem_bus_if.req_valid;
    assign mem_req_rw     = mem_bus_if.req_data.rw;
    assign mem_req_addr   = mem_bus_if.req_data.addr;
    assign mem_req_data   = mem_bus_if.req_data.data;
    assign mem_req_byteen = mem_bus_if.req_data.byteen;
    mem_bus_attr_t mem_req_attr;
    assign mem_req_attr    = mem_bus_attr_t'(mem_bus_if.req_data.attr);
    assign mem_req_flush   = mem_req_attr.is_flush;
    assign mem_req_io      = mem_req_attr.is_addr_io;
    assign mem_req_noalloc = mem_req_attr.is_noalloc;
    `UNUSED_VAR (mem_req_attr)
    assign mem_req_tag    = mem_bus_if.req_data.tag;
    assign mem_bus_if.req_ready = mem_req_ready;

    assign mem_bus_if.rsp_valid = mem_rsp_valid;
    assign mem_bus_if.rsp_data.data = mem_rsp_data;
    assign mem_bus_if.rsp_data.tag  = mem_rsp_tag;
    assign mem_rsp_ready = mem_bus_if.rsp_ready;

endmodule
