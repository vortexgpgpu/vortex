`include "VX_rop_define.vh"

// Module for handling memory requests
module VX_rop_mem #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4
) (
    input wire clk,
    input wire reset,

    // Input request
    input wire                           req_valid,
    input wire                           req_rw,
    input wire [NUM_REQS-1:0]            req_mask,
    input wire [WORD_SIZE-1:0]           req_byteen,
    input wire [NUM_REQS-1:0][ADDRW-1:0] req_addr,
    input wire [NUM_REQS-1:0][DATAW-1:0] req_data,
    input wire [TAGW-1:0]                req_tag,
    output wire                          req_ready,

    // Output response
    output wire                           rsp_valid,
    output wire [NUM_REQS-1:0]            rsp_mask,
    output wire [NUM_REQS-1:0][DATAW-1:0] rsp_data,
    output wire [TAGW-1:0]                rsp_tag,
    input wire                            rsp_ready
);

    // Memory interface
    VX_dcache_req_if.master cache_req_if;
    VX_dcache_rsp_if.slave  cache_rsp_if;

    VX_mem_streamer #(
        .NUM_REQS (NUM_LANES),
        .ADDRW (32),
        .DATAW (32),
        .TAGW (32),
        .WORD_SIZE (4),
        .QUEUE_SIZE (16),
        .PARTIAL_RESPONSE (1)
    ) mem_streamer (
        .clk            (clk),
        .reset          (reset),

        .req_valid      (req_valid),
        .req_rw         (req_rw),
        .req_mask       (req_mask),
        .req_byteen     (req_byteen),
        .req_addr       (req_addr),
        .req_data       (req_data),
        .req_tag        (req_tag),
        .req_ready      (req_ready),

        .rsp_valid      (rsp_valid),
        .rsp_mask       (rsp_mask),
        .rsp_data       (rsp_data),
        .rsp_tag        (rsp_tag),
        .rsp_ready      (rsp_ready),

        .mem_req_valid  (cache_req_if.valid),
        .mem_req_rw     (cache_req_if.rw),
        .mem_req_byteen (cache_req_if.byteen),
        .mem_req_addr   (cache_req_if.addr),
        .mem_req_data   (cache_req_if.data),
        .mem_req_tag    (cache_req_if.tag),
        .mem_req_ready  (cache_req_if.ready),

        .mem_rsp_valid  (cache_rsp_if.valid),
        .mem_rsp_mask   (cache_rsp_if.mask),
        .mem_rsp_data   (cache_rsp_if.data),
        .mem_rsp_tag    (cache_rsp_if.tag),
        .mem_rsp_ready  (cache_rsp_if.ready),
    );
    
endmodule