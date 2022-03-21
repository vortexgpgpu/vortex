`include "VX_rop_define.vh"

// Module for handling memory requests
module VX_rop_mem #(
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4,
    parameter TAG_WIDTH  = 1
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_rop_perf_if.master rop_perf_if,
`endif

    // Device configuration
    input rop_dcrs_t dcrs,

    // Memory interface
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Request interface
    input wire                                      req_valid,
    input wire [NUM_LANES-1:0]                      req_tmask,
    input wire                                      req_rw,
    input wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0]   req_pos_x,
    input wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0]   req_pos_y,
    input rgba_t [NUM_LANES-1:0]                    req_color, 
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] req_depth,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] req_stencil,
    input wire [TAG_WIDTH-1:0]                      req_tag,
    output wire                                     req_ready,

    // Response interface
    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0]         rsp_tmask,
    output rgba_t [NUM_LANES-1:0]       rsp_color, 
    output wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] rsp_depth,
    output wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] rsp_stencil,
    output wire [TAG_WIDTH-1:0]         rsp_tag,
    input wire                          rsp_ready
);

    localparam NUM_REQS = 2 * NUM_LANES;

    wire [NUM_REQS-1:0]       req_mask;
    wire [NUM_REQS-1:0]       write_mask;
    wire [NUM_REQS-1:0][31:0] req_addr;
    wire [NUM_REQS-1:0][31:0] req_data;
    wire [NUM_REQS-1:0][31:0] rsp_data;

`ifdef PERF_ENABLE
    // TODO
    assign rop_perf_if.mem_reads = 0;
    assign rop_perf_if.mem_writes = 0;
    assign rop_perf_if.mem_latency = 0;
`endif

    //////////////////////////////////////////////////////////////////

    assign write_mask = { {NUM_LANES{~dcrs.depth_writemask}}, {NUM_LANES{dcrs.depth_writemask}} };
    assign req_mask = {2{req_tmask}} & write_mask;
    assign cache_rsp_if.ready = rsp_ready;

    for (genvar i = 0;  i < NUM_LANES; ++i) begin
        assign req_addr[i]    = dcrs.zbuf_addr + (req_pos_y[i] * dcrs.zbuf_pitch) + (req_pos_x[i] * 4);
        assign req_data[i]    = {req_stencil[i], req_depth[i]};
        assign rsp_depth[i]   = rsp_data[i] & `ROP_DEPTH_MASK;
        assign rsp_stencil[i] = (rsp_data[i] >> `ROP_DEPTH_BITS) & `ROP_STENCIL_MASK;
    end

    for (genvar i = NUM_LANES; i < NUM_REQS; ++i) begin
        assign req_addr[i] = dcrs.cbuf_addr + (req_pos_y[i - NUM_LANES] * dcrs.cbuf_pitch) + (req_pos_x[i - NUM_LANES] * 4);
        assign req_data[i] = req_color[i - NUM_LANES];
        assign rsp_color[i - NUM_LANES] = rsp_data[i];
    end

    VX_mem_streamer #(
        .NUM_REQS         (NUM_REQS),
        .ADDRW            (32),
        .DATAW            (32),
        .TAGW             (TAG_WIDTH),
        .WORD_SIZE        (4),
        .QUEUE_SIZE       (16),
        .PARTIAL_RESPONSE (0)
    ) mem_streamer (
        .clk            (clk),
        .reset          (reset),

        .req_valid      (req_valid),
        .req_rw         (req_rw),
        .req_mask       (req_mask),
        .req_byteen     (8'hff),
        .req_addr       (req_addr),
        .req_data       (req_data),
        .req_tag        (req_tag),
        .req_ready      (req_ready),

        .rsp_valid      (rsp_valid),
        .rsp_mask       (rsp_tmask),
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
        .mem_rsp_mask   (cache_rsp_if.tmask),
        .mem_rsp_data   (cache_rsp_if.data),
        .mem_rsp_tag    (cache_rsp_if.tag),
        .mem_rsp_ready  (cache_rsp_if.ready)
    );
    
endmodule
