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
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if,

    // Request interface
    input wire                                      req_valid,
    input wire [NUM_LANES-1:0]                      req_mask,
    input wire [NUM_LANES-1:0]                      req_ds_pass,
    input wire                                      req_rw,
    input wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0]   req_pos_x,
    input wire [NUM_LANES-1:0][`ROP_DIM_BITS-1:0]   req_pos_y,
    input rgba_t [NUM_LANES-1:0]                    req_color, 
    input wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] req_depth,
    input wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] req_stencil,
    input wire [NUM_LANES-1:0]                      req_backface,
    input wire [TAG_WIDTH-1:0]                      req_tag,
    output wire                                     req_ready,

    // Response interface
    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0]         rsp_mask,
    output rgba_t [NUM_LANES-1:0]       rsp_color, 
    output wire [NUM_LANES-1:0][`ROP_DEPTH_BITS-1:0] rsp_depth,
    output wire [NUM_LANES-1:0][`ROP_STENCIL_BITS-1:0] rsp_stencil,
    output wire [TAG_WIDTH-1:0]         rsp_tag,
    input wire                          rsp_ready
);

    localparam NUM_REQS = 2 * NUM_LANES;

    wire                        mreq_valid;
    wire                        mreq_rw;
    wire [NUM_REQS-1:0]         mreq_mask;
    wire [NUM_REQS-1:0][`OCACHE_ADDR_WIDTH-1:0] mreq_addr;
    wire [NUM_REQS-1:0][31:0]   mreq_data;
    wire [NUM_REQS-1:0][3:0]    mreq_byteen;
    wire [TAG_WIDTH-1:0]        mreq_tag;
    wire                        mreq_ready;
    
    wire                        mrsp_valid;
    wire [NUM_REQS-1:0]         mrsp_mask;
    wire [NUM_REQS-1:0][31:0]   mrsp_data;
    wire [TAG_WIDTH-1:0]        mrsp_tag;
    wire                        mrsp_ready;

    `UNUSED_VAR (dcrs)

`ifdef PERF_ENABLE
    wire [$clog2(NUM_REQS+1)-1:0] perf_mem_rd_req_per_cycle;
    wire [$clog2(NUM_REQS+1)-1:0] perf_mem_wr_req_per_cycle;
    wire [$clog2(NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;

    wire [NUM_REQS-1:0] perf_mem_rd_req_per_mask = cache_req_if.valid & ~cache_req_if.rw & cache_req_if.ready;
    wire [NUM_REQS-1:0] perf_mem_wr_req_per_mask = cache_req_if.valid & cache_req_if.rw & cache_req_if.ready;
    wire [NUM_REQS-1:0] perf_mem_rsp_per_mask = cache_rsp_if.valid & cache_rsp_if.ready;

    `POP_COUNT(perf_mem_rd_req_per_cycle, perf_mem_rd_req_per_mask);    
    `POP_COUNT(perf_mem_wr_req_per_cycle, perf_mem_wr_req_per_mask);    
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_per_mask);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    wire [$clog2(NUM_REQS+1)+1-1:0] perf_pending_reads_cycle = perf_mem_rd_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= 0;
        end else begin
            perf_pending_reads <= perf_pending_reads + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads   <= 0;
            perf_mem_writes  <= 0;
            perf_mem_latency <= 0;
        end else begin
            perf_mem_reads   <= perf_mem_reads   + `PERF_CTR_BITS'(perf_mem_rd_req_per_cycle);
            perf_mem_writes  <= perf_mem_writes  + `PERF_CTR_BITS'(perf_mem_wr_req_per_cycle);
            perf_mem_latency <= perf_mem_latency + `PERF_CTR_BITS'(perf_pending_reads);
        end
    end

    assign rop_perf_if.mem_reads   = perf_mem_reads;
    assign rop_perf_if.mem_writes  = perf_mem_writes;
    assign rop_perf_if.mem_latency = perf_mem_latency;
`endif

    //////////////////////////////////////////////////////////////////

    wire color_enable = req_rw ? (dcrs.cbuf_writemask != 0) : dcrs.blend_enable;
    wire [3:0] color_byteen = dcrs.cbuf_writemask;

    wire depth_enable = dcrs.depth_enable && (~req_rw || (dcrs.depth_writemask != 0));
    wire [2:0] depth_byteen = {3{dcrs.depth_writemask}};

    wire stencil_front_enable = dcrs.stencil_back_enable && (~req_rw || (dcrs.stencil_back_writemask != 0));
    wire stencil_back_enable  = dcrs.stencil_front_enable && (~req_rw || (dcrs.stencil_front_writemask != 0));
    wire stencil_enable = stencil_back_enable | stencil_front_enable;

    wire [NUM_LANES-1:0] stencil_byteen;
    for (genvar i = 0;  i < NUM_LANES; ++i) begin        
        assign stencil_byteen[i] = req_backface[i] ? (dcrs.stencil_back_writemask != 0) : (dcrs.stencil_front_writemask != 0);
    end

    for (genvar i = 0;  i < NUM_LANES; ++i) begin
        wire [31:0] byte_addr = dcrs.zbuf_addr + (req_pos_y[i] * dcrs.zbuf_pitch) + (req_pos_x[i] * 4);
        `UNUSED_VAR (byte_addr)
        assign mreq_mask[i]   = req_mask[i] && (depth_enable | stencil_enable);
        assign mreq_addr[i]   = byte_addr[(32-`OCACHE_ADDR_WIDTH) +: `OCACHE_ADDR_WIDTH];
        assign mreq_data[i]   = {req_stencil[i], req_depth[i]};
        assign mreq_byteen[i] = {stencil_byteen[i], depth_byteen};
    end

    for (genvar i = NUM_LANES; i < NUM_REQS; ++i) begin
        wire [31:0] byte_addr = dcrs.cbuf_addr + (req_pos_y[i - NUM_LANES] * dcrs.cbuf_pitch) + (req_pos_x[i - NUM_LANES] * 4);
        `UNUSED_VAR (byte_addr)        
        assign mreq_mask[i]   = req_mask[i - NUM_LANES] && color_enable && (~req_rw || req_ds_pass[i - NUM_LANES]);
        assign mreq_addr[i]   = byte_addr[(32-`OCACHE_ADDR_WIDTH) +: `OCACHE_ADDR_WIDTH];
        assign mreq_data[i]   = req_color[i - NUM_LANES];
        assign mreq_byteen[i] = color_byteen;
    end

    assign mreq_valid = req_valid && (| mreq_mask);
    assign req_ready  = mreq_ready;
    assign mreq_rw    = req_rw;
    assign mreq_tag   = req_tag;

    VX_mem_streamer #(
        .NUM_REQS         (NUM_REQS),
        .ADDRW            (`OCACHE_ADDR_WIDTH),
        .DATAW            (32),
        .TAGW             (TAG_WIDTH),
        .QUEUE_SIZE       (`ROP_MEM_QUEUE_SIZE),
        .PARTIAL_RESPONSE (0)
    ) mem_streamer (
        .clk            (clk),
        .reset          (reset),

        .req_valid      (mreq_valid),
        .req_rw         (mreq_rw),
        .req_mask       (mreq_mask),
        .req_byteen     (mreq_byteen),
        .req_addr       (mreq_addr),
        .req_data       (mreq_data),
        .req_tag        (mreq_tag),
        .req_ready      (mreq_ready),

        .rsp_valid      (mrsp_valid),
        .rsp_mask       (mrsp_mask),
        .rsp_data       (mrsp_data),
        .rsp_tag        (mrsp_tag),
        .rsp_ready      (mrsp_ready),

        .mem_req_valid  (cache_req_if.valid),
        .mem_req_rw     (cache_req_if.rw),
        .mem_req_byteen (cache_req_if.byteen),
        .mem_req_addr   (cache_req_if.addr),
        .mem_req_data   (cache_req_if.data),
        .mem_req_tag    (cache_req_if.tag),
        .mem_req_ready  (cache_req_if.ready),

        .mem_rsp_valid  (cache_rsp_if.valid),
        .mem_rsp_data   (cache_rsp_if.data),
        .mem_rsp_tag    (cache_rsp_if.tag),
        .mem_rsp_ready  (cache_rsp_if.ready)
    );    

    assign rsp_valid = mrsp_valid;

    assign rsp_mask = (mrsp_mask[0 +: NUM_LANES] | mrsp_mask[NUM_LANES +: NUM_LANES]);

    for (genvar i = 0;  i < NUM_LANES; ++i) begin        
        assign rsp_depth[i]   = `ROP_DEPTH_BITS'(mrsp_data[i] >> 0) & `ROP_DEPTH_BITS'(`ROP_DEPTH_MASK);
        assign rsp_stencil[i] = `ROP_STENCIL_BITS'(mrsp_data[i] >> `ROP_DEPTH_BITS) & `ROP_STENCIL_BITS'(`ROP_STENCIL_MASK);        
    end

    for (genvar i = NUM_LANES; i < NUM_REQS; ++i) begin
        assign rsp_color[i - NUM_LANES] = mrsp_data[i];        
    end

    assign rsp_tag = mrsp_tag;

    assign mrsp_ready = rsp_ready;
    
endmodule
