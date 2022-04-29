`include "VX_rop_define.vh"

module VX_rop_unit #(    
    parameter CLUSTER_ID = 0,    
    parameter NUM_SLICES = 1,
    parameter NUM_LANES  = 1
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_rop_perf_if.master rop_perf_if,
`endif

    // Memory interface
    VX_cache_req_if.master cache_req_if,
    VX_cache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_dcr_if.slave rop_dcr_if,
    VX_rop_req_if.slave rop_req_if
);
    rop_dcrs_t dcrs;
    assign dcrs = rop_dcr_if.data;

    VX_rop_req_if #(
        .NUM_LANES (NUM_LANES)
    ) per_slice_rop_req_if[NUM_SLICES]();

    VX_cache_req_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_ID_BITS)
    ) per_slice_cache_req_if[NUM_SLICES]();

    VX_cache_rsp_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_ID_BITS)
    ) per_slice_cache_rsp_if[NUM_SLICES]();

    for (genvar i = 0; i < NUM_SLICES; ++i) begin
        VX_rop_slice #(
            .CLUSTER_ID (CLUSTER_ID),
            .NUM_LANES  (NUM_LANES)
        ) rop_slice (
            .clk            (clk),
            .reset          (reset),
            .dcrs           (dcrs),
            .cache_req_if   (per_slice_cache_req_if[i]),
            .cache_rsp_if   (per_slice_cache_rsp_if[i]),
            .rop_req_if     (per_slice_rop_req_if[i])
        );
    end

    VX_rop_req_demux #(
        .NUM_REQS  (NUM_SLICES),
        .NUM_LANES (NUM_LANES),
        .BUFFERED  (1)
    ) rop_req_demux (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (rop_req_if),
        .req_out_if (per_slice_rop_req_if)
    );

    VX_cache_mux #(
        .NUM_REQS     (NUM_SLICES),
        .NUM_LANES    (`OCACHE_NUM_REQS),
        .DATA_SIZE    (`OCACHE_WORD_SIZE),
        .TAG_IN_WIDTH (`OCACHE_TAG_ID_BITS),
        .TAG_SEL_IDX  (0),
        .BUFFERED_REQ (1),
        .BUFFERED_RSP (1)
    ) cache_req_mux (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_slice_cache_req_if),        
        .rsp_in_if  (per_slice_cache_rsp_if),
        .req_out_if (cache_req_if),
        .rsp_out_if (cache_rsp_if)
    );

`ifdef PERF_ENABLE

    wire [$clog2(`OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_req_per_cycle;
    wire [$clog2(`OCACHE_NUM_REQS+1)-1:0] perf_mem_wr_req_per_cycle;
    wire [$clog2(`OCACHE_NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;
    wire [$clog2(`OCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [`OCACHE_NUM_REQS-1:0] perf_mem_rd_req_per_mask = cache_req_if.valid & ~cache_req_if.rw & cache_req_if.ready;
    wire [`OCACHE_NUM_REQS-1:0] perf_mem_wr_req_per_mask = cache_req_if.valid & cache_req_if.rw & cache_req_if.ready;
    wire [`OCACHE_NUM_REQS-1:0] perf_mem_rsp_per_mask    = cache_rsp_if.valid & cache_rsp_if.ready;

    `POP_COUNT(perf_mem_rd_req_per_cycle, perf_mem_rd_req_per_mask);    
    `POP_COUNT(perf_mem_wr_req_per_cycle, perf_mem_wr_req_per_mask);    
    `POP_COUNT(perf_mem_rsp_per_cycle,    perf_mem_rsp_per_mask);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    assign perf_pending_reads_cycle = perf_mem_rd_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= 0;
        end else begin
            perf_pending_reads <= perf_pending_reads + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = rop_req_if.valid & ~rop_req_if.ready;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= 0;
            perf_mem_writes   <= 0;
            perf_mem_latency  <= 0;
            perf_stall_cycles <= 0;
        end else begin
            perf_mem_reads    <= perf_mem_reads    + `PERF_CTR_BITS'(perf_mem_rd_req_per_cycle);
            perf_mem_writes   <= perf_mem_writes   + `PERF_CTR_BITS'(perf_mem_wr_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency  + `PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign rop_perf_if.mem_reads    = perf_mem_reads;
    assign rop_perf_if.mem_writes   = perf_mem_writes;
    assign rop_perf_if.mem_latency  = perf_mem_latency;
    assign rop_perf_if.stall_cycles = perf_stall_cycles;

`endif

endmodule
