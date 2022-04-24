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
    rop_dcrs_t dcrs = rop_dcr_if.data;

`ifdef PERF_ENABLE
    VX_rop_perf_if per_slice_rop_perf_if[NUM_SLICES]();
`endif

    VX_rop_req_if #(
        .NUM_LANES (NUM_LANES)
    ) per_slice_rop_req_if[NUM_SLICES]();

    VX_cache_req_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_SEL_BITS)
    ) per_slice_cache_req_if[NUM_SLICES]();

    VX_cache_rsp_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_SEL_BITS)
    ) per_slice_cache_rsp_if[NUM_SLICES]();

    for (genvar i = 0; i < NUM_SLICES; ++i) begin
        VX_rop_slice #(
            .CLUSTER_ID (CLUSTER_ID),
            .NUM_LANES  (NUM_LANES)
        ) rop_slice (
            .clk            (clk),
            .reset          (reset),
        `ifdef PERF_ENABLE
            .rop_perf_if    (per_slice_rop_perf_if[i]),
        `endif
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
        .TAG_IN_WIDTH (`OCACHE_TAG_SEL_BITS),
        .TAG_SEL_IDX  (0),
        .BUFFERED_REQ (1),
        .BUFFERED_RSP (1),
    ) cache_req_mux (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_slice_cache_req_if),        
        .rsp_in_if  (per_slice_cache_rsp_if),
        .req_out_if (cache_req_if),
        .rsp_out_if (cache_rsp_if)
    );

`ifdef PERF_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_idle_cycles;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    wire perf_idle_cycle = ~rop_req_if.valid & rop_req_if.ready;
    wire perf_stall_cycle = rop_req_if.valid & ~rop_req_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            perf_idle_cycles  <= 0;
            perf_stall_cycles <= 0;
        end else begin
            perf_idle_cycles  <= perf_idle_cycles + `PERF_CTR_BITS'(perf_idle_cycle);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign rop_perf_if.idle_cycles = perf_idle_cycles;
    assign rop_perf_if.stall_cycles = perf_stall_cycles;
`endif

endmodule
