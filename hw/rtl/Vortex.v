`include "VX_define.vh"

module Vortex (
    `SCOPE_IO_Vortex

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DRAM request
    output wire                             dram_req_valid,
    output wire                             dram_req_rw,    
    output wire [`VX_DRAM_BYTEEN_WIDTH-1:0] dram_req_byteen,    
    output wire [`VX_DRAM_ADDR_WIDTH-1:0]   dram_req_addr,
    output wire [`VX_DRAM_LINE_WIDTH-1:0]   dram_req_data,
    output wire [`VX_DRAM_TAG_WIDTH-1:0]    dram_req_tag,
    input  wire                             dram_req_ready,

    // DRAM response    
    input wire                              dram_rsp_valid,        
    input wire [`VX_DRAM_LINE_WIDTH-1:0]    dram_rsp_data,
    input wire [`VX_DRAM_TAG_WIDTH-1:0]     dram_rsp_tag,
    output wire                             dram_rsp_ready,

    // CSR Request
    input  wire                             csr_req_valid,
    input  wire [`VX_CSR_ID_WIDTH-1:0]      csr_req_coreid,
    input  wire [11:0]                      csr_req_addr,
    input  wire                             csr_req_rw,
    input  wire [31:0]                      csr_req_data,
    output wire                             csr_req_ready,

    // CSR Response
    output wire                             csr_rsp_valid,
    output wire [31:0]                      csr_rsp_data,
    input wire                              csr_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_valid;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_rw;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] per_cluster_dram_req_byteen;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_dram_req_addr;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_req_data;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_req_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_rsp_data;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_rsp_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_req_valid;
    wire [`NUM_CLUSTERS-1:0][11:0]                   per_cluster_csr_req_addr;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_req_rw;
    wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_csr_req_data;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_csr_rsp_data;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_rsp_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_busy;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_ebreak;

    wire [`LOG2UP(`NUM_CLUSTERS)-1:0] csr_cluster_id = `LOG2UP(`NUM_CLUSTERS)'(csr_req_coreid >> `CLOG2(`NUM_CORES));
    wire [`NC_BITS-1:0] csr_core_id = `NC_BITS'(csr_req_coreid);

    for (genvar i = 0; i < `NUM_CLUSTERS; i++) begin

        wire cluster_reset;
        VX_reset_relay #(
            .PASSTHRU (`NUM_CLUSTERS == 1)
        ) reset_relay (
            .clk       (clk),
            .reset     (reset),
            .reset_out (cluster_reset)
        );

        VX_cluster #(
            .CLUSTER_ID(i)
        ) cluster (
            `SCOPE_BIND_Vortex_cluster(i)

            .clk            (clk),
            .reset          (cluster_reset),

            .dram_req_valid (per_cluster_dram_req_valid [i]),
            .dram_req_rw    (per_cluster_dram_req_rw    [i]),
            .dram_req_byteen(per_cluster_dram_req_byteen[i]),
            .dram_req_addr  (per_cluster_dram_req_addr  [i]),
            .dram_req_data  (per_cluster_dram_req_data  [i]),
            .dram_req_tag   (per_cluster_dram_req_tag   [i]),
            .dram_req_ready (per_cluster_dram_req_ready [i]),

            .dram_rsp_valid (per_cluster_dram_rsp_valid [i]),
            .dram_rsp_data  (per_cluster_dram_rsp_data  [i]),
            .dram_rsp_tag   (per_cluster_dram_rsp_tag   [i]),
            .dram_rsp_ready (per_cluster_dram_rsp_ready [i]),

            .csr_req_valid  (per_cluster_csr_req_valid  [i]),
            .csr_req_coreid (csr_core_id),
            .csr_req_rw     (per_cluster_csr_req_rw     [i]),
            .csr_req_addr   (per_cluster_csr_req_addr   [i]),
            .csr_req_data   (per_cluster_csr_req_data   [i]),
            .csr_req_ready  (per_cluster_csr_req_ready  [i]),

            .csr_rsp_valid  (per_cluster_csr_rsp_valid  [i]),
            .csr_rsp_data   (per_cluster_csr_rsp_data   [i]),
            .csr_rsp_ready  (per_cluster_csr_rsp_ready  [i]),

            .busy           (per_cluster_busy           [i]),
            .ebreak         (per_cluster_ebreak         [i])
        );
    end

    VX_csr_arb #(
        .NUM_REQS     (`NUM_CLUSTERS),
        .DATA_WIDTH   (32),
        .ADDR_WIDTH   (12),
        .BUFFERED_REQ (`NUM_CLUSTERS >= 4),
        .BUFFERED_RSP (1)
    ) csr_arb (
        .clk            (clk),
        .reset          (reset),

        .request_id     (csr_cluster_id),

        // input requests
        .req_valid_in   (csr_req_valid),
        .req_addr_in    (csr_req_addr),
        .req_rw_in      (csr_req_rw),
        .req_data_in    (csr_req_data),
        .req_ready_in   (csr_req_ready),

        // output request
        .req_valid_out  (per_cluster_csr_req_valid),
        .req_addr_out   (per_cluster_csr_req_addr),
        .req_rw_out     (per_cluster_csr_req_rw),
        .req_data_out   (per_cluster_csr_req_data),
        .req_ready_out  (per_cluster_csr_req_ready),

        // input responses
        .rsp_valid_in   (per_cluster_csr_rsp_valid),
        .rsp_data_in    (per_cluster_csr_rsp_data),
        .rsp_ready_in   (per_cluster_csr_rsp_ready),
        
        // output response
        .rsp_valid_out  (csr_rsp_valid),
        .rsp_data_out   (csr_rsp_data),
        .rsp_ready_out  (csr_rsp_ready)
    );

    assign busy   = (| per_cluster_busy);
    assign ebreak = (| per_cluster_ebreak);

    if (`L3_ENABLE) begin
    `ifdef PERF_ENABLE
        VX_perf_cache_if perf_l3cache_if();
    `endif

        VX_cache #(
            .CACHE_ID           (`L3CACHE_ID),
            .CACHE_SIZE         (`L3CACHE_SIZE),
            .CACHE_LINE_SIZE     (`L3CACHE_LINE_SIZE),
            .NUM_BANKS          (`L3NUM_BANKS),
            .WORD_SIZE          (`L3WORD_SIZE),
            .NUM_REQS           (`NUM_CLUSTERS),
            .CREQ_SIZE          (`L3CREQ_SIZE),
            .MSHR_SIZE          (`L3MSHR_SIZE),
            .DRSQ_SIZE          (`L3DRSQ_SIZE),
            .CRSQ_SIZE          (`L3CRSQ_SIZE),
            .DREQ_SIZE          (`L3DREQ_SIZE),
            .DRAM_ENABLE        (1),
            .WRITE_ENABLE       (1),
            .CORE_TAG_WIDTH     (`L2DRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .DRAM_TAG_WIDTH     (`L3DRAM_TAG_WIDTH)
        ) l3cache (
            `SCOPE_BIND_Vortex_l3cache
 
            .clk                (clk),
            .reset              (reset),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_l3cache_if),
        `endif

            // Core request    
            .core_req_valid     (per_cluster_dram_req_valid),
            .core_req_rw        (per_cluster_dram_req_rw),
            .core_req_byteen    (per_cluster_dram_req_byteen),
            .core_req_addr      (per_cluster_dram_req_addr),
            .core_req_data      (per_cluster_dram_req_data),
            .core_req_tag       (per_cluster_dram_req_tag),
            .core_req_ready     (per_cluster_dram_req_ready),

            // Core response
            .core_rsp_valid     (per_cluster_dram_rsp_valid),
            .core_rsp_data      (per_cluster_dram_rsp_data),
            .core_rsp_tag       (per_cluster_dram_rsp_tag),              
            .core_rsp_ready     (per_cluster_dram_rsp_ready),

            // DRAM request
            .dram_req_valid     (dram_req_valid),
            .dram_req_rw        (dram_req_rw),
            .dram_req_byteen    (dram_req_byteen),
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),

            // DRAM response
            .dram_rsp_valid     (dram_rsp_valid),            
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_ready     (dram_rsp_ready)
        );

    end else begin

        VX_mem_arb #(
            .NUM_REQS      (`NUM_CLUSTERS),
            .DATA_WIDTH    (`L3DRAM_LINE_WIDTH),            
            .TAG_IN_WIDTH  (`L2DRAM_TAG_WIDTH),
            .TAG_OUT_WIDTH (`L3DRAM_TAG_WIDTH),
            .BUFFERED_REQ  (1),
            .BUFFERED_RSP  (`NUM_CLUSTERS >= 4)
        ) dram_arb (
            .clk            (clk),
            .reset          (reset),

            // Core request
            .req_valid_in   (per_cluster_dram_req_valid),
            .req_rw_in      (per_cluster_dram_req_rw),
            .req_byteen_in  (per_cluster_dram_req_byteen),
            .req_addr_in    (per_cluster_dram_req_addr),
            .req_data_in    (per_cluster_dram_req_data),  
            .req_tag_in     (per_cluster_dram_req_tag),  
            .req_ready_in   (per_cluster_dram_req_ready),

            // DRAM request
            .req_valid_out  (dram_req_valid),
            .req_rw_out     (dram_req_rw),        
            .req_byteen_out (dram_req_byteen),        
            .req_addr_out   (dram_req_addr),
            .req_data_out   (dram_req_data),
            .req_tag_out    (dram_req_tag),
            .req_ready_out  (dram_req_ready),

            // Core response
            .rsp_valid_out  (per_cluster_dram_rsp_valid),
            .rsp_data_out   (per_cluster_dram_rsp_data),
            .rsp_tag_out    (per_cluster_dram_rsp_tag),
            .rsp_ready_out  (per_cluster_dram_rsp_ready),
            
            // DRAM response
            .rsp_valid_in   (dram_rsp_valid),
            .rsp_tag_in     (dram_rsp_tag),
            .rsp_data_in    (dram_rsp_data),
            .rsp_ready_in   (dram_rsp_ready)
        );

    end

    `SCOPE_ASSIGN (reset, reset);

    `SCOPE_ASSIGN (dram_req_fire, dram_req_valid && dram_req_ready);
    `SCOPE_ASSIGN (dram_req_addr, `TO_FULL_ADDR(dram_req_addr));
    `SCOPE_ASSIGN (dram_req_rw,   dram_req_rw);
    `SCOPE_ASSIGN (dram_req_byteen, dram_req_byteen);
    `SCOPE_ASSIGN (dram_req_data, dram_req_data);
    `SCOPE_ASSIGN (dram_req_tag,  dram_req_tag);
    `SCOPE_ASSIGN (dram_rsp_fire, dram_rsp_valid && dram_rsp_ready);
    `SCOPE_ASSIGN (dram_rsp_data, dram_rsp_data);
    `SCOPE_ASSIGN (dram_rsp_tag,  dram_rsp_tag);
    `SCOPE_ASSIGN (busy, busy);

`ifdef DBG_PRINT_DRAM
    always @(posedge clk) begin
        if (dram_req_valid && dram_req_ready) begin
            if (dram_req_rw)
                $display("%t: DRAM Wr Req: addr=%0h, tag=%0h, byteen=%0h data=%0h", $time, `TO_FULL_ADDR(dram_req_addr), dram_req_tag, dram_req_byteen, dram_req_data);
            else
                $display("%t: DRAM Rd Req: addr=%0h, tag=%0h, byteen=%0h", $time, `TO_FULL_ADDR(dram_req_addr), dram_req_tag, dram_req_byteen);
        end
        if (dram_rsp_valid && dram_rsp_ready) begin
            $display("%t: DRAM Rsp: tag=%0h, data=%0h", $time, dram_rsp_tag, dram_rsp_data);
        end
    end
`endif


`ifndef NDEBUG
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule