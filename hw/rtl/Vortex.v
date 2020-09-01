`include "VX_define.vh"

module Vortex (
    `SCOPE_SIGNALS_ISTAGE_IO
    `SCOPE_SIGNALS_LSU_IO
    `SCOPE_SIGNALS_CORE_IO
    `SCOPE_SIGNALS_CACHE_IO
    `SCOPE_SIGNALS_PIPELINE_IO
    `SCOPE_SIGNALS_EX_IO

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

    // Snoop request
    input wire                              snp_req_valid,
    input wire [`VX_DRAM_ADDR_WIDTH-1:0]    snp_req_addr,
    input wire                              snp_req_invalidate,
    input wire [`VX_SNP_TAG_WIDTH-1:0]      snp_req_tag,
    output wire                             snp_req_ready, 

    // Snoop response
    output wire                             snp_rsp_valid,
    output wire [`VX_SNP_TAG_WIDTH-1:0]     snp_rsp_tag,
    input wire                              snp_rsp_ready,     

    // I/O request
    output wire                             io_req_valid,
    output wire                             io_req_rw,  
    output wire [3:0]                       io_req_byteen,  
    output wire [29:0]                      io_req_addr,
    output wire [31:0]                      io_req_data,    
    output wire [`VX_CORE_TAG_WIDTH-1:0]    io_req_tag,    
    input wire                              io_req_ready,

    // I/O response
    input wire                              io_rsp_valid,
    input wire [31:0]                       io_rsp_data,
    input wire [`VX_CORE_TAG_WIDTH-1:0]     io_rsp_tag,
    output wire                             io_rsp_ready,

    // CSR I/O Request
    input  wire                             csr_io_req_valid,
    input  wire [`VX_CSR_ID_WIDTH-1:0]      csr_io_req_coreid,
    input  wire [11:0]                      csr_io_req_addr,
    input  wire                             csr_io_req_rw,
    input  wire [31:0]                      csr_io_req_data,
    output wire                             csr_io_req_ready,

    // CSR I/O Response
    output wire                             csr_io_rsp_valid,
    output wire [31:0]                      csr_io_rsp_data,
    input wire                              csr_io_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
    if (`NUM_CLUSTERS == 1) begin

        VX_cluster #(
            .CLUSTER_ID(0)
        ) cluster (
            `SCOPE_SIGNALS_ISTAGE_BIND
            `SCOPE_SIGNALS_LSU_BIND
            `SCOPE_SIGNALS_CORE_BIND
            `SCOPE_SIGNALS_CACHE_BIND
            `SCOPE_SIGNALS_PIPELINE_BIND
            `SCOPE_SIGNALS_EX_BIND

            .clk                (clk),
            .reset              (reset),
            
            .dram_req_valid     (dram_req_valid),
            .dram_req_rw        (dram_req_rw),
            .dram_req_byteen    (dram_req_byteen),
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),

            .dram_rsp_valid     (dram_rsp_valid),            
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_ready     (dram_rsp_ready),

            .snp_req_valid      (snp_req_valid),
            .snp_req_addr       (snp_req_addr),
            .snp_req_invalidate (snp_req_invalidate),
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            .snp_rsp_valid      (snp_rsp_valid),
            .snp_rsp_tag        (snp_rsp_tag),
            .snp_rsp_ready      (snp_rsp_ready),

            .io_req_valid       (io_req_valid),
            .io_req_rw          (io_req_rw),
            .io_req_byteen      (io_req_byteen),
            .io_req_addr        (io_req_addr),
            .io_req_data        (io_req_data),
            .io_req_tag         (io_req_tag),
            .io_req_ready       (io_req_ready),

            .io_rsp_valid       (io_rsp_valid),            
            .io_rsp_data        (io_rsp_data),
            .io_rsp_tag         (io_rsp_tag),
            .io_rsp_ready       (io_rsp_ready),

            .csr_io_req_valid   (csr_io_req_valid),
            .csr_io_req_coreid  (csr_io_req_coreid),
            .csr_io_req_rw      (csr_io_req_rw),
            .csr_io_req_addr    (csr_io_req_addr),
            .csr_io_req_data    (csr_io_req_data),
            .csr_io_req_ready   (csr_io_req_ready),

            .csr_io_rsp_valid   (csr_io_rsp_valid),            
            .csr_io_rsp_data    (csr_io_rsp_data),
            .csr_io_rsp_ready   (csr_io_rsp_ready),

            .busy               (busy),
            .ebreak             (ebreak)
        );

    end else begin

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_valid;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_rw;        
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] per_cluster_dram_req_byteen;   
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_dram_req_addr;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_req_data;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_req_tag;
        wire                                            l3_core_req_ready;
  
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_valid;        
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_rsp_data;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_rsp_tag; 
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_valid;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_snp_req_addr;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_invalidate;
        wire [`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]   per_cluster_snp_req_tag;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_rsp_valid;
        wire [`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]   per_cluster_snp_rsp_tag;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_rsp_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_req_valid;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_req_rw;
        wire [`NUM_CLUSTERS-1:0][3:0]                    per_cluster_io_req_byteen;
        wire [`NUM_CLUSTERS-1:0][29:0]                   per_cluster_io_req_addr;
        wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_io_req_data;        
        wire [`NUM_CLUSTERS-1:0][`L2CORE_TAG_WIDTH-1:0]  per_cluster_io_req_tag;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_req_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_rsp_valid;
        wire [`NUM_CLUSTERS-1:0][`L2CORE_TAG_WIDTH-1:0]  per_cluster_io_rsp_tag;
        wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_io_rsp_data;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_rsp_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_req_valid;
        wire [`NUM_CLUSTERS-1:0][11:0]                   per_cluster_csr_io_req_addr;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_req_rw;
        wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_csr_io_req_data;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_req_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_rsp_valid;
        wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_csr_io_rsp_data;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_rsp_ready;

        wire [`NUM_CLUSTERS-1:0]                         per_cluster_busy;
        wire [`NUM_CLUSTERS-1:0]                         per_cluster_ebreak;

        wire [`CLOG2(`NUM_CLUSTERS)-1:0] csr_io_request_id = `CLOG2(`NUM_CLUSTERS)'(csr_io_req_coreid >> `CLOG2(`NUM_CLUSTERS));
        wire [`NC_BITS-1:0] per_cluster_csr_io_req_coreid = `NC_BITS'(csr_io_req_coreid);

        for (genvar i = 0; i < `NUM_CLUSTERS; i++) begin        
            VX_cluster #(
                .CLUSTER_ID(i)
            ) cluster (
                `SCOPE_SIGNALS_ISTAGE_BIND
                `SCOPE_SIGNALS_LSU_BIND
                `SCOPE_SIGNALS_CORE_BIND
                `SCOPE_SIGNALS_CACHE_BIND
                `SCOPE_SIGNALS_PIPELINE_BIND
                `SCOPE_SIGNALS_EX_BIND

                .clk                (clk),
                .reset              (reset),

                .dram_req_valid     (per_cluster_dram_req_valid [i]),
                .dram_req_rw        (per_cluster_dram_req_rw    [i]),
                .dram_req_byteen    (per_cluster_dram_req_byteen[i]),
                .dram_req_addr      (per_cluster_dram_req_addr  [i]),
                .dram_req_data      (per_cluster_dram_req_data  [i]),
                .dram_req_tag       (per_cluster_dram_req_tag   [i]), 
                .dram_req_ready     (l3_core_req_ready),

                .dram_rsp_valid     (per_cluster_dram_rsp_valid [i]),                
                .dram_rsp_data      (per_cluster_dram_rsp_data  [i]),
                .dram_rsp_tag       (per_cluster_dram_rsp_tag   [i]),
                .dram_rsp_ready     (per_cluster_dram_rsp_ready [i]),

                .snp_req_valid      (per_cluster_snp_req_valid  [i]),
                .snp_req_addr       (per_cluster_snp_req_addr   [i]),
                .snp_req_invalidate (per_cluster_snp_req_invalidate[i]),
                .snp_req_tag        (per_cluster_snp_req_tag    [i]),
                .snp_req_ready      (per_cluster_snp_req_ready  [i]),

                .snp_rsp_valid      (per_cluster_snp_rsp_valid  [i]),
                .snp_rsp_tag        (per_cluster_snp_rsp_tag    [i]),
                .snp_rsp_ready      (per_cluster_snp_rsp_ready  [i]),

                .io_req_valid       (per_cluster_io_req_valid   [i]),
                .io_req_rw          (per_cluster_io_req_rw      [i]),
                .io_req_byteen      (per_cluster_io_req_byteen  [i]),
                .io_req_addr        (per_cluster_io_req_addr    [i]),
                .io_req_data        (per_cluster_io_req_data    [i]),                
                .io_req_tag         (per_cluster_io_req_tag     [i]),
                .io_req_ready       (per_cluster_io_req_ready   [i]),

                .io_rsp_valid       (per_cluster_io_rsp_valid   [i]),            
                .io_rsp_data        (per_cluster_io_rsp_data    [i]),
                .io_rsp_tag         (per_cluster_io_rsp_tag     [i]),
                .io_rsp_ready       (per_cluster_io_rsp_ready   [i]),

                .csr_io_req_valid   (per_cluster_csr_io_req_valid[i]),
                .csr_io_req_coreid  (per_cluster_csr_io_req_coreid),
                .csr_io_req_rw      (per_cluster_csr_io_req_rw  [i]),
                .csr_io_req_addr    (per_cluster_csr_io_req_addr[i]),
                .csr_io_req_data    (per_cluster_csr_io_req_data[i]),
                .csr_io_req_ready   (per_cluster_csr_io_req_ready[i]),

                .csr_io_rsp_valid   (per_cluster_csr_io_rsp_valid[i]),            
                .csr_io_rsp_data    (per_cluster_csr_io_rsp_data[i]),
                .csr_io_rsp_ready   (per_cluster_csr_io_rsp_ready[i]),

                .busy               (per_cluster_busy           [i]),
                .ebreak             (per_cluster_ebreak         [i])
            );
        end

        VX_mem_arb #(
            .NUM_REQUESTS  (`NUM_CLUSTERS),
            .WORD_SIZE     (4),
            .TAG_IN_WIDTH  (`L2CORE_TAG_WIDTH),
            .TAG_OUT_WIDTH (`L3CORE_TAG_WIDTH)
        ) io_arb (
            .clk                    (clk),
            .reset                  (reset),

            // input requests
            .in_mem_req_valid       (per_cluster_io_req_valid),
            .in_mem_req_rw          (per_cluster_io_req_rw),
            .in_mem_req_byteen      (per_cluster_io_req_byteen),
            .in_mem_req_addr        (per_cluster_io_req_addr),
            .in_mem_req_data        (per_cluster_io_req_data),  
            .in_mem_req_tag         (per_cluster_io_req_tag),  
            .in_mem_req_ready       (per_cluster_io_req_ready),

            // input responses
            .in_mem_rsp_valid       (per_cluster_io_rsp_valid),
            .in_mem_rsp_data        (per_cluster_io_rsp_data),
            .in_mem_rsp_tag         (per_cluster_io_rsp_tag),
            .in_mem_rsp_ready       (per_cluster_io_rsp_ready),

            // output request
            .out_mem_req_valid      (io_req_valid),
            .out_mem_req_rw         (io_req_rw),        
            .out_mem_req_byteen     (io_req_byteen),        
            .out_mem_req_addr       (io_req_addr),
            .out_mem_req_data       (io_req_data),
            .out_mem_req_tag        (io_req_tag),
            .out_mem_req_ready      (io_req_ready),
            
            // output response
            .out_mem_rsp_valid      (io_rsp_valid),
            .out_mem_rsp_tag        (io_rsp_tag),
            .out_mem_rsp_data       (io_rsp_data),
            .out_mem_rsp_ready      (io_rsp_ready)
        );

        VX_csr_io_arb #(
            .NUM_REQUESTS (`NUM_CLUSTERS)
        ) csr_io_arb (
            .clk                    (clk),
            .reset                  (reset),

            .request_id             (csr_io_request_id),

            // input requests
            .in_csr_io_req_valid    (csr_io_req_valid),    
            .in_csr_io_req_addr     (csr_io_req_addr),
            .in_csr_io_req_rw       (csr_io_req_rw),
            .in_csr_io_req_data     (csr_io_req_data),
            .in_csr_io_req_ready    (csr_io_req_ready),

            // input responses
            .in_csr_io_rsp_valid    (per_cluster_csr_io_rsp_valid),
            .in_csr_io_rsp_data     (per_cluster_csr_io_rsp_data),
            .in_csr_io_rsp_ready    (per_cluster_csr_io_rsp_ready),

            // output request
            .out_csr_io_req_valid   (per_cluster_csr_io_req_valid),
            .out_csr_io_req_addr    (per_cluster_csr_io_req_addr),            
            .out_csr_io_req_rw      (per_cluster_csr_io_req_rw),
            .out_csr_io_req_data    (per_cluster_csr_io_req_data),  
            .out_csr_io_req_ready   (per_cluster_csr_io_req_ready),            
            
            // output response
            .out_csr_io_rsp_valid   (csr_io_rsp_valid),
            .out_csr_io_rsp_data    (csr_io_rsp_data),
            .out_csr_io_rsp_ready   (csr_io_rsp_ready)
        );

        assign busy   = (| per_cluster_busy);
        assign ebreak = (& per_cluster_ebreak);

        // L3 Cache ///////////////////////////////////////////////////////////

        wire [`L3NUM_REQUESTS-1:0]                           l3_core_req_valid;
        wire [`L3NUM_REQUESTS-1:0]                           l3_core_req_rw;
        wire [`L3NUM_REQUESTS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] l3_core_req_byteen;
        wire [`L3NUM_REQUESTS-1:0][`L2DRAM_ADDR_WIDTH-1:0]   l3_core_req_addr;
        wire [`L3NUM_REQUESTS-1:0][`L2DRAM_LINE_WIDTH-1:0]   l3_core_req_data;
        wire [`L3NUM_REQUESTS-1:0][`L2DRAM_TAG_WIDTH-1:0]    l3_core_req_tag;

        wire [`L3NUM_REQUESTS-1:0]                           l3_core_rsp_valid;        
        wire [`L3NUM_REQUESTS-1:0][`L2DRAM_LINE_WIDTH-1:0]   l3_core_rsp_data;
        wire [`L3NUM_REQUESTS-1:0][`L2DRAM_TAG_WIDTH-1:0]    l3_core_rsp_tag;
        wire                                                 l3_core_rsp_ready;    

        wire [`NUM_CLUSTERS-1:0]                             l3_snp_fwdout_valid;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0]     l3_snp_fwdout_addr;
        wire [`NUM_CLUSTERS-1:0]                             l3_snp_fwdout_invalidate;
        wire [`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]       l3_snp_fwdout_tag;
        wire [`NUM_CLUSTERS-1:0]                             l3_snp_fwdout_ready;    

        wire [`NUM_CLUSTERS-1:0]                             l3_snp_fwdin_valid;
        wire [`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]       l3_snp_fwdin_tag;
        wire [`NUM_CLUSTERS-1:0]                             l3_snp_fwdin_ready;

        for (genvar i = 0; i < `L3NUM_REQUESTS; i++) begin
            // Core Request
            assign l3_core_req_valid  [i] = per_cluster_dram_req_valid [i];
            assign l3_core_req_rw     [i] = per_cluster_dram_req_rw    [i];
            assign l3_core_req_byteen [i] = per_cluster_dram_req_byteen[i];
            assign l3_core_req_addr   [i] = per_cluster_dram_req_addr  [i];
            assign l3_core_req_tag    [i] = per_cluster_dram_req_tag   [i];
            assign l3_core_req_data   [i] = per_cluster_dram_req_data  [i];            

            // Core Response
            assign per_cluster_dram_rsp_valid [i] = l3_core_rsp_valid [i] && l3_core_rsp_ready;
            assign per_cluster_dram_rsp_data  [i] = l3_core_rsp_data [i];
            assign per_cluster_dram_rsp_tag   [i] = l3_core_rsp_tag [i];

            // Snoop Forwarding out
            assign per_cluster_snp_req_valid      [i] = l3_snp_fwdout_valid[i];
            assign per_cluster_snp_req_addr       [i] = l3_snp_fwdout_addr[i];
            assign per_cluster_snp_req_invalidate [i] = l3_snp_fwdout_invalidate[i];
            assign per_cluster_snp_req_tag        [i] = l3_snp_fwdout_tag[i];
            assign l3_snp_fwdout_ready            [i] = per_cluster_snp_req_ready[i];

            // Snoop Forwarding in
            assign l3_snp_fwdin_valid        [i] = per_cluster_snp_rsp_valid [i];
            assign l3_snp_fwdin_tag          [i] = per_cluster_snp_rsp_tag [i];
            assign per_cluster_snp_rsp_ready [i] = l3_snp_fwdin_ready [i];
        end

        assign l3_core_rsp_ready = (& per_cluster_dram_rsp_ready);

        VX_cache #(
            .CACHE_ID           (0),
            .CACHE_SIZE         (`L3CACHE_SIZE),
            .BANK_LINE_SIZE     (`L3BANK_LINE_SIZE),
            .NUM_BANKS          (`L3NUM_BANKS),
            .WORD_SIZE          (`L3WORD_SIZE),
            .NUM_REQUESTS       (`L3NUM_REQUESTS),
            .CREQ_SIZE          (`L3CREQ_SIZE),
            .MRVQ_SIZE          (`L3MRVQ_SIZE),
            .DFPQ_SIZE          (`L3DFPQ_SIZE),
            .SNRQ_SIZE          (`L3SNRQ_SIZE),
            .CWBQ_SIZE          (`L3CWBQ_SIZE),
            .DWBQ_SIZE          (`L3DWBQ_SIZE),
            .DFQQ_SIZE          (`L3DFQQ_SIZE),
            .PRFQ_SIZE          (`L3PRFQ_SIZE),
            .PRFQ_STRIDE        (`L3PRFQ_STRIDE),
            .DRAM_ENABLE        (1),
            .WRITE_ENABLE       (1),
            .SNOOP_FORWARDING   (1),
            .CORE_TAG_WIDTH     (`L2DRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .DRAM_TAG_WIDTH     (`L3DRAM_TAG_WIDTH),
            .NUM_SNP_REQUESTS   (`NUM_CLUSTERS),
            .SNP_REQ_TAG_WIDTH  (`L3SNP_TAG_WIDTH),
            .SNP_FWD_TAG_WIDTH  (`L2SNP_TAG_WIDTH)
        ) l3cache (
            `SCOPE_SIGNALS_CACHE_UNBIND

            .clk                (clk),
            .reset              (reset),

            // Core request    
            .core_req_valid     (l3_core_req_valid),
            .core_req_rw        (l3_core_req_rw),
            .core_req_byteen    (l3_core_req_byteen),
            .core_req_addr      (l3_core_req_addr),
            .core_req_data      (l3_core_req_data),
            .core_req_tag       (l3_core_req_tag),
            .core_req_ready     (l3_core_req_ready),

            // Core response
            .core_rsp_valid     (l3_core_rsp_valid),
            .core_rsp_data      (l3_core_rsp_data),
            .core_rsp_tag       (l3_core_rsp_tag),              
            .core_rsp_ready     (l3_core_rsp_ready),

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
            .dram_rsp_ready     (dram_rsp_ready),

            // Snoop request
            .snp_req_valid      (snp_req_valid),
            .snp_req_addr       (snp_req_addr),
            .snp_req_invalidate (snp_req_invalidate),
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            // Snoop response
            .snp_rsp_valid      (snp_rsp_valid),
            .snp_rsp_tag        (snp_rsp_tag),
            .snp_rsp_ready      (snp_rsp_ready),

            // Snoop forwarding out
            .snp_fwdout_valid   (l3_snp_fwdout_valid),
            .snp_fwdout_addr    (l3_snp_fwdout_addr),
            .snp_fwdout_invalidate(l3_snp_fwdout_invalidate),
            .snp_fwdout_tag     (l3_snp_fwdout_tag),
            .snp_fwdout_ready   (l3_snp_fwdout_ready),

            // Snoop forwarding in
            .snp_fwdin_valid    (l3_snp_fwdin_valid),
            .snp_fwdin_tag      (l3_snp_fwdin_tag),
            .snp_fwdin_ready    (l3_snp_fwdin_ready)       
        );
    end

`ifdef DBG_PRINT_DRAM
    always @(posedge clk) begin
        if (dram_req_valid && dram_req_ready) begin
            $display("%t: DRAM req: rw=%b addr=%0h, tag=%0h, byteen=%0h data=%0h", $time, dram_req_rw, `DRAM_TO_BYTE_ADDR(dram_req_addr), dram_req_tag, dram_req_byteen, dram_req_data);
        end
        if (dram_rsp_valid && dram_rsp_ready) begin
            $display("%t: DRAM rsp: tag=%0h, data=%0h", $time, dram_rsp_tag, dram_rsp_data);
        end
    end
`endif

endmodule