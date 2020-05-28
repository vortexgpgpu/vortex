`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex_Socket (
    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DRAM request
    output wire                             dram_req_valid,
    output wire                             dram_req_rw,    
    output wire[`L3DRAM_BYTEEN_WIDTH-1:0]   dram_req_byteen,    
    output wire[`L3DRAM_ADDR_WIDTH-1:0]     dram_req_addr,
    output wire[`L3DRAM_LINE_WIDTH-1:0]     dram_req_data,
    output wire[`L3DRAM_TAG_WIDTH-1:0]      dram_req_tag,
    input  wire                             dram_req_ready,

    // DRAM response    
    input wire                              dram_rsp_valid,        
    input wire[`L3DRAM_LINE_WIDTH-1:0]      dram_rsp_data,
    input wire[`L3DRAM_TAG_WIDTH-1:0]       dram_rsp_tag,
    output wire                             dram_rsp_ready,

    // Snoop request
    input wire                              snp_req_valid,
    input wire[`L3DRAM_ADDR_WIDTH-1:0]      snp_req_addr,
    input wire[`L3SNP_TAG_WIDTH-1:0]        snp_req_tag,
    output wire                             snp_req_ready, 

    // Snoop response
    output wire                             snp_rsp_valid,
    output wire[`L3SNP_TAG_WIDTH-1:0]       snp_rsp_tag,
    input wire                              snp_rsp_ready,     

    // I/O request
    output wire                             io_req_valid,
    output wire                             io_req_rw,  
    output wire[3:0]                        io_req_byteen,  
    output wire[29:0]                       io_req_addr,
    output wire[31:0]                       io_req_data,    
    output wire[`DCORE_TAG_WIDTH-1:0]       io_req_tag,    
    input wire                              io_req_ready,

    // I/O response
    input wire                              io_rsp_valid,
    input wire[31:0]                        io_rsp_data,
    input wire[`DCORE_TAG_WIDTH-1:0]        io_rsp_tag,
    output wire                             io_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
    if (`NUM_CLUSTERS == 1) begin

        Vortex_Cluster #(
            .CLUSTER_ID(`L3CACHE_ID)
        ) Vortex_Cluster (
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

            .busy               (busy),
            .ebreak             (ebreak)
        );

    end else begin

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_valid;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_rw;        
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] per_cluster_dram_req_byteen;   
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_dram_req_addr;
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_req_data;
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_req_tag;
        wire                                            l3_core_req_ready;
  
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_valid;        
        wire[`NUM_CLUSTERS-1:0][`L3DRAM_LINE_WIDTH-1:0] per_cluster_dram_rsp_data;
        wire[`NUM_CLUSTERS-1:0][`L3DRAM_TAG_WIDTH-1:0]  per_cluster_dram_rsp_tag; 
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_ready;

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_valid;
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_snp_req_addr;
        wire[`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]   per_cluster_snp_req_tag;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_ready;

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_snp_rsp_valid;
        wire[`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]   per_cluster_snp_rsp_tag;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_snp_rsp_ready;

    `IGNORE_WARNINGS_BEGIN
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_io_req_valid;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_io_req_rw;
        wire[`NUM_CLUSTERS-1:0][3:0]                    per_cluster_io_req_byteen;
        wire[`NUM_CLUSTERS-1:0][29:0]                   per_cluster_io_req_addr;
        wire[`NUM_CLUSTERS-1:0][31:0]                   per_cluster_io_req_data;        
        wire[`NUM_CLUSTERS-1:0][`DCORE_TAG_WIDTH-1:0]   per_cluster_io_req_tag;

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_io_rsp_ready;
    `IGNORE_WARNINGS_END

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_busy;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_ebreak;

        genvar i;
        for (i = 0; i < `NUM_CLUSTERS; i++) begin        
            Vortex_Cluster #(
                .CLUSTER_ID(i)
            ) Vortex_Cluster (
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
                .io_req_ready       (io_req_ready),

                .io_rsp_valid       (io_rsp_valid),            
                .io_rsp_data        (io_rsp_data),
                .io_rsp_tag         (io_rsp_tag),
                .io_rsp_ready       (per_cluster_io_rsp_ready   [i]),

                .busy               (per_cluster_busy           [i]),
                .ebreak             (per_cluster_ebreak         [i])
            );
        end        

        assign io_req_valid  = per_cluster_io_req_valid[0];
        assign io_req_rw     = per_cluster_io_req_rw[0];
        assign io_req_byteen = per_cluster_io_req_byteen[0];
        assign io_req_addr   = per_cluster_io_req_addr[0];
        assign io_req_data   = per_cluster_io_req_data[0];
        assign io_req_tag    = per_cluster_io_req_tag[0];

        assign io_rsp_ready  = per_cluster_io_rsp_ready[0];

        assign busy   = (| per_cluster_busy);
        assign ebreak = (& per_cluster_ebreak);

        // L3 Cache ///////////////////////////////////////////////////////////

        wire[`L3NUM_REQUESTS-1:0]                           l3_core_req_valid;
        wire[`L3NUM_REQUESTS-1:0]                           l3_core_req_rw;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] l3_core_req_byteen;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_ADDR_WIDTH-1:0]   l3_core_req_addr;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_LINE_WIDTH-1:0]   l3_core_req_data;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_TAG_WIDTH-1:0]    l3_core_req_tag;

        wire[`L3NUM_REQUESTS-1:0]                           l3_core_rsp_valid;        
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_LINE_WIDTH-1:0]   l3_core_rsp_data;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_TAG_WIDTH-1:0]    l3_core_rsp_tag;
        wire                                                l3_core_rsp_ready;    

        wire[`NUM_CLUSTERS-1:0]                             l3_snp_fwdout_valid;
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0]     l3_snp_fwdout_addr;
        wire[`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]       l3_snp_fwdout_tag;
        wire[`NUM_CLUSTERS-1:0]                             l3_snp_fwdout_ready;    

        wire[`NUM_CLUSTERS-1:0]                             l3_snp_fwdin_valid;
        wire[`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]       l3_snp_fwdin_tag;
        wire[`NUM_CLUSTERS-1:0]                             l3_snp_fwdin_ready;

        for (i = 0; i < `L3NUM_REQUESTS; i++) begin
            // Core Request
            assign l3_core_req_valid [i] = per_cluster_dram_req_valid [i];
            assign l3_core_req_rw    [i] = per_cluster_dram_req_rw    [i];
            assign l3_core_req_byteen[i] = per_cluster_dram_req_byteen[i];
            assign l3_core_req_addr  [i] = per_cluster_dram_req_addr  [i];
            assign l3_core_req_tag   [i] = per_cluster_dram_req_tag   [i];
            assign l3_core_req_data  [i] = per_cluster_dram_req_data  [i];            

            // Core Response
            assign per_cluster_dram_rsp_valid [i] = l3_core_rsp_valid [i] && l3_core_rsp_ready;
            assign per_cluster_dram_rsp_data  [i] = l3_core_rsp_data [i];
            assign per_cluster_dram_rsp_tag   [i] = l3_core_rsp_tag [i];

            // Snoop Forwarding out
            assign per_cluster_snp_req_valid [i] = l3_snp_fwdout_valid[i];
            assign per_cluster_snp_req_addr  [i] = l3_snp_fwdout_addr[i];
            assign per_cluster_snp_req_tag   [i] = l3_snp_fwdout_tag[i];
            assign l3_snp_fwdout_ready       [i] = per_cluster_snp_req_ready[i];

            // Snoop Forwarding in
            assign l3_snp_fwdin_valid        [i] = per_cluster_snp_rsp_valid [i];
            assign l3_snp_fwdin_tag          [i] = per_cluster_snp_rsp_tag [i];
            assign per_cluster_snp_rsp_ready [i] = l3_snp_fwdin_ready [i];
        end

        assign l3_core_rsp_ready = (& per_cluster_dram_rsp_ready);

        VX_cache #(
            .CACHE_ID               (0),
            .CACHE_SIZE             (`L3CACHE_SIZE),
            .BANK_LINE_SIZE         (`L3BANK_LINE_SIZE),
            .NUM_BANKS              (`L3NUM_BANKS),
            .WORD_SIZE              (`L3WORD_SIZE),
            .NUM_REQUESTS           (`L3NUM_REQUESTS),
            .STAGE_1_CYCLES         (`L3STAGE_1_CYCLES),
            .CREQ_SIZE              (`L3CREQ_SIZE),
            .MRVQ_SIZE              (`L3MRVQ_SIZE),
            .DFPQ_SIZE              (`L3DFPQ_SIZE),
            .SNRQ_SIZE              (`L3SNRQ_SIZE),
            .CWBQ_SIZE              (`L3CWBQ_SIZE),
            .DWBQ_SIZE              (`L3DWBQ_SIZE),
            .DFQQ_SIZE              (`L3DFQQ_SIZE),
            .PRFQ_SIZE              (`L3PRFQ_SIZE),
            .PRFQ_STRIDE            (`L3PRFQ_STRIDE),
            .DRAM_ENABLE            (1),
            .WRITE_ENABLE           (1),
            .SNOOP_FORWARDING       (1),
            .CORE_TAG_WIDTH         (`L2DRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS       (0),
            .DRAM_TAG_WIDTH         (`L3DRAM_TAG_WIDTH),
            .NUM_SNP_REQUESTS       (`NUM_CLUSTERS),
            .SNP_REQ_TAG_WIDTH      (`L3SNP_TAG_WIDTH),
            .SNP_FWD_TAG_WIDTH      (`L2SNP_TAG_WIDTH)
        ) gpu_l3cache (
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
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            // Snoop response
            .snp_rsp_valid      (snp_rsp_valid),
            .snp_rsp_tag        (snp_rsp_tag),
            .snp_rsp_ready      (snp_rsp_ready),

            // Snoop forwarding out
            .snp_fwdout_valid   (l3_snp_fwdout_valid),
            .snp_fwdout_addr    (l3_snp_fwdout_addr),
            .snp_fwdout_tag     (l3_snp_fwdout_tag),
            .snp_fwdout_ready   (l3_snp_fwdout_ready),

            // Snoop forwarding in
            .snp_fwdin_valid    (l3_snp_fwdin_valid),
            .snp_fwdin_tag      (l3_snp_fwdin_tag),
            .snp_fwdin_ready    (l3_snp_fwdin_ready)       
        );
    end

`ifdef DBG_PRINT_DRAM
    always_ff @(posedge clk) begin
        if (dram_req_valid && dram_req_ready) begin
            $display("%t: DRAM req: rw=%b addr=%0h, tag=%0h, byteen=%0h data=%0h", $time, dram_req_rw, {dram_req_addr, `CLOG2(`GLOBAL_BLOCK_SIZE)'(0)}, dram_req_tag, dram_req_byteen, dram_req_data);
        end
        if (dram_rsp_valid && dram_rsp_ready) begin
            $display("%t: DRAM rsp: tag=%0h, data=%0h", $time, dram_rsp_tag, dram_rsp_data);
        end
    end
`endif

endmodule