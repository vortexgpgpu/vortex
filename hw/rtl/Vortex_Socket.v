`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex_Socket (
    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // DRAM request
    output wire                         dram_req_read,
    output wire                         dram_req_write,    
    output wire[`L3DRAM_ADDR_WIDTH-1:0] dram_req_addr,
    output wire[`L3DRAM_LINE_WIDTH-1:0] dram_req_data,
    output wire[`L3DRAM_TAG_WIDTH-1:0]  dram_req_tag,
    input  wire                         dram_req_ready,

    // DRAM response    
    input  wire                         dram_rsp_valid,        
    input  wire[`L3DRAM_LINE_WIDTH-1:0] dram_rsp_data,
    input  wire[`L3DRAM_TAG_WIDTH-1:0]  dram_rsp_tag,
    output wire                         dram_rsp_ready,

    // Cache snooping
    input  wire                         llc_snp_req_valid,
    input  wire[`L3DRAM_ADDR_WIDTH-1:0] llc_snp_req_addr,
    output wire                         llc_snp_req_ready, 

    // I/O request
    output wire                         io_req_read,
    output wire                         io_req_write,    
    output wire[31:0]                   io_req_addr,
    output wire[31:0]                   io_req_data,
    output wire[`BYTE_EN_BITS-1:0]      io_req_byteen,
    output wire[`CORE_REQ_TAG_WIDTH-1:0] io_req_tag,    
    input wire                          io_req_ready,

    // I/O response
    input wire                          io_rsp_valid,
    input wire[31:0]                    io_rsp_data,
    input wire[`CORE_REQ_TAG_WIDTH-1:0] io_rsp_tag,
    output wire                         io_rsp_ready,

    // Status
    output wire                         busy, 
    output wire                         ebreak
);
    if (`NUM_CLUSTERS == 1) begin

        Vortex_Cluster #(
            .CLUSTER_ID(0)
        ) Vortex_Cluster (
            .clk                (clk),
            .reset              (reset),
            
            .dram_req_read      (dram_req_read),
            .dram_req_write     (dram_req_write),
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),

            .dram_rsp_valid     (dram_rsp_valid),            
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_ready     (dram_rsp_ready),

            .llc_snp_req_valid  (llc_snp_req_valid),
            .llc_snp_req_addr   (llc_snp_req_addr),
            .llc_snp_req_ready  (llc_snp_req_ready),

            .io_req_read        (io_req_read),
            .io_req_write       (io_req_write),
            .io_req_addr        (io_req_addr),
            .io_req_data        (io_req_data),
            .io_req_byteen      (io_req_byteen),
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

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_read;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_write;        
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_dram_req_addr;
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_req_data;
        wire[`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_req_tag;
        wire                                            l3_core_req_ready;
  
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_valid;        
        wire[`NUM_CLUSTERS-1:0][`L3DRAM_LINE_WIDTH-1:0] per_cluster_dram_rsp_data;
        wire[`NUM_CLUSTERS-1:0][`L3DRAM_TAG_WIDTH-1:0]  per_cluster_dram_rsp_tag; 
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_ready;

        wire                                            snp_fwd_valid;
        wire[`L3DRAM_ADDR_WIDTH-1:0]                    snp_fwd_addr;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_snp_fwd_ready;

    `IGNORE_WARNINGS_BEGIN
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_io_req_read;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_io_req_write;
        wire[`NUM_CLUSTERS-1:0][31:0]                   per_cluster_io_req_addr;
        wire[`NUM_CLUSTERS-1:0][31:0]                   per_cluster_io_req_data;
        wire[`NUM_CLUSTERS-1:0][`BYTE_EN_BITS-1:0]      per_cluster_io_req_byteen;
        wire[`NUM_CLUSTERS-1:0][`CORE_REQ_TAG_WIDTH-1:0] per_cluster_io_req_tag;

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_io_rsp_ready;
    `IGNORE_WARNINGS_END

        wire[`NUM_CLUSTERS-1:0]                         per_cluster_busy;
        wire[`NUM_CLUSTERS-1:0]                         per_cluster_ebreak;

        genvar i;
        for (i = 0; i < `NUM_CLUSTERS; i=i+1) begin        
            Vortex_Cluster #(
                .CLUSTER_ID(i)
            ) Vortex_Cluster (
                .clk                (clk),
                .reset              (reset),

                .dram_req_write     (per_cluster_dram_req_write [i]),
                .dram_req_read      (per_cluster_dram_req_read  [i]),
                .dram_req_addr      (per_cluster_dram_req_addr  [i]),
                .dram_req_data      (per_cluster_dram_req_data  [i]),
                .dram_req_tag       (per_cluster_dram_req_tag   [i]), 
                .dram_req_ready     (l3_core_req_ready),

                .dram_rsp_valid     (per_cluster_dram_rsp_valid [i]),                
                .dram_rsp_data      (per_cluster_dram_rsp_data  [i]),
                .dram_rsp_tag       (per_cluster_dram_rsp_tag   [i]),
                .dram_rsp_ready     (per_cluster_dram_rsp_ready [i]),

                .llc_snp_req_valid  (snp_fwd_valid),
                .llc_snp_req_addr   (snp_fwd_addr),
                .llc_snp_req_ready  (per_cluster_snp_fwd_ready  [i]),

                .io_req_read        (per_cluster_io_req_read    [i]),
                .io_req_write       (per_cluster_io_req_write   [i]),
                .io_req_addr        (per_cluster_io_req_addr    [i]),
                .io_req_data        (per_cluster_io_req_data    [i]),
                .io_req_byteen      (per_cluster_io_req_byteen  [i]),
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

        assign io_req_read   = per_cluster_io_req_read[0];
        assign io_req_write  = per_cluster_io_req_write[0];
        assign io_req_addr   = per_cluster_io_req_addr[0];
        assign io_req_data   = per_cluster_io_req_data[0];
        assign io_req_byteen = per_cluster_io_req_byteen[0];
        assign io_req_tag    = per_cluster_io_req_tag[0];

        assign io_rsp_ready  = per_cluster_io_rsp_ready[0];

        assign busy   = (| per_cluster_busy);
        assign ebreak = (& per_cluster_ebreak);

        // L3 Cache ///////////////////////////////////////////////////////////

        wire[`L3NUM_REQUESTS-1:0]                           l3_core_req_valid;
        wire[`L3NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]        l3_core_req_read;
        wire[`L3NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]        l3_core_req_write;
        wire[`L3NUM_REQUESTS-1:0][31:0]                     l3_core_req_addr;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_LINE_WIDTH-1:0]   l3_core_req_data;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_TAG_WIDTH-1:0]    l3_core_req_tag;

        wire[`L3NUM_REQUESTS-1:0]                           l3_core_rsp_valid;        
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_LINE_WIDTH-1:0]   l3_core_rsp_data;
        wire[`L3NUM_REQUESTS-1:0][`L2DRAM_TAG_WIDTH-1:0]    l3_core_rsp_tag;
        wire[`L3NUM_REQUESTS-1:0]                           l3_core_rsp_ready;

        for (i = 0; i < `L3NUM_REQUESTS; i=i+1) begin
            // Core Request
            assign l3_core_req_valid [i] = (per_cluster_dram_req_read [i] | per_cluster_dram_req_write [i]);
            assign l3_core_req_read  [i] = per_cluster_dram_req_read  [i] ? `BYTE_EN_LW : `BYTE_EN_NO;
            assign l3_core_req_write [i] = per_cluster_dram_req_write [i] ? `BYTE_EN_LW : `BYTE_EN_NO;
            assign l3_core_req_addr  [i] = {per_cluster_dram_req_addr [i], {`LOG2UP(`L2BANK_LINE_SIZE){1'b0}}};
            assign l3_core_req_tag   [i] = per_cluster_dram_req_tag   [i];
            assign l3_core_req_data  [i] = per_cluster_dram_req_data  [i];

            // Core  Response
            assign l3_core_rsp_ready [i] = per_cluster_dram_rsp_ready[i];  

            // Cache Fill Response
            assign per_cluster_dram_rsp_valid [i] = l3_core_rsp_valid [i];
            assign per_cluster_dram_rsp_data  [i] = l3_core_rsp_data [i];
            assign per_cluster_dram_rsp_tag   [i] = l3_core_rsp_tag [i];
        end

        VX_cache #(
            .CACHE_SIZE             (`L3CACHE_SIZE),
            .BANK_LINE_SIZE         (`L3BANK_LINE_SIZE),
            .NUM_BANKS              (`L3NUM_BANKS),
            .WORD_SIZE              (`L3WORD_SIZE),
            .NUM_REQUESTS           (`L3NUM_REQUESTS),
            .STAGE_1_CYCLES         (`L3STAGE_1_CYCLES),
            .REQQ_SIZE              (`L3REQQ_SIZE),
            .MRVQ_SIZE              (`L3MRVQ_SIZE),
            .DFPQ_SIZE              (`L3DFPQ_SIZE),
            .SNRQ_SIZE              (`L3SNRQ_SIZE),
            .CWBQ_SIZE              (`L3CWBQ_SIZE),
            .DWBQ_SIZE              (`L3DWBQ_SIZE),
            .DFQQ_SIZE              (`L3DFQQ_SIZE),
            .LLVQ_SIZE              (`L3LLVQ_SIZE),
            .FFSQ_SIZE              (`L3FFSQ_SIZE),
            .PRFQ_SIZE              (`L3PRFQ_SIZE),
            .PRFQ_STRIDE            (`L3PRFQ_STRIDE),
            .FILL_INVALIDAOR_SIZE   (`L3FILL_INVALIDAOR_SIZE),
            .DRAM_ENABLE            (1),
            .WRITE_ENABLE           (1),
            .SNOOP_FORWARDING       (1),
            .CORE_TAG_WIDTH         (`L2DRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS       (0),
            .DRAM_TAG_WIDTH         (`L3DRAM_TAG_WIDTH)
        ) gpu_l3cache (
            .clk                (clk),
            .reset              (reset),

            // Core request    
            .core_req_valid     (l3_core_req_valid),
            .core_req_read      (l3_core_req_read),
            .core_req_write     (l3_core_req_write),
            .core_req_addr      (l3_core_req_addr),
            .core_req_data      (l3_core_req_data),
            .core_req_tag       (l3_core_req_tag),
            .core_req_ready     (l3_core_req_ready),

            // Core response
            .core_rsp_valid     (l3_core_rsp_valid),
            .core_rsp_data      (l3_core_rsp_data),
            .core_rsp_tag       (l3_core_rsp_tag),              
            .core_rsp_ready     (& l3_core_rsp_ready),

            // DRAM request
            .dram_req_write     (dram_req_write),
            .dram_req_read      (dram_req_read),
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
            .snp_req_valid      (llc_snp_req_valid),
            .snp_req_addr       (llc_snp_req_addr),
            .snp_req_ready      (llc_snp_req_ready),

            // Snoop forwarding
            .snp_fwd_valid      (snp_fwd_valid),
            .snp_fwd_addr       (snp_fwd_addr),
            .snp_fwd_ready      (& per_cluster_snp_fwd_ready)
        );
    end

endmodule