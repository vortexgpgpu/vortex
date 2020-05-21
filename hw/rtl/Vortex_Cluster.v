`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex_Cluster #(
    parameter CLUSTER_ID = 0
) ( 
    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // DRAM request
    output wire                         dram_req_read,
    output wire                         dram_req_write,    
    output wire[`L2DRAM_ADDR_WIDTH-1:0] dram_req_addr,
    output wire[`L2DRAM_LINE_WIDTH-1:0] dram_req_data,
    output wire[`L2DRAM_TAG_WIDTH-1:0]  dram_req_tag,
    input  wire                         dram_req_ready,

    // DRAM response  
    input  wire                         dram_rsp_valid,
    input  wire[`L2DRAM_LINE_WIDTH-1:0] dram_rsp_data,
    input  wire[`L2DRAM_TAG_WIDTH-1:0]  dram_rsp_tag,
    output wire                         dram_rsp_ready,

    // Snoop request
    input wire                          snp_req_valid,
    input wire[`L2DRAM_ADDR_WIDTH-1:0]  snp_req_addr,
    input wire[`L2SNP_TAG_WIDTH-1:0]    snp_req_tag,
    output wire                         snp_req_ready, 

    // Snoop response
    output wire                         snp_rsp_valid,
    output wire[`L2SNP_TAG_WIDTH-1:0]   snp_rsp_tag,
    input wire                          snp_rsp_ready,     

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
    wire[`NUM_CORES-1:0]                        per_core_D_dram_req_read;
    wire[`NUM_CORES-1:0]                        per_core_D_dram_req_write;    
    wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0] per_core_D_dram_req_addr;
    wire[`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_D_dram_req_data;
    wire[`NUM_CORES-1:0][`DDRAM_TAG_WIDTH-1:0]  per_core_D_dram_req_tag;
    wire[`NUM_CORES-1:0]                        per_core_D_dram_req_ready;

    wire[`NUM_CORES-1:0]                        per_core_D_dram_rsp_valid;            
    wire[`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_D_dram_rsp_data;
    wire[`NUM_CORES-1:0][`DDRAM_TAG_WIDTH-1:0]  per_core_D_dram_rsp_tag;
    wire[`NUM_CORES-1:0]                        per_core_D_dram_rsp_ready;

    wire[`NUM_CORES-1:0]                        per_core_I_dram_req_read;  
    wire[`NUM_CORES-1:0][`IDRAM_ADDR_WIDTH-1:0] per_core_I_dram_req_addr;
    wire[`NUM_CORES-1:0][`IDRAM_LINE_WIDTH-1:0] per_core_I_dram_req_data;
    wire[`NUM_CORES-1:0][`IDRAM_TAG_WIDTH-1:0]  per_core_I_dram_req_tag;
    wire[`NUM_CORES-1:0]                        per_core_I_dram_req_ready;
   
    wire[`NUM_CORES-1:0]                        per_core_I_dram_rsp_valid;        
    wire[`NUM_CORES-1:0][`IDRAM_LINE_WIDTH-1:0] per_core_I_dram_rsp_data;        
    wire[`NUM_CORES-1:0][`IDRAM_TAG_WIDTH-1:0]  per_core_I_dram_rsp_tag;
    wire[`NUM_CORES-1:0]                        per_core_I_dram_rsp_ready;

    wire[`NUM_CORES-1:0]                        per_core_snp_req_valid;
    wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0] per_core_snp_req_addr;
    wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]   per_core_snp_req_tag;
    wire[`NUM_CORES-1:0]                        per_core_snp_req_ready;
    
    wire[`NUM_CORES-1:0]                        per_core_snp_rsp_valid;
    wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]   per_core_snp_rsp_tag;
    wire[`NUM_CORES-1:0]                        per_core_snp_rsp_ready;

`IGNORE_WARNINGS_BEGIN
    wire[`NUM_CORES-1:0]                        per_core_io_req_read;
    wire[`NUM_CORES-1:0]                        per_core_io_req_write;
    wire[`NUM_CORES-1:0][31:0]                  per_core_io_req_addr;
    wire[`NUM_CORES-1:0][31:0]                  per_core_io_req_data;
    wire[`NUM_CORES-1:0][`BYTE_EN_BITS-1:0]     per_core_io_req_byteen;
    wire[`NUM_CORES-1:0][`CORE_REQ_TAG_WIDTH-1:0] per_core_io_req_tag;
    
    wire[`NUM_CORES-1:0]                        per_core_io_rsp_ready;
`IGNORE_WARNINGS_END

    wire[`NUM_CORES-1:0]                        per_core_busy;
    wire[`NUM_CORES-1:0]                        per_core_ebreak;

    genvar i;
    for (i = 0; i < `NUM_CORES; i++) begin    
        Vortex #(
            .CORE_ID(i + (CLUSTER_ID * `NUM_CORES))
        ) vortex_core (
            .clk                (clk),
            .reset              (reset),
            .D_dram_req_read    (per_core_D_dram_req_read   [i]),
            .D_dram_req_write   (per_core_D_dram_req_write  [i]),                
            .D_dram_req_addr    (per_core_D_dram_req_addr   [i]),
            .D_dram_req_data    (per_core_D_dram_req_data   [i]),
            .D_dram_req_tag     (per_core_D_dram_req_tag    [i]),
            .D_dram_req_ready   (per_core_D_dram_req_ready  [i]),         
            .D_dram_rsp_valid   (per_core_D_dram_rsp_valid  [i]),                
            .D_dram_rsp_data    (per_core_D_dram_rsp_data   [i]),
            .D_dram_rsp_tag     (per_core_D_dram_rsp_tag    [i]),
            .D_dram_rsp_ready   (per_core_D_dram_rsp_ready  [i]),

            .I_dram_req_read    (per_core_I_dram_req_read   [i]),
            `UNUSED_PIN (I_dram_req_write),                            
            .I_dram_req_addr    (per_core_I_dram_req_addr   [i]),                
            .I_dram_req_data    (per_core_I_dram_req_data   [i]),
            .I_dram_req_tag     (per_core_I_dram_req_tag    [i]),                
            .I_dram_req_ready   (per_core_I_dram_req_ready  [i]),          
            .I_dram_rsp_valid   (per_core_I_dram_rsp_valid  [i]),
            .I_dram_rsp_tag     (per_core_I_dram_rsp_tag    [i]),
            .I_dram_rsp_data    (per_core_I_dram_rsp_data   [i]),
            .I_dram_rsp_ready   (per_core_I_dram_rsp_ready  [i]),   

            .snp_req_valid      (per_core_snp_req_valid     [i]),
            .snp_req_addr       (per_core_snp_req_addr      [i]),
            .snp_req_tag        (per_core_snp_req_tag       [i]),
            .snp_req_ready      (per_core_snp_req_ready     [i]),

            .snp_rsp_valid      (per_core_snp_rsp_valid     [i]),
            .snp_rsp_tag        (per_core_snp_rsp_tag       [i]),
            .snp_rsp_ready      (per_core_snp_rsp_ready     [i]),

            .io_req_read        (per_core_io_req_read       [i]),
            .io_req_write       (per_core_io_req_write      [i]),
            .io_req_addr        (per_core_io_req_addr       [i]),
            .io_req_data        (per_core_io_req_data       [i]),
            .io_req_byteen      (per_core_io_req_byteen     [i]),
            .io_req_tag         (per_core_io_req_tag        [i]),
            .io_req_ready       (io_req_ready),

            .io_rsp_valid       (io_rsp_valid),            
            .io_rsp_data        (io_rsp_data),
            .io_rsp_tag         (io_rsp_tag),
            .io_rsp_ready       (per_core_io_rsp_ready      [i]),

            .busy               (per_core_busy              [i]),
            .ebreak             (per_core_ebreak            [i])
        );
    end   

    assign io_req_read   = per_core_io_req_read[0];
    assign io_req_write  = per_core_io_req_write[0];
    assign io_req_addr   = per_core_io_req_addr[0];
    assign io_req_data   = per_core_io_req_data[0];
    assign io_req_byteen = per_core_io_req_byteen[0];
    assign io_req_tag    = per_core_io_req_tag[0];

    assign io_rsp_ready  = per_core_io_rsp_ready[0];
    
    assign busy = (| per_core_busy);
    assign ebreak = (& per_core_ebreak);

    if (`L2_ENABLE) begin

        // L2 Cache ///////////////////////////////////////////////////////////

        wire[`L2NUM_REQUESTS-1:0]                           l2_core_req_valid;
        wire[`L2NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]        l2_core_req_write;
        wire[`L2NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]        l2_core_req_read;
        wire[`L2NUM_REQUESTS-1:0][31:0]                     l2_core_req_addr;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]     l2_core_req_tag;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0]    l2_core_req_data;
        wire                                                l2_core_req_ready;

        wire[`L2NUM_REQUESTS-1:0]                           l2_core_rsp_valid;        
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0]    l2_core_rsp_data;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]     l2_core_rsp_tag;
        wire                                                l2_core_rsp_ready;

        wire[`NUM_CORES-1:0]                                l2_snp_fwdout_valid;
        wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0]         l2_snp_fwdout_addr;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]           l2_snp_fwdout_tag;
        wire[`NUM_CORES-1:0]                                l2_snp_fwdout_ready;    

        wire[`NUM_CORES-1:0]                                l2_snp_fwdin_valid;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]           l2_snp_fwdin_tag;
        wire[`NUM_CORES-1:0]                                l2_snp_fwdin_ready;

        for (i = 0; i < `L2NUM_REQUESTS; i = i + 2) begin
            assign l2_core_req_valid [i]   = (per_core_D_dram_req_read[(i/2)] | per_core_D_dram_req_write[(i/2)]);
            assign l2_core_req_valid [i+1] = per_core_I_dram_req_read[(i/2)];

            assign l2_core_req_read  [i]   = per_core_D_dram_req_read[(i/2)] ? `BYTE_EN_LW : `BYTE_EN_NO;
            assign l2_core_req_read  [i+1] = per_core_I_dram_req_read[(i/2)] ? `BYTE_EN_LW : `BYTE_EN_NO;
            
            assign l2_core_req_write [i]   = per_core_D_dram_req_write[(i/2)] ? `BYTE_EN_LW : `BYTE_EN_NO;
            assign l2_core_req_write [i+1] = `BYTE_EN_NO;

            assign l2_core_req_addr  [i]   = {per_core_D_dram_req_addr[(i/2)], {`LOG2UP(`DBANK_LINE_SIZE){1'b0}}};
            assign l2_core_req_addr  [i+1] = {per_core_I_dram_req_addr[(i/2)], {`LOG2UP(`IBANK_LINE_SIZE){1'b0}}};

            assign l2_core_req_data  [i]   = per_core_D_dram_req_data[(i/2)];
            assign l2_core_req_data  [i+1] = per_core_I_dram_req_data[(i/2)];

            assign l2_core_req_tag   [i]   = per_core_D_dram_req_tag[(i/2)];
            assign l2_core_req_tag   [i+1] = per_core_I_dram_req_tag[(i/2)];

            assign per_core_D_dram_req_ready [(i/2)] = l2_core_req_ready;
            assign per_core_I_dram_req_ready [(i/2)] = l2_core_req_ready;

            assign per_core_D_dram_rsp_valid [(i/2)] = l2_core_rsp_valid[i] && l2_core_rsp_ready;
            assign per_core_I_dram_rsp_valid [(i/2)] = l2_core_rsp_valid[i+1] && l2_core_rsp_ready;

            assign per_core_D_dram_rsp_data  [(i/2)] = l2_core_rsp_data[i];
            assign per_core_I_dram_rsp_data  [(i/2)] = l2_core_rsp_data[i+1];

            assign per_core_D_dram_rsp_tag   [(i/2)] = l2_core_rsp_tag[i];
            assign per_core_I_dram_rsp_tag   [(i/2)] = l2_core_rsp_tag[i+1];  

            assign per_core_snp_req_valid [(i/2)] = l2_snp_fwdout_valid [(i/2)];
            assign per_core_snp_req_addr  [(i/2)] = l2_snp_fwdout_addr [(i/2)];    
            assign per_core_snp_req_tag   [(i/2)] = l2_snp_fwdout_tag [(i/2)];
            assign l2_snp_fwdout_ready    [(i/2)] = per_core_snp_req_ready[(i/2)];

            assign l2_snp_fwdin_valid     [(i/2)] = per_core_snp_rsp_valid [(i/2)];
            assign l2_snp_fwdin_tag       [(i/2)] = per_core_snp_rsp_tag [(i/2)];
            assign per_core_snp_rsp_ready [(i/2)] = l2_snp_fwdin_ready [(i/2)];
        end

        assign l2_core_rsp_ready = (& per_core_D_dram_rsp_ready) && (& per_core_I_dram_rsp_ready);

        VX_cache #(
            .CACHE_ID               (`L2CACHE_ID),
            .CACHE_SIZE             (`L2CACHE_SIZE),
            .BANK_LINE_SIZE         (`L2BANK_LINE_SIZE),
            .NUM_BANKS              (`L2NUM_BANKS),
            .WORD_SIZE              (`L2WORD_SIZE),
            .NUM_REQUESTS           (`L2NUM_REQUESTS),
            .STAGE_1_CYCLES         (`L2STAGE_1_CYCLES),
            .REQQ_SIZE              (`L2REQQ_SIZE),
            .MRVQ_SIZE              (`L2MRVQ_SIZE),
            .DFPQ_SIZE              (`L2DFPQ_SIZE),
            .SNRQ_SIZE              (`L2SNRQ_SIZE),
            .CWBQ_SIZE              (`L2CWBQ_SIZE),
            .DWBQ_SIZE              (`L2DWBQ_SIZE),
            .DFQQ_SIZE              (`L2DFQQ_SIZE),
            .LLVQ_SIZE              (`L2LLVQ_SIZE),
            .PRFQ_SIZE              (`L2PRFQ_SIZE),
            .PRFQ_STRIDE            (`L2PRFQ_STRIDE),
            .FILL_INVALIDAOR_SIZE   (`L2FILL_INVALIDAOR_SIZE),            
            .DRAM_ENABLE            (1),
            .WRITE_ENABLE           (1),
            .SNOOP_FORWARDING       (1),
            .CORE_TAG_WIDTH         (`DDRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS       (0),
            .DRAM_TAG_WIDTH         (`L2DRAM_TAG_WIDTH),
            .NUM_SNP_REQUESTS       (`NUM_CORES),
            .SNP_REQ_TAG_WIDTH      (`L2SNP_TAG_WIDTH),
            .SNP_FWD_TAG_WIDTH      (`DSNP_TAG_WIDTH)
        ) gpu_l2cache (
            .clk                (clk),
            .reset              (reset),

            // Core request
            .core_req_valid     (l2_core_req_valid),
            .core_req_read      (l2_core_req_read),
            .core_req_write     (l2_core_req_write),
            .core_req_addr      (l2_core_req_addr),
            .core_req_data      (l2_core_req_data),  
            .core_req_tag       (l2_core_req_tag),  
            .core_req_ready     (l2_core_req_ready),

            // Core response
            .core_rsp_valid     (l2_core_rsp_valid),
            .core_rsp_data      (l2_core_rsp_data),
            .core_rsp_tag       (l2_core_rsp_tag),
            .core_rsp_ready     (l2_core_rsp_ready),

            // DRAM request
            .dram_req_read      (dram_req_read),
            .dram_req_write     (dram_req_write),        
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),
            
            // DRAM response
            .dram_rsp_valid     (dram_rsp_valid),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_data      (dram_rsp_data),
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
            .snp_fwdout_valid   (l2_snp_fwdout_valid),
            .snp_fwdout_addr    (l2_snp_fwdout_addr),
            .snp_fwdout_tag     (l2_snp_fwdout_tag),
            .snp_fwdout_ready   (l2_snp_fwdout_ready),

            // Snoop forwarding in
            .snp_fwdin_valid    (l2_snp_fwdin_valid),
            .snp_fwdin_tag      (l2_snp_fwdin_tag),
            .snp_fwdin_ready    (l2_snp_fwdin_ready)      
        );

    end else begin
    
        wire[`L2NUM_REQUESTS-1:0]                        arb_core_req_read;
        wire[`L2NUM_REQUESTS-1:0]                        arb_core_req_write;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_ADDR_WIDTH-1:0] arb_core_req_addr;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]  arb_core_req_tag;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0] arb_core_req_data;
        wire[`L2NUM_REQUESTS-1:0]                        arb_core_req_ready;

        wire[`L2NUM_REQUESTS-1:0]                        arb_core_rsp_valid;        
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0] arb_core_rsp_data;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]  arb_core_rsp_tag;
        wire[`L2NUM_REQUESTS-1:0]                        arb_core_rsp_ready;

        wire[`NUM_CORES-1:0]                             arb_snp_fwdout_valid;
        wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0]      arb_snp_fwdout_addr;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]        arb_snp_fwdout_tag;
        wire[`NUM_CORES-1:0]                             arb_snp_fwdout_ready;    

        wire[`NUM_CORES-1:0]                             arb_snp_fwdin_valid;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]        arb_snp_fwdin_tag;
        wire[`NUM_CORES-1:0]                             arb_snp_fwdin_ready;

        for (i = 0; i < `L2NUM_REQUESTS; i = i + 2) begin            
            assign arb_core_req_read  [i]   = per_core_D_dram_req_read[(i/2)];
            assign arb_core_req_read  [i+1] = per_core_I_dram_req_read[(i/2)];

            assign arb_core_req_write [i]   = per_core_D_dram_req_write[(i/2)];
            assign arb_core_req_write [i+1] = 0;

            assign arb_core_req_addr  [i]   = per_core_D_dram_req_addr[(i/2)];
            assign arb_core_req_addr  [i+1] = per_core_I_dram_req_addr[(i/2)];

            assign arb_core_req_data  [i]   = per_core_D_dram_req_data[(i/2)];
            assign arb_core_req_data  [i+1] = per_core_I_dram_req_data[(i/2)];

            assign arb_core_req_tag   [i]   = per_core_D_dram_req_tag[(i/2)];
            assign arb_core_req_tag   [i+1] = per_core_I_dram_req_tag[(i/2)];

            assign per_core_D_dram_req_ready [(i/2)] = arb_core_req_ready[i];
            assign per_core_I_dram_req_ready [(i/2)] = arb_core_req_ready[i+1];

            assign per_core_D_dram_rsp_valid [(i/2)] = arb_core_rsp_valid[i];
            assign per_core_I_dram_rsp_valid [(i/2)] = arb_core_rsp_valid[i+1];

            assign per_core_D_dram_rsp_data  [(i/2)] = arb_core_rsp_data[i];
            assign per_core_I_dram_rsp_data  [(i/2)] = arb_core_rsp_data[i+1];

            assign per_core_D_dram_rsp_tag   [(i/2)] = arb_core_rsp_tag[i];
            assign per_core_I_dram_rsp_tag   [(i/2)] = arb_core_rsp_tag[i+1];

            assign arb_core_rsp_ready [i]   = per_core_D_dram_rsp_ready[(i/2)];
            assign arb_core_rsp_ready [i+1] = per_core_I_dram_rsp_ready[(i/2)];  

            assign per_core_snp_req_valid [(i/2)] = arb_snp_fwdout_valid [(i/2)];
            assign per_core_snp_req_addr  [(i/2)] = arb_snp_fwdout_addr [(i/2)];    
            assign per_core_snp_req_tag   [(i/2)] = arb_snp_fwdout_tag [(i/2)];
            assign arb_snp_fwdout_ready   [(i/2)] = per_core_snp_req_ready[(i/2)];

            assign arb_snp_fwdin_valid    [(i/2)] = per_core_snp_rsp_valid [(i/2)];
            assign arb_snp_fwdin_tag      [(i/2)] = per_core_snp_rsp_tag [(i/2)];
            assign per_core_snp_rsp_ready [(i/2)] = arb_snp_fwdin_ready [(i/2)];
        end

        VX_snp_forwarder #(
            .BANK_LINE_SIZE(`L2BANK_LINE_SIZE), 
            .NUM_REQUESTS(`NUM_CORES), 
            .SNRQ_SIZE(`L2SNRQ_SIZE),
            .SNP_REQ_TAG_WIDTH(`L2SNP_TAG_WIDTH),
            .SNP_FWD_TAG_WIDTH(`DSNP_TAG_WIDTH)
        ) snp_forwarder (
            .clk                (clk),
            .reset              (reset),

            .snp_req_valid      (snp_req_valid),
            .snp_req_addr       (snp_req_addr),
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            .snp_rsp_valid      (snp_rsp_valid),       
            `UNUSED_PIN (snp_rsp_addr),
            .snp_rsp_tag        (snp_rsp_tag),
            .snp_rsp_ready      (snp_rsp_ready),   

            .snp_fwdout_valid   (arb_snp_fwdout_valid),
            .snp_fwdout_addr    (arb_snp_fwdout_addr),
            .snp_fwdout_tag     (arb_snp_fwdout_tag),
            .snp_fwdout_ready   (arb_snp_fwdout_ready),

            .snp_fwdin_valid    (arb_snp_fwdin_valid),
            .snp_fwdin_tag      (arb_snp_fwdin_tag),
            .snp_fwdin_ready    (arb_snp_fwdin_ready)      
        );

        VX_dram_arb #(
            .BANK_LINE_SIZE (`L2BANK_LINE_SIZE),
            .NUM_REQUESTS   (`L2NUM_REQUESTS),
            .CORE_TAG_WIDTH (`DDRAM_TAG_WIDTH),
            .DRAM_TAG_WIDTH (`L2DRAM_TAG_WIDTH)
        ) dram_arb (
            .clk                (clk),
            .reset              (reset),

            // Core request
            .core_req_read      (arb_core_req_read),
            .core_req_write     (arb_core_req_write),
            .core_req_addr      (arb_core_req_addr),
            .core_req_data      (arb_core_req_data),  
            .core_req_tag       (arb_core_req_tag),  
            .core_req_ready     (arb_core_req_ready),

            // Core response
            .core_rsp_valid     (arb_core_rsp_valid),
            .core_rsp_data      (arb_core_rsp_data),
            .core_rsp_tag       (arb_core_rsp_tag),
            .core_rsp_ready     (arb_core_rsp_ready),

            // DRAM request
            .dram_req_read      (dram_req_read),
            .dram_req_write     (dram_req_write),        
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),
            
            // DRAM response
            .dram_rsp_valid     (dram_rsp_valid),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_ready     (dram_rsp_ready)
        );

    end

endmodule