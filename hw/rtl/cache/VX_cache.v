`include "VX_cache_config.vh"

module VX_cache #(
    parameter CACHE_ID                      = 0,

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 4096, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks
    parameter NUM_BANKS                     = 4, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle
    parameter NUM_REQUESTS                  = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 8, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 8, 
    // DRAM Response Queue Size
    parameter DRFQ_SIZE                     = 8, 
    // Snoop Req Queue Size
    parameter SNRQ_SIZE                     = 8, 

    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 8, 
    // DRAM Request Queue Size
    parameter DREQ_SIZE                     = 8, 
    // Snoop Response Size
    parameter SNPQ_SIZE                     = 8,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

    // Enable dram update
    parameter DRAM_ENABLE                   = 1,

    // Enable cache flush
    parameter FLUSH_ENABLE                  = 1,

    // Enable snoop forwarding
    parameter SNOOP_FORWARDING              = 1,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 4,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // dram request tag size
    parameter DRAM_TAG_WIDTH                = 28,

    // Number of snoop forwarding requests
    parameter NUM_SNP_REQUESTS              = (SNOOP_FORWARDING ? 4 : 1), 

    // Snooping request tag width
    parameter SNP_REQ_TAG_WIDTH             = (SNOOP_FORWARDING ? 4 : 1),

    // Snooping forward tag width
    parameter SNP_FWD_TAG_WIDTH             = (SNOOP_FORWARDING ? 4 : 1)
 ) (
    `SCOPE_IO_VX_cache
    
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQUESTS-1:0]                           core_req_valid,
    input wire [`CORE_REQ_TAG_COUNT-1:0]                    core_req_rw,
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]            core_req_byteen,
    input wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]     core_req_addr,
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]          core_req_data,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire                                             core_req_ready,

    // Core response
    output wire [NUM_REQUESTS-1:0]                          core_rsp_valid,    
    output wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]         core_rsp_data,
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire                                             core_rsp_ready,   
    
    // DRAM request
    output wire                             dram_req_valid,
    output wire                             dram_req_rw,    
    output wire [BANK_LINE_SIZE-1:0]        dram_req_byteen,    
    output wire [`DRAM_ADDR_WIDTH-1:0]      dram_req_addr,
    output wire [`BANK_LINE_WIDTH-1:0]      dram_req_data,
    output wire [DRAM_TAG_WIDTH-1:0]        dram_req_tag,
    input  wire                             dram_req_ready,
    
    // DRAM response
    input  wire                             dram_rsp_valid,    
    input  wire [`BANK_LINE_WIDTH-1:0]      dram_rsp_data,
    input  wire [DRAM_TAG_WIDTH-1:0]        dram_rsp_tag,
    output wire                             dram_rsp_ready,    

    // Snoop request
    input wire                              snp_req_valid,
    input wire [`DRAM_ADDR_WIDTH-1:0]       snp_req_addr,
    input wire                              snp_req_invalidate,
    input wire [SNP_REQ_TAG_WIDTH-1:0]      snp_req_tag,
    output wire                             snp_req_ready,

    // Snoop response
    output wire                             snp_rsp_valid,    
    output wire [SNP_REQ_TAG_WIDTH-1:0]     snp_rsp_tag,
    input  wire                             snp_rsp_ready,

    // Snoop Forwarding out
    output wire [NUM_SNP_REQUESTS-1:0]      snp_fwdout_valid,
    output wire [NUM_SNP_REQUESTS-1:0][`DRAM_ADDR_WIDTH-1:0] snp_fwdout_addr,
    output wire [NUM_SNP_REQUESTS-1:0]      snp_fwdout_invalidate,
    output wire [NUM_SNP_REQUESTS-1:0][SNP_FWD_TAG_WIDTH-1:0] snp_fwdout_tag,
`IGNORE_WARNINGS_BEGIN
    input wire [NUM_SNP_REQUESTS-1:0]       snp_fwdout_ready,

    // Snoop forwarding in
    input wire [NUM_SNP_REQUESTS-1:0]       snp_fwdin_valid,    
    input wire [NUM_SNP_REQUESTS-1:0][SNP_FWD_TAG_WIDTH-1:0] snp_fwdin_tag,
`IGNORE_WARNINGS_END
    output wire [NUM_SNP_REQUESTS-1:0]      snp_fwdin_ready,

    output wire [NUM_BANKS-1:0] miss_vec
);

    wire [NUM_BANKS-1:0][NUM_REQUESTS-1:0]      per_bank_valid;

    wire [NUM_BANKS-1:0]                        per_bank_core_req_ready;
    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]        per_bank_core_rsp_tid; 
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]       per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]    per_bank_core_rsp_tag;    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_dram_req_valid;    
    wire [NUM_BANKS-1:0]                        per_bank_dram_req_rw;
    wire [NUM_BANKS-1:0][BANK_LINE_SIZE-1:0]    per_bank_dram_req_byteen;    
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0]  per_bank_dram_req_addr;
    wire [NUM_BANKS-1:0][`BANK_LINE_WIDTH-1:0]  per_bank_dram_req_data;
    wire [NUM_BANKS-1:0]                        per_bank_dram_req_ready;

    wire [NUM_BANKS-1:0]                        per_bank_dram_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_snp_req_ready;

    wire [NUM_BANKS-1:0]                        per_bank_snp_rsp_valid;
    wire [NUM_BANKS-1:0][SNP_REQ_TAG_WIDTH-1:0] per_bank_snp_rsp_tag;
    wire [NUM_BANKS-1:0]                        per_bank_snp_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_miss; 
    assign miss_vec = per_bank_miss; 

   

    wire                         snp_req_valid_qual;    
    wire [`DRAM_ADDR_WIDTH-1:0]  snp_req_addr_qual;
    wire                         snp_req_invalidate_qual;    
    wire [SNP_REQ_TAG_WIDTH-1:0] snp_req_tag_qual;
    wire                         snp_req_ready_qual;

    if (SNOOP_FORWARDING) begin
        VX_snp_forwarder #(
            .CACHE_ID          (CACHE_ID),
            .BANK_LINE_SIZE    (BANK_LINE_SIZE), 
            .NUM_REQUESTS      (NUM_SNP_REQUESTS), 
            .SNRQ_SIZE         (SNRQ_SIZE),
            .SNP_REQ_TAG_WIDTH (SNP_REQ_TAG_WIDTH)
        ) snp_forwarder (
            .clk                (clk),
            .reset              (reset),

            .snp_req_valid      (snp_req_valid),
            .snp_req_addr       (snp_req_addr),
            .snp_req_invalidate (snp_req_invalidate),
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            .snp_rsp_valid      (snp_req_valid_qual),
            .snp_rsp_addr       (snp_req_addr_qual),
            .snp_rsp_invalidate (snp_req_invalidate_qual),
            .snp_rsp_tag        (snp_req_tag_qual),
            .snp_rsp_ready      (snp_req_ready_qual),   

            .snp_fwdout_valid   (snp_fwdout_valid),
            .snp_fwdout_addr    (snp_fwdout_addr),
            .snp_fwdout_invalidate(snp_fwdout_invalidate),
            .snp_fwdout_tag     (snp_fwdout_tag),
            .snp_fwdout_ready   (snp_fwdout_ready),

            .snp_fwdin_valid    (snp_fwdin_valid),
            .snp_fwdin_tag      (snp_fwdin_tag),
            .snp_fwdin_ready    (snp_fwdin_ready)      
        );
    end else begin
        assign snp_fwdout_valid = 0;
        assign snp_fwdout_addr  = 0;
        assign snp_fwdout_invalidate = 0;
        assign snp_fwdout_tag   = 0;

        assign snp_fwdin_ready  = 0;

        assign snp_req_valid_qual      = snp_req_valid;
        assign snp_req_addr_qual       = snp_req_addr;
        assign snp_req_invalidate_qual = snp_req_invalidate;
        assign snp_req_tag_qual        = snp_req_tag;
        assign snp_req_ready           = snp_req_ready_qual;
    end    

    if (NUM_BANKS == 1) begin
        assign snp_req_ready_qual = per_bank_snp_req_ready;
    end else begin
        assign snp_req_ready_qual = per_bank_snp_req_ready[`DRAM_ADDR_BANK(snp_req_addr_qual)];
    end    

    VX_cache_core_req_bank_sel #(
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .NUM_REQUESTS   (NUM_REQUESTS)
    ) cache_core_req_bank_sel (
        .core_req_valid  (core_req_valid),
        .per_bank_ready  (per_bank_core_req_ready),
        .core_req_addr   (core_req_addr),
        .per_bank_valid  (per_bank_valid),
        .core_req_ready  (core_req_ready)
    );

    assign dram_req_tag   = dram_req_addr;
    assign dram_rsp_ready = (& per_bank_dram_rsp_ready);
    
    for (genvar i = 0; i < NUM_BANKS; i++) begin
        wire [NUM_REQUESTS-1:0]                             curr_bank_core_req_valid;            
        wire [`CORE_REQ_TAG_COUNT-1:0]                      curr_bank_core_req_rw;  
        wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              curr_bank_core_req_byteen;
        wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       curr_bank_core_req_addr;
        wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  curr_bank_core_req_tag;
        wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            curr_bank_core_req_data;
        wire                                                curr_bank_core_req_ready;

        wire                            curr_bank_core_rsp_valid;
        wire [`REQS_BITS-1:0]           curr_bank_core_rsp_tid;
        wire [`WORD_WIDTH-1:0]          curr_bank_core_rsp_data;
        wire [CORE_TAG_WIDTH-1:0]       curr_bank_core_rsp_tag;
        wire                            curr_bank_core_rsp_ready;

        wire                            curr_bank_dram_req_valid;
        wire                            curr_bank_dram_req_rw;
        wire [BANK_LINE_SIZE-1:0]       curr_bank_dram_req_byteen;
        wire [`LINE_ADDR_WIDTH-1:0]     curr_bank_dram_req_addr;
        wire[`BANK_LINE_WIDTH-1:0]      curr_bank_dram_req_data;
        wire                            curr_bank_dram_req_ready;

        wire                            curr_bank_dram_rsp_valid;            
        wire [`BANK_LINE_WIDTH-1:0]     curr_bank_dram_rsp_data;
        wire [`LINE_ADDR_WIDTH-1:0]     curr_bank_dram_rsp_addr;
        wire                            curr_bank_dram_rsp_ready;

        wire                            curr_bank_snp_req_valid;
        wire [`LINE_ADDR_WIDTH-1:0]     curr_bank_snp_req_addr;
        wire                            curr_bank_snp_req_invalidate;
        wire [SNP_REQ_TAG_WIDTH-1:0]    curr_bank_snp_req_tag;
        wire                            curr_bank_snp_req_ready;    

        wire                            curr_bank_snp_rsp_valid;
        wire [SNP_REQ_TAG_WIDTH-1:0]    curr_bank_snp_rsp_tag;
        wire                            curr_bank_snp_rsp_ready;                    

        wire                            curr_bank_miss; 

        // Core Req
        assign curr_bank_core_req_valid   = (per_bank_valid[i] & {NUM_REQUESTS{core_req_ready}});
        assign curr_bank_core_req_addr    = core_req_addr;
        assign curr_bank_core_req_rw      = core_req_rw;
        assign curr_bank_core_req_byteen  = core_req_byteen;
        assign curr_bank_core_req_data    = core_req_data;
        assign curr_bank_core_req_tag     = core_req_tag;
        assign per_bank_core_req_ready[i] = curr_bank_core_req_ready;

        // Core WB
        assign curr_bank_core_rsp_ready    = per_bank_core_rsp_ready[i];
        assign per_bank_core_rsp_valid [i] = curr_bank_core_rsp_valid;
        assign per_bank_core_rsp_tid   [i] = curr_bank_core_rsp_tid;
        assign per_bank_core_rsp_tag   [i] = curr_bank_core_rsp_tag;
        assign per_bank_core_rsp_data  [i] = curr_bank_core_rsp_data;

        // DRAM request            
        assign per_bank_dram_req_valid[i] = curr_bank_dram_req_valid;          
        assign per_bank_dram_req_rw[i] = curr_bank_dram_req_rw;
        assign per_bank_dram_req_byteen[i] = curr_bank_dram_req_byteen;
        if (NUM_BANKS == 1) begin  
            assign per_bank_dram_req_addr[i] = curr_bank_dram_req_addr;
        end else begin
            assign per_bank_dram_req_addr[i] = `LINE_TO_DRAM_ADDR(curr_bank_dram_req_addr, i);            
        end
        assign per_bank_dram_req_data[i] = curr_bank_dram_req_data;
        assign curr_bank_dram_req_ready = per_bank_dram_req_ready[i];

        // DRAM response
        if (NUM_BANKS == 1) begin
            assign curr_bank_dram_rsp_valid = dram_rsp_valid;
            assign curr_bank_dram_rsp_addr  = dram_rsp_tag;
        end else begin
            assign curr_bank_dram_rsp_valid = dram_rsp_valid && (`DRAM_ADDR_BANK(dram_rsp_tag) == i);
            assign curr_bank_dram_rsp_addr  = `DRAM_TO_LINE_ADDR(dram_rsp_tag);    
        end
        assign curr_bank_dram_rsp_data    = dram_rsp_data;
        assign per_bank_dram_rsp_ready[i] = curr_bank_dram_rsp_ready;

        // Snoop request
        if (NUM_BANKS == 1) begin
            assign curr_bank_snp_req_valid = snp_req_valid_qual;
            assign curr_bank_snp_req_addr  = snp_req_addr_qual;
        end else begin
            assign curr_bank_snp_req_valid = snp_req_valid_qual && (`DRAM_ADDR_BANK(snp_req_addr_qual) == i);
            assign curr_bank_snp_req_addr  = `DRAM_TO_LINE_ADDR(snp_req_addr_qual);
        end
        assign curr_bank_snp_req_invalidate = snp_req_invalidate_qual;
        assign curr_bank_snp_req_tag        = snp_req_tag_qual;
        assign per_bank_snp_req_ready[i]    = curr_bank_snp_req_ready;

        // Snoop response            
        assign per_bank_snp_rsp_valid[i] = curr_bank_snp_rsp_valid;
        assign per_bank_snp_rsp_tag[i]   = curr_bank_snp_rsp_tag;
        assign curr_bank_snp_rsp_ready   = per_bank_snp_rsp_ready[i];

        //Misses
        assign per_bank_miss[i] = curr_bank_miss; 
        
        VX_bank #(                
            .BANK_ID            (i),
            .CACHE_ID           (CACHE_ID),
            .CACHE_SIZE         (CACHE_SIZE),
            .BANK_LINE_SIZE     (BANK_LINE_SIZE),
            .NUM_BANKS          (NUM_BANKS),
            .WORD_SIZE          (WORD_SIZE),
            .NUM_REQUESTS       (NUM_REQUESTS),
            .CREQ_SIZE          (CREQ_SIZE),
            .MRVQ_SIZE          (MRVQ_SIZE),
            .DRFQ_SIZE          (DRFQ_SIZE),
            .SNRQ_SIZE          (SNRQ_SIZE),
            .CWBQ_SIZE          (CWBQ_SIZE),
            .DREQ_SIZE          (DREQ_SIZE),
            .SNPQ_SIZE          (SNPQ_SIZE),
            .DRAM_ENABLE        (DRAM_ENABLE),
            .FLUSH_ENABLE       (FLUSH_ENABLE),
            .WRITE_ENABLE       (WRITE_ENABLE),
            .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),                
            .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
            .SNP_REQ_TAG_WIDTH  (SNP_REQ_TAG_WIDTH)
        ) bank (
            `SCOPE_BIND_VX_cache_bank(i)
            
            .clk                (clk),
            .reset              (reset),                
            // Core request
            .core_req_valid     (curr_bank_core_req_valid),                  
            .core_req_rw        (curr_bank_core_req_rw),
            .core_req_byteen    (curr_bank_core_req_byteen),              
            .core_req_addr      (curr_bank_core_req_addr),
            .core_req_data      (curr_bank_core_req_data),
            .core_req_tag       (curr_bank_core_req_tag),
            .core_req_ready     (curr_bank_core_req_ready),

            // Core response                
            .core_rsp_valid     (curr_bank_core_rsp_valid),
            .core_rsp_tid       (curr_bank_core_rsp_tid),
            .core_rsp_data      (curr_bank_core_rsp_data),
            .core_rsp_tag       (curr_bank_core_rsp_tag),
            .core_rsp_ready     (curr_bank_core_rsp_ready),

            // DRAM request
            .dram_req_valid     (curr_bank_dram_req_valid),
            .dram_req_rw        (curr_bank_dram_req_rw),
            .dram_req_byteen    (curr_bank_dram_req_byteen),
            .dram_req_addr      (curr_bank_dram_req_addr),
            .dram_req_data      (curr_bank_dram_req_data),   
            .dram_req_ready     (curr_bank_dram_req_ready),

            // DRAM response
            .dram_rsp_valid     (curr_bank_dram_rsp_valid),                
            .dram_rsp_data      (curr_bank_dram_rsp_data),
            .dram_rsp_addr      (curr_bank_dram_rsp_addr),
            .dram_rsp_ready     (curr_bank_dram_rsp_ready),

            // Snoop request
            .snp_req_valid      (curr_bank_snp_req_valid),
            .snp_req_addr       (curr_bank_snp_req_addr),
            .snp_req_invalidate (curr_bank_snp_req_invalidate),
            .snp_req_tag        (curr_bank_snp_req_tag),
            .snp_req_ready      (curr_bank_snp_req_ready),

            // Snoop response
            .snp_rsp_valid      (curr_bank_snp_rsp_valid),
            .snp_rsp_tag        (curr_bank_snp_rsp_tag),
            .snp_rsp_ready      (curr_bank_snp_rsp_ready),

            //Misses
            .misses             (curr_bank_miss)
        );
    end   

    VX_cache_core_rsp_merge #(
        .NUM_BANKS          (NUM_BANKS),
        .WORD_SIZE          (WORD_SIZE),
        .NUM_REQUESTS       (NUM_REQUESTS),
        .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),        
        .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS)
    ) cache_core_rsp_merge (
        .clk                     (clk),
        .reset                   (reset),                    
        .per_bank_core_rsp_valid (per_bank_core_rsp_valid),   
        .per_bank_core_rsp_tag   (per_bank_core_rsp_tag),
        .per_bank_core_rsp_tid   (per_bank_core_rsp_tid),   
        .per_bank_core_rsp_data  (per_bank_core_rsp_data),
        .per_bank_core_rsp_ready (per_bank_core_rsp_ready),
        .core_rsp_valid          (core_rsp_valid),      
        .core_rsp_tag            (core_rsp_tag),
        .core_rsp_data           (core_rsp_data),  
        .core_rsp_ready          (core_rsp_ready)
    ); 

    if (DRAM_ENABLE) begin
        VX_cache_dram_req_arb #(
            .BANK_LINE_SIZE (BANK_LINE_SIZE),
            .NUM_BANKS      (NUM_BANKS),
            .WORD_SIZE      (WORD_SIZE)
        ) cache_dram_req_arb (
            .clk                        (clk),
            .reset                      (reset),        
            .per_bank_dram_req_valid    (per_bank_dram_req_valid),        
            .per_bank_dram_req_rw       (per_bank_dram_req_rw),        
            .per_bank_dram_req_byteen   (per_bank_dram_req_byteen),
            .per_bank_dram_req_addr     (per_bank_dram_req_addr),
            .per_bank_dram_req_data     (per_bank_dram_req_data),
            .per_bank_dram_req_ready    (per_bank_dram_req_ready),
            .dram_req_valid             (dram_req_valid),
            .dram_req_rw                (dram_req_rw),        
            .dram_req_byteen            (dram_req_byteen),        
            .dram_req_addr              (dram_req_addr),
            .dram_req_data              (dram_req_data),  
            .dram_req_ready             (dram_req_ready)
        );    
    end else begin
        `UNUSED_VAR (per_bank_dram_req_valid)
        `UNUSED_VAR (per_bank_dram_req_rw)
        `UNUSED_VAR (per_bank_dram_req_byteen)
        `UNUSED_VAR (per_bank_dram_req_addr)
        `UNUSED_VAR (per_bank_dram_req_data)
        assign per_bank_dram_req_ready = 0;
        assign dram_req_valid  = 0;
        assign dram_req_rw     = 0;
        assign dram_req_byteen = 0;
        assign dram_req_addr   = 0;
        assign dram_req_data   = 0;
        `UNUSED_VAR (dram_req_ready)
    end

    if (FLUSH_ENABLE) begin
        VX_snp_rsp_arb #(
            .NUM_BANKS         (NUM_BANKS),
            .BANK_LINE_SIZE    (BANK_LINE_SIZE),
            .SNP_REQ_TAG_WIDTH (SNP_REQ_TAG_WIDTH)
        ) snp_rsp_arb ( 
            .clk                    (clk),
            .reset                  (reset),
            .per_bank_snp_rsp_valid (per_bank_snp_rsp_valid),
            .per_bank_snp_rsp_tag   (per_bank_snp_rsp_tag),
            .per_bank_snp_rsp_ready (per_bank_snp_rsp_ready),
            .snp_rsp_valid          (snp_rsp_valid),
            .snp_rsp_tag            (snp_rsp_tag),
            .snp_rsp_ready          (snp_rsp_ready)
        );
    end else begin
        `UNUSED_VAR (per_bank_snp_rsp_valid)
        `UNUSED_VAR (per_bank_snp_rsp_tag)
        assign per_bank_snp_rsp_ready = 0;
        assign snp_rsp_valid = 0;
        assign snp_rsp_tag   = 0;
        `UNUSED_VAR (snp_rsp_ready)        
    end
    
endmodule
