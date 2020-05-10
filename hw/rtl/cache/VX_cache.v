`include "VX_cache_config.vh"

module VX_cache #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 16, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 2, 
    // Number of cycles to complete stage 1 (read from memory)
    parameter STAGE_1_CYCLES                = 2,

    // Queues feeding into banks Knobs {1, 2, 4, 8, ...}

    // Core Request Queue Size
    parameter REQQ_SIZE                     = 8, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 8, 
    // Dram Fill Rsp Queue Size
    parameter DFPQ_SIZE                     = 2, 
    // Snoop Req Queue
    parameter SNRQ_SIZE                     = 8, 

    // Queues for writebacks Knobs {1, 2, 4, 8, ...}
    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 8, 
    // Dram Writeback Queue Size
    parameter DWBQ_SIZE                     = 4, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 8, 
    // Lower Level Cache Hit Queue Size
    parameter LLVQ_SIZE                     = 16, 
    // Fill Forward SNP Queue
    parameter FFSQ_SIZE                     = 8,

    // Fill Invalidator Size {Fill invalidator must be active}
    parameter FILL_INVALIDAOR_SIZE          = 16,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

    // Enable dram update
    parameter DRAM_ENABLE                   = 1,

    // Enable snoop forwarding
    parameter SNOOP_FORWARDING              = 0,

    // Prefetcher
    parameter PRFQ_SIZE                     = 64,
    parameter PRFQ_STRIDE                   = 0,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // dram request tag size
    parameter DRAM_TAG_WIDTH                = 1
 ) (
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQUESTS-1:0]                       core_req_valid,
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]    core_req_read,
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]    core_req_write,
    input wire [NUM_REQUESTS-1:0][31:0]                 core_req_addr,
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]      core_req_data,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire                                         core_req_ready,

    // Core response
    output wire [NUM_REQUESTS-1:0]                      core_rsp_valid,    
    output wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]     core_rsp_data,
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire                                         core_rsp_ready,   
    
    // DRAM request
    output wire                             dram_req_read,
    output wire                             dram_req_write,    
    output wire [`DRAM_ADDR_WIDTH-1:0]      dram_req_addr,
    output wire [`BANK_LINE_WIDTH-1:0]      dram_req_data,
    output wire [DRAM_TAG_WIDTH-1:0]        dram_req_tag,
    input  wire                             dram_req_ready,
    
    // DRAM response
    input  wire                             dram_rsp_valid,    
    input  wire [`BANK_LINE_WIDTH-1:0]      dram_rsp_data,
    input  wire [DRAM_TAG_WIDTH-1:0]        dram_rsp_tag,
    output wire                             dram_rsp_ready,    

    // Snoop Req
    input wire                              snp_req_valid,
    input wire [`DRAM_ADDR_WIDTH-1:0]       snp_req_addr,
    output wire                             snp_req_ready,

    // Snoop Forward
    output wire                             snp_fwd_valid,
    output wire [`DRAM_ADDR_WIDTH-1:0]      snp_fwd_addr,
    input  wire                             snp_fwd_ready
);

    wire [NUM_BANKS-1:0][NUM_REQUESTS-1:0]                per_bank_valids;
    
    wire [NUM_BANKS-1:0]                                  per_bank_core_rsp_pop;
    wire [NUM_BANKS-1:0]                                  per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]                  per_bank_core_rsp_tid; 
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]                 per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]              per_bank_core_rsp_tag;    

    wire                                                  dfqq_full;
    wire [NUM_BANKS-1:0]                                  per_bank_dram_fill_req_valid;
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0]            per_bank_dram_fill_req_addr;
    wire [NUM_BANKS-1:0]                                  per_bank_dram_fill_rsp_ready;

    wire [NUM_BANKS-1:0]                                  per_bank_dram_wb_queue_pop;
    wire [NUM_BANKS-1:0]                                  per_bank_dram_wb_req_valid;    
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0]            per_bank_dram_wb_req_addr;
    wire [NUM_BANKS-1:0][`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] per_bank_dram_wb_req_data;

    wire [NUM_BANKS-1:0]                                  per_bank_reqq_full;
    wire [NUM_BANKS-1:0]                                  per_bank_snp_req_full;

    wire [NUM_BANKS-1:0]                                  per_bank_snp_fwd_valid;
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0]            per_bank_snp_fwd_addr;
    wire [NUM_BANKS-1:0]                                  per_bank_snp_fwd_pop;

`DEBUG_BEGIN  
    wire [NUM_BANKS-1:0]                                  per_bank_dram_fill_req_is_snp;
`DEBUG_END

    assign dram_req_tag   = dram_req_addr;
    assign core_req_ready = ~(| per_bank_reqq_full);    
    assign snp_req_ready  = ~(| per_bank_snp_req_full);    
    assign dram_rsp_ready = (| per_bank_dram_fill_rsp_ready);    

    VX_cache_core_req_bank_sel #(
        .BANK_LINE_SIZE          (BANK_LINE_SIZE),
        .NUM_BANKS               (NUM_BANKS),
        .WORD_SIZE               (WORD_SIZE),
        .NUM_REQUESTS            (NUM_REQUESTS)
    ) cache_core_req_bank_sell (
        .core_req_valid  (core_req_valid),
        .core_req_addr   (core_req_addr),
        .per_bank_valids (per_bank_valids)
    );

    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            wire [NUM_REQUESTS-1:0]                curr_bank_core_req_valids;
            wire [NUM_REQUESTS-1:0][31:0]          curr_bank_core_req_addr;
            wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] curr_bank_core_req_tag;
            wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0] curr_bank_core_req_data;
            wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0] curr_bank_core_req_read;  
            wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0] curr_bank_core_req_write;

            wire                                   curr_bank_core_rsp_pop;
            wire                                   curr_bank_core_rsp_valid;
            wire [`REQS_BITS-1:0]                  curr_bank_core_rsp_tid;
            wire [`WORD_WIDTH-1:0]                 curr_bank_core_rsp_data;
            wire [CORE_TAG_WIDTH-1:0]              curr_bank_core_rsp_tag;

            wire                                   curr_bank_dram_fill_rsp_valid;            
            wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] curr_bank_dram_fill_rsp_data;
            wire [`LINE_ADDR_WIDTH-1:0]            curr_bank_dram_fill_rsp_addr;
            wire                                   curr_bank_dram_fill_rsp_ready;

            wire                                   curr_bank_dram_fill_req_full;
            wire                                   curr_bank_dram_fill_req_valid;
            wire                                   curr_bank_dram_fill_req_is_snp;
            wire [`LINE_ADDR_WIDTH-1:0]            curr_bank_dram_fill_req_addr;

            wire                                   curr_bank_dram_wb_req_pop;
            wire                                   curr_bank_dram_wb_req_valid;
            wire [`LINE_ADDR_WIDTH-1:0]            curr_bank_dram_wb_req_addr;
            wire[`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] curr_bank_dram_wb_req_data;

            wire                                   curr_bank_snp_req_valid;
            wire [`LINE_ADDR_WIDTH-1:0]            curr_bank_snp_req_addr;
            wire                                   curr_bank_snp_req_full;    

            wire                                   curr_bank_snp_fwd_valid;
            wire [`LINE_ADDR_WIDTH-1:0]            curr_bank_snp_fwd_addr;
            wire                                   curr_bank_snp_fwd_pop;                    

            wire                                   curr_bank_reqq_full;

            // Core Req
            assign curr_bank_core_req_valids  = per_bank_valids[i];
            assign curr_bank_core_req_addr    = core_req_addr;
            assign curr_bank_core_req_data    = core_req_data;
            assign curr_bank_core_req_tag     = core_req_tag;
            assign curr_bank_core_req_read    = core_req_read;
            assign curr_bank_core_req_write   = core_req_write;
            assign per_bank_reqq_full[i]      = curr_bank_reqq_full;

            // Core WB
            assign curr_bank_core_rsp_pop      = per_bank_core_rsp_pop[i];
            assign per_bank_core_rsp_valid [i] = curr_bank_core_rsp_valid;
            assign per_bank_core_rsp_tid   [i] = curr_bank_core_rsp_tid;
            assign per_bank_core_rsp_tag   [i] = curr_bank_core_rsp_tag;
            assign per_bank_core_rsp_data  [i] = curr_bank_core_rsp_data;

            // Dram fill request
            assign curr_bank_dram_fill_req_full     = dfqq_full;
            assign per_bank_dram_fill_req_valid[i]  = curr_bank_dram_fill_req_valid;
            assign per_bank_dram_fill_req_addr[i]   = `LINE_TO_DRAM_ADDR(curr_bank_dram_fill_req_addr, i);
            assign per_bank_dram_fill_req_is_snp[i] = curr_bank_dram_fill_req_is_snp;

            // Dram fill response
            assign curr_bank_dram_fill_rsp_valid   = dram_rsp_valid && (`DRAM_ADDR_BANK(dram_rsp_tag) == i);
            assign curr_bank_dram_fill_rsp_addr    = `DRAM_TO_LINE_ADDR(dram_rsp_tag);
            assign curr_bank_dram_fill_rsp_data    = dram_rsp_data;
            assign per_bank_dram_fill_rsp_ready[i] = curr_bank_dram_fill_rsp_ready;

            // Dram writeback request
            assign curr_bank_dram_wb_req_pop     = per_bank_dram_wb_queue_pop[i];
            assign per_bank_dram_wb_req_valid[i] = curr_bank_dram_wb_req_valid;            
            assign per_bank_dram_wb_req_addr[i]  = `LINE_TO_DRAM_ADDR(curr_bank_dram_wb_req_addr, i);
            assign per_bank_dram_wb_req_data[i]  = curr_bank_dram_wb_req_data;

            // Snoop Request
            assign curr_bank_snp_req_valid  = snp_req_valid && (`DRAM_ADDR_BANK(snp_req_addr) == i);
            assign curr_bank_snp_req_addr   = `DRAM_TO_LINE_ADDR(snp_req_addr);
            assign per_bank_snp_req_full[i] = curr_bank_snp_req_full;

            // Snoop Fwd            
            assign per_bank_snp_fwd_valid[i] = curr_bank_snp_fwd_valid;
            assign per_bank_snp_fwd_addr[i]  = `LINE_TO_DRAM_ADDR(curr_bank_snp_fwd_addr, i);
            assign curr_bank_snp_fwd_pop     = per_bank_snp_fwd_pop[i];
            
            VX_bank #(
                .CACHE_SIZE             (CACHE_SIZE),
                .BANK_LINE_SIZE         (BANK_LINE_SIZE),
                .NUM_BANKS              (NUM_BANKS),
                .WORD_SIZE              (WORD_SIZE),
                .NUM_REQUESTS           (NUM_REQUESTS),
                .STAGE_1_CYCLES         (STAGE_1_CYCLES),
                .REQQ_SIZE              (REQQ_SIZE),
                .MRVQ_SIZE              (MRVQ_SIZE),
                .DFPQ_SIZE              (DFPQ_SIZE),
                .SNRQ_SIZE              (SNRQ_SIZE),
                .CWBQ_SIZE              (CWBQ_SIZE),
                .DWBQ_SIZE              (DWBQ_SIZE),
                .DFQQ_SIZE              (DFQQ_SIZE),
                .LLVQ_SIZE              (LLVQ_SIZE),
                .FFSQ_SIZE              (FFSQ_SIZE),
                .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE),
                .DRAM_ENABLE            (DRAM_ENABLE),
                .WRITE_ENABLE           (WRITE_ENABLE),
                .SNOOP_FORWARDING       (SNOOP_FORWARDING),
                .CORE_TAG_WIDTH         (CORE_TAG_WIDTH),                
                .CORE_TAG_ID_BITS       (CORE_TAG_ID_BITS)
            ) bank (
                .clk                     (clk),
                .reset                   (reset),                
                // Core request
                .core_req_valids         (curr_bank_core_req_valids),                  
                .core_req_read           (curr_bank_core_req_read),
                .core_req_write          (curr_bank_core_req_write),              
                .core_req_addr           (curr_bank_core_req_addr),
                .core_req_data           (curr_bank_core_req_data),
                .core_req_tag            (curr_bank_core_req_tag),
                .core_req_full           (curr_bank_reqq_full),
                .core_req_ready          (core_req_ready),                

                // Core response                
                .core_rsp_valid          (curr_bank_core_rsp_valid),
                .core_rsp_tid            (curr_bank_core_rsp_tid),
                .core_rsp_data           (curr_bank_core_rsp_data),
                .core_rsp_tag            (curr_bank_core_rsp_tag),
                .core_rsp_pop            (curr_bank_core_rsp_pop),

                // Dram fill request
                .dram_fill_req_valid     (curr_bank_dram_fill_req_valid),
                .dram_fill_req_addr      (curr_bank_dram_fill_req_addr),
                .dram_fill_req_is_snp    (curr_bank_dram_fill_req_is_snp),
                .dram_fill_req_full      (curr_bank_dram_fill_req_full),

                // Dram fill response
                .dram_fill_rsp_valid     (curr_bank_dram_fill_rsp_valid),                
                .dram_fill_rsp_data      (curr_bank_dram_fill_rsp_data),
                .dram_fill_rsp_addr      (curr_bank_dram_fill_rsp_addr),
                .dram_fill_rsp_ready     (curr_bank_dram_fill_rsp_ready),

                // Dram writeback request               
                .dram_wb_req_valid       (curr_bank_dram_wb_req_valid),
                .dram_wb_req_addr        (curr_bank_dram_wb_req_addr),
                .dram_wb_req_data        (curr_bank_dram_wb_req_data),   
                .dram_wb_req_pop         (curr_bank_dram_wb_req_pop),

                // Snoop request
                .snp_req_valid           (curr_bank_snp_req_valid),
                .snp_req_addr            (curr_bank_snp_req_addr),
                .snp_req_full            (curr_bank_snp_req_full),

                // Snoop forwarding
                .snp_fwd_valid           (curr_bank_snp_fwd_valid),
                .snp_fwd_addr            (curr_bank_snp_fwd_addr),
                .snp_fwd_pop             (curr_bank_snp_fwd_pop)
            );
        end   
    endgenerate   

    VX_cache_core_rsp_merge #(
        .NUM_BANKS              (NUM_BANKS),
        .WORD_SIZE              (WORD_SIZE),
        .NUM_REQUESTS           (NUM_REQUESTS),
        .CORE_TAG_WIDTH         (CORE_TAG_WIDTH),        
        .CORE_TAG_ID_BITS       (CORE_TAG_ID_BITS)
    ) cache_core_rsp_merge (
        .per_bank_core_rsp_tid   (per_bank_core_rsp_tid),                
        .per_bank_core_rsp_valid (per_bank_core_rsp_valid),   
        .per_bank_core_rsp_data  (per_bank_core_rsp_data),
        .per_bank_core_rsp_tag   (per_bank_core_rsp_tag),
        .per_bank_core_rsp_pop   (per_bank_core_rsp_pop),

        .core_rsp_valid (core_rsp_valid),
        .core_rsp_data  (core_rsp_data),        
        .core_rsp_tag   (core_rsp_tag),
        .core_rsp_ready (core_rsp_ready)
    );

    VX_cache_dram_req_arb #(
        .BANK_LINE_SIZE         (BANK_LINE_SIZE),
        .NUM_BANKS              (NUM_BANKS),
        .WORD_SIZE              (WORD_SIZE),
        .DFQQ_SIZE              (DFQQ_SIZE),
        .PRFQ_SIZE              (PRFQ_SIZE),
        .PRFQ_STRIDE            (PRFQ_STRIDE)
    ) cache_dram_req_arb (
        .clk                           (clk),
        .reset                         (reset),
        .dfqq_full                     (dfqq_full),
        .per_bank_dram_fill_req_valid  (per_bank_dram_fill_req_valid),
        .per_bank_dram_fill_req_addr   (per_bank_dram_fill_req_addr),
        .per_bank_dram_wb_queue_pop    (per_bank_dram_wb_queue_pop),
        .per_bank_dram_wb_req_valid    (per_bank_dram_wb_req_valid),        
        .per_bank_dram_wb_req_addr     (per_bank_dram_wb_req_addr),
        .per_bank_dram_wb_req_data     (per_bank_dram_wb_req_data),
        .dram_req_read                 (dram_req_read),
        .dram_req_write                (dram_req_write),        
        .dram_req_addr                 (dram_req_addr),
        .dram_req_data                 (dram_req_data),  
        .dram_req_ready                (dram_req_ready)
    );   

    VX_snp_fwd_arb #(
        .NUM_BANKS(NUM_BANKS),
        .BANK_LINE_SIZE(BANK_LINE_SIZE)
    ) snp_fwd_arb (
        .per_bank_snp_fwd_valid (per_bank_snp_fwd_valid),
        .per_bank_snp_fwd_addr  (per_bank_snp_fwd_addr),
        .per_bank_snp_fwd_pop   (per_bank_snp_fwd_pop),
        .snp_fwd_valid          (snp_fwd_valid),
        .snp_fwd_addr           (snp_fwd_addr),
        .snp_fwd_ready          (snp_fwd_ready)
    );
    
endmodule