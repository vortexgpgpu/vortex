`include "VX_cache_config.vh"

module VX_cache #(
    parameter CACHE_ID                      = 0,

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 8092, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks
    parameter NUM_BANKS                     = 4, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle
    parameter NUM_REQS                      = NUM_BANKS, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 4, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 8, 
    // DRAM Response Queue Size
    parameter DRSQ_SIZE                     = 4, 
    // Snoop Request Queue Size
    parameter SREQ_SIZE                     = 4, 

    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 4, 
    // DRAM Request Queue Size
    parameter DREQ_SIZE                     = 4, 
    // Snoop Response Size
    parameter SRSQ_SIZE                     = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

    // Enable dram update
    parameter DRAM_ENABLE                   = 1,

    // Enable cache flush
    parameter FLUSH_ENABLE                  = 1,

    // core request tag size
    parameter CORE_TAG_WIDTH                = $clog2(MSHR_SIZE),
    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // dram request tag size
    parameter DRAM_TAG_WIDTH                = (32 - $clog2(BANK_LINE_SIZE)),

    // Snooping request tag width
    parameter SNP_TAG_WIDTH                 = 1
 ) (
    `SCOPE_IO_VX_cache
    
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQS-1:0]                           core_req_valid,
    input wire [`CORE_REQ_TAG_COUNT-1:0]                core_req_rw,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]            core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0]     core_req_addr,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]          core_req_data,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire [`CORE_REQ_TAG_COUNT-1:0]               core_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                          core_rsp_valid,    
    output wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]         core_rsp_data,
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire [`CORE_REQ_TAG_COUNT-1:0]               core_rsp_ready,   

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if                        perf_cache_if,
`endif

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
    input wire                              snp_req_inv,
    input wire [SNP_TAG_WIDTH-1:0]          snp_req_tag,
    output wire                             snp_req_ready,

    // Snoop response
    output wire                             snp_rsp_valid,    
    output wire [SNP_TAG_WIDTH-1:0]         snp_rsp_tag,
    input  wire                             snp_rsp_ready,

    output wire [NUM_BANKS-1:0] miss_vec
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))

    wire [NUM_BANKS-1:0][NUM_REQS-1:0]          per_bank_valid;

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
    wire [NUM_BANKS-1:0][SNP_TAG_WIDTH-1:0]     per_bank_snp_rsp_tag;
    wire [NUM_BANKS-1:0]                        per_bank_snp_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_miss; 
    assign miss_vec = per_bank_miss;  


`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_mshr_stall_per_bank;
    wire [NUM_BANKS-1:0] perf_pipe_stall_per_bank;
    wire [NUM_BANKS-1:0] perf_evict_per_bank;
    wire [NUM_BANKS-1:0] perf_read_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_write_miss_per_bank;
`endif

    if (NUM_BANKS == 1) begin
        assign snp_req_ready = per_bank_snp_req_ready;
    end else begin
        assign snp_req_ready = per_bank_snp_req_ready[`DRAM_ADDR_BANK(snp_req_addr)];
    end    

    VX_cache_core_req_bank_sel #(
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .NUM_REQS       (NUM_REQS),
        .CORE_TAG_ID_BITS (CORE_TAG_ID_BITS)
    ) cache_core_req_bank_sel (
        .core_req_valid  (core_req_valid),
        .core_req_addr   (core_req_addr),
        .core_req_ready  (core_req_ready),
        .per_bank_valid  (per_bank_valid),
        .per_bank_ready  (per_bank_core_req_ready)        
    );

    assign dram_req_tag = dram_req_addr;
    if (NUM_BANKS == 1) begin
        assign dram_rsp_ready = per_bank_dram_rsp_ready;
    end else begin
        assign dram_rsp_ready = per_bank_dram_rsp_ready[`DRAM_ADDR_BANK(dram_rsp_tag)];
    end
    
    for (genvar i = 0; i < NUM_BANKS; i++) begin
        wire [NUM_REQS-1:0]                             curr_bank_core_req_valid;            
        wire [`CORE_REQ_TAG_COUNT-1:0]                  curr_bank_core_req_rw;  
        wire [NUM_REQS-1:0][WORD_SIZE-1:0]              curr_bank_core_req_byteen;
        wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0]       curr_bank_core_req_addr;
        wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] curr_bank_core_req_tag;
        wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]            curr_bank_core_req_data;
        wire                                            curr_bank_core_req_ready;

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
        wire                            curr_bank_snp_req_inv;
        wire [SNP_TAG_WIDTH-1:0]        curr_bank_snp_req_tag;
        wire                            curr_bank_snp_req_ready;    

        wire                            curr_bank_snp_rsp_valid;
        wire [SNP_TAG_WIDTH-1:0]        curr_bank_snp_rsp_tag;
        wire                            curr_bank_snp_rsp_ready;                    

        wire                            curr_bank_miss; 

        // Core Req
        assign curr_bank_core_req_valid   = per_bank_valid[i];
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
            assign curr_bank_snp_req_valid = snp_req_valid;
            assign curr_bank_snp_req_addr  = snp_req_addr;
        end else begin
            assign curr_bank_snp_req_valid = snp_req_valid && (`DRAM_ADDR_BANK(snp_req_addr) == i);
            assign curr_bank_snp_req_addr  = `DRAM_TO_LINE_ADDR(snp_req_addr);
        end
        assign curr_bank_snp_req_inv     = snp_req_inv;
        assign curr_bank_snp_req_tag     = snp_req_tag;
        assign per_bank_snp_req_ready[i] = curr_bank_snp_req_ready;

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
            .NUM_REQS           (NUM_REQS),
            .CREQ_SIZE          (CREQ_SIZE),
            .MSHR_SIZE          (MSHR_SIZE),
            .DRSQ_SIZE          (DRSQ_SIZE),
            .SREQ_SIZE          (SREQ_SIZE),
            .CRSQ_SIZE          (CRSQ_SIZE),
            .DREQ_SIZE          (DREQ_SIZE),
            .SRSQ_SIZE          (SRSQ_SIZE),
            .DRAM_ENABLE        (DRAM_ENABLE),
            .FLUSH_ENABLE       (FLUSH_ENABLE),
            .WRITE_ENABLE       (WRITE_ENABLE),
            .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),                
            .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
            .SNP_TAG_WIDTH      (SNP_TAG_WIDTH)
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

        `ifdef PERF_ENABLE
            .perf_mshr_stall    (perf_mshr_stall_per_bank[i]),
            .perf_pipe_stall    (perf_pipe_stall_per_bank[i]),
            .perf_evict         (perf_evict_per_bank[i]),
            .perf_read_miss     (perf_read_miss_per_bank[i]),
            .perf_write_miss    (perf_write_miss_per_bank[i]),
        `endif

            // Snoop request
            .snp_req_valid      (curr_bank_snp_req_valid),
            .snp_req_addr       (curr_bank_snp_req_addr),
            .snp_req_inv        (curr_bank_snp_req_inv),
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
        .NUM_REQS           (NUM_REQS),
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
        wire [NUM_BANKS-1:0][(`DRAM_ADDR_WIDTH + 1 + BANK_LINE_SIZE + `BANK_LINE_WIDTH)-1:0] data_in;
        for (genvar i = 0; i < NUM_BANKS; i++) begin
            assign data_in[i] = {per_bank_dram_req_addr[i], per_bank_dram_req_rw[i], per_bank_dram_req_byteen[i], per_bank_dram_req_data[i]};
        end

        VX_stream_arbiter #(
            .NUM_REQS   (NUM_BANKS),
            .DATAW      (`DRAM_ADDR_WIDTH + 1 + BANK_LINE_SIZE + `BANK_LINE_WIDTH),
            .OUT_BUFFER (NUM_BANKS >= 4)
        ) dram_req_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (per_bank_dram_req_valid),
            .data_in   (data_in),
            .ready_in  (per_bank_dram_req_ready),   
            .valid_out (dram_req_valid),   
            .data_out  ({dram_req_addr, dram_req_rw, dram_req_byteen, dram_req_data}),
            .ready_out (dram_req_ready)
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
        VX_stream_arbiter #(
            .NUM_REQS   (NUM_BANKS),
            .DATAW      (SNP_TAG_WIDTH),
            .OUT_BUFFER (NUM_BANKS >= 4)
        ) snp_rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (per_bank_snp_rsp_valid),
            .data_in   (per_bank_snp_rsp_tag),
            .ready_in  (per_bank_snp_rsp_ready),
            .valid_out (snp_rsp_valid),
            .data_out  (snp_rsp_tag),         
            .ready_out (snp_rsp_ready)
        );
    end else begin
        `UNUSED_VAR (per_bank_snp_rsp_valid)
        `UNUSED_VAR (per_bank_snp_rsp_tag)
        assign per_bank_snp_rsp_ready = 0;
        assign snp_rsp_valid = 0;
        assign snp_rsp_tag   = 0;
        `UNUSED_VAR (snp_rsp_ready)        
    end
    
`ifdef PERF_ENABLE
    // per cycle: core_req_r, core_req_w
    reg [($clog2(NUM_REQS+1)-1):0] perf_core_req_r_per_cycle, perf_core_req_w_per_cycle;
    reg [($clog2(NUM_REQS+1)-1):0] perf_crsp_stall_per_cycle;

    if (CORE_TAG_ID_BITS != 0) begin
        VX_countones #( // core_req_r
            .N(NUM_REQS) 
        ) perf_countones_core_req_r_count (
            .valids (core_req_valid & {NUM_REQS{core_req_ready & ~core_req_rw}}),
            .count  (perf_core_req_r_per_cycle)
        );

        VX_countones #( // core_req_w
            .N(NUM_REQS) 
        ) perf_countones_core_req_w_count (
            .valids (core_req_valid & {NUM_REQS{core_req_ready & core_req_rw}}),
            .count  (perf_core_req_w_per_cycle)
        );

        VX_countones #( // core_rsp
            .N(NUM_REQS) 
        ) perf_countones_core_rsp_count (
            .valids (core_rsp_valid & {NUM_REQS{!core_rsp_ready}}),
            .count  (perf_crsp_stall_per_cycle)
        );
    end else begin
        VX_countones #( // core_req_r
            .N(NUM_REQS) 
        ) perf_countones_core_req_r_count (
            .valids (core_req_valid & core_req_ready & ~core_req_rw),
            .count  (perf_core_req_r_per_cycle)
        );
        
        VX_countones #( // core_req_w
            .N(NUM_REQS) 
        ) perf_countones_core_req_w_count (
            .valids (core_req_valid & core_req_ready & core_req_rw),
            .count  (perf_core_req_w_per_cycle)
        );

        VX_countones #( // core_rsp
            .N(NUM_REQS) 
        ) perf_countones_core_rsp_count (
            .valids (core_rsp_valid & ~core_rsp_ready),
            .count  (perf_crsp_stall_per_cycle)
        );
    end

    // per cycle: msrq stalls, pipeline stalls, evictions, read misses, write misses
    reg [($clog2(NUM_BANKS+1)-1):0] perf_mshr_stall_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_pipe_stall_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_evictions_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_read_miss_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_write_miss_per_cycle;

    VX_countones #(
        .N(NUM_BANKS) 
    ) perf_countones_mshr_stall_count (
        .valids (perf_mshr_stall_per_bank),
        .count  (perf_mshr_stall_per_cycle)
    );

    VX_countones #(
        .N(NUM_BANKS) 
    ) perf_countones_total_stall_count (
        .valids (perf_pipe_stall_per_bank),
        .count  (perf_pipe_stall_per_cycle)
    );

    VX_countones #(
        .N(NUM_BANKS) 
    ) perf_countones_EVICTSict_count (
        .valids (perf_evict_per_bank),
        .count  (perf_evictions_per_cycle)
    );

    VX_countones #(
        .N(NUM_BANKS) 
    ) perf_countones_read_miss_count (
        .valids (perf_read_miss_per_bank),
        .count  (perf_read_miss_per_cycle)
    );

    VX_countones #(
        .N(NUM_BANKS) 
    ) perf_countones_write_miss_count (
        .valids (perf_write_miss_per_bank),
        .count  (perf_write_miss_per_cycle)
    );

    reg [63:0] perf_core_req_r;
    reg [63:0] perf_core_req_w;
    reg [63:0] perf_mshr_stall;
    reg [63:0] perf_pipe_stall;
    reg [63:0] perf_evictions;
    reg [63:0] perf_read_miss;
    reg [63:0] perf_write_miss;
    reg [63:0] perf_crsp_stall;
    reg [63:0] perf_dreq_stall;

    always @(posedge clk) begin
        if (reset) begin
            perf_core_req_r <= 0;
            perf_core_req_w <= 0;
            perf_crsp_stall <= 0;
            perf_mshr_stall <= 0;
            perf_pipe_stall <= 0;
            perf_evictions  <= 0;
            perf_read_miss  <= 0;
            perf_write_miss <= 0;
            perf_dreq_stall <= 0;
        end else begin
            // core requests
            perf_core_req_r <= perf_core_req_r + $bits(perf_core_req_r)'(perf_core_req_r_per_cycle);
            perf_core_req_w <= perf_core_req_w + $bits(perf_core_req_w)'(perf_core_req_w_per_cycle);
            // core response stalls
            perf_crsp_stall <= perf_crsp_stall + $bits(perf_crsp_stall)'(perf_crsp_stall_per_cycle);
            // miss reserve queue stalls
            perf_mshr_stall <= perf_mshr_stall + $bits(perf_mshr_stall)'(perf_mshr_stall_per_cycle);
            // pipeline stalls
            perf_pipe_stall <= perf_pipe_stall + $bits(perf_pipe_stall)'(perf_pipe_stall_per_cycle);
            // total evictions
            perf_evictions <= perf_evictions + $bits(perf_evictions)'(perf_evictions_per_cycle);
            // read misses
            perf_read_miss <= perf_read_miss + $bits(perf_read_miss)'(perf_read_miss_per_cycle);
            // write misses
            perf_write_miss <= perf_write_miss + $bits(perf_write_miss)'(perf_write_miss_per_cycle);
            // dram request stalls
            if (dram_req_valid & !dram_req_ready) begin
                perf_dreq_stall <= perf_dreq_stall + 64'd1;
            end
        end
    end

    assign perf_cache_if.reads = perf_core_req_r;
    assign perf_cache_if.writes = perf_core_req_w;
    assign perf_cache_if.read_misses = perf_read_miss;
    assign perf_cache_if.write_misses = perf_write_miss;
    assign perf_cache_if.evictions = perf_evictions;
    assign perf_cache_if.mshr_stalls = perf_mshr_stall;
    assign perf_cache_if.pipe_stalls = perf_pipe_stall;
    assign perf_cache_if.crsp_stalls = perf_crsp_stall;
    assign perf_cache_if.dreq_stalls = perf_dreq_stall;
`endif

endmodule
