`include "VX_cache_config.vh"

module VX_cache #(
    parameter CACHE_ID                      = 0,

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 16384, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE               = 64, 
    // Number of banks
    parameter NUM_BANKS                     = 4, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle
    parameter NUM_REQS                      = NUM_BANKS, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 4, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 16, 
    // DRAM Response Queue Size
    parameter DRSQ_SIZE                     = 4,

    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 4, 
    // DRAM Request Queue Size
    parameter DREQ_SIZE                     = 4, 

    // Enable dram update
    parameter DRAM_ENABLE                   = 1,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

    // Enable write-through
    parameter WRITE_THROUGH                 = 1,

    // core request tag size
    parameter CORE_TAG_WIDTH                = $clog2(MSHR_SIZE),
    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = CORE_TAG_WIDTH,

    // dram request tag size
    parameter DRAM_TAG_WIDTH                = `LOG2UP(NUM_BANKS),

    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET              = 0
 ) (
    `SCOPE_IO_VX_cache
    
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQS-1:0]                           core_req_valid,
    input wire [NUM_REQS-1:0]                           core_req_rw,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]            core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0]     core_req_addr,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]          core_req_data,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0]       core_req_tag,
    output wire [NUM_REQS-1:0]                          core_req_ready,

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
    output wire [CACHE_LINE_SIZE-1:0]       dram_req_byteen,    
    output wire [`DRAM_ADDR_WIDTH-1:0]      dram_req_addr,
    output wire [`CACHE_LINE_WIDTH-1:0]     dram_req_data,
    output wire [DRAM_TAG_WIDTH-1:0]        dram_req_tag,
    input  wire                             dram_req_ready,
    
    // DRAM response
    input  wire                             dram_rsp_valid,    
    input  wire [`CACHE_LINE_WIDTH-1:0]     dram_rsp_data,
    input  wire [DRAM_TAG_WIDTH-1:0]        dram_rsp_tag,
    output wire                             dram_rsp_ready
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))

    wire [NUM_BANKS-1:0]                        per_bank_core_req_valid; 
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]        per_bank_core_req_tid;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_rw;  
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]         per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][`WORD_ADDR_WIDTH-1:0]  per_bank_core_req_addr;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]    per_bank_core_req_tag;
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]       per_bank_core_req_data;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_ready;
    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]        per_bank_core_rsp_tid; 
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]       per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]    per_bank_core_rsp_tag;    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_dram_req_valid;    
    wire [NUM_BANKS-1:0]                        per_bank_dram_req_rw;
    wire [NUM_BANKS-1:0][CACHE_LINE_SIZE-1:0]   per_bank_dram_req_byteen;    
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0]  per_bank_dram_req_addr;
    wire [NUM_BANKS-1:0][`CACHE_LINE_WIDTH-1:0] per_bank_dram_req_data;
    wire [NUM_BANKS-1:0]                        per_bank_dram_req_ready;

    wire [NUM_BANKS-1:0]                        per_bank_dram_rsp_ready;

`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_read_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_write_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_mshr_stall_per_bank;
    wire [NUM_BANKS-1:0] perf_pipe_stall_per_bank;
`endif   

    VX_cache_core_req_bank_sel #(
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS       (NUM_BANKS),
        .WORD_SIZE       (WORD_SIZE),
        .NUM_REQS        (NUM_REQS),
        .CORE_TAG_WIDTH  (CORE_TAG_WIDTH),
        .BANK_ADDR_OFFSET(BANK_ADDR_OFFSET),
        .BUFFERED        ((NUM_BANKS > 1) && DRAM_ENABLE)
    ) cache_core_req_bank_sel (        
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE        
        .bank_stalls    (perf_cache_if.bank_stalls),
    `else
        `UNUSED_PIN (bank_stalls),
    `endif     
        .core_req_valid (core_req_valid),
        .core_req_rw    (core_req_rw), 
        .core_req_byteen(core_req_byteen),
        .core_req_addr  (core_req_addr),
        .core_req_data  (core_req_data),
        .core_req_tag   (core_req_tag),
        .core_req_ready (core_req_ready),
        .per_bank_core_req_valid (per_bank_core_req_valid), 
        .per_bank_core_req_tid   (per_bank_core_req_tid),
        .per_bank_core_req_rw    (per_bank_core_req_rw),
        .per_bank_core_req_byteen(per_bank_core_req_byteen),
        .per_bank_core_req_addr  (per_bank_core_req_addr),
        .per_bank_core_req_tag   (per_bank_core_req_tag),
        .per_bank_core_req_data  (per_bank_core_req_data),
        .per_bank_core_req_ready (per_bank_core_req_ready)
    );

    if (NUM_BANKS == 1) begin
        `UNUSED_VAR (dram_rsp_tag)
        assign dram_rsp_ready = per_bank_dram_rsp_ready;
    end else begin
        assign dram_rsp_ready = per_bank_dram_rsp_ready[dram_rsp_tag];
    end
    
    for (genvar i = 0; i < NUM_BANKS; i++) begin
        wire                        curr_bank_core_req_valid;     
        wire [`REQS_BITS-1:0]       curr_bank_core_req_tid;       
        wire                        curr_bank_core_req_rw;  
        wire [WORD_SIZE-1:0]        curr_bank_core_req_byteen;
        wire [`WORD_ADDR_WIDTH-1:0] curr_bank_core_req_addr;
        wire [CORE_TAG_WIDTH-1:0]   curr_bank_core_req_tag;
        wire [`WORD_WIDTH-1:0]      curr_bank_core_req_data;
        wire                        curr_bank_core_req_ready;

        wire                        curr_bank_core_rsp_valid;
        wire [`REQS_BITS-1:0]       curr_bank_core_rsp_tid;
        wire [`WORD_WIDTH-1:0]      curr_bank_core_rsp_data;
        wire [CORE_TAG_WIDTH-1:0]   curr_bank_core_rsp_tag;
        wire                        curr_bank_core_rsp_ready;

        wire                        curr_bank_dram_req_valid;
        wire                        curr_bank_dram_req_rw;
        wire [CACHE_LINE_SIZE-1:0]  curr_bank_dram_req_byteen;
        wire [`LINE_ADDR_WIDTH-1:0] curr_bank_dram_req_addr;
        wire[`CACHE_LINE_WIDTH-1:0] curr_bank_dram_req_data;
        wire                        curr_bank_dram_req_ready;

        wire                        curr_bank_dram_rsp_valid;            
        wire [`CACHE_LINE_WIDTH-1:0] curr_bank_dram_rsp_data;
        wire                        curr_bank_dram_rsp_ready;

        // Core Req
        assign curr_bank_core_req_valid   = per_bank_core_req_valid[i];
        assign curr_bank_core_req_tid     = per_bank_core_req_tid[i];
        assign curr_bank_core_req_addr    = per_bank_core_req_addr[i];
        assign curr_bank_core_req_rw      = per_bank_core_req_rw[i];
        assign curr_bank_core_req_byteen  = per_bank_core_req_byteen[i];
        assign curr_bank_core_req_data    = per_bank_core_req_data[i];
        assign curr_bank_core_req_tag     = per_bank_core_req_tag[i];
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
        end else begin
            assign curr_bank_dram_rsp_valid = dram_rsp_valid && (dram_rsp_tag == i);
        end
        assign curr_bank_dram_rsp_data    = dram_rsp_data;
        assign per_bank_dram_rsp_ready[i] = curr_bank_dram_rsp_ready;
        
        VX_bank #(                
            .BANK_ID            (i),
            .CACHE_ID           (CACHE_ID),
            .CACHE_SIZE         (CACHE_SIZE),
            .CACHE_LINE_SIZE     (CACHE_LINE_SIZE),
            .NUM_BANKS          (NUM_BANKS),
            .WORD_SIZE          (WORD_SIZE),
            .NUM_REQS           (NUM_REQS),
            .CREQ_SIZE          (CREQ_SIZE),
            .MSHR_SIZE          (MSHR_SIZE),
            .DRSQ_SIZE          (DRSQ_SIZE),
            .CRSQ_SIZE          (CRSQ_SIZE),
            .DREQ_SIZE          (DREQ_SIZE),
            .DRAM_ENABLE        (DRAM_ENABLE),
            .WRITE_ENABLE       (WRITE_ENABLE),
            .WRITE_THROUGH      (WRITE_THROUGH),
            .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),                
            .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
            .BANK_ADDR_OFFSET   (BANK_ADDR_OFFSET)
        ) bank (
            `SCOPE_BIND_VX_cache_bank(i)
            
            .clk                (clk),
            .reset              (reset),     

        `ifdef PERF_ENABLE
            .perf_read_misses   (perf_read_miss_per_bank[i]),
            .perf_write_misses  (perf_write_miss_per_bank[i]),
            .perf_mshr_stalls   (perf_mshr_stall_per_bank[i]),
            .perf_pipe_stalls   (perf_pipe_stall_per_bank[i]),
        `endif
                       
            // Core request
            .core_req_valid     (curr_bank_core_req_valid),                  
            .core_req_tid       (curr_bank_core_req_tid),
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
            .dram_rsp_ready     (curr_bank_dram_rsp_ready)
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
        wire [NUM_BANKS-1:0][(DRAM_TAG_WIDTH + `DRAM_ADDR_WIDTH + 1 + CACHE_LINE_SIZE + `CACHE_LINE_WIDTH)-1:0] data_in;
        for (genvar i = 0; i < NUM_BANKS; i++) begin
            assign data_in[i] = {DRAM_TAG_WIDTH'(i), per_bank_dram_req_addr[i], per_bank_dram_req_rw[i], per_bank_dram_req_byteen[i], per_bank_dram_req_data[i]};
        end

        VX_stream_arbiter #(
            .NUM_REQS (NUM_BANKS),
            .DATAW    (DRAM_TAG_WIDTH + `DRAM_ADDR_WIDTH + 1 + CACHE_LINE_SIZE + `CACHE_LINE_WIDTH),
            .BUFFERED (1)
        ) dram_req_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (per_bank_dram_req_valid),
            .data_in   (data_in),
            .ready_in  (per_bank_dram_req_ready),   
            .valid_out (dram_req_valid),   
            .data_out  ({dram_req_tag, dram_req_addr, dram_req_rw, dram_req_byteen, dram_req_data}),
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
        assign dram_req_tag    = 0;
        `UNUSED_VAR (dram_req_ready)
    end

`ifdef PERF_ENABLE
    // per cycle: core_reads, core_writes
    reg [($clog2(NUM_REQS+1)-1):0] perf_core_reads_per_cycle, perf_core_writes_per_cycle;
    reg [($clog2(NUM_REQS+1)-1):0] perf_crsp_stall_per_cycle;

    VX_countones #(
        .N(NUM_REQS) 
    ) perf_countones_core_reads_count (
        .valids (core_req_valid & core_req_ready & ~core_req_rw),
        .count  (perf_core_reads_per_cycle)
    );
    
    VX_countones #(
        .N(NUM_REQS) 
    ) perf_countones_core_writes_count (
        .valids (core_req_valid & core_req_ready & core_req_rw),
        .count  (perf_core_writes_per_cycle)
    );

    if (CORE_TAG_ID_BITS != 0) begin
        VX_countones #(
            .N(NUM_REQS) 
        ) perf_countones_core_rsp_count (
            .valids (core_rsp_valid & {NUM_REQS{!core_rsp_ready}}),
            .count  (perf_crsp_stall_per_cycle)
        );
    end else begin
        VX_countones #(
            .N(NUM_REQS) 
        ) perf_countones_core_rsp_count (
            .valids (core_rsp_valid & ~core_rsp_ready),
            .count  (perf_crsp_stall_per_cycle)
        );
    end

    // per cycle: read misses, write misses, msrq stalls, pipeline stalls
    reg [($clog2(NUM_BANKS+1)-1):0] perf_read_miss_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_write_miss_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_mshr_stall_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_pipe_stall_per_cycle;

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

    reg [63:0] perf_core_reads;
    reg [63:0] perf_core_writes;
    reg [63:0] perf_read_misses;
    reg [63:0] perf_write_misses;
    reg [63:0] perf_mshr_stalls;
    reg [63:0] perf_pipe_stalls;
    reg [63:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_core_reads   <= 0;
            perf_core_writes  <= 0;
            perf_read_misses  <= 0;
            perf_write_misses <= 0;
            perf_mshr_stalls  <= 0;
            perf_pipe_stalls  <= 0;
            perf_crsp_stalls  <= 0;
        end else begin
            perf_core_reads   <= perf_core_reads + 64'(perf_core_reads_per_cycle);
            perf_core_writes  <= perf_core_writes + 64'(perf_core_writes_per_cycle);
            perf_read_misses  <= perf_read_misses + 64'(perf_read_miss_per_cycle);
            perf_write_misses <= perf_write_misses + 64'(perf_write_miss_per_cycle);
            perf_mshr_stalls  <= perf_mshr_stalls + 64'(perf_mshr_stall_per_cycle);
            perf_pipe_stalls  <= perf_pipe_stalls + 64'(perf_pipe_stall_per_cycle);
            perf_crsp_stalls  <= perf_crsp_stalls + 64'(perf_crsp_stall_per_cycle);
        end
    end

    assign perf_cache_if.reads        = perf_core_reads;
    assign perf_cache_if.writes       = perf_core_writes;
    assign perf_cache_if.read_misses  = perf_read_misses;
    assign perf_cache_if.write_misses = perf_write_misses;
    assign perf_cache_if.mshr_stalls  = perf_mshr_stalls;
    assign perf_cache_if.pipe_stalls  = perf_pipe_stalls;
    assign perf_cache_if.crsp_stalls  = perf_crsp_stalls;
`endif

endmodule
