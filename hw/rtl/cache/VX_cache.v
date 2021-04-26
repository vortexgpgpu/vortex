`include "VX_cache_config.vh"

module VX_cache #(
    parameter CACHE_ID                      = 0,

    // Number of Word requests per cycle
    parameter NUM_REQS                      = 4,

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 16384, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE               = 64, 
    // Number of banks
    parameter NUM_BANKS                     = NUM_REQS,
    // Number of ports per banks
    parameter NUM_PORTS                     = 1,
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 4, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 8, 
    // Memory Response Queue Size
    parameter MRSQ_SIZE                     = 4,
    // Memory Request Queue Size
    parameter MREQ_SIZE                     = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

    // core request tag size
    parameter CORE_TAG_WIDTH                = $clog2(MSHR_SIZE),
    
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = CORE_TAG_WIDTH,

    // Memory request tag size
    parameter MEM_TAG_WIDTH                = (32 - $clog2(CACHE_LINE_SIZE)),

    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET              = 0,

    // in-order DRAN
    parameter IN_ORDER_MEM                 = 0
 ) (
    `SCOPE_IO_VX_cache
    
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQS-1:0]                           core_req_valid,
    input wire [NUM_REQS-1:0]                           core_req_rw,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0]     core_req_addr,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]            core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]          core_req_data,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0]       core_req_tag,
    output wire [NUM_REQS-1:0]                          core_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                          core_rsp_valid,    
    output wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]         core_rsp_data,
    output wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire [`CORE_REQ_TAG_COUNT-1:0]               core_rsp_ready,   

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,    
    output wire [CACHE_LINE_SIZE-1:0]       mem_req_byteen,    
    output wire [`MEM_ADDR_WIDTH-1:0]       mem_req_addr,
    output wire [`CACHE_LINE_WIDTH-1:0]     mem_req_data,
    output wire [MEM_TAG_WIDTH-1:0]         mem_req_tag,
    input  wire                             mem_req_ready,
    
    // Memory response
    input  wire                             mem_rsp_valid,    
    input  wire [`CACHE_LINE_WIDTH-1:0]     mem_rsp_data,
    input  wire [MEM_TAG_WIDTH-1:0]         mem_rsp_tag,
    output wire                             mem_rsp_ready,    
    
    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if                        perf_cache_if,
`endif

    // device flush
    input wire flush
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))
    
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0]         per_bank_core_req_valid; 
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`UP(`WORD_SELECT_BITS)-1:0] per_bank_core_req_wsel;
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0][WORD_SIZE-1:0] per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`WORD_WIDTH-1:0] per_bank_core_req_data;
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`REQS_BITS-1:0] per_bank_core_req_tid;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_rw;  
    wire [NUM_BANKS-1:0][`LINE_ADDR_WIDTH-1:0]  per_bank_core_req_addr;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]    per_bank_core_req_tag;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_ready;
    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0]         per_bank_core_rsp_pmask;
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`WORD_WIDTH-1:0] per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][NUM_PORTS-1:0][`REQS_BITS-1:0] per_bank_core_rsp_tid;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0]    per_bank_core_rsp_tag;    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_mem_req_valid;    
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_rw;
    wire [NUM_BANKS-1:0][CACHE_LINE_SIZE-1:0]   per_bank_mem_req_byteen;    
    wire [NUM_BANKS-1:0][`MEM_ADDR_WIDTH-1:0]   per_bank_mem_req_addr;
    wire [NUM_BANKS-1:0][`CACHE_LINE_WIDTH-1:0] per_bank_mem_req_data;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_ready;

    wire [NUM_BANKS-1:0]                        per_bank_mem_rsp_ready;

    wire [`CACHE_LINE_WIDTH-1:0]                mem_rsp_data_qual;
    wire [MEM_TAG_WIDTH-1:0]                    mem_rsp_tag_qual;
    wire [`LINE_SELECT_BITS-1:0]                flush_addr;
    wire                                        flush_enable;

`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_read_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_write_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_mshr_stall_per_bank;
    wire [NUM_BANKS-1:0] perf_pipe_stall_per_bank;
`endif

    ///////////////////////////////////////////////////////////////////////////

    wire mrsq_full, mrsq_empty;
    wire mrsq_push, mrsq_pop;

    assign mrsq_push = mem_rsp_valid && mem_rsp_ready;
    assign mem_rsp_ready = !mrsq_full; 

    VX_fifo_queue #(
        .DATAW    (MEM_TAG_WIDTH + `CACHE_LINE_WIDTH), 
        .SIZE     (MRSQ_SIZE),
        .BUFFERED (1)
    ) mem_rsp_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (mrsq_push),
        .pop        (mrsq_pop),
        .data_in    ({mem_rsp_tag, mem_rsp_data}),                
        .data_out   ({mem_rsp_tag_qual, mem_rsp_data_qual}),
        .empty      (mrsq_empty),
        .full       (mrsq_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );
    
    if (NUM_BANKS == 1) begin
        `UNUSED_VAR (mem_rsp_tag_qual)
        assign mrsq_pop = !mrsq_empty && per_bank_mem_rsp_ready;
    end else begin
        assign mrsq_pop = !mrsq_empty && per_bank_mem_rsp_ready[`MEM_ADDR_BANK(mem_rsp_tag_qual)];
    end

    ///////////////////////////////////////////////////////////////////////////

    VX_flush_ctrl #( 
        .CACHE_SIZE (CACHE_SIZE),
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS)
    ) flush_ctrl (
        .clk       (clk),
        .reset     (reset || flush),
        .addr_out  (flush_addr),
        .valid_out (flush_enable)
    );

    ///////////////////////////////////////////////////////////////////////////

    VX_cache_core_req_bank_sel #(
        .CACHE_ID        (CACHE_ID),
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS       (NUM_BANKS),
        .NUM_PORTS       (NUM_PORTS),
        .WORD_SIZE       (WORD_SIZE),
        .NUM_REQS        (NUM_REQS),
        .CORE_TAG_WIDTH  (CORE_TAG_WIDTH),
        .BANK_ADDR_OFFSET(BANK_ADDR_OFFSET)
    ) core_req_bank_sel (        
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE        
        .bank_stalls(perf_cache_if.bank_stalls),
    `endif     
        .core_req_valid (core_req_valid),
        .core_req_rw    (core_req_rw), 
        .core_req_addr  (core_req_addr),
        .core_req_byteen(core_req_byteen),
        .core_req_data  (core_req_data),
        .core_req_tag   (core_req_tag),
        .core_req_ready (core_req_ready),
        .per_bank_core_req_valid (per_bank_core_req_valid), 
        .per_bank_core_req_rw    (per_bank_core_req_rw),
        .per_bank_core_req_addr  (per_bank_core_req_addr),
        .per_bank_core_req_wsel  (per_bank_core_req_wsel),
        .per_bank_core_req_byteen(per_bank_core_req_byteen),
        .per_bank_core_req_data  (per_bank_core_req_data),
        .per_bank_core_req_tag   (per_bank_core_req_tag),
        .per_bank_core_req_tid   (per_bank_core_req_tid),
        .per_bank_core_req_ready (per_bank_core_req_ready)
    );

    ///////////////////////////////////////////////////////////////////////////
    
    for (genvar i = 0; i < NUM_BANKS; i++) begin
        wire [NUM_PORTS-1:0]        curr_bank_core_req_valid;
        wire [NUM_PORTS-1:0][`UP(`WORD_SELECT_BITS)-1:0] curr_bank_core_req_wsel;
        wire [NUM_PORTS-1:0][WORD_SIZE-1:0] curr_bank_core_req_byteen;
        wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] curr_bank_core_req_data;
        wire [NUM_PORTS-1:0][`REQS_BITS-1:0] curr_bank_core_req_tid;     
        wire                        curr_bank_core_req_rw;  
        wire [`LINE_ADDR_WIDTH-1:0] curr_bank_core_req_addr;
        wire [CORE_TAG_WIDTH-1:0]   curr_bank_core_req_tag;  
        wire                        curr_bank_core_req_ready;

        wire                        curr_bank_core_rsp_valid;
        wire [NUM_PORTS-1:0]        curr_bank_core_rsp_pmask;        
        wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] curr_bank_core_rsp_data;
        wire [NUM_PORTS-1:0][`REQS_BITS-1:0] curr_bank_core_rsp_tid;
        wire [CORE_TAG_WIDTH-1:0]   curr_bank_core_rsp_tag;
        wire                        curr_bank_core_rsp_ready;

        wire                        curr_bank_mem_req_valid;
        wire                        curr_bank_mem_req_rw;
        wire [CACHE_LINE_SIZE-1:0]  curr_bank_mem_req_byteen;
        wire [`LINE_ADDR_WIDTH-1:0] curr_bank_mem_req_addr;
        wire[`CACHE_LINE_WIDTH-1:0] curr_bank_mem_req_data;
        wire                        curr_bank_mem_req_ready;

        wire                        curr_bank_mem_rsp_valid;    
        wire [`LINE_ADDR_WIDTH-1:0] curr_bank_mem_rsp_addr;        
        wire [`CACHE_LINE_WIDTH-1:0] curr_bank_mem_rsp_data;
        wire                        curr_bank_mem_rsp_ready;

        // Core Req
        assign curr_bank_core_req_valid   = per_bank_core_req_valid[i];
        assign curr_bank_core_req_addr    = per_bank_core_req_addr[i];
        assign curr_bank_core_req_rw      = per_bank_core_req_rw[i];
        assign curr_bank_core_req_wsel    = per_bank_core_req_wsel[i];
        assign curr_bank_core_req_byteen  = per_bank_core_req_byteen[i];
        assign curr_bank_core_req_data    = per_bank_core_req_data[i];
        assign curr_bank_core_req_tag     = per_bank_core_req_tag[i];
        assign curr_bank_core_req_tid     = per_bank_core_req_tid[i];
        assign per_bank_core_req_ready[i] = curr_bank_core_req_ready;

        // Core WB
        assign curr_bank_core_rsp_ready   = per_bank_core_rsp_ready[i];
        assign per_bank_core_rsp_valid[i] = curr_bank_core_rsp_valid;
        assign per_bank_core_rsp_pmask[i] = curr_bank_core_rsp_pmask;
        assign per_bank_core_rsp_tid  [i] = curr_bank_core_rsp_tid;
        assign per_bank_core_rsp_tag  [i] = curr_bank_core_rsp_tag;
        assign per_bank_core_rsp_data [i] = curr_bank_core_rsp_data;

        // Memory request            
        assign per_bank_mem_req_valid[i] = curr_bank_mem_req_valid;          
        assign per_bank_mem_req_rw[i] = curr_bank_mem_req_rw;
        assign per_bank_mem_req_byteen[i] = curr_bank_mem_req_byteen;
        if (NUM_BANKS == 1) begin  
            assign per_bank_mem_req_addr[i] = curr_bank_mem_req_addr;
        end else begin
            assign per_bank_mem_req_addr[i] = `LINE_TO_MEM_ADDR(curr_bank_mem_req_addr, i); 
        end
        assign per_bank_mem_req_data[i] = curr_bank_mem_req_data;
        assign curr_bank_mem_req_ready = per_bank_mem_req_ready[i];

        // Memory response
        if (NUM_BANKS == 1) begin
            assign curr_bank_mem_rsp_valid = !mrsq_empty;
            assign curr_bank_mem_rsp_addr  = mem_rsp_tag_qual;
        end else begin
            assign curr_bank_mem_rsp_valid = !mrsq_empty && (`MEM_ADDR_BANK(mem_rsp_tag_qual) == i);
            assign curr_bank_mem_rsp_addr  = `MEM_TO_LINE_ADDR(mem_rsp_tag_qual); 
        end
        assign curr_bank_mem_rsp_data    = mem_rsp_data_qual;
        assign per_bank_mem_rsp_ready[i] = curr_bank_mem_rsp_ready;
        
        VX_bank #(                
            .BANK_ID            (i),
            .CACHE_ID           (CACHE_ID),
            .CACHE_SIZE         (CACHE_SIZE),
            .CACHE_LINE_SIZE    (CACHE_LINE_SIZE),
            .NUM_BANKS          (NUM_BANKS),
            .NUM_PORTS          (NUM_PORTS),
            .WORD_SIZE          (WORD_SIZE),
            .NUM_REQS           (NUM_REQS),
            .CREQ_SIZE          (CREQ_SIZE),
            .MSHR_SIZE          (MSHR_SIZE),
            .MREQ_SIZE          (MREQ_SIZE),
            .WRITE_ENABLE       (WRITE_ENABLE),
            .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),                
            .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
            .BANK_ADDR_OFFSET   (BANK_ADDR_OFFSET),
            .IN_ORDER_MEM       (IN_ORDER_MEM)
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
            .core_req_rw        (curr_bank_core_req_rw),
            .core_req_byteen    (curr_bank_core_req_byteen),              
            .core_req_addr      (curr_bank_core_req_addr),
            .core_req_wsel      (curr_bank_core_req_wsel),
            .core_req_data      (curr_bank_core_req_data),
            .core_req_tag       (curr_bank_core_req_tag),
            .core_req_tid       (curr_bank_core_req_tid),
            .core_req_ready     (curr_bank_core_req_ready),

            // Core response                
            .core_rsp_valid     (curr_bank_core_rsp_valid),
            .core_rsp_pmask     (curr_bank_core_rsp_pmask),
            .core_rsp_tid       (curr_bank_core_rsp_tid),
            .core_rsp_data      (curr_bank_core_rsp_data),
            .core_rsp_tag       (curr_bank_core_rsp_tag),
            .core_rsp_ready     (curr_bank_core_rsp_ready),

            // Memory request
            .mem_req_valid      (curr_bank_mem_req_valid),
            .mem_req_rw         (curr_bank_mem_req_rw),
            .mem_req_byteen     (curr_bank_mem_req_byteen),
            .mem_req_addr       (curr_bank_mem_req_addr),
            .mem_req_data       (curr_bank_mem_req_data),   
            .mem_req_ready      (curr_bank_mem_req_ready),

            // Memory response
            .mem_rsp_valid      (curr_bank_mem_rsp_valid),    
            .mem_rsp_addr       (curr_bank_mem_rsp_addr),            
            .mem_rsp_data       (curr_bank_mem_rsp_data),
            .mem_rsp_ready      (curr_bank_mem_rsp_ready),

            // flush    
            .flush_enable       (flush_enable),
            .flush_addr         (flush_addr)
        );
    end   

    VX_cache_core_rsp_merge #(
        .CACHE_ID           (CACHE_ID),
        .NUM_BANKS          (NUM_BANKS),
        .NUM_PORTS          (NUM_PORTS),
        .WORD_SIZE          (WORD_SIZE),
        .NUM_REQS           (NUM_REQS),
        .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),        
        .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS)
    ) core_rsp_merge (
        .clk                     (clk),
        .reset                   (reset),                    
        .per_bank_core_rsp_valid (per_bank_core_rsp_valid),   
        .per_bank_core_rsp_pmask (per_bank_core_rsp_pmask),   
        .per_bank_core_rsp_data  (per_bank_core_rsp_data),
        .per_bank_core_rsp_tag   (per_bank_core_rsp_tag),
        .per_bank_core_rsp_tid   (per_bank_core_rsp_tid),   
        .per_bank_core_rsp_ready (per_bank_core_rsp_ready),
        .core_rsp_valid          (core_rsp_valid),      
        .core_rsp_tag            (core_rsp_tag),
        .core_rsp_data           (core_rsp_data),  
        .core_rsp_ready          (core_rsp_ready)
    ); 

    wire [NUM_BANKS-1:0][(`MEM_ADDR_WIDTH + 1 + CACHE_LINE_SIZE + `CACHE_LINE_WIDTH)-1:0] data_in;
    for (genvar i = 0; i < NUM_BANKS; i++) begin
        assign data_in[i] = {per_bank_mem_req_addr[i], per_bank_mem_req_rw[i], per_bank_mem_req_byteen[i], per_bank_mem_req_data[i]};
    end

    VX_stream_arbiter #(
        .NUM_REQS (NUM_BANKS),
        .DATAW    (`MEM_ADDR_WIDTH + 1 + CACHE_LINE_SIZE + `CACHE_LINE_WIDTH),
        .BUFFERED (1)
    ) mem_req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_bank_mem_req_valid),
        .data_in   (data_in),
        .ready_in  (per_bank_mem_req_ready),   
        .valid_out (mem_req_valid),   
        .data_out  ({mem_req_addr, mem_req_rw, mem_req_byteen, mem_req_data}),
        .ready_out (mem_req_ready)
    );

    assign mem_req_tag = mem_req_addr;

`ifdef PERF_ENABLE
    // per cycle: core_reads, core_writes
    reg [($clog2(NUM_REQS+1)-1):0] perf_core_reads_per_cycle;
    reg [($clog2(NUM_REQS+1)-1):0] perf_core_writes_per_cycle;
    reg [($clog2(NUM_REQS+1)-1):0] perf_crsp_stall_per_cycle;

    assign perf_core_reads_per_cycle  = $countones(core_req_valid & core_req_ready & ~core_req_rw);
    assign perf_core_writes_per_cycle = $countones(core_req_valid & core_req_ready & core_req_rw);
    
    if (CORE_TAG_ID_BITS != 0) begin
        assign perf_crsp_stall_per_cycle = $countones(core_rsp_valid & {NUM_REQS{!core_rsp_ready}});
    end else begin
        assign perf_crsp_stall_per_cycle = $countones(core_rsp_valid & ~core_rsp_ready);
    end

    // per cycle: read misses, write misses, msrq stalls, pipeline stalls
    reg [($clog2(NUM_BANKS+1)-1):0] perf_read_miss_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_write_miss_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_mshr_stall_per_cycle;
    reg [($clog2(NUM_BANKS+1)-1):0] perf_pipe_stall_per_cycle; 
    
    assign perf_read_miss_per_cycle  = $countones(perf_read_miss_per_bank);
    assign perf_write_miss_per_cycle = $countones(perf_write_miss_per_bank);    
    assign perf_mshr_stall_per_cycle = $countones(perf_mshr_stall_per_bank);
    assign perf_pipe_stall_per_cycle = $countones(perf_pipe_stall_per_bank);

    reg [`PERF_CTR_BITS-1:0] perf_core_reads;
    reg [`PERF_CTR_BITS-1:0] perf_core_writes;
    reg [`PERF_CTR_BITS-1:0] perf_read_misses;
    reg [`PERF_CTR_BITS-1:0] perf_write_misses;
    reg [`PERF_CTR_BITS-1:0] perf_mshr_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_pipe_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_crsp_stalls;

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
            perf_core_reads   <= perf_core_reads  + `PERF_CTR_BITS'(perf_core_reads_per_cycle);
            perf_core_writes  <= perf_core_writes + `PERF_CTR_BITS'(perf_core_writes_per_cycle);
            perf_read_misses  <= perf_read_misses + `PERF_CTR_BITS'(perf_read_miss_per_cycle);
            perf_write_misses <= perf_write_misses+ `PERF_CTR_BITS'(perf_write_miss_per_cycle);
            perf_mshr_stalls  <= perf_mshr_stalls + `PERF_CTR_BITS'(perf_mshr_stall_per_cycle);
            perf_pipe_stalls  <= perf_pipe_stalls + `PERF_CTR_BITS'(perf_pipe_stall_per_cycle);
            perf_crsp_stalls  <= perf_crsp_stalls + `PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
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
