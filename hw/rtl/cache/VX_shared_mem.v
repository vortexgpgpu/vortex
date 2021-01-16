`include "VX_cache_config.vh"

module VX_shared_mem #(
    parameter CACHE_ID                      = 0,

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 16384, 
    // Number of banks
    parameter NUM_BANKS                     = 4, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle
    parameter NUM_REQS                      = NUM_BANKS, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 4,

    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 4,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET              = 0    
 ) (
    `SCOPE_IO_VX_cache
    
    input wire clk,
    input wire reset,   

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if                            perf_cache_if,
`endif

    // Core request    
    input wire [NUM_REQS-1:0]                   core_req_valid,
    input wire [NUM_REQS-1:0]                   core_req_rw,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]    core_req_byteen,
    input wire [NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr,
    input wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]  core_req_data,
    input wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire [NUM_REQS-1:0]                  core_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                  core_rsp_valid,    
    output wire [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]            core_rsp_tag,
    input  wire                                 core_rsp_ready
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid value"))

    localparam CACHE_LINE_SIZE = WORD_SIZE;

`ifdef DBG_CACHE_REQ_INFO
    /* verilator lint_off UNUSED */
    wire [31:0]         debug_pc_st0;
    wire [`NW_BITS-1:0] debug_wid_st0;
    /* verilator lint_on UNUSED */
`endif

    wire [NUM_BANKS-1:0]                    per_bank_core_req_valid_unqual; 
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]    per_bank_core_req_tid_unqual;
    wire [NUM_BANKS-1:0]                    per_bank_core_req_rw_unqual;  
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_core_req_byteen_unqual;
    wire [NUM_BANKS-1:0][`WORD_ADDR_WIDTH-1:0] per_bank_core_req_addr_unqual;
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]   per_bank_core_req_data_unqual;
    wire [NUM_BANKS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_req_tag_unqual;
    wire [NUM_BANKS-1:0]                    per_bank_core_req_ready_unqual;
    
    VX_cache_core_req_bank_sel #(
        .CACHE_LINE_SIZE (WORD_SIZE),
        .NUM_BANKS       (NUM_BANKS),
        .WORD_SIZE       (WORD_SIZE),
        .NUM_REQS        (NUM_REQS),
        .CORE_TAG_WIDTH  (CORE_TAG_WIDTH),
        .BANK_ADDR_OFFSET(BANK_ADDR_OFFSET),
        .BUFFERED        (0)
    ) core_req_bank_sel (        
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE        
        .bank_stalls(perf_cache_if.bank_stalls),
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
        .per_bank_core_req_valid (per_bank_core_req_valid_unqual), 
        .per_bank_core_req_tid   (per_bank_core_req_tid_unqual),
        .per_bank_core_req_rw    (per_bank_core_req_rw_unqual),
        .per_bank_core_req_byteen(per_bank_core_req_byteen_unqual),
        .per_bank_core_req_addr  (per_bank_core_req_addr_unqual),
        .per_bank_core_req_tag   (per_bank_core_req_tag_unqual),
        .per_bank_core_req_data  (per_bank_core_req_data_unqual),
        .per_bank_core_req_ready (per_bank_core_req_ready_unqual)
    );

    `UNUSED_VAR (per_bank_core_req_tag_unqual)
    `UNUSED_VAR (per_bank_core_req_rw_unqual)

    wire [NUM_BANKS-1:0]                    per_bank_core_req_valid; 
    wire [NUM_BANKS-1:0][`REQS_BITS-1:0]    per_bank_core_req_tid;
    wire [NUM_REQS-1:0]                     per_bank_core_req_rw;  
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][`LINE_SELECT_BITS-1:0] per_bank_core_req_addr;    
    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0]   per_bank_core_req_data;
    wire [NUM_REQS-1:0][CORE_TAG_WIDTH-1:0] per_bank_core_req_tag;

    wire creq_push, creq_pop, creq_empty, creq_full;
    wire crsq_full;
        
    assign creq_push = (| core_req_valid) && !creq_full;
    assign creq_pop = ~creq_empty && ~crsq_full;
    
    assign per_bank_core_req_ready_unqual = {NUM_BANKS{~creq_full}}; 

    wire [NUM_REQS-1:0][`LINE_SELECT_BITS-1:0] per_bank_core_req_addr_qual;
    `UNUSED_VAR (per_bank_core_req_addr_unqual)
    for (genvar i = 0; i < NUM_REQS; i++) begin        
        wire [`LINE_ADDR_WIDTH-1:0] tmp = `LINE_SELECT_ADDRX(per_bank_core_req_addr_unqual[i]);
        assign per_bank_core_req_addr_qual[i] = tmp[`LINE_SELECT_BITS-1:0];
        `UNUSED_VAR (tmp)
    end

    VX_fifo_queue #(
        .DATAW    (NUM_BANKS * (1 + `REQS_BITS + 1 + WORD_SIZE + `LINE_SELECT_BITS + `WORD_WIDTH + CORE_TAG_WIDTH)), 
        .SIZE     (CREQ_SIZE),
        .BUFFERED (1),
        .FASTRAM  (1)
    ) core_req_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (creq_push),
        .pop     (creq_pop),
        .data_in ({per_bank_core_req_valid_unqual, 
                   per_bank_core_req_tid_unqual, 
                   per_bank_core_req_rw_unqual, 
                   per_bank_core_req_byteen_unqual, 
                   per_bank_core_req_addr_qual, 
                   per_bank_core_req_data_unqual, 
                   per_bank_core_req_tag_unqual}),
        .data_out({per_bank_core_req_valid, 
                   per_bank_core_req_tid, 
                   per_bank_core_req_rw, 
                   per_bank_core_req_byteen, 
                   per_bank_core_req_addr, 
                   per_bank_core_req_data, 
                   per_bank_core_req_tag}),
        .empty   (creq_empty),
        .full    (creq_full),
        `UNUSED_PIN (size)
    );

    wire [NUM_BANKS-1:0][`WORD_WIDTH-1:0] per_bank_core_rsp_data;    

    for (genvar i = 0; i < NUM_BANKS; i++) begin
        VX_sp_ram #(
            .DATAW(`WORD_WIDTH),
            .SIZE(`LINES_PER_BANK),
            .BYTEENW(WORD_SIZE),
            .RWCHECK(1)
        ) data (
            .clk(clk),
            .addr(per_bank_core_req_addr[i]),          
            .wren(per_bank_core_req_valid[i] && per_bank_core_req_rw[i] && ~crsq_full),  
            .byteen(per_bank_core_req_byteen[i]),
            .rden(1'b1),
            .din(per_bank_core_req_data[i]),
            .dout(per_bank_core_rsp_data[i])
        );
    end  
    
    reg [NUM_REQS-1:0] core_rsp_valid_unqual;
    reg [NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data_unqual;
    reg [CORE_TAG_WIDTH-1:0] core_rsp_tag_unqual;
    
    always @(*) begin              
        core_rsp_valid_unqual = 0;
        core_rsp_data_unqual  = 'x;        
        core_rsp_tag_unqual   = 'x;
        for (integer i = 0; i < NUM_BANKS; i++) begin 
            if (per_bank_core_req_valid[i]) begin
                core_rsp_valid_unqual[per_bank_core_req_tid[i]] = 1;
                core_rsp_data_unqual[per_bank_core_req_tid[i]]  = per_bank_core_rsp_data[i];
                core_rsp_tag_unqual = per_bank_core_req_tag[i];
            end
        end
    end    
    
`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st0, debug_wid_st0} = core_rsp_tag_unqual[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin
        assign {debug_pc_st0, debug_wid_st0} = 0;
    end
`endif
    
    wire [NUM_REQS-1:0] core_rsp_valid_tmask;
    wire crsq_push, crsq_pop, crsq_empty;

    wire core_rsp_rw = | (per_bank_core_req_valid & per_bank_core_req_rw);

    assign crsq_push = ~creq_empty && ~core_rsp_rw && ~crsq_full;    
    assign crsq_pop = ~crsq_empty && core_rsp_ready;  
    
    VX_fifo_queue #(
        .DATAW    (NUM_BANKS * (1 + `WORD_WIDTH) + CORE_TAG_WIDTH), 
        .SIZE     (CRSQ_SIZE),
        .BUFFERED (1),
        .FASTRAM  (1)
    ) core_rsp_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (crsq_push),
        .pop     (crsq_pop),
        .data_in ({core_rsp_valid_unqual, core_rsp_data_unqual, core_rsp_tag_unqual}),
        .data_out({core_rsp_valid_tmask,  core_rsp_data,        core_rsp_tag}),
        .empty   (crsq_empty),
        .full    (crsq_full),
        `UNUSED_PIN (size)
    );

    assign core_rsp_valid = core_rsp_valid_tmask & {NUM_REQS{~crsq_empty}};

`ifdef DBG_PRINT_CACHE_BANK
    always @(posedge clk) begin        
        if (crsq_full) begin
            $display("%t: cache%0d pipeline-stall", $time, CACHE_ID);
        end
        if (creq_pop) begin
            if (core_rsp_rw)
                $display("%t: cache%0d core-wr-req: tmask=%0b, addr=%0h, tag=%0h, byteen=%b, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, per_bank_core_req_valid, per_bank_core_req_addr, per_bank_core_req_tag, per_bank_core_req_byteen, per_bank_core_req_data, debug_wid_st0, debug_pc_st0);
            else
                $display("%t: cache%0d core-rd-req: tmask=%0b, addr=%0h, tag=%0h, byteen=%b, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, per_bank_core_req_valid, per_bank_core_req_addr, per_bank_core_req_tag, per_bank_core_req_byteen, per_bank_core_rsp_data, debug_wid_st0, debug_pc_st0);
        end
    end    
`endif

`ifdef PERF_ENABLE
    assign perf_cache_if.reads        = '0;
    assign perf_cache_if.writes       = '0;
    assign perf_cache_if.read_misses  = '0;
    assign perf_cache_if.write_misses = '0;
    assign perf_cache_if.mshr_stalls  = '0;
    assign perf_cache_if.pipe_stalls  = '0;
    assign perf_cache_if.crsp_stalls  = '0;
`endif

endmodule
