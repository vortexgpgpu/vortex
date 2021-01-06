`include "VX_cache_config.vh"

module VX_bank #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0, 

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE               = 1, 
    // Number of bankS
    parameter NUM_BANKS                     = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQS                      = 1, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 1, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 1, 
    // DRAM Response Queue Size
    parameter DRSQ_SIZE                     = 1, 

    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 1, 
    // DRAM Request Queue Size
    parameter DREQ_SIZE                     = 1,

    // Enable dram update
    parameter DRAM_ENABLE                   = 1,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

    // Enable write-through
    parameter WRITE_THROUGH                 = 1,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET              = 0
) (
    `SCOPE_IO_VX_bank

    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output wire perf_read_misses,
    output wire perf_write_misses,
    output wire perf_mshr_stalls,
    output wire perf_pipe_stalls,
`endif

    // Core Request    
    input wire                          core_req_valid,   
    input wire [`REQS_BITS-1:0]         core_req_tid,
    input wire                          core_req_rw,  
    input wire [WORD_SIZE-1:0]          core_req_byteen,
    input wire [`WORD_ADDR_WIDTH-1:0]   core_req_addr,
    input wire [`WORD_WIDTH-1:0]        core_req_data,
    input wire [CORE_TAG_WIDTH-1:0]     core_req_tag,
    output wire                         core_req_ready,
    
    // Core Response    
    output wire                         core_rsp_valid,
    output wire [`REQS_BITS-1:0]        core_rsp_tid,
    output wire [`WORD_WIDTH-1:0]       core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]    core_rsp_tag,
    input  wire                         core_rsp_ready,

    // DRAM request
    output wire                         dram_req_valid,
    output wire                         dram_req_rw,
    output wire [CACHE_LINE_SIZE-1:0]   dram_req_byteen,    
    output wire [`LINE_ADDR_WIDTH-1:0]  dram_req_addr,
    output wire [`CACHE_LINE_WIDTH-1:0] dram_req_data,
    input  wire                         dram_req_ready,
    
    // DRAM response
    input  wire                         dram_rsp_valid,  
    input  wire [`CACHE_LINE_WIDTH-1:0] dram_rsp_data,
    output wire                         dram_rsp_ready
);

    localparam MSHR_SIZE_BITS = $clog2(MSHR_SIZE+1);

`ifdef DBG_CACHE_REQ_INFO
    /* verilator lint_off UNUSED */
    wire [31:0]         debug_pc_st0,  debug_pc_st1,  debug_pc_st01;
    wire [`NW_BITS-1:0] debug_wid_st0, debug_wid_st1, debug_wid_st01;
    /* verilator lint_on UNUSED */
`endif

    wire drsq_pop;
    wire drsq_empty, drsp_empty_next;

    wire [`CACHE_LINE_WIDTH-1:0] drsq_filldata;    

    wire drsq_push = dram_rsp_valid && dram_rsp_ready;
    
    if (DRAM_ENABLE) begin
        wire drsq_full;
        assign dram_rsp_ready = !drsq_full;

        VX_input_queue #(
            .DATAW    ($bits(dram_rsp_data)), 
            .SIZE     (DRSQ_SIZE),
            .FASTRAM  (1)
        ) dram_rsp_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (drsq_push),
            .pop     (drsq_pop),
            .data_in (dram_rsp_data),        
            .data_out(drsq_filldata),
            .empty   (drsq_empty),
            `UNUSED_PIN (data_out_next),
            .empty_next(drsp_empty_next),
            .full    (drsq_full),
            `UNUSED_PIN (size)
        );
    end else begin
        `UNUSED_VAR (dram_rsp_valid)
        `UNUSED_VAR (dram_rsp_data)
        assign drsq_empty      = 1;
        assign drsp_empty_next = 1;
        assign drsq_filldata   = 0;
        assign dram_rsp_ready  = 0;        
    end

    wire                        creq_pop;    
    wire                        creq_full;
    wire                        creq_empty;
    wire [`REQS_BITS-1:0]       creq_tid_next;
    wire                        creq_rw_next;  
    wire [WORD_SIZE-1:0]        creq_byteen_next;
`IGNORE_WARNINGS_BEGIN
    wire [`WORD_ADDR_WIDTH-1:0] creq_addr_next_unqual;
`IGNORE_WARNINGS_END
    wire [`LINE_ADDR_WIDTH-1:0] creq_addr_next;
    wire [`UP(`WORD_SELECT_BITS)-1:0] creq_wsel_next;
    wire [`WORD_WIDTH-1:0]      creq_writeword_next;
    wire [CORE_TAG_WIDTH-1:0]   creq_tag_next;

    wire creq_push = (| core_req_valid) && core_req_ready;
    assign core_req_ready = !creq_full;

    if (BANK_ADDR_OFFSET == 0) begin
        assign creq_addr_next = `LINE_SELECT_ADDR0(creq_addr_next_unqual);
    end else begin
        assign creq_addr_next = `LINE_SELECT_ADDRX(creq_addr_next_unqual);
    end    

    if (`WORD_SELECT_BITS != 0) begin
        assign creq_wsel_next = creq_addr_next_unqual[`WORD_SELECT_BITS-1:0];
    end else begin
        assign creq_wsel_next = 0;
    end  

    VX_input_queue #(
        .DATAW    (CORE_TAG_WIDTH + `REQS_BITS + 1 + WORD_SIZE + `WORD_ADDR_WIDTH + `WORD_WIDTH), 
        .SIZE     (CREQ_SIZE),
        .FASTRAM  (1)
    ) core_req_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (creq_push),
        .pop     (creq_pop),
        .data_in ({core_req_tag, core_req_tid, core_req_rw, core_req_byteen, core_req_addr, core_req_data}),        
        .data_out_next({creq_tag_next, creq_tid_next, creq_rw_next, creq_byteen_next, creq_addr_next_unqual, creq_writeword_next}),
        `UNUSED_PIN (empty_next),
        `UNUSED_PIN (data_out),
        .empty   (creq_empty),        
        .full    (creq_full),
        `UNUSED_PIN (size)
    );      

    wire                            mshr_valid;
    wire                            mshr_valid_next;
    wire [`REQS_BITS-1:0]           mshr_tid_next;
    wire [`LINE_ADDR_WIDTH-1:0]     mshr_addr_next;
    wire [`UP(`WORD_SELECT_BITS)-1:0] mshr_wsel_next;
    wire [`WORD_WIDTH-1:0]          mshr_writeword_next;
    wire [`REQ_TAG_WIDTH-1:0]       mshr_tag_next;
    wire                            mshr_rw_next;  
    wire [WORD_SIZE-1:0]            mshr_byteen_next;
    
    reg [`LINE_ADDR_WIDTH-1:0]      creq_addr;
    reg [`UP(`WORD_SELECT_BITS)-1:0] creq_wsel;    
    reg [`REQ_TAG_WIDTH-1:0]        creq_tag;
    reg                             creq_mem_rw;
    reg [WORD_SIZE-1:0]             creq_byteen;
    reg [`WORD_WIDTH-1:0]           creq_writeword;
    reg [`REQS_BITS-1:0]            creq_tid;

    always @(posedge clk) begin   
        creq_addr      <= (mshr_valid_next || !drsp_empty_next) ? mshr_addr_next : creq_addr_next;
        creq_wsel      <= mshr_valid_next ? mshr_wsel_next : creq_wsel_next;           
        creq_mem_rw    <= mshr_valid_next ? mshr_rw_next : creq_rw_next;
        creq_byteen    <= mshr_valid_next ? mshr_byteen_next : creq_byteen_next;              
        creq_writeword <= mshr_valid_next ? mshr_writeword_next : creq_writeword_next;
        creq_tid       <= mshr_valid_next ? mshr_tid_next : creq_tid_next;
        creq_tag       <= mshr_valid_next ? `REQ_TAG_WIDTH'(mshr_tag_next) : `REQ_TAG_WIDTH'(creq_tag_next);
    end

    wire                            mshr_pop;
    reg [MSHR_SIZE_BITS-1:0]        mshr_pending_size;   
    wire [MSHR_SIZE_BITS-1:0]       mshr_pending_size_n;   
    reg                             mshr_going_full;
    wire                            mshr_pending_hazard_unqual_st0;
    
    wire                            valid_st0, valid_st1;    
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_mshr_st0, is_mshr_st1;
    wire [`LINE_ADDR_WIDTH-1:0]     addr_st0, addr_st1;
    wire [`UP(`WORD_SELECT_BITS)-1:0] wsel_st0, wsel_st1;    
    wire [`CACHE_LINE_WIDTH-1:0]    readdata_st0, readdata_st1;    
    wire [`WORD_WIDTH-1:0]          writeword_st0, writeword_st1;
    wire [`CACHE_LINE_WIDTH-1:0]    filldata_st0, filldata_st1;
    wire [`TAG_SELECT_BITS-1:0]     readtag_st0, readtag_st1;   
    wire                            miss_st0, miss_st1; 
    wire                            force_miss_st0, force_miss_st1;
    wire                            dirty_st0;
    wire [CACHE_LINE_SIZE-1:0]      dirtyb_st0, dirtyb_st1;
    wire [`REQ_TAG_WIDTH-1:0]       tag_st0, tag_st1;
    wire                            mem_rw_st0, mem_rw_st1;
    wire [WORD_SIZE-1:0]            byteen_st0, byteen_st1;
    wire [`REQS_BITS-1:0]           req_tid_st0, req_tid_st1;    
    wire                            do_writeback_st0, do_writeback_st1;
    wire                            writeen_unqual_st0, writeen_unqual_st1;
    wire                            mshr_push_unqual_st0, mshr_push_unqual_st1;
    wire                            dreq_push_unqual_st0, dreq_push_unqual_st1;
    wire                            writeen_st1;
    wire                            core_req_hit_st1;

    wire                            valid_st01;
    wire                            writeen_st01;
    wire [`LINE_ADDR_WIDTH-1:0]     addr_st01;
    wire [`UP(`WORD_SELECT_BITS)-1:0] wsel_st01;
    wire [WORD_SIZE-1:0]            byteen_st01;
    wire [`WORD_WIDTH-1:0]          writeword_st01;
    wire [`REQ_TAG_WIDTH-1:0]       tag_st01;
    
    wire mshr_push_stall;
    wire crsq_push_stall;    
    wire dreq_push_stall;
    wire pipeline_stall;

    wire is_mshr_miss_st1 = valid_st1 && is_mshr_st1 && (miss_st1 || force_miss_st1);

    wire creq_commit = valid_st1 && !is_fill_st1
                    && (core_req_hit_st1 || (WRITE_THROUGH && mem_rw_st1))
                    && !pipeline_stall;

    // determine which queue to pop next in piority order
    wire mshr_pop_unqual = mshr_valid;
    wire drsq_pop_unqual = !mshr_pop_unqual && !drsq_empty;
    wire creq_pop_unqual = !mshr_pop_unqual && !drsq_pop_unqual && !creq_empty && !mshr_going_full;

    assign mshr_pop = mshr_pop_unqual && !pipeline_stall 
                   && !is_mshr_miss_st1; // stop if previous request was a miss
    assign drsq_pop = drsq_pop_unqual && !pipeline_stall;
    assign creq_pop = creq_pop_unqual && !pipeline_stall;

    // MSHR pending size    
    assign mshr_pending_size_n = mshr_pending_size + 
                ((creq_pop && !creq_commit) ? 1 : ((creq_commit && !creq_pop) ? -1 : 0));
    always @(posedge clk) begin
        if (reset) begin
            mshr_pending_size <= 0;
            mshr_going_full   <= 0;
        end else begin
            mshr_pending_size <= mshr_pending_size_n;
            mshr_going_full   <= (mshr_pending_size_n == MSHR_SIZE_BITS'(MSHR_SIZE));
        end        
    end

    assign valid_st0     = mshr_pop || drsq_pop || creq_pop;
    assign is_mshr_st0   = mshr_pop_unqual;
    assign is_fill_st0   = drsq_pop_unqual;
    assign addr_st0      = creq_addr;
    assign wsel_st0      = creq_wsel;
    assign mem_rw_st0    = creq_mem_rw;
    assign byteen_st0    = creq_byteen;
    assign writeword_st0 = creq_writeword;
    assign req_tid_st0   = creq_tid;
    assign tag_st0       = creq_tag;
    assign filldata_st0  = drsq_filldata;

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st0, debug_wid_st0} = tag_st0[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin
        assign {debug_pc_st0, debug_wid_st0} = 0;
    end
`endif

if (DRAM_ENABLE) begin       
    VX_tag_access #(
        .BANK_ID        (BANK_ID),
        .CACHE_ID       (CACHE_ID),
        .CORE_TAG_ID_BITS(CORE_TAG_ID_BITS),
        .CACHE_SIZE     (CACHE_SIZE),
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),        
        .WRITE_ENABLE   (WRITE_ENABLE),
        .BANK_ADDR_OFFSET (BANK_ADDR_OFFSET)
     ) tag_access (
        .clk            (clk),
        .reset          (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc       (debug_pc_st0),
        .debug_wid      (debug_wid_st0),
    `endif

        .stall          (pipeline_stall),

        // read/Fill
        .lookup_in      (creq_pop || mshr_pop),
        .raddr_in       (addr_st0),        
        .do_fill_in     (drsq_pop),
        .miss_out       (miss_st0),
        .readtag_out    (readtag_st0),        
        .dirty_out      (dirty_st0),

        // write
        .waddr_in       (addr_st1),
        .writeen_in     (valid_st1 && writeen_st1)
    );

    assign valid_st01     = valid_st1;
    assign writeen_st01   = writeen_st1;
    assign addr_st01      = addr_st1;
    assign wsel_st01      = wsel_st1;
    assign byteen_st01    = byteen_st1;
    assign writeword_st01 = writeword_st1;
    assign tag_st01       = tag_st1;

    // redundant fills
    wire is_redundant_fill = is_fill_st0 && !miss_st0;

    // we have a miss in mshr for the current address
    wire mshr_pending_hazard_st0 = mshr_pending_hazard_unqual_st0 
                                || (valid_st1 && (miss_st1 || force_miss_st1) && (addr_st0 == addr_st1));

    // force miss to ensure commit order when a new request has pending previous requests to same block
    assign force_miss_st0 = !is_mshr_st0 && !is_fill_st0 && mshr_pending_hazard_st0;
    
    assign writeen_unqual_st0 = (!is_fill_st0 && !miss_st0 && mem_rw_st0) 
                             || (is_fill_st0 && !is_redundant_fill);

    wire send_fill_req_st0 = !is_fill_st0 && miss_st0
                          && !(WRITE_THROUGH && mem_rw_st0);

    assign do_writeback_st0 = (WRITE_THROUGH && !is_fill_st0 && mem_rw_st0) 
                           || (!WRITE_THROUGH && is_fill_st0 && dirty_st0 && !is_redundant_fill);

    assign dreq_push_unqual_st0 = send_fill_req_st0 || do_writeback_st0;

    assign mshr_push_unqual_st0 = !is_fill_st0 && !(WRITE_THROUGH && mem_rw_st0);
    
end else begin

    `UNUSED_VAR (mshr_pending_hazard_unqual_st0)
    `UNUSED_VAR (drsq_push)
    `UNUSED_VAR (dirty_st0)
    `UNUSED_VAR (writeen_st1)

`ifdef DBG_CACHE_REQ_INFO
    assign debug_pc_st1  = debug_pc_st0;
    assign debug_wid_st1 = debug_wid_st0;
`endif

    assign valid_st01       = valid_st0;
    assign writeen_st01     = mem_rw_st0;
    assign addr_st01        = addr_st0;
    assign wsel_st01        = wsel_st0;
    assign byteen_st01      = byteen_st0;
    assign writeword_st01   = writeword_st0;
    assign tag_st01         = tag_st0;
    
    assign miss_st0             = 0;
    assign dirty_st0            = 0;    
    assign force_miss_st0       = 0;
    assign readtag_st0          = 0;   
    assign do_writeback_st0     = 0;
    assign writeen_unqual_st0   = mem_rw_st0;
    assign dreq_push_unqual_st0 = 0;
    assign mshr_push_unqual_st0 = 0;
end 

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `UP(`WORD_SELECT_BITS) + CACHE_LINE_SIZE + `CACHE_LINE_WIDTH + `WORD_WIDTH + `TAG_SELECT_BITS + `CACHE_LINE_WIDTH + 1 + WORD_SIZE + `REQS_BITS + `REQ_TAG_WIDTH),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!pipeline_stall),
        .data_in  ({valid_st0, is_mshr_st0, is_fill_st0, writeen_unqual_st0, mshr_push_unqual_st0, dreq_push_unqual_st0, do_writeback_st0, miss_st0, force_miss_st0, addr_st0, wsel_st0, dirtyb_st0, readdata_st0, writeword_st0, readtag_st0, filldata_st0, mem_rw_st0, byteen_st0, req_tid_st0, tag_st0}),
        .data_out ({valid_st1, is_mshr_st1, is_fill_st1, writeen_unqual_st1, mshr_push_unqual_st1, dreq_push_unqual_st1, do_writeback_st1, miss_st1, force_miss_st1, addr_st1, wsel_st1, dirtyb_st1, readdata_st1, writeword_st1, readtag_st1, filldata_st1, mem_rw_st1, byteen_st1, req_tid_st1, tag_st1})
    );

    assign core_req_hit_st1 = !is_fill_st1 && !miss_st1 && !force_miss_st1;
    
    assign writeen_st1 = writeen_unqual_st1 && (is_fill_st1 || !force_miss_st1);

    wire dreq_push_st1 = dreq_push_unqual_st1 && (do_writeback_st1 || !force_miss_st1);

    wire mshr_push_st1 = mshr_push_unqual_st1 && (miss_st1 || force_miss_st1);

    wire crsq_push_st1 = core_req_hit_st1 && !mem_rw_st1;

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st01, debug_wid_st01} = tag_st01[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin        
        assign {debug_pc_st01, debug_wid_st01} = 0;
    end
`endif
    `UNUSED_VAR (tag_st01)

    VX_data_access #(
        .BANK_ID        (BANK_ID),
        .CACHE_ID       (CACHE_ID),
        .CORE_TAG_ID_BITS(CORE_TAG_ID_BITS),
        .CACHE_SIZE     (CACHE_SIZE),
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .DRAM_ENABLE    (DRAM_ENABLE),
        .WORD_SIZE      (WORD_SIZE),
        .WRITE_ENABLE   (WRITE_ENABLE),
        .WRITE_THROUGH  (WRITE_THROUGH)
     ) data_access (
        .clk            (clk),
        .reset          (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .rdebug_pc      (debug_pc_st0),
        .rdebug_wid     (debug_wid_st0),
        .wdebug_pc      (debug_pc_st01),
        .wdebug_wid     (debug_wid_st01),
    `endif
        .stall          (pipeline_stall),

        // reading
        .readen_in      (valid_st0 && !mem_rw_st0 && !is_fill_st0),
        .raddr_in       (addr_st0),    
        .readdata_out   (readdata_st0),
        .dirtyb_out     (dirtyb_st0),

        // writing
        .writeen_in     (valid_st01 && writeen_st01),
        .waddr_in       (addr_st01),        
        .wfill_in       (is_fill_st1),
        .wwsel_in       (wsel_st01),
        .wbyteen_in     (byteen_st01),
        .writeword_in   (writeword_st01),
        .readdata_in    (readdata_st1),
        .filldata_in    (filldata_st1)
    ); 

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st1, debug_wid_st1} = tag_st1[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin        
        assign {debug_pc_st1, debug_wid_st1} = 0;
    end
`endif

    wire mshr_push_unqual = valid_st1 && mshr_push_st1;
    assign mshr_push_stall = 0;

    wire mshr_push = mshr_push_unqual
                  && !crsq_push_stall 
                  && !dreq_push_stall;

    wire incoming_fill_st1 = (!drsq_empty && (addr_st1 == addr_st0));

    if (DRAM_ENABLE) begin

        wire mshr_dequeue_st1 = valid_st1 && is_mshr_st1 && !mshr_push_unqual && !pipeline_stall;

        // push a missed request as 'ready' if it was a forced miss that actually had a hit 
        // or the fill request for this block is comming
        wire mshr_init_ready_state_st1 = !miss_st1 || incoming_fill_st1;

        VX_miss_resrv #(
            .BANK_ID            (BANK_ID),
            .CACHE_ID           (CACHE_ID),      
            .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
            .CACHE_LINE_SIZE    (CACHE_LINE_SIZE),
            .NUM_BANKS          (NUM_BANKS),
            .WORD_SIZE          (WORD_SIZE),
            .NUM_REQS           (NUM_REQS),
            .MSHR_SIZE          (MSHR_SIZE),
            .CORE_TAG_WIDTH     (CORE_TAG_WIDTH)
        ) miss_resrv (
            .clk                (clk),
            .reset              (reset),

        `ifdef DBG_CACHE_REQ_INFO
            .deq_debug_pc       (debug_pc_st0),
            .deq_debug_wid      (debug_wid_st0),
            .enq_debug_pc       (debug_pc_st1),
            .enq_debug_wid      (debug_wid_st1),
        `endif

            // enqueue
            .enqueue            (mshr_push),        
            .enqueue_addr       (addr_st1),
            .enqueue_data       ({writeword_st1, req_tid_st1, tag_st1, mem_rw_st1, byteen_st1, wsel_st1}),
            .enqueue_is_mshr    (is_mshr_st1),
            .enqueue_ready      (mshr_init_ready_state_st1),

            // lookup
            .lookup_ready       (drsq_pop),
            .lookup_addr        (addr_st0),
            .lookup_match       (mshr_pending_hazard_unqual_st0),
            
            // schedule
            .schedule           (mshr_pop),        
            .schedule_valid     (mshr_valid),
            .schedule_valid_next(mshr_valid_next),
            .schedule_addr_next (mshr_addr_next),
            .schedule_data_next ({mshr_writeword_next, mshr_tid_next, mshr_tag_next, mshr_rw_next, mshr_byteen_next, mshr_wsel_next}),            
            `UNUSED_PIN (schedule_addr),
            `UNUSED_PIN (schedule_data),

            // dequeue
            .dequeue            (mshr_dequeue_st1)
        );
    end else begin
        `UNUSED_VAR (valid_st1)        
        `UNUSED_VAR (mshr_push)
        `UNUSED_VAR (wsel_st1)
        `UNUSED_VAR (writeword_st1)
        `UNUSED_VAR (mem_rw_st1)
        `UNUSED_VAR (byteen_st1)
        `UNUSED_VAR (incoming_fill_st1)
        assign mshr_pending_hazard_unqual_st0 = 0;
        assign mshr_valid      = 0;
        assign mshr_valid_next = 0;
        assign mshr_addr_next = 0;
        assign mshr_wsel_next = 0;
        assign mshr_writeword_next = 0;
        assign mshr_tid_next = 0;
        assign mshr_tag_next = 0;
        assign mshr_rw_next = 0;
        assign mshr_byteen_next = 0;
    end

    // Enqueue core response
     
    wire crsq_empty, crsq_full;

    wire crsq_push_unqual = valid_st1 && crsq_push_st1;
    assign crsq_push_stall = crsq_push_unqual && crsq_full;

    wire crsq_push = crsq_push_unqual
                  && !crsq_full
                  && !mshr_push_stall
                  && !dreq_push_stall;

    wire crsq_pop = core_rsp_valid && core_rsp_ready;

    wire [`REQS_BITS-1:0]     crsq_tid_st1 = req_tid_st1;
    wire [CORE_TAG_WIDTH-1:0] crsq_tag_st1 = CORE_TAG_WIDTH'(tag_st1);
    wire [`WORD_WIDTH-1:0]    crsq_data_st1;  

     if (`WORD_SELECT_BITS != 0) begin
        wire [`WORD_WIDTH-1:0] readword = readdata_st1[wsel_st1 * `WORD_WIDTH +: `WORD_WIDTH];
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign crsq_data_st1[i * 8 +: 8] = readword[i * 8 +: 8] & {8{byteen_st1[i]}};
        end
    end else begin
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign crsq_data_st1[i * 8 +: 8] = readdata_st1[i * 8 +: 8] & {8{byteen_st1[i]}};
        end
    end
  
    VX_fifo_queue #(
        .DATAW    (`REQS_BITS + CORE_TAG_WIDTH + `WORD_WIDTH), 
        .SIZE     (CRSQ_SIZE),
        .BUFFERED (1),    
        .FASTRAM  (1)
    ) core_rsp_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (crsq_push),
        .pop     (crsq_pop),
        .data_in ({crsq_tid_st1, crsq_tag_st1, crsq_data_st1}),
        .data_out({core_rsp_tid, core_rsp_tag, core_rsp_data}),
        .empty   (crsq_empty),
        .full    (crsq_full),
        `UNUSED_PIN (size)
    );

    assign core_rsp_valid = !crsq_empty;

    // Enqueue DRAM request

    wire dreq_empty, dreq_full;
    
    wire dreq_push_unqual = valid_st1 && dreq_push_st1;
    assign dreq_push_stall = dreq_push_unqual && dreq_full;
    
    wire dreq_push = dreq_push_unqual                  
                  && !dreq_full
                  && !mshr_push_stall
                  && !crsq_push_stall;

    wire dreq_pop = dram_req_valid && dram_req_ready;

    wire writeback = WRITE_ENABLE && do_writeback_st1;

    wire [`LINE_ADDR_WIDTH-1:0] dreq_addr = (WRITE_THROUGH || !writeback) ? addr_st1 :
                                                {readtag_st1, addr_st1[`LINE_SELECT_BITS-1:0]};

    wire [`CACHE_LINE_WIDTH-1:0] dreq_data;
    wire [CACHE_LINE_SIZE-1:0] dreq_byteen, dreq_byteen_unqual;

    if (WRITE_THROUGH) begin
        `UNUSED_VAR (dirtyb_st1)
        if (`WORD_SELECT_BITS != 0) begin
            for (genvar i = 0; i < `WORDS_PER_LINE; i++) begin
                assign dreq_byteen_unqual[i * WORD_SIZE +: WORD_SIZE] = (wsel_st1 == `WORD_SELECT_BITS'(i)) ? byteen_st1 : {WORD_SIZE{1'b0}};
                assign dreq_data[i * `WORD_WIDTH +: `WORD_WIDTH] = writeword_st1;
            end
        end else begin
            assign dreq_byteen_unqual = byteen_st1;
            assign dreq_data = writeword_st1;
        end
    end else begin
        assign dreq_byteen_unqual = dirtyb_st1;
        assign dreq_data = readdata_st1;
    end

    assign dreq_byteen = writeback ? dreq_byteen_unqual : {CACHE_LINE_SIZE{1'b1}};

    if (DRAM_ENABLE) begin 
        always @(posedge clk) begin        
            assert (!(dreq_push && !do_writeback_st1 && incoming_fill_st1))
                else $error("%t: incoming fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));
        end

        VX_fifo_queue #(
            .DATAW    (1 + CACHE_LINE_SIZE + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH), 
            .SIZE     (DREQ_SIZE),
            .BUFFERED (1),
            .FASTRAM  (1)
        ) dram_req_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (dreq_push),
            .pop     (dreq_pop),
            .data_in ({writeback,   dreq_byteen,     dreq_addr,     dreq_data}),        
            .data_out({dram_req_rw, dram_req_byteen, dram_req_addr, dram_req_data}),
            .empty   (dreq_empty),
            .full    (dreq_full),
            `UNUSED_PIN (size)
        );
    end else begin
        `UNUSED_VAR (dreq_push)
        `UNUSED_VAR (dreq_pop)
        `UNUSED_VAR (dreq_addr)
        `UNUSED_VAR (dreq_data)
        `UNUSED_VAR (dreq_byteen)
        `UNUSED_VAR (readtag_st1)
        `UNUSED_VAR (dirtyb_st1)
        `UNUSED_VAR (readdata_st1)         
        `UNUSED_VAR (writeback)
        `UNUSED_VAR (dram_req_ready)
        assign dreq_empty   = 1;       
        assign dreq_full    = 0;
        assign dram_req_rw  = 0;
        assign dram_req_byteen = 0;
        assign dram_req_addr = 0;
        assign dram_req_data = 0;
    end  

    assign dram_req_valid = !dreq_empty;    

    // bank pipeline stall
    assign pipeline_stall = mshr_push_stall
                         || crsq_push_stall 
                         || dreq_push_stall;
                         
    `SCOPE_ASSIGN (valid_st0, valid_st0);
    `SCOPE_ASSIGN (valid_st1, valid_st1);
    `SCOPE_ASSIGN (is_fill_st0, is_fill_st0);
    `SCOPE_ASSIGN (is_mshr_st0, is_mshr_st0);
    `SCOPE_ASSIGN (miss_st0,    miss_st0);
    `SCOPE_ASSIGN (dirty_st0,   dirty_st0);
    `SCOPE_ASSIGN (force_miss_st0, force_miss_st0);
    `SCOPE_ASSIGN (mshr_push,   mshr_push);    
    `SCOPE_ASSIGN (pipeline_stall, pipeline_stall);
    `SCOPE_ASSIGN (addr_st0, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID));
    `SCOPE_ASSIGN (addr_st1, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));

`ifdef PERF_ENABLE
    assign perf_read_misses  = !pipeline_stall && miss_st1 && !is_mshr_st1 && !mem_rw_st1;
    assign perf_write_misses = !pipeline_stall && miss_st1 && !is_mshr_st1 && mem_rw_st1;
    assign perf_mshr_stalls  = mshr_going_full;
    assign perf_pipe_stalls  = pipeline_stall || mshr_going_full;
`endif

`ifdef DBG_PRINT_CACHE_BANK
    always @(posedge clk) begin        
        if (pipeline_stall) begin
            $display("%t: cache%0d:%0d pipeline-stall: mshr=%b, cwbq=%b, dwbq=%b", $time, CACHE_ID, BANK_ID, mshr_push_stall, crsq_push_stall, dreq_push_stall);
        end
        if (drsq_pop) begin
            $display("%t: cache%0d:%0d fill-rsp: addr=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), drsq_filldata);
        end
        if (creq_pop || mshr_pop) begin
            if (creq_mem_rw)
                $display("%t: cache%0d:%0d core-wr-req: addr=%0h, is_mshr=%b, tag=%0h, tid=%0d, byteen=%b, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), is_mshr_st0, creq_tag, creq_tid, creq_byteen, creq_writeword, debug_wid_st0, debug_pc_st0);
            else
                $display("%t: cache%0d:%0d core-rd-req: addr=%0h, is_mshr=%b, tag=%0h, tid=%0d, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), is_mshr_st0, creq_tag, creq_tid, creq_byteen, debug_wid_st0, debug_pc_st0);
        end
        if (crsq_push) begin
            $display("%t: cache%0d:%0d core-rsp: addr=%0h, tag=%0h, tid=%0d, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), crsq_tag_st1, crsq_tid_st1, crsq_data_st1, debug_wid_st1, debug_pc_st1);
        end
        if (dreq_push) begin
            if (do_writeback_st1)
                $display("%t: cache%0d:%0d writeback: addr=%0h, data=%0h, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dreq_addr, BANK_ID), dreq_data, dreq_byteen, debug_wid_st1, debug_pc_st1);
            else
                $display("%t: cache%0d:%0d fill-req: addr=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dreq_addr, BANK_ID), debug_wid_st1, debug_pc_st1);
        end
    end    
`endif

endmodule