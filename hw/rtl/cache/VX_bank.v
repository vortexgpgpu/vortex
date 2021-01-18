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

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 1,

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
    input wire                          dram_rsp_valid,  
    input wire [`LINE_ADDR_WIDTH-1:0]   dram_rsp_addr,
    input wire [`CACHE_LINE_WIDTH-1:0]  dram_rsp_data,
    input wire                          dram_rsp_flush,
    output wire                         dram_rsp_ready
);

`ifdef DBG_CACHE_REQ_INFO
    /* verilator lint_off UNUSED */
    wire [31:0]         debug_pc_st0,  debug_pc_st1;
    wire [`NW_BITS-1:0] debug_wid_st0, debug_wid_st1;
    /* verilator lint_on UNUSED */
`endif

    wire drsq_pop;
    wire drsq_empty, drsq_empty_next;
    wire [`LINE_ADDR_WIDTH-1:0]  drsq_addr_next;
    wire [`CACHE_LINE_WIDTH-1:0] drsq_filldata_next;
    wire drsq_flush_next;

    wire drsq_push = dram_rsp_valid && dram_rsp_ready;
    
    wire drsq_full;
    assign dram_rsp_ready = !drsq_full;

    VX_fifo_queue_xt #(
        .DATAW    (`LINE_ADDR_WIDTH + $bits(dram_rsp_data) + 1), 
        .SIZE     (DRSQ_SIZE),
        .FASTRAM  (1)
    ) dram_rsp_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (drsq_push),
        .pop     (drsq_pop),
        .data_in ({dram_rsp_addr, dram_rsp_data, dram_rsp_flush}),        
        `UNUSED_PIN (data_out),
        .empty   (drsq_empty),
        .data_out_next ({drsq_addr_next, drsq_filldata_next, drsq_flush_next}),
        .empty_next (drsq_empty_next),
        .full    (drsq_full),
        `UNUSED_PIN (size)
    );

    wire                        creq_pop;    
    wire                        creq_full, creq_empty;
    wire                        creq_rw_next;  
    wire [WORD_SIZE-1:0]        creq_byteen_next;
    wire [`REQS_BITS-1:0]       creq_tid_next;
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

    VX_fifo_queue_xt #(
        .DATAW    (CORE_TAG_WIDTH + `REQS_BITS + 1 + WORD_SIZE + `WORD_ADDR_WIDTH + `WORD_WIDTH), 
        .SIZE     (CREQ_SIZE),
        .FASTRAM  (1)
    ) core_req_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (creq_push),
        .pop     (creq_pop),
        .data_in ({core_req_tag, core_req_tid, core_req_rw, core_req_byteen, core_req_addr, core_req_data}),        
        `UNUSED_PIN (data_out),
        .empty   (creq_empty),        
        .data_out_next({creq_tag_next, creq_tid_next, creq_rw_next, creq_byteen_next, creq_addr_next_unqual, creq_writeword_next}),
        `UNUSED_PIN (empty_next),
        .full    (creq_full),
        `UNUSED_PIN (size)
    );  

    wire                            crsq_alm_full;
    wire                            dreq_alm_full;
    wire                            mshr_alm_full;    

    wire                            mshr_pop;            
    wire                            mshr_pending_unqual_st0;
    wire                            mshr_valid;
    wire                            mshr_valid_next;
    wire [`REQS_BITS-1:0]           mshr_tid_next;
    wire [`LINE_ADDR_WIDTH-1:0]     mshr_addr_next;
    wire [`UP(`WORD_SELECT_BITS)-1:0] mshr_wsel_next;
    wire [`WORD_WIDTH-1:0]          mshr_writeword_next;
    wire [`REQ_TAG_WIDTH-1:0]       mshr_tag_next;
    wire                            mshr_rw_next;  
    wire [WORD_SIZE-1:0]            mshr_byteen_next;

    wire [`LINE_ADDR_WIDTH-1:0]     addr_st0, addr_st1;
    wire [`UP(`WORD_SELECT_BITS)-1:0] wsel_st0, wsel_st1;    
    wire                            mem_rw_st0, mem_rw_st1;
    wire [WORD_SIZE-1:0]            byteen_st0, byteen_st1;
    wire [`CACHE_LINE_WIDTH-1:0]    data_st0, data_st1;
    wire [`REQS_BITS-1:0]           req_tid_st0, req_tid_st1;    
    wire [`REQ_TAG_WIDTH-1:0]       tag_st0, tag_st1;
    wire                            valid_st0, valid_st1;        
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_mshr_st0, is_mshr_st1;    
    wire [`CACHE_LINE_WIDTH-1:0]    readdata_st1;
    wire                            miss_st0, miss_st1; 
    wire                            force_miss_st0, force_miss_st1;
    wire                            do_writeback_st0, do_writeback_st1;
    wire                            writeen_unqual_st0, writeen_unqual_st1;
    wire                            dreq_push_unqual_st0, dreq_push_unqual_st1;
    wire                            writeen_st1;
    wire                            core_req_hit_st1;
    wire                            is_flush_st0;

    wire is_mshr_miss_st1 = valid_st1 && is_mshr_st1 && (miss_st1 || force_miss_st1);

    // determine which queue to pop next in piority order    
    wire mshr_pop_unqual = mshr_valid; 
    wire drsq_pop_unqual = !mshr_pop_unqual && !drsq_empty;
    wire creq_pop_unqual = !mshr_pop_unqual && !drsq_pop_unqual && !creq_empty;

    assign mshr_pop = mshr_pop_unqual
                   && !crsq_alm_full        // ensure core response ready
                   && !is_mshr_miss_st1;    // do not schedule another mshr request when the previous one missed
    
    assign drsq_pop = drsq_pop_unqual;

    assign creq_pop = creq_pop_unqual 
                   && !crsq_alm_full        // ensure core response ready
                   && !dreq_alm_full        // ensure dram request ready
                   && !mshr_alm_full;       // ensure mshr enqueue ready

    assign valid_st0   = mshr_pop || drsq_pop || creq_pop;
    assign is_mshr_st0 = mshr_pop_unqual;
    assign is_fill_st0 = drsq_pop_unqual;

    VX_pipe_register #(
        .DATAW  (`LINE_ADDR_WIDTH + `UP(`WORD_SELECT_BITS) + 1 + WORD_SIZE + `CACHE_LINE_WIDTH + `REQS_BITS + `REQ_TAG_WIDTH + 1),
        .RESETW (0)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({
            mshr_valid_next ? mshr_addr_next : (!drsq_empty_next ? drsq_addr_next : creq_addr_next),
            mshr_valid_next ? mshr_wsel_next : creq_wsel_next,
            mshr_valid_next ? mshr_rw_next : creq_rw_next,
            mshr_valid_next ? mshr_byteen_next : creq_byteen_next,
            mshr_valid_next ? {`WORDS_PER_LINE{mshr_writeword_next}} : (!drsq_empty_next ? drsq_filldata_next : {`WORDS_PER_LINE{creq_writeword_next}}),
            mshr_valid_next ? mshr_tid_next : creq_tid_next,
            mshr_valid_next ? `REQ_TAG_WIDTH'(mshr_tag_next) : `REQ_TAG_WIDTH'(creq_tag_next),
            drsq_flush_next
        }),
        .data_out ({addr_st0, wsel_st0, mem_rw_st0, byteen_st0, data_st0, req_tid_st0, tag_st0, is_flush_st0})
    );

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st0, debug_wid_st0} = tag_st0[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin
        assign {debug_pc_st0, debug_wid_st0} = 0;
    end
`endif
        
    VX_tag_access #(
        .BANK_ID          (BANK_ID),
        .CACHE_ID         (CACHE_ID),
        .CORE_TAG_ID_BITS (CORE_TAG_ID_BITS),
        .CACHE_SIZE       (CACHE_SIZE),
        .CACHE_LINE_SIZE  (CACHE_LINE_SIZE),
        .NUM_BANKS        (NUM_BANKS),
        .WORD_SIZE        (WORD_SIZE),   
        .BANK_ADDR_OFFSET (BANK_ADDR_OFFSET)
        ) tag_access (
        .clk            (clk),
        .reset          (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc       (debug_pc_st0),
        .debug_wid      (debug_wid_st0),
    `endif    

        // read/Fill
        .lookup         (creq_pop || mshr_pop),
        .addr           (addr_st0),        
        .fill           (drsq_pop),
        .is_flush       (is_flush_st0),
        .missed         (miss_st0)
    );

    // redundant fills
    wire is_redundant_fill = is_fill_st0 && !miss_st0;

    // we have a miss in mshr or going to it for the current address
    wire mshr_pending_st0 = mshr_pending_unqual_st0 
                         || (valid_st1 && (miss_st1 || force_miss_st1) && (addr_st0 == addr_st1));

    // force miss to ensure commit order when a new request has pending previous requests to same block
    assign force_miss_st0 = !is_mshr_st0 && !is_fill_st0 && mshr_pending_st0;

    assign writeen_unqual_st0 = (!is_fill_st0 && !miss_st0 && mem_rw_st0) 
                             || (is_fill_st0 && !is_redundant_fill);

    wire send_fill_req_st0 = !is_fill_st0 && miss_st0 && !mem_rw_st0;

    assign do_writeback_st0 = !is_fill_st0 && mem_rw_st0;

    assign dreq_push_unqual_st0 = send_fill_req_st0 || do_writeback_st0;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `UP(`WORD_SELECT_BITS) + `CACHE_LINE_WIDTH + 1 + WORD_SIZE + `REQS_BITS + `REQ_TAG_WIDTH),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({valid_st0, is_mshr_st0, is_fill_st0, writeen_unqual_st0, dreq_push_unqual_st0, do_writeback_st0, miss_st0, force_miss_st0, addr_st0, wsel_st0, data_st0, mem_rw_st0, byteen_st0, req_tid_st0, tag_st0}),
        .data_out ({valid_st1, is_mshr_st1, is_fill_st1, writeen_unqual_st1, dreq_push_unqual_st1, do_writeback_st1, miss_st1, force_miss_st1, addr_st1, wsel_st1, data_st1, mem_rw_st1, byteen_st1, req_tid_st1, tag_st1})
    );

    assign core_req_hit_st1 = !is_fill_st1 && !miss_st1 && !force_miss_st1;
    
    assign writeen_st1 = writeen_unqual_st1 && (is_fill_st1 || !force_miss_st1);

    wire dreq_push_st1 = dreq_push_unqual_st1 && (do_writeback_st1 || !force_miss_st1);

    wire mshr_push_st1 = !is_fill_st1 && !mem_rw_st1 && (miss_st1 || force_miss_st1);

    wire crsq_push_st1 = core_req_hit_st1 && !mem_rw_st1;

    VX_data_access #(
        .BANK_ID          (BANK_ID),
        .CACHE_ID         (CACHE_ID),
        .CORE_TAG_ID_BITS (CORE_TAG_ID_BITS),
        .CACHE_SIZE       (CACHE_SIZE),
        .CACHE_LINE_SIZE  (CACHE_LINE_SIZE),
        .NUM_BANKS        (NUM_BANKS),
        .WORD_SIZE        (WORD_SIZE),
        .WRITE_ENABLE     (WRITE_ENABLE)
     ) data_access (
        .clk        (clk),
        .reset      (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc   (debug_pc_st1),
        .debug_wid  (debug_wid_st1),
    `endif

        .addr       (addr_st1),

        // reading
        .readen     (valid_st1 && !mem_rw_st1 && !is_fill_st1),        
        .rddata     (readdata_st1),

        // writing        
        .writeen    (valid_st1 && writeen_st1),
        .is_fill    (is_fill_st1),
        .wsel       (wsel_st1),
        .byteen     (byteen_st1),
        .wrdata     (data_st1)
    ); 

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st1, debug_wid_st1} = tag_st1[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin        
        assign {debug_pc_st1, debug_wid_st1} = 0;
    end
`endif

    wire incoming_fill_st1 = valid_st0 && is_fill_st0 && (addr_st1 == addr_st0);

    wire mshr_push    = valid_st1 && mshr_push_st1;
    wire mshr_dequeue = valid_st1 && is_mshr_st1 && !mshr_push_st1;

    // push a missed request as 'ready' if it was a forced miss that actually had a hit 
    // or the fill request for this block is comming
    wire mshr_init_ready_state = !miss_st1 || incoming_fill_st1;

    VX_miss_resrv #(
        .BANK_ID            (BANK_ID),
        .CACHE_ID           (CACHE_ID),      
        .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
        .CACHE_LINE_SIZE    (CACHE_LINE_SIZE),
        .NUM_BANKS          (NUM_BANKS),
        .WORD_SIZE          (WORD_SIZE),
        .NUM_REQS           (NUM_REQS),
        .MSHR_SIZE          (MSHR_SIZE),
        .ALM_FULL           (MSHR_SIZE-1),
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
        .enqueue_data       ({data_st1[`WORD_WIDTH-1:0], req_tid_st1, tag_st1, mem_rw_st1, byteen_st1, wsel_st1}),
        .enqueue_is_mshr    (is_mshr_st1),
        .enqueue_as_ready   (mshr_init_ready_state),
        .enqueue_almfull    (mshr_alm_full),

        // lookup
        .lookup_ready       (drsq_pop && !is_flush_st0),
        .lookup_addr        (addr_st0),
        .lookup_match       (mshr_pending_unqual_st0),
        
        // schedule
        .schedule           (mshr_pop),        
        .schedule_valid     (mshr_valid),
        `UNUSED_PIN (schedule_addr),
        `UNUSED_PIN (schedule_data),
        .schedule_valid_next(mshr_valid_next),
        .schedule_addr_next (mshr_addr_next),
        .schedule_data_next ({mshr_writeword_next, mshr_tid_next, mshr_tag_next, mshr_rw_next, mshr_byteen_next, mshr_wsel_next}),

        // dequeue
        .dequeue            (mshr_dequeue)
    );

    // Enqueue core response
     
    wire crsq_empty;

    wire crsq_push = valid_st1 && crsq_push_st1;
    wire crsq_pop  = core_rsp_valid && core_rsp_ready;

    wire [`REQS_BITS-1:0]     crsq_tid_st1 = req_tid_st1;
    wire [CORE_TAG_WIDTH-1:0] crsq_tag_st1 = CORE_TAG_WIDTH'(tag_st1);
    wire [`WORD_WIDTH-1:0]    crsq_data_st1;  

     if (`WORD_SELECT_BITS != 0) begin
        assign crsq_data_st1 = readdata_st1[wsel_st1 * `WORD_WIDTH +: `WORD_WIDTH];
    end else begin
        assign crsq_data_st1 = readdata_st1;
    end
  
    VX_fifo_queue #(
        .DATAW    (`REQS_BITS + CORE_TAG_WIDTH + `WORD_WIDTH), 
        .SIZE     (CRSQ_SIZE),
        .ALM_FULL (CRSQ_SIZE-1),
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
        .alm_full(crsq_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),        
        `UNUSED_PIN (size)
    );

    assign core_rsp_valid = !crsq_empty;

    // Enqueue DRAM request

    wire dreq_empty;
        
    wire dreq_push = valid_st1 && dreq_push_st1 
                  && (do_writeback_st1 || !incoming_fill_st1);

    wire dreq_pop = dram_req_valid && dram_req_ready;

    wire writeback = WRITE_ENABLE && do_writeback_st1;

    wire [`LINE_ADDR_WIDTH-1:0] dreq_addr = addr_st1;

    wire [`CACHE_LINE_WIDTH-1:0] dreq_data;
    wire [CACHE_LINE_SIZE-1:0] dreq_byteen, dreq_byteen_unqual;

    if (`WORD_SELECT_BITS != 0) begin
        for (genvar i = 0; i < `WORDS_PER_LINE; i++) begin
            assign dreq_byteen_unqual[i * WORD_SIZE +: WORD_SIZE] = (wsel_st1 == `WORD_SELECT_BITS'(i)) ? byteen_st1 : {WORD_SIZE{1'b0}};
        end
    end else begin
        assign dreq_byteen_unqual = byteen_st1;
    end
    assign dreq_data = data_st1;

    assign dreq_byteen = writeback ? dreq_byteen_unqual : {CACHE_LINE_SIZE{1'b1}};

    VX_fifo_queue #(
        .DATAW    (1 + CACHE_LINE_SIZE + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH), 
        .SIZE     (DREQ_SIZE),
        .ALM_FULL (DREQ_SIZE-1),
        .BUFFERED (NUM_BANKS == 1),
        .FASTRAM  (1)
    ) dram_req_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (dreq_push),
        .pop     (dreq_pop),
        .data_in ({writeback,   dreq_byteen,     dreq_addr,     dreq_data}),        
        .data_out({dram_req_rw, dram_req_byteen, dram_req_addr, dram_req_data}),
        .empty   (dreq_empty),
        .alm_full(dreq_alm_full),
        `UNUSED_PIN (full),        
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    assign dram_req_valid = !dreq_empty;    
                         
    `SCOPE_ASSIGN (valid_st0, valid_st0);
    `SCOPE_ASSIGN (valid_st1, valid_st1);
    `SCOPE_ASSIGN (is_fill_st0, is_fill_st0);
    `SCOPE_ASSIGN (is_mshr_st0, is_mshr_st0);
    `SCOPE_ASSIGN (miss_st0,    miss_st0);
    `SCOPE_ASSIGN (force_miss_st0, force_miss_st0);
    `SCOPE_ASSIGN (mshr_push,   mshr_push);    
    `SCOPE_ASSIGN (crsq_alm_full, crsq_alm_full);
    `SCOPE_ASSIGN (dreq_alm_full, dreq_alm_full);
    `SCOPE_ASSIGN (mshr_alm_full, mshr_alm_full);
    `SCOPE_ASSIGN (addr_st0, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID));
    `SCOPE_ASSIGN (addr_st1, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));

`ifdef PERF_ENABLE
    assign perf_read_misses  = valid_st1 && !is_fill_st1 && !is_mshr_st1 && miss_st1 && !mem_rw_st1;
    assign perf_write_misses = valid_st1 && !is_fill_st1 && !is_mshr_st1 && miss_st1 && mem_rw_st1;
    assign perf_pipe_stalls  = crsq_alm_full || dreq_alm_full || mshr_alm_full;
    assign perf_mshr_stalls  = mshr_alm_full;
`endif

`ifdef DBG_PRINT_CACHE_BANK
    always @(posedge clk) begin        
        if (valid_st1 && !is_fill_st1 && miss_st1 && incoming_fill_st1) begin
            $display("%t: miss with incoming fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));
            assert(!is_mshr_st1);
        end
        if (crsq_alm_full || dreq_alm_full || mshr_alm_full) begin
            $display("%t: cache%0d:%0d pipeline-stall: cwbq=%b, dwbq=%b, mshr=%b", $time, CACHE_ID, BANK_ID, crsq_alm_full, dreq_alm_full, mshr_alm_full);
        end
        if (drsq_pop) begin
            if (is_flush_st0)
                $display("%t: cache%0d:%0d flush: addr=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID));
            else    
                $display("%t: cache%0d:%0d fill-rsp: addr=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), filldata_st0);
        end
        if (creq_pop || mshr_pop) begin
            if (mem_rw_st0)
                $display("%t: cache%0d:%0d core-wr-req: addr=%0h, is_mshr=%b, tag=%0h, tid=%0d, byteen=%b, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), is_mshr_st0, tag_st0, req_tid_st0, byteen_st0, writeword_st0, debug_wid_st0, debug_pc_st0);
            else
                $display("%t: cache%0d:%0d core-rd-req: addr=%0h, is_mshr=%b, tag=%0h, tid=%0d, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), is_mshr_st0, tag_st0, req_tid_st0, byteen_st0, debug_wid_st0, debug_pc_st0);
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