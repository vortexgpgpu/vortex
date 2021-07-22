`include "VX_cache_define.vh"

module VX_bank #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0,

    // Number of Word requests per cycle
    parameter NUM_REQS                      = 1,  

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE               = 1, 
    // Number of bankS
    parameter NUM_BANKS                     = 1,
    // Number of ports per banks
    parameter NUM_PORTS                     = 1,
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 1, 
    // Core Response Queue Size
    parameter CRSQ_SIZE                     = 1,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 1, 
    // Memory Request Queue Size
    parameter MREQ_SIZE                     = 1,

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
    input wire [NUM_PORTS-1:0]          core_req_pmask,
    input wire [NUM_PORTS-1:0][`UP(`WORD_SELECT_BITS)-1:0] core_req_wsel,
    input wire [NUM_PORTS-1:0][WORD_SIZE-1:0] core_req_byteen,
    input wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] core_req_data,  
    input wire [NUM_PORTS-1:0][`REQS_BITS-1:0] core_req_tid,
    input wire                          core_req_rw,  
    input wire [`LINE_ADDR_WIDTH-1:0]   core_req_addr,
    input wire [CORE_TAG_WIDTH-1:0]     core_req_tag,
    output wire                         core_req_ready,
    
    // Core Response    
    output wire                         core_rsp_valid,
    output wire [NUM_PORTS-1:0]         core_rsp_pmask,
    output wire [NUM_PORTS-1:0][`REQS_BITS-1:0] core_rsp_tid,
    output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]    core_rsp_tag,
    input  wire                         core_rsp_ready,

    // Memory request
    output wire                         mem_req_valid,
    output wire                         mem_req_rw,
    output wire [CACHE_LINE_SIZE-1:0]   mem_req_byteen,    
    output wire [`LINE_ADDR_WIDTH-1:0]  mem_req_addr,
    output wire [`CACHE_LINE_WIDTH-1:0] mem_req_data,
    input  wire                         mem_req_ready,
    
    // Memory response
    input wire                          mem_rsp_valid,  
    input wire [`LINE_ADDR_WIDTH-1:0]   mem_rsp_addr,
    input wire [`CACHE_LINE_WIDTH-1:0]  mem_rsp_data,
    output wire                         mem_rsp_ready,

    // flush
    input wire                          flush_enable,
    input wire [`LINE_SELECT_BITS-1:0]  flush_addr
);

    `UNUSED_PARAM (CORE_TAG_ID_BITS)
    
`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    wire [31:0]         debug_pc_sel, debug_pc_st0,  debug_pc_st1;
    wire [`NW_BITS-1:0] debug_wid_sel, debug_wid_st0, debug_wid_st1;
`IGNORE_WARNINGS_END
`endif    

    wire [NUM_PORTS-1:0] creq_pmask;
    wire [NUM_PORTS-1:0][`UP(`WORD_SELECT_BITS)-1:0] creq_wsel;
    wire [NUM_PORTS-1:0][WORD_SIZE-1:0] creq_byteen;
    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] creq_data;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] creq_tid;  
    wire                        creq_rw;  
    wire [`LINE_ADDR_WIDTH-1:0] creq_addr;
    wire [CORE_TAG_WIDTH-1:0]   creq_tag;

    wire creq_out_valid, creq_out_ready;

    VX_elastic_buffer #(
        .DATAW    (CORE_TAG_WIDTH + 1 + `LINE_ADDR_WIDTH + (1 + `UP(`WORD_SELECT_BITS) + WORD_SIZE + `WORD_WIDTH + `REQS_BITS) * NUM_PORTS),
        .SIZE     (CREQ_SIZE),
        .BUFFERED (CREQ_SIZE > 2)
    ) core_req_queue (
        .clk        (clk),
        .reset      (reset),
        .ready_in   (core_req_ready),
        .valid_in   (core_req_valid),
        .data_in    ({core_req_tag, core_req_rw, core_req_addr, core_req_pmask, core_req_wsel, core_req_byteen, core_req_data, core_req_tid}),                
        .data_out   ({creq_tag,     creq_rw,     creq_addr,     creq_pmask,     creq_wsel,     creq_byteen,     creq_data,     creq_tid}),
        .ready_out  (creq_out_ready),
        .valid_out  (creq_out_valid)
    );  

    wire                            mshr_alm_full;
    wire                            mshr_pop;
    wire                            mshr_pending;
    wire                            mshr_valid;
    wire [`LINE_ADDR_WIDTH-1:0]     mshr_addr;
    wire [CORE_TAG_WIDTH-1:0]       mshr_tag;
    wire [NUM_PORTS-1:0]            mshr_pmask;
    wire [NUM_PORTS-1:0][`UP(`WORD_SELECT_BITS)-1:0] mshr_wsel;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] mshr_tid;
    
    wire [`LINE_ADDR_WIDTH-1:0]     addr_st0, addr_st1;
    wire                            mem_rw_st0, mem_rw_st1;
    wire [NUM_PORTS-1:0][`UP(`WORD_SELECT_BITS)-1:0] wsel_st0, wsel_st1;        
    wire [NUM_PORTS-1:0][WORD_SIZE-1:0] byteen_st0, byteen_st1;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] req_tid_st0, req_tid_st1;
    wire [NUM_PORTS-1:0]            pmask_st0, pmask_st1;
    wire [`CACHE_LINE_WIDTH-1:0]    rdata_st1;
    wire [`CACHE_LINE_WIDTH-1:0]    wdata_st0, wdata_st1;    
    wire [CORE_TAG_WIDTH-1:0]       tag_st0, tag_st1;
    wire                            valid_st0, valid_st1;        
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_mshr_st0, is_mshr_st1;        
    wire                            miss_st0, miss_st1; 
    wire                            prev_miss_dep_st0;
    wire                            fill_req_unqual_st0, fill_req_unqual_st1;
    wire                            force_miss_st0, force_miss_st1;
    wire                            writeen_unqual_st0, writeen_unqual_st1;
    wire                            incoming_fill_st0, incoming_fill_st1;
    wire                            mshr_pending_st0;
    wire                            is_flush_st0;

    wire crsq_in_valid, crsq_in_ready, crsq_in_stall;
    wire mreq_alm_full;

    wire creq_out_fire = creq_out_valid && creq_out_ready;
    wire crsq_in_fire = crsq_in_valid && crsq_in_ready;
    
    VX_pending_size #( 
        .SIZE (MSHR_SIZE)
    ) mshr_pending_size (
        .clk   (clk),
        .reset (reset),
        .push  (creq_out_fire && !creq_rw),
        .pop   (crsq_in_fire),
        .full  (mshr_alm_full),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (size)
    );

    // determine which queue to pop next in priority order 
    wire mshr_grant  = !mreq_alm_full;  // ensure memory request queue not full (deadlock prevention) 
    wire mshr_enable = mshr_grant && mshr_valid; 

    wire mrsq_grant  = !mshr_enable;
    wire mrsq_enable = mrsq_grant && mem_rsp_valid;
    
    wire creq_grant  = !mshr_enable && !mrsq_enable && !flush_enable;

    wire is_miss_st1 = valid_st1 && (miss_st1 || force_miss_st1);
    
    assign mshr_pop = mshr_enable
                   && !(is_miss_st1 && is_mshr_st1)  // do not schedule another mshr request if the previous one missed
                   && !crsq_in_stall;       // ensure core response ready
                   
    assign creq_out_ready = creq_grant 
                        && !mreq_alm_full   // ensure memory request ready                   
                        && !mshr_alm_full   // ensure mshr enqueue ready
                        && !crsq_in_stall;  // ensure core response ready

    assign mem_rsp_ready = mrsq_grant
                        && !crsq_in_stall;  // ensure core response ready

    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;

    // we have a miss in mshr or entering it for the current address
    wire mshr_pending_sel = mshr_pending
                         || (is_miss_st1 && (creq_addr == addr_st1));

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_sel, debug_wid_sel} = mshr_enable ? mshr_tag[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS] : creq_tag[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin        
        assign {debug_pc_sel, debug_wid_sel} = 0;
    end
`endif

    wire [`CACHE_LINE_WIDTH-1:0] creq_line_data;

    if (`WORDS_PER_LINE > 1) begin
        if (NUM_PORTS > 1) begin
            reg [`CACHE_LINE_WIDTH-1:0] creq_line_data_r;
            always @(*) begin
                creq_line_data_r = 'x;
                for (integer p = 0; p < NUM_PORTS; p++) begin
                    if (creq_pmask[p]) begin
                        creq_line_data_r[creq_wsel[p] * `WORD_WIDTH +: `WORD_WIDTH] = creq_data[p];
                    end
                end
            end
            assign creq_line_data = creq_line_data_r;
        end else begin
            assign creq_line_data = {`WORDS_PER_LINE{creq_data}};
        end
    end else begin
        assign creq_line_data = creq_data;
    end
    
    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH + (`UP(`WORD_SELECT_BITS) + WORD_SIZE + `REQS_BITS + 1) * NUM_PORTS + CORE_TAG_WIDTH + 1 + 1),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!crsq_in_stall),
        .data_in  ({
            flush_enable || mshr_pop || mem_rsp_fire || creq_out_fire,
            flush_enable,
            mshr_enable,
            mrsq_enable || flush_enable,
            mshr_enable ? 1'b0 : creq_rw,
            mshr_enable ? mshr_addr : (mem_rsp_valid ? mem_rsp_addr : (flush_enable ? `LINE_ADDR_WIDTH'(flush_addr) : creq_addr)),
            (mem_rsp_valid || !WRITE_ENABLE) ? mem_rsp_data : creq_line_data,
            mshr_enable ? mshr_wsel : creq_wsel,
            creq_byteen,
            mshr_enable ? mshr_tid : creq_tid,
            mshr_enable ? mshr_pmask : creq_pmask,
            mshr_enable ? mshr_tag : creq_tag,
            mshr_pending_sel
        }),
        .data_out ({valid_st0, is_flush_st0, is_mshr_st0, is_fill_st0, mem_rw_st0, addr_st0, wdata_st0, wsel_st0, byteen_st0, req_tid_st0, pmask_st0, tag_st0, mshr_pending_st0})
    );

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st0, debug_wid_st0} = tag_st0[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin
        assign {debug_pc_st0, debug_wid_st0} = 0;
    end
`endif

    wire do_lookup_st0 = valid_st0 && ~is_fill_st0;
    wire do_fill_st0   = valid_st0 && is_fill_st0 && !crsq_in_stall;

    wire tag_match_st0;
        
    VX_tag_access #(
        .BANK_ID          (BANK_ID),
        .CACHE_ID         (CACHE_ID),
        .CACHE_SIZE       (CACHE_SIZE),
        .CACHE_LINE_SIZE  (CACHE_LINE_SIZE),
        .NUM_BANKS        (NUM_BANKS),
        .WORD_SIZE        (WORD_SIZE),   
        .BANK_ADDR_OFFSET (BANK_ADDR_OFFSET)
    ) tag_access (
        .clk        (clk),
        .reset      (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc   (debug_pc_st0),
        .debug_wid  (debug_wid_st0),
    `endif    

        // read/Fill
        .lookup     (do_lookup_st0),
        .addr       (addr_st0),        
        .fill       (do_fill_st0),
        .is_flush   (is_flush_st0),
        .tag_match  (tag_match_st0)
    );

    // redundant fills
    wire is_redundant_fill_st0 = is_fill_st0 && tag_match_st0;

    // we had a miss with prior request for the current address
    assign prev_miss_dep_st0 = is_miss_st1 && (addr_st0 == addr_st1);

    assign miss_st0 = !is_fill_st0 && !tag_match_st0;

    // force miss to ensure commit order when a new request has pending previous requests to same block
    // also force a miss for mshr requests when previous requests got a miss    
    assign force_miss_st0 = (!is_fill_st0 && !is_mshr_st0 && (mshr_pending_st0 || prev_miss_dep_st0))
                         || (is_mshr_st0 && is_mshr_st1 && is_miss_st1);

    assign writeen_unqual_st0 = (WRITE_ENABLE && !is_fill_st0 && tag_match_st0 && mem_rw_st0)
                             || (is_fill_st0 && !is_redundant_fill_st0);

    assign incoming_fill_st0 = mem_rsp_valid && (addr_st0 == mem_rsp_addr);

    assign fill_req_unqual_st0 = !mem_rw_st0 && (!force_miss_st0 || (is_mshr_st0 && !prev_miss_dep_st0));

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH + (`UP(`WORD_SELECT_BITS) + WORD_SIZE + `REQS_BITS + 1) * NUM_PORTS + CORE_TAG_WIDTH),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!crsq_in_stall),
        .data_in  ({valid_st0, is_mshr_st0, is_fill_st0, writeen_unqual_st0, fill_req_unqual_st0, incoming_fill_st0, miss_st0, force_miss_st0, mem_rw_st0, addr_st0, wdata_st0, wsel_st0, byteen_st0, req_tid_st0, pmask_st0, tag_st0}),
        .data_out ({valid_st1, is_mshr_st1, is_fill_st1, writeen_unqual_st1, fill_req_unqual_st1, incoming_fill_st1, miss_st1, force_miss_st1, mem_rw_st1, addr_st1, wdata_st1, wsel_st1, byteen_st1, req_tid_st1, pmask_st1, tag_st1})
    ); 

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st1, debug_wid_st1} = tag_st1[CORE_TAG_WIDTH-1:CORE_TAG_ID_BITS];
    end else begin        
        assign {debug_pc_st1, debug_wid_st1} = 0;
    end
`endif

    wire writeen_st1 = writeen_unqual_st1 && (is_fill_st1 || !force_miss_st1);

    wire crsq_push_st1 = !is_fill_st1 && !mem_rw_st1 && !miss_st1 && !force_miss_st1;

    wire mshr_push_st1 = !is_fill_st1 && !mem_rw_st1 && (miss_st1 || force_miss_st1);

    wire incoming_fill_qual_st1 = (mem_rsp_valid && (addr_st1 == mem_rsp_addr)) 
                               || incoming_fill_st1;

    wire do_writeback_st1 = !is_fill_st1 && mem_rw_st1;    

    wire mreq_push_st1 = (miss_st1 && fill_req_unqual_st1 && !incoming_fill_qual_st1)
                      || do_writeback_st1;

    wire [`WORDS_PER_LINE-1:0][WORD_SIZE-1:0] line_byteen_st1;

    if (`WORDS_PER_LINE > 1) begin
        reg [CACHE_LINE_SIZE-1:0] line_byteen_r;
        always @(*) begin
            line_byteen_r = 0;           
            for (integer p = 0; p < NUM_PORTS; p++) begin
                if ((NUM_PORTS == 1) || pmask_st1[p]) begin
                    line_byteen_r[wsel_st1[p] * WORD_SIZE +: WORD_SIZE] = byteen_st1[p];
                end
            end
        end
        assign line_byteen_st1 = line_byteen_r;
    end else begin
        assign line_byteen_st1 = byteen_st1;
    end   

    VX_data_access #(
        .BANK_ID        (BANK_ID),
        .CACHE_ID       (CACHE_ID),
        .CACHE_SIZE     (CACHE_SIZE),
        .CACHE_LINE_SIZE(CACHE_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .WRITE_ENABLE   (WRITE_ENABLE)
     ) data_access (
        .clk        (clk),
        .reset      (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc   (debug_pc_st1),
        .debug_wid  (debug_wid_st1),
    `endif

        .addr       (addr_st1),

        // reading
        .readen     (valid_st1 && !is_fill_st1 && !mem_rw_st1),
        .rdata      (rdata_st1),

        // writing        
        .writeen    (valid_st1 && writeen_st1),
        .is_fill    (is_fill_st1),
        .byteen     (line_byteen_st1),        
        .wdata      (wdata_st1)
    );

    wire mshr_push    = valid_st1 && mshr_push_st1;
    wire mshr_dequeue = valid_st1 && is_mshr_st1 && !mshr_push_st1 && crsq_in_ready;
    wire mshr_restore = is_mshr_st1;

    // push a missed request as 'ready' if it was a forced miss that actually had a hit 
    // or the fill request for this block is comming
    wire mshr_init_ready_state = !miss_st1 || incoming_fill_qual_st1;

    // use memory rsp or core req address to lookup the mshr
    wire [`LINE_ADDR_WIDTH-1:0] lookup_addr = mem_rsp_valid ? mem_rsp_addr : creq_addr;

    VX_miss_resrv #(
        .BANK_ID            (BANK_ID),
        .CACHE_ID           (CACHE_ID),
        .CACHE_LINE_SIZE    (CACHE_LINE_SIZE),
        .NUM_BANKS          (NUM_BANKS),
        .NUM_PORTS          (NUM_PORTS),
        .WORD_SIZE          (WORD_SIZE),
        .NUM_REQS           (NUM_REQS),
        .MSHR_SIZE          (MSHR_SIZE),
        .ALM_FULL           (MSHR_SIZE-2),
        .CORE_TAG_WIDTH     (CORE_TAG_WIDTH)
    ) miss_resrv (
        .clk                (clk),
        .reset              (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .deq_debug_pc       (debug_pc_sel),
        .deq_debug_wid      (debug_wid_sel),
        .enq_debug_pc       (debug_pc_st1),
        .enq_debug_wid      (debug_wid_st1),
    `endif

        // enqueue
        .enqueue            (mshr_push),        
        .enqueue_addr       (addr_st1),
        .enqueue_data       ({wsel_st1, tag_st1, req_tid_st1, pmask_st1}),
        .enqueue_is_mshr    (mshr_restore),
        .enqueue_as_ready   (mshr_init_ready_state),                
        `UNUSED_PIN (enqueue_almfull),
        `UNUSED_PIN (enqueue_full),

        // lookup        
        .lookup_addr        (lookup_addr),
        .lookup_match       (mshr_pending),

        // fill update
        .fill_update        (mem_rsp_fire),
        
        // schedule
        .schedule           (mshr_pop),        
        .schedule_valid     (mshr_valid),
        .schedule_addr      (mshr_addr),
        .schedule_data      ({mshr_wsel, mshr_tag, mshr_tid, mshr_pmask}),

        // dequeue
        .dequeue            (mshr_dequeue)
    );

    // Enqueue core response
     
    wire [NUM_PORTS-1:0] crsq_pmask;
    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] crsq_data;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] crsq_tid;
    wire [CORE_TAG_WIDTH-1:0] crsq_tag;

    assign crsq_in_valid = valid_st1 && crsq_push_st1;
    assign crsq_in_stall = crsq_in_valid && !crsq_in_ready;

    assign crsq_pmask = pmask_st1;
    assign crsq_tid   = req_tid_st1;
    assign crsq_tag   = tag_st1;

    if (`WORDS_PER_LINE > 1) begin
        for (genvar p = 0; p < NUM_PORTS; ++p) begin
            assign crsq_data[p] = rdata_st1[wsel_st1[p] * `WORD_WIDTH +: `WORD_WIDTH];
        end
    end else begin
        assign crsq_data = rdata_st1;
    end

    VX_elastic_buffer #(
        .DATAW (CORE_TAG_WIDTH + (1 + `WORD_WIDTH + `REQS_BITS) * NUM_PORTS),
        .SIZE  (CRSQ_SIZE)
    ) core_rsp_req (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsq_in_valid),        
        .data_in   ({crsq_tag, crsq_pmask, crsq_data, crsq_tid}),
        .ready_in  (crsq_in_ready),      
        .valid_out (core_rsp_valid),
        .data_out  ({core_rsp_tag, core_rsp_pmask, core_rsp_data, core_rsp_tid}),
        .ready_out (core_rsp_ready)
    );

    // Enqueue memory request

    wire [CACHE_LINE_SIZE-1:0] mreq_byteen;
    wire [`LINE_ADDR_WIDTH-1:0] mreq_addr;
    wire [`CACHE_LINE_WIDTH-1:0] mreq_data;
    wire mreq_push, mreq_pop, mreq_empty, mreq_rw;
        
    assign mreq_push = valid_st1 && mreq_push_st1;

    assign mreq_pop = mem_req_valid && mem_req_ready;

    assign mreq_rw     = WRITE_ENABLE && do_writeback_st1;
    assign mreq_byteen = mreq_rw ? line_byteen_st1 : {CACHE_LINE_SIZE{1'b1}};
    assign mreq_addr   = addr_st1;
    assign mreq_data   = wdata_st1;

    VX_fifo_queue #(
        .DATAW    (1 + CACHE_LINE_SIZE + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH), 
        .SIZE     (MREQ_SIZE),
        .ALM_FULL (MREQ_SIZE-2)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (mreq_push),
        .pop        (mreq_pop),
        .data_in    ({mreq_rw,    mreq_byteen,    mreq_addr,    mreq_data}),
        .data_out   ({mem_req_rw, mem_req_byteen, mem_req_addr, mem_req_data}),
        .empty      (mreq_empty),        
        .alm_full   (mreq_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),        
        `UNUSED_PIN (size)
    );

    assign mem_req_valid = !mreq_empty;    
                         
    `SCOPE_ASSIGN (valid_st0, valid_st0);
    `SCOPE_ASSIGN (valid_st1, valid_st1);
    `SCOPE_ASSIGN (is_fill_st0, is_fill_st0);
    `SCOPE_ASSIGN (is_mshr_st0, is_mshr_st0);
    `SCOPE_ASSIGN (miss_st0, miss_st0);
    `SCOPE_ASSIGN (force_miss_st0, force_miss_st0);
    `SCOPE_ASSIGN (mshr_push, mshr_push);    
    `SCOPE_ASSIGN (crsq_in_stall, crsq_in_stall);
    `SCOPE_ASSIGN (mreq_alm_full, mreq_alm_full);
    `SCOPE_ASSIGN (mshr_alm_full, mshr_alm_full);
    `SCOPE_ASSIGN (addr_st0, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID));
    `SCOPE_ASSIGN (addr_st1, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));

`ifdef PERF_ENABLE
    assign perf_read_misses  = valid_st1 && !is_fill_st1 && !is_mshr_st1 && miss_st1 && !mem_rw_st1;
    assign perf_write_misses = valid_st1 && !is_fill_st1 && !is_mshr_st1 && miss_st1 && mem_rw_st1;
    assign perf_pipe_stalls  = crsq_in_stall || mreq_alm_full || mshr_alm_full;
    assign perf_mshr_stalls  = mshr_alm_full;
`endif

`ifdef DBG_PRINT_CACHE_BANK
    always @(posedge clk) begin  
        /*if (crsq_in_fire && (NUM_PORTS > 1) && $countones(crsq_pmask) > 1) begin
            $display("%t: *** cache%0d:%0d multi-port-out: pmask=%b, addr=%0h, tag=%0h", $time, CACHE_ID, BANK_ID, crsq_pmask, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), crsq_tag);
        end*/     
        if (valid_st1 && !is_fill_st1 && miss_st1 && incoming_fill_qual_st1) begin
            $display("%t: *** cache%0d:%0d miss with incoming fill - addr=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));
            assert(!is_mshr_st1);
        end
        if (crsq_in_stall || mreq_alm_full || mshr_alm_full) begin
            $display("%t: *** cache%0d:%0d pipeline-stall: cwbq=%b, dwbq=%b, mshr=%b", $time, CACHE_ID, BANK_ID, crsq_in_stall, mreq_alm_full, mshr_alm_full);
        end
        if (flush_enable) begin
            $display("%t: cache%0d:%0d flush: addr=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(flush_addr, BANK_ID));
        end
        if (mem_rsp_fire) begin
            $display("%t: cache%0d:%0d fill-rsp: addr=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mem_rsp_addr, BANK_ID), mem_rsp_data);
        end
        if (mshr_pop) begin
            $display("%t: cache%0d:%0d mshr-rd-req: addr=%0h, tag=%0h, pmask=%b, tid=%0d, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mshr_addr, BANK_ID), mshr_tag, mshr_pmask, mshr_tid, debug_wid_sel, debug_pc_sel);
        end
        if (creq_out_fire) begin
            if (creq_rw)
                $display("%t: cache%0d:%0d core-wr-req: addr=%0h, tag=%0h, pmask=%b, tid=%0d, byteen=%b, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(creq_addr, BANK_ID), creq_tag, creq_pmask, creq_tid, creq_byteen, creq_data, debug_wid_sel, debug_pc_sel);
            else
                $display("%t: cache%0d:%0d core-rd-req: addr=%0h, tag=%0h, pmask=%b, tid=%0d, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(creq_addr, BANK_ID), creq_tag, creq_pmask, creq_tid, creq_byteen, debug_wid_sel, debug_pc_sel);
        end
        if (crsq_in_fire) begin
            $display("%t: cache%0d:%0d core-rsp: addr=%0h, tag=%0h, pmask=%b, tid=%0d, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), crsq_tag, crsq_pmask, crsq_tid, crsq_data, debug_wid_st1, debug_pc_st1);
        end
        if (mreq_push) begin
            if (do_writeback_st1)
                $display("%t: cache%0d:%0d writeback: addr=%0h, data=%0h, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mreq_addr, BANK_ID), mreq_data, mreq_byteen, debug_wid_st1, debug_pc_st1);
            else
                $display("%t: cache%0d:%0d fill-req: addr=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mreq_addr, BANK_ID), debug_wid_st1, debug_pc_st1);
        end
    end    
`endif

endmodule