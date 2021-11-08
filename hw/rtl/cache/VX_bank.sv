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
    parameter BANK_ADDR_OFFSET              = 0,

    parameter MSHR_ADDR_WIDTH  = $clog2(MSHR_SIZE),
    parameter WORD_SELECT_BITS = `UP(`WORD_SELECT_BITS)
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
    input wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] core_req_wsel,
    input wire [NUM_PORTS-1:0][WORD_SIZE-1:0] core_req_byteen,
    input wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] core_req_data,  
    input wire [NUM_PORTS-1:0][`REQS_BITS-1:0] core_req_tid,
    input wire [NUM_PORTS-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    input wire                          core_req_rw,  
    input wire [`LINE_ADDR_WIDTH-1:0]   core_req_addr,
    output wire                         core_req_ready,
    
    // Core Response    
    output wire                         core_rsp_valid,
    output wire [NUM_PORTS-1:0]         core_rsp_pmask,
    output wire [NUM_PORTS-1:0][`REQS_BITS-1:0] core_rsp_tid,
    output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] core_rsp_data,
    output wire [NUM_PORTS-1:0][CORE_TAG_WIDTH-1:0] core_rsp_tag,
    input  wire                         core_rsp_ready,

    // Memory request
    output wire                         mem_req_valid,
    output wire                         mem_req_rw,
    output wire [NUM_PORTS-1:0]         mem_req_pmask,
    output wire [NUM_PORTS-1:0][WORD_SIZE-1:0] mem_req_byteen,
    output wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] mem_req_wsel,
    output wire [`LINE_ADDR_WIDTH-1:0]  mem_req_addr,
    output wire [MSHR_ADDR_WIDTH-1:0]   mem_req_id,
    output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] mem_req_data,
    input  wire                         mem_req_ready,
    
    // Memory response
    input wire                          mem_rsp_valid,
    input wire [MSHR_ADDR_WIDTH-1:0]    mem_rsp_id,
    input wire [`CACHE_LINE_WIDTH-1:0]  mem_rsp_data,
    output wire                         mem_rsp_ready,

    // flush
    input wire                          flush_enable,
    input wire [`LINE_SELECT_BITS-1:0]  flush_addr
);

    `UNUSED_PARAM (CORE_TAG_ID_BITS)
    
`ifdef DBG_CACHE_REQ_INFO
`IGNORE_UNUSED_BEGIN
    wire [31:0]         debug_pc_sel, debug_pc_st0,  debug_pc_st1;
    wire [`NW_BITS-1:0] debug_wid_sel, debug_wid_st0, debug_wid_st1;
`IGNORE_UNUSED_END
`endif    

    wire [NUM_PORTS-1:0] creq_pmask;
    wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] creq_wsel;
    wire [NUM_PORTS-1:0][WORD_SIZE-1:0] creq_byteen;
    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] creq_data;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] creq_tid;  
    wire [NUM_PORTS-1:0][CORE_TAG_WIDTH-1:0] creq_tag;
    wire                        creq_rw;  
    wire [`LINE_ADDR_WIDTH-1:0] creq_addr;
    
    wire creq_valid, creq_ready;

    VX_elastic_buffer #(
        .DATAW (1 + `LINE_ADDR_WIDTH + NUM_PORTS * (1 + WORD_SELECT_BITS + WORD_SIZE + `WORD_WIDTH + `REQS_BITS + CORE_TAG_WIDTH)),
        .SIZE  (CREQ_SIZE)
    ) core_req_queue (
        .clk        (clk),
        .reset      (reset),
        .ready_in   (core_req_ready),
        .valid_in   (core_req_valid),
        .data_in    ({core_req_rw, core_req_addr, core_req_pmask, core_req_wsel, core_req_byteen, core_req_data, core_req_tid, core_req_tag}),                
        .data_out   ({creq_rw,     creq_addr,     creq_pmask,     creq_wsel,     creq_byteen,     creq_data,     creq_tid,     creq_tag}),
        .ready_out  (creq_ready),
        .valid_out  (creq_valid)
    );
    
    wire                            mreq_alm_full;    
    wire [`LINE_ADDR_WIDTH-1:0]     mem_rsp_addr;
    wire                            crsq_valid, crsq_ready;
    wire                            crsq_stall;
        
    wire                            mshr_valid;
    wire                            mshr_ready;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_alloc_id;
    wire                            mshr_alm_full;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_dequeue_id;
    wire [`LINE_ADDR_WIDTH-1:0]     mshr_addr;
    wire [NUM_PORTS-1:0][CORE_TAG_WIDTH-1:0] mshr_tag;    
    wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] mshr_wsel;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] mshr_tid;
    wire [NUM_PORTS-1:0]            mshr_pmask;
    
    wire [`LINE_ADDR_WIDTH-1:0]     addr_st0, addr_st1;
    wire                            is_read_st0, is_read_st1;
    wire                            is_write_st0, is_write_st1;
    wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] wsel_st0, wsel_st1;        
    wire [NUM_PORTS-1:0][WORD_SIZE-1:0] byteen_st0, byteen_st1;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] req_tid_st0, req_tid_st1;
    wire [NUM_PORTS-1:0]            pmask_st0, pmask_st1;
    wire [NUM_PORTS-1:0][CORE_TAG_WIDTH-1:0] tag_st0, tag_st1;
    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] rdata_st1;
    wire [`CACHE_LINE_WIDTH-1:0]    wdata_st0, wdata_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_id_st0, mshr_id_st1;
    wire                            valid_st0, valid_st1;        
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_mshr_st0, is_mshr_st1;    
    wire                            miss_st0, miss_st1;
    wire                            is_flush_st0;
    wire                            mshr_pending_st0, mshr_pending_st1;

    // prevent read-during-write hazard when accessing tags/data block RAMs
    wire rdw_fill_hazard  = valid_st0 && is_fill_st0;
    wire rdw_write_hazard = valid_st0 && is_write_st0 && ~creq_rw;
    
    // determine which queue to pop next in priority order 
    wire mshr_grant  = !flush_enable;
    wire mshr_enable = mshr_grant && mshr_valid; 

    wire mrsq_grant  = !flush_enable && !mshr_enable;
    wire mrsq_enable = mrsq_grant && mem_rsp_valid;
    wire creq_grant  = !flush_enable && !mshr_enable && !mrsq_enable;                    

    wire creq_enable = creq_grant && creq_valid;
    
    assign mshr_ready = mshr_grant
                     && !rdw_fill_hazard    // prevent read-during-write hazard
                     && !crsq_stall;        // ensure core_rsp_queue not full
                     

    assign mem_rsp_ready = mrsq_grant
                        && !crsq_stall;     // ensure core_rsp_queue not full
    
    assign creq_ready = creq_grant
                     && !rdw_write_hazard   // prevent read-during-write hazard
                     && !mreq_alm_full      // ensure mem_req_queue not full
                     && !mshr_alm_full      // ensure mshr not full
                     && !crsq_stall;        // ensure core_rsp_queue not full

    wire flush_fire   = flush_enable;
    wire mshr_fire    = mshr_valid && mshr_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
    wire creq_fire    = creq_valid && creq_ready;

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_wid_sel, debug_pc_sel} = mshr_enable ? mshr_tag[0][`CACHE_REQ_INFO_RNG] : creq_tag[0][`CACHE_REQ_INFO_RNG];
    end else begin        
        assign {debug_wid_sel, debug_pc_sel} = 0;
    end
`endif

    wire [`CACHE_LINE_WIDTH-1:0] wdata_sel;    
    assign wdata_sel[(NUM_PORTS * `WORD_WIDTH)-1:0] = (mem_rsp_valid || !WRITE_ENABLE) ? mem_rsp_data[(NUM_PORTS * `WORD_WIDTH)-1:0] : creq_data;
    for (genvar i = NUM_PORTS * `WORD_WIDTH; i < `CACHE_LINE_WIDTH; ++i) begin
        assign wdata_sel[i] = mem_rsp_data[i];
    end

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH + NUM_PORTS * (WORD_SELECT_BITS + WORD_SIZE + `REQS_BITS + 1 + CORE_TAG_WIDTH) + MSHR_ADDR_WIDTH),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!crsq_stall),
        .data_in  ({
            flush_fire || mshr_fire || mem_rsp_fire || creq_fire,
            flush_enable,
            mshr_enable,
            mrsq_enable,
            creq_enable && ~creq_rw,
            creq_enable && creq_rw,
            flush_enable ? `LINE_ADDR_WIDTH'(flush_addr) : (mshr_valid ? mshr_addr : (mem_rsp_valid ? mem_rsp_addr : creq_addr)),
            wdata_sel,
            mshr_valid ? mshr_wsel : creq_wsel,
            creq_byteen,
            mshr_valid ? mshr_tid : creq_tid,
            mshr_valid ? mshr_pmask : creq_pmask,
            mshr_valid ? mshr_tag : creq_tag,
            mshr_valid ? mshr_dequeue_id : mem_rsp_id
        }),
        .data_out ({valid_st0, is_flush_st0, is_mshr_st0, is_fill_st0, is_read_st0, is_write_st0, addr_st0, wdata_st0, wsel_st0, byteen_st0, req_tid_st0, pmask_st0, tag_st0, mshr_id_st0})
    );

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_wid_st0, debug_pc_st0} = tag_st0[0][`CACHE_REQ_INFO_RNG];
    end else begin
        assign {debug_wid_st0, debug_pc_st0} = 0;
    end
`endif

    wire do_fill_st0   = valid_st0 && is_fill_st0;
    wire do_flush_st0  = valid_st0 && is_flush_st0;
    wire do_lookup_st0 = valid_st0 && ~(is_fill_st0 || is_flush_st0);

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
        .clk       (clk),
        .reset     (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc  (debug_pc_st0),
        .debug_wid (debug_wid_st0),
    `endif
        .stall      (crsq_stall),

        // read/Fill
        .lookup    (do_lookup_st0),
        .addr      (addr_st0),        
        .fill      (do_fill_st0),
        .flush     (do_flush_st0),    
        .tag_match (tag_match_st0)
    );

    // we have a core request hit
    assign miss_st0 = (is_read_st0 || is_write_st0) && ~tag_match_st0;

    wire [MSHR_ADDR_WIDTH-1:0] mshr_id_a_st0 = (is_read_st0 || is_write_st0) ? mshr_alloc_id : mshr_id_st0;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `CACHE_LINE_WIDTH + NUM_PORTS * (WORD_SELECT_BITS + WORD_SIZE + `REQS_BITS + 1 + CORE_TAG_WIDTH) + MSHR_ADDR_WIDTH + 1),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (!crsq_stall),
        .data_in  ({valid_st0, is_mshr_st0, is_fill_st0, is_read_st0, is_write_st0, miss_st0, addr_st0, wdata_st0, wsel_st0, byteen_st0, req_tid_st0, pmask_st0, tag_st0, mshr_id_a_st0, mshr_pending_st0}),
        .data_out ({valid_st1, is_mshr_st1, is_fill_st1, is_read_st1, is_write_st1, miss_st1, addr_st1, wdata_st1, wsel_st1, byteen_st1, req_tid_st1, pmask_st1, tag_st1, mshr_id_st1,   mshr_pending_st1})
    ); 

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_wid_st1, debug_pc_st1} = tag_st1[0][`CACHE_REQ_INFO_RNG];
    end else begin        
        assign {debug_wid_st1, debug_pc_st1} = 0;
    end
`endif

    wire do_read_st0  = valid_st0 && is_read_st0;
    wire do_read_st1  = valid_st1 && is_read_st1;
    wire do_fill_st1  = valid_st1 && is_fill_st1;
    wire do_write_st1 = valid_st1 && is_write_st1;
    wire do_mshr_st1  = valid_st1 && is_mshr_st1;

    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] creq_data_st1 = wdata_st1[0 +: NUM_PORTS * `WORD_WIDTH];
    `UNUSED_VAR (wdata_st1)
    
    VX_data_access #(
        .BANK_ID        (BANK_ID),
        .CACHE_ID       (CACHE_ID),
        .CACHE_SIZE     (CACHE_SIZE),
        .CACHE_LINE_SIZE(CACHE_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .NUM_PORTS      (NUM_PORTS),
        .WORD_SIZE      (WORD_SIZE),
        .WRITE_ENABLE   (WRITE_ENABLE)
     ) data_access (
        .clk        (clk),
        .reset      (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc   (debug_pc_st1),
        .debug_wid  (debug_wid_st1),
    `endif
        .stall      (crsq_stall),

        .read       (do_read_st1 || do_mshr_st1),      
        .fill       (do_fill_st1),        
        .write      (do_write_st1 && !miss_st1),
        .addr       (addr_st1),
        .wsel       (wsel_st1),
        .pmask      (pmask_st1),
        .byteen     (byteen_st1),
        .fill_data  (wdata_st1),  
        .write_data (creq_data_st1),
        .read_data  (rdata_st1)
    );
    
    wire mshr_allocate = do_read_st0 && !crsq_stall;
    wire mshr_replay   = do_fill_st0 && !crsq_stall;
    wire mshr_lookup   = mshr_allocate;
    wire mshr_release  = do_read_st1 && !miss_st1 && !crsq_stall;

    VX_pending_size #( 
        .SIZE (MSHR_SIZE)
    ) mshr_pending_size (
        .clk   (clk),
        .reset (reset),
        .incr  (creq_fire && ~creq_rw),
        .decr  (mshr_fire || mshr_release),
        .full  (mshr_alm_full),
        `UNUSED_PIN (size),
        `UNUSED_PIN (empty)
    );

    VX_miss_resrv #(
        .BANK_ID            (BANK_ID),
        .CACHE_ID           (CACHE_ID),
        .CACHE_LINE_SIZE    (CACHE_LINE_SIZE),
        .NUM_BANKS          (NUM_BANKS),
        .NUM_PORTS          (NUM_PORTS),
        .WORD_SIZE          (WORD_SIZE),
        .NUM_REQS           (NUM_REQS),
        .MSHR_SIZE          (MSHR_SIZE),
        .CORE_TAG_WIDTH     (CORE_TAG_WIDTH)
    ) miss_resrv (
        .clk                (clk),
        .reset              (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .deq_debug_pc       (debug_pc_sel),
        .deq_debug_wid      (debug_wid_sel),
        .lkp_debug_pc       (debug_pc_st0),
        .lkp_debug_wid      (debug_wid_st0),
        .rel_debug_pc       (debug_pc_st1),
        .rel_debug_wid      (debug_wid_st1),
    `endif

        // allocate
        .allocate_valid     (mshr_allocate),
        .allocate_addr      (addr_st0),
        .allocate_data      ({wsel_st0, tag_st0, req_tid_st0, pmask_st0}),
        .allocate_id        (mshr_alloc_id),
        `UNUSED_PIN (allocate_ready),

        // lookup
        .lookup_valid       (mshr_lookup),
        .lookup_replay      (mshr_replay),
        .lookup_id          (mshr_alloc_id),
        .lookup_addr        (addr_st0),
        .lookup_match       (mshr_pending_st0),

        // fill
        .fill_valid         (mem_rsp_fire),
        .fill_id            (mem_rsp_id),
        .fill_addr          (mem_rsp_addr),

        // dequeue
        .dequeue_valid      (mshr_valid),
        .dequeue_id         (mshr_dequeue_id),
        .dequeue_addr       (mshr_addr),
        .dequeue_data       ({mshr_wsel, mshr_tag, mshr_tid, mshr_pmask}),
        .dequeue_ready      (mshr_ready),

        // release
        .release_valid      (mshr_release),
        .release_id         (mshr_id_st1)
    );

    // Enqueue core response

    wire [NUM_PORTS-1:0] crsq_pmask;
    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] crsq_data;
    wire [NUM_PORTS-1:0][`REQS_BITS-1:0] crsq_tid;
    wire [NUM_PORTS-1:0][CORE_TAG_WIDTH-1:0] crsq_tag;
    
    assign crsq_valid = (do_read_st1 && !miss_st1) 
                     || do_mshr_st1;

    assign crsq_stall = crsq_valid && !crsq_ready;

    assign crsq_pmask = pmask_st1;
    assign crsq_tid   = req_tid_st1;
    assign crsq_data  = rdata_st1;
    assign crsq_tag   = tag_st1;

    VX_elastic_buffer #(
        .DATAW   (NUM_PORTS * (CORE_TAG_WIDTH + 1 + `WORD_WIDTH + `REQS_BITS)),
        .SIZE    (CRSQ_SIZE),
        .OUT_REG (1)
    ) core_rsp_req (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsq_valid),        
        .data_in   ({crsq_tag, crsq_pmask, crsq_data, crsq_tid}),
        .ready_in  (crsq_ready),      
        .valid_out (core_rsp_valid),
        .data_out  ({core_rsp_tag, core_rsp_pmask, core_rsp_data, core_rsp_tid}),
        .ready_out (core_rsp_ready)
    );

    // Enqueue memory request

    wire mreq_push, mreq_pop, mreq_empty;
    wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] mreq_data;
    wire [NUM_PORTS-1:0][WORD_SIZE-1:0] mreq_byteen;
    wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] mreq_wsel;
    wire [NUM_PORTS-1:0] mreq_pmask;
    wire [`LINE_ADDR_WIDTH-1:0] mreq_addr;
    wire [MSHR_ADDR_WIDTH-1:0] mreq_id;
    wire mreq_rw;

    assign mreq_push = (do_read_st1 && miss_st1 && !mshr_pending_st1)
                     || do_write_st1;

    assign mreq_pop = mem_req_valid && mem_req_ready;

    assign mreq_rw   = WRITE_ENABLE && is_write_st1;
    assign mreq_addr = addr_st1;
    assign mreq_id   = mshr_id_st1;
    assign mreq_pmask= pmask_st1;
    assign mreq_wsel = wsel_st1;
    assign mreq_byteen = byteen_st1;
    assign mreq_data = creq_data_st1;

    VX_fifo_queue #(
        .DATAW    (1 + `LINE_ADDR_WIDTH + MSHR_ADDR_WIDTH + NUM_PORTS * (1 + WORD_SIZE + WORD_SELECT_BITS + `WORD_WIDTH)), 
        .SIZE     (MREQ_SIZE),
        .ALM_FULL (MREQ_SIZE-2),
        .OUT_REG  (1 == NUM_BANKS)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (mreq_push),
        .pop        (mreq_pop),
        .data_in    ({mreq_rw,    mreq_addr,    mreq_id,    mreq_pmask,    mreq_byteen,    mreq_wsel,    mreq_data}),
        .data_out   ({mem_req_rw, mem_req_addr, mem_req_id, mem_req_pmask, mem_req_byteen, mem_req_wsel, mem_req_data}),
        .empty      (mreq_empty),        
        .alm_full   (mreq_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),        
        `UNUSED_PIN (size)
    );

    assign mem_req_valid = !mreq_empty;

///////////////////////////////////////////////////////////////////////////////
                         
    `SCOPE_ASSIGN (valid_st0, valid_st0);
    `SCOPE_ASSIGN (valid_st1, valid_st1);
    `SCOPE_ASSIGN (is_fill_st0, is_fill_st0);
    `SCOPE_ASSIGN (is_mshr_st0, is_mshr_st0);
    `SCOPE_ASSIGN (miss_st0, miss_st0);  
    `SCOPE_ASSIGN (crsq_stall, crsq_stall);
    `SCOPE_ASSIGN (mreq_alm_full, mreq_alm_full);
    `SCOPE_ASSIGN (mshr_alm_full, mshr_alm_full);
    `SCOPE_ASSIGN (addr_st0, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID));
    `SCOPE_ASSIGN (addr_st1, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));

`ifdef PERF_ENABLE
    assign perf_read_misses  = do_read_st1 && miss_st1;
    assign perf_write_misses = do_write_st1 && miss_st1;
    assign perf_pipe_stalls  = crsq_stall || mreq_alm_full || mshr_alm_full;
    assign perf_mshr_stalls  = mshr_alm_full;
`endif

`ifdef DBG_TRACE_CACHE_BANK
    wire crsq_fire = crsq_valid && crsq_ready;
    wire pipeline_stall = (mshr_valid || mem_rsp_valid || creq_valid) 
                       && ~(mshr_fire || mem_rsp_fire || creq_fire);

    always @(posedge clk) begin
        if (pipeline_stall) begin
            dpi_trace("%d: *** cache%0d:%0d stall: crsq=%b, mreq=%b, mshr=%b\n", $time, CACHE_ID, BANK_ID, crsq_stall, mreq_alm_full, mshr_alm_full);
        end
        if (flush_enable) begin
            dpi_trace("%d: cache%0d:%0d flush: addr=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(flush_addr, BANK_ID));
        end
        if (mem_rsp_fire) begin
            dpi_trace("%d: cache%0d:%0d fill-rsp: addr=%0h, id=%0d, data=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mem_rsp_addr, BANK_ID), mem_rsp_id, mem_rsp_data);
        end
        if (mshr_fire) begin
            dpi_trace("%d: cache%0d:%0d mshr-pop: addr=%0h, tag=%0h, pmask=%b, tid=%0d, wid=%0d, PC=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mshr_addr, BANK_ID), mshr_tag, mshr_pmask, mshr_tid, debug_wid_sel, debug_pc_sel);
        end
        if (creq_fire) begin
            if (creq_rw)
                dpi_trace("%d: cache%0d:%0d core-wr-req: addr=%0h, tag=%0h, pmask=%b, tid=%0d, byteen=%b, data=%0h, wid=%0d, PC=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(creq_addr, BANK_ID), creq_tag, creq_pmask, creq_tid, creq_byteen, creq_data, debug_wid_sel, debug_pc_sel);
            else
                dpi_trace("%d: cache%0d:%0d core-rd-req: addr=%0h, tag=%0h, pmask=%b, tid=%0d, byteen=%b, wid=%0d, PC=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(creq_addr, BANK_ID), creq_tag, creq_pmask, creq_tid, creq_byteen, debug_wid_sel, debug_pc_sel);
        end
        if (crsq_fire) begin
            dpi_trace("%d: cache%0d:%0d core-rsp: addr=%0h, tag=%0h, pmask=%b, tid=%0d, data=%0h, wid=%0d, PC=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), crsq_tag, crsq_pmask, crsq_tid, crsq_data, debug_wid_st1, debug_pc_st1);
        end
        if (mreq_push) begin
            if (is_write_st1)
                dpi_trace("%d: cache%0d:%0d writeback: addr=%0h, data=%0h, byteen=%b, wid=%0d, PC=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mreq_addr, BANK_ID), mreq_data, mreq_byteen, debug_wid_st1, debug_pc_st1);
            else
                dpi_trace("%d: cache%0d:%0d fill-req: addr=%0h, id=%0d, wid=%0d, PC=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(mreq_addr, BANK_ID), mreq_id, debug_wid_st1, debug_pc_st1);
        end
    end    
`endif

endmodule