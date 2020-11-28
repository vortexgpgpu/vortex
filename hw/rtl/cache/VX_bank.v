`include "VX_cache_config.vh"

module VX_bank #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0, 

    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 1, 
    // Number of bankS
    parameter NUM_BANKS                     = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1, 
    // Number of Word requests per cycle
    parameter NUM_REQUESTS                  = 1, 

    // Core Request Queue Size
    parameter CREQ_SIZE                     = 1, 
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE                     = 1, 
    // DRAM Response Queue Size
    parameter DRFQ_SIZE                     = 1, 
    // Snoop Req Queue Size
    parameter SNRQ_SIZE                     = 1, 

    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 1, 
    // DRAM Request Queue Size
    parameter DREQ_SIZE                     = 1,
    // Snoop Response Size
    parameter SNPQ_SIZE                     = 1,

    // Enable cache writeable
     parameter WRITE_ENABLE                 = 0,

    // Enable dram update
    parameter DRAM_ENABLE                   = 0,
     
    // Enable cache flush
    parameter FLUSH_ENABLE                  = 0,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // Snooping request tag width
    parameter SNP_TAG_WIDTH                 = 1
) (
    `SCOPE_IO_VX_bank

    input wire clk,
    input wire reset,

    // Core Request    
    input wire [NUM_REQUESTS-1:0]                               core_req_valid,        
    input wire [`CORE_REQ_TAG_COUNT-1:0]                        core_req_rw,  
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]                core_req_byteen,
    input wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]         core_req_addr,
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]              core_req_data,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]    core_req_tag,
    output wire                                                 core_req_ready,
    
    // Core Response    
    output wire                         core_rsp_valid,
    output wire [`REQS_BITS-1:0]        core_rsp_tid,
    output wire [`WORD_WIDTH-1:0]       core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]    core_rsp_tag,
    input  wire                         core_rsp_ready,

    // DRAM request
    output wire                         dram_req_valid,
    output wire                         dram_req_rw,
    output wire [BANK_LINE_SIZE-1:0]    dram_req_byteen,    
    output wire [`LINE_ADDR_WIDTH-1:0]  dram_req_addr,
    output wire [`BANK_LINE_WIDTH-1:0]  dram_req_data,
    input  wire                         dram_req_ready,
    
    // DRAM response
    input  wire                         dram_rsp_valid,    
    input  wire [`LINE_ADDR_WIDTH-1:0]  dram_rsp_addr,
    input  wire [`BANK_LINE_WIDTH-1:0]  dram_rsp_data,
    output wire                         dram_rsp_ready,

    // Snoop Request
    input  wire                         snp_req_valid,
    input  wire [`LINE_ADDR_WIDTH-1:0]  snp_req_addr,
    input  wire                         snp_req_invalidate,
    input  wire [SNP_TAG_WIDTH-1:0]     snp_req_tag,
    output wire                         snp_req_ready,

    // Snoop Response
    output wire                         snp_rsp_valid,
    output wire [SNP_TAG_WIDTH-1:0]     snp_rsp_tag,
    input  wire                         snp_rsp_ready,

    // Misses
    output wire                         misses
);

`ifdef DBG_CACHE_REQ_INFO
    /* verilator lint_off UNUSED */
    wire[31:0]           debug_pc_st0;
    wire[`NR_BITS-1:0]   debug_rd_st0;
    wire[`NW_BITS-1:0]   debug_wid_st0;
    wire                 debug_rw_st0;    
    wire[WORD_SIZE-1:0]  debug_byteen_st0;
    wire[`REQS_BITS-1:0] debug_tid_st0;
    wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st0;

    wire[31:0]           debug_pc_st1;
    wire[`NR_BITS-1:0]   debug_rd_st1;
    wire[`NW_BITS-1:0]   debug_wid_st1;
    wire                 debug_rw_st1;    
    wire[WORD_SIZE-1:0]  debug_byteen_st1;
    wire[`REQS_BITS-1:0] debug_tid_st1;
    wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st1;

    wire[31:0]           debug_pc_st2;
    wire[`NR_BITS-1:0]   debug_rd_st2;
    wire[`NW_BITS-1:0]   debug_wid_st2;
    wire                 debug_rw_st2;    
    wire[WORD_SIZE-1:0]  debug_byteen_st2;
    wire[`REQS_BITS-1:0] debug_tid_st2;
    wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st2;

    wire[31:0]           debug_pc_st3;
    wire[`NR_BITS-1:0]   debug_rd_st3;
    wire[`NW_BITS-1:0]   debug_wid_st3;
    wire                 debug_rw_st3;    
    wire[WORD_SIZE-1:0]  debug_byteen_st3;
    wire[`REQS_BITS-1:0] debug_tid_st3;
    wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st3;
    /* verilator lint_on UNUSED */
`endif

    wire snrq_pop;
    wire snrq_empty;
    wire snrq_full;
    
    wire [`LINE_ADDR_WIDTH-1:0] snrq_addr_st0;
    wire snrq_invalidate_st0;
    wire [SNP_TAG_WIDTH-1:0] snrq_tag_st0;

    wire snp_req_fire = snp_req_valid && snp_req_ready;
    assign snp_req_ready = !snrq_full;

    VX_generic_queue #(
        .DATAW(`LINE_ADDR_WIDTH + 1 + SNP_TAG_WIDTH), 
        .SIZE(SNRQ_SIZE)
    ) snp_req_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (snp_req_fire),
        .pop     (snrq_pop),
        .data_in ({snp_req_addr,  snp_req_invalidate,  snp_req_tag}),        
        .data_out({snrq_addr_st0, snrq_invalidate_st0, snrq_tag_st0}),
        .empty   (snrq_empty),
        .full    (snrq_full),
        `UNUSED_PIN (size)
    );

    wire dfpq_pop;
    wire dfpq_empty;
    wire dfpq_full;
    wire [`LINE_ADDR_WIDTH-1:0] dfpq_addr_st0;
    wire [`BANK_LINE_WIDTH-1:0] dfpq_filldata_st0;    

    wire dram_rsp_fire = dram_rsp_valid && dram_rsp_ready;
    assign dram_rsp_ready = !dfpq_full;

    if (DRAM_ENABLE) begin
        VX_generic_queue #(
            .DATAW(`LINE_ADDR_WIDTH + $bits(dram_rsp_data)), 
            .SIZE(DRFQ_SIZE)
        ) dfp_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (dram_rsp_fire),
            .pop     (dfpq_pop),
            .data_in ({dram_rsp_addr, dram_rsp_data}),        
            .data_out({dfpq_addr_st0, dfpq_filldata_st0}),
            .empty   (dfpq_empty),
            .full    (dfpq_full),
            `UNUSED_PIN (size)
        );
    end else begin
        `UNUSED_VAR (dram_rsp_valid)
        `UNUSED_VAR (dram_rsp_addr)
        `UNUSED_VAR (dram_rsp_data)
        assign dfpq_empty        = 1;
        assign dfpq_full         = 0;
        assign dfpq_addr_st0     = 0;
        assign dfpq_filldata_st0 = 0;        
    end

    wire                        reqq_pop;
    wire                        reqq_empty;
    wire                        reqq_full;
    wire [`REQS_BITS-1:0]       reqq_tid_st0;
    wire                        reqq_rw_st0;  
    wire [WORD_SIZE-1:0]        reqq_byteen_st0;
`IGNORE_WARNINGS_BEGIN
    wire [`WORD_ADDR_WIDTH-1:0] reqq_addr_st0;
`IGNORE_WARNINGS_END    
    wire [`WORD_WIDTH-1:0]      reqq_writeword_st0;
    wire [CORE_TAG_WIDTH-1:0]   reqq_tag_st0;

    wire core_req_fire = (| core_req_valid) && core_req_ready;
    assign core_req_ready = !reqq_full;

    VX_bank_core_req_arb #(
        .WORD_SIZE        (WORD_SIZE),
        .NUM_REQUESTS     (NUM_REQUESTS),
        .CREQ_SIZE        (CREQ_SIZE),
        .CORE_TAG_WIDTH   (CORE_TAG_WIDTH),        
        .CORE_TAG_ID_BITS (CORE_TAG_ID_BITS)
    ) core_req_arb (
        .clk            (clk),
        .reset          (reset),

        // Enqueue
        .push           (core_req_fire),
        .tag_in         (core_req_tag),      
        .valids_in      (core_req_valid),
        .rw_in          (core_req_rw),
        .byteen_in      (core_req_byteen),
        .addr_in        (core_req_addr),
        .writedata_in   (core_req_data),  

        // Dequeue
        .pop            (reqq_pop),
        .tag_out        (reqq_tag_st0),
        .tid_out        (reqq_tid_st0),
        .rw_out         (reqq_rw_st0),
        .byteen_out     (reqq_byteen_st0),
        .addr_out       (reqq_addr_st0),
        .writedata_out  (reqq_writeword_st0),
        
        // States
        .empty          (reqq_empty),
        .full           (reqq_full)
    );    

    wire                                  mshr_pop;
    wire                                  mshr_full;
    wire                                  mshr_almfull;
    wire                                  mshr_valid_st0;
    wire[`REQS_BITS-1:0]                  mshr_tid_st0;
    wire [`LINE_ADDR_WIDTH-1:0]           mshr_addr_st0;
    wire [`UP(`WORD_SELECT_WIDTH)-1:0]    mshr_wsel_st0;
    wire [`WORD_WIDTH-1:0]                mshr_writeword_st0;
    wire [`REQ_TAG_WIDTH-1:0]             mshr_tag_st0;
    wire                                  mshr_rw_st0;  
    wire [WORD_SIZE-1:0]                  mshr_byteen_st0;
    wire                                  mshr_is_snp_st0;
    wire                                  mshr_snp_invalidate_st0;
    wire                                  is_mshr_miss_st2;
    wire                                  is_mshr_miss_st3;

    wire mshr_push_stall;
    wire cwbq_push_stall;    
    wire dwbq_push_stall;    
    wire snpq_push_stall;
    wire pipeline_stall;
    
    wire is_fill_st1;
    
    // determine which queue to pop next in piority order
    wire mshr_pop_unqual = mshr_valid_st0;
    wire dfpq_pop_unqual = !mshr_pop_unqual && !dfpq_empty;
    wire reqq_pop_unqual = !mshr_pop_unqual && !dfpq_pop_unqual && !reqq_empty && !mshr_almfull;
    wire snrq_pop_unqual = !mshr_pop_unqual && !dfpq_pop_unqual && !reqq_pop_unqual && !snrq_empty && !mshr_almfull;

    assign mshr_pop = mshr_pop_unqual && !pipeline_stall 
                   && !(is_mshr_miss_st2 || is_mshr_miss_st3); // stop if previous request was a miss
    assign dfpq_pop = dfpq_pop_unqual && !pipeline_stall;
    assign reqq_pop = reqq_pop_unqual && !pipeline_stall;
    assign snrq_pop = snrq_pop_unqual && !pipeline_stall;
    
    wire                                  is_fill_st0;
    wire                                  valid_st0;
    wire [`LINE_ADDR_WIDTH-1:0]           addr_st0;
    wire [`UP(`WORD_SELECT_WIDTH)-1:0]    wsel_st0;
    wire                                  is_mshr_st0;

    wire [`WORD_WIDTH-1:0]                writeword_st0;
    wire [`BANK_LINE_WIDTH-1:0]           writedata_st0;
    wire [`REQ_INST_META_WIDTH-1:0]       inst_meta_st0;
    wire                                  is_snp_st0;
    wire                                  snp_invalidate_st0;
    wire                                  mshr_pending_hazard_unqual_st0;
    
    wire                                  valid_st1;
    wire [`LINE_ADDR_WIDTH-1:0]           addr_st1;
    wire [`UP(`WORD_SELECT_WIDTH)-1:0]    wsel_st1;
    wire [`WORD_WIDTH-1:0]                writeword_st1;
    wire [`REQ_INST_META_WIDTH-1:0]       inst_meta_st1;    
    wire [`BANK_LINE_WIDTH-1:0]           writedata_st1;
    wire                                  is_snp_st1;
    wire                                  snp_invalidate_st1;
    wire                                  is_mshr_st1;
    wire                                  mshr_pending_hazard_st1;
    wire                                  miss_st3;
    wire                                  force_miss_st3;  
    wire [`LINE_ADDR_WIDTH-1:0]           addr_st3;

    assign is_mshr_st0 = mshr_pop_unqual;

    assign is_fill_st0 = dfpq_pop_unqual;

    assign valid_st0 = dfpq_pop || mshr_pop || reqq_pop || snrq_pop;

    assign addr_st0 = mshr_pop_unqual ? mshr_addr_st0 :
                      dfpq_pop_unqual ? dfpq_addr_st0 :
                      reqq_pop_unqual ? reqq_addr_st0[`LINE_SELECT_ADDR_RNG] :
                      snrq_pop_unqual ? snrq_addr_st0 :
                                        0;
    
    if (`WORD_SELECT_WIDTH != 0) begin
        assign wsel_st0 = reqq_pop_unqual ? reqq_addr_st0[`WORD_SELECT_WIDTH-1:0] :
                            mshr_pop_unqual ? mshr_wsel_st0 :
                                0; 
    end else begin 
        `UNUSED_VAR (mshr_wsel_st0)
        assign wsel_st0 = 0;
    end

    assign writedata_st0 = dfpq_filldata_st0;

    assign inst_meta_st0 = mshr_pop_unqual ? {`REQ_TAG_WIDTH'(mshr_tag_st0), mshr_rw_st0, mshr_byteen_st0, mshr_tid_st0} :
                           reqq_pop_unqual ? {`REQ_TAG_WIDTH'(reqq_tag_st0), reqq_rw_st0, reqq_byteen_st0, reqq_tid_st0} :
                           snrq_pop_unqual ? {`REQ_TAG_WIDTH'(snrq_tag_st0), 1'b0,        WORD_SIZE'(0),   `REQS_BITS'(0)} :
                                             0;

    assign is_snp_st0 = mshr_pop_unqual ? mshr_is_snp_st0 :
                            snrq_pop_unqual ? 1 :
                                0;

    assign snp_invalidate_st0 = mshr_pop_unqual ? mshr_snp_invalidate_st0 :
                                    snrq_pop_unqual ? snrq_invalidate_st0 :
                                        0;

    assign writeword_st0 = mshr_pop_unqual ? mshr_writeword_st0 :
                                reqq_pop_unqual ? reqq_writeword_st0 :
                                    0;

    // we have a miss in msrq or in stage 3 for the current address
    wire mshr_pending_hazard_st0 = mshr_pending_hazard_unqual_st0 
                                || ((miss_st3 || force_miss_st3) && (addr_st3 == addr_st0));

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st0, debug_rd_st0, debug_wid_st0, debug_tagid_st0, debug_rw_st0, debug_byteen_st0, debug_tid_st0} = inst_meta_st0;
    end else begin
        assign {debug_pc_st0, debug_rd_st0, debug_wid_st0, debug_tagid_st0, debug_rw_st0, debug_byteen_st0, debug_tid_st0} = 0;
    end
`endif

    VX_generic_register #(
        .N(1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `UP(`WORD_SELECT_WIDTH) + `WORD_WIDTH + `REQ_INST_META_WIDTH + 1 + `BANK_LINE_WIDTH)
    ) pipe_reg0 (
        .clk   (clk),
        .reset (reset),
        .stall (pipeline_stall),
        .flush (1'b0),
        .in    ({is_mshr_st0, is_snp_st0, snp_invalidate_st0, mshr_pending_hazard_st0, valid_st0, addr_st0, wsel_st0, writeword_st0, inst_meta_st0, is_fill_st0, writedata_st0}),
        .out   ({is_mshr_st1, is_snp_st1, snp_invalidate_st1, mshr_pending_hazard_st1, valid_st1, addr_st1, wsel_st1, writeword_st1, inst_meta_st1, is_fill_st1, writedata_st1})
    );

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st1, debug_rd_st1, debug_wid_st1, debug_tagid_st1, debug_rw_st1, debug_byteen_st1, debug_tid_st1} = inst_meta_st1;
    end else begin
        assign {debug_pc_st1, debug_rd_st1, debug_wid_st1, debug_tagid_st1, debug_rw_st1, debug_byteen_st1, debug_tid_st1} = 0;
    end
`endif

    wire[`TAG_SELECT_BITS-1:0]  readtag_st1;
    wire                        writeen_st1;
    wire                        writeen_st2;
    wire                        miss_st1;
    wire                        miss_st2;
    wire                        dirty_st1;
    wire                        mem_rw_st1;  
    wire [WORD_SIZE-1:0]        mem_byteen_st1;  
    wire                        force_miss_st2; 
    wire[`LINE_ADDR_WIDTH-1:0]  addr_st2;
`DEBUG_BEGIN
    wire [`REQ_TAG_WIDTH-1:0]   tag_st1;
    wire [`REQS_BITS-1:0]       tid_st1;
`DEBUG_END    

    assign {tag_st1, mem_rw_st1, mem_byteen_st1, tid_st1} = inst_meta_st1;

    // we have a matching previous request that missed alreedy
    wire st2_pending_hazard_st1 = (miss_st2 || force_miss_st2) && (addr_st2 == addr_st1);
    wire st3_pending_hazard_st1 = (miss_st3 || force_miss_st3) && (addr_st3 == addr_st1);

    // force miss to ensure commit order when a new request has pending previous requests to same block
    // also force a miss for msrq requests when previous requests got a miss
    wire force_miss_st1 = (valid_st1 && !is_mshr_st1 && !is_fill_st1 
                        && (mshr_pending_hazard_st1 || st2_pending_hazard_st1 || st3_pending_hazard_st1)) 
                       || (valid_st1 && is_mshr_st1 && is_mshr_miss_st2);
    
    VX_tag_access #(
        .BANK_ID        (BANK_ID),
        .CACHE_ID       (CACHE_ID),
        .CORE_TAG_ID_BITS(CORE_TAG_ID_BITS),
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .DRAM_ENABLE    (DRAM_ENABLE),
        .WRITE_ENABLE   (WRITE_ENABLE)
     ) tag_access (
        .clk            (clk),
        .reset          (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc       (debug_pc_st1),
        .debug_rd       (debug_rd_st1),
        .debug_wid      (debug_wid_st1),
        .debug_tagid    (debug_tagid_st1),
    `endif

        .stall          (pipeline_stall),

        // Inputs
        .valid_in       (valid_st1),
        .addr_in        (addr_st1),
        .is_write_in    (mem_rw_st1),
        .is_fill_in     (is_fill_st1),
        .is_snp_in      (is_snp_st1),
        .snp_invalidate_in(snp_invalidate_st1),
        .force_miss_in  (force_miss_st1),

        // Outputs
        .readtag_out    (readtag_st1),
        .miss_out       (miss_st1),
        .dirty_out      (dirty_st1),
        .writeen_out    (writeen_st1)
    );

    assign misses = miss_st1;
    
    wire                            valid_st2;    
    wire [`UP(`WORD_SELECT_WIDTH)-1:0] wsel_st2;
    wire [`WORD_WIDTH-1:0]          writeword_st2;
    wire [`WORD_WIDTH-1:0]          readword_st2;
    wire [`BANK_LINE_WIDTH-1:0]     readdata_st2;
    wire [`BANK_LINE_WIDTH-1:0]     writedata_st2;
    wire [WORD_SIZE-1:0]            mem_byteen_st2;  
    wire                            dirty_st2;
    wire [BANK_LINE_SIZE-1:0]       dirtyb_st2;
    wire [`REQ_INST_META_WIDTH-1:0] inst_meta_st2;
    wire [`TAG_SELECT_BITS-1:0]     readtag_st2;    
    wire                            is_fill_st2;
    wire                            is_snp_st2;
    wire                            snp_invalidate_st2;
    wire                            is_mshr_st2;
    
    VX_generic_register #(
        .N(1 + 1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `UP(`WORD_SELECT_WIDTH) + `WORD_WIDTH + `TAG_SELECT_BITS + 1 + 1 + `BANK_LINE_WIDTH + WORD_SIZE + `REQ_INST_META_WIDTH)
    ) pipe_reg1 (
        .clk   (clk),
        .reset (reset),
        .stall (pipeline_stall),
        .flush (1'b0),
        .in    ({is_mshr_st1, writeen_st1, force_miss_st1, is_snp_st1, snp_invalidate_st1, is_fill_st1, valid_st1, addr_st1, wsel_st1, writeword_st1, readtag_st1, miss_st1, dirty_st1, writedata_st1, mem_byteen_st1, inst_meta_st1}),
        .out   ({is_mshr_st2, writeen_st2, force_miss_st2, is_snp_st2, snp_invalidate_st2, is_fill_st2, valid_st2, addr_st2, wsel_st2, writeword_st2, readtag_st2, miss_st2, dirty_st2, writedata_st2, mem_byteen_st2, inst_meta_st2})
    );    

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st2, debug_rd_st2, debug_wid_st2, debug_tagid_st2, debug_rw_st2, debug_byteen_st2, debug_tid_st2} = inst_meta_st2;
    end else begin
        assign {debug_pc_st2, debug_rd_st2, debug_wid_st2, debug_tagid_st2, debug_rw_st2, debug_byteen_st2, debug_tid_st2} = 0;
    end
`endif

    assign is_mshr_miss_st2 = (miss_st2 || force_miss_st2) && is_mshr_st2;

    VX_data_access #(
        .BANK_ID        (BANK_ID),
        .CACHE_ID       (CACHE_ID),
        .CORE_TAG_ID_BITS(CORE_TAG_ID_BITS),
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .DRAM_ENABLE    (DRAM_ENABLE),
        .WRITE_ENABLE   (WRITE_ENABLE)
     ) data_access (
        .clk            (clk),
        .reset          (reset),

    `ifdef DBG_CACHE_REQ_INFO
        .debug_pc       (debug_pc_st2),
        .debug_rd       (debug_rd_st2),
        .debug_wid      (debug_wid_st2),
        .debug_tagid    (debug_tagid_st2),
    `endif

        .stall          (pipeline_stall),

        // Inputs
        .valid_in       (valid_st2),
        .addr_in        (addr_st2),
        .writeen_in     (writeen_st2),
        .is_fill_in     (is_fill_st2),
        .wordsel_in     (wsel_st2),
        .byteen_in      (mem_byteen_st2),
        .writeword_in   (writeword_st2),
        .writedata_in   (writedata_st2),

        // Outputs
        .readword_out   (readword_st2),
        .readdata_out   (readdata_st2),
        .dirtyb_out     (dirtyb_st2)
    );

    wire                            valid_st3;  
    wire [`UP(`WORD_SELECT_WIDTH)-1:0] wsel_st3;
    wire [`WORD_WIDTH-1:0]          writeword_st3;
    wire [`WORD_WIDTH-1:0]          readword_st3;
    wire [`BANK_LINE_WIDTH-1:0]     readdata_st3;
    wire [BANK_LINE_SIZE-1:0]       dirtyb_st3;
    wire [`REQ_INST_META_WIDTH-1:0] inst_meta_st3;
    wire [`TAG_SELECT_BITS-1:0]     readtag_st3;  
    wire                            is_snp_st3;
    wire                            snp_invalidate_st3;
    wire                            is_mshr_st3;
    wire                            send_core_rsp_st3;
    wire                            send_dwb_req_st3;    
    wire                            do_writeback_st3;
    wire                            send_snp_rsp_st3;
    wire                            incoming_fill_st3;

    wire send_core_rsp_st2 = valid_st2 && !is_fill_st2 && !is_snp_st2 && !miss_st2 && !force_miss_st2;

    // check if a matching fill request is comming
    wire incoming_fill_dfp_st2 = dram_rsp_fire && (addr_st2 == dram_rsp_addr);
    wire incoming_fill_st0_st2 = !dfpq_empty   && (addr_st2 == dfpq_addr_st0);
    wire incoming_fill_st1_st2 = is_fill_st1   && (addr_st2 == addr_st1);
    wire incoming_fill_st2 = incoming_fill_dfp_st2 
                          || incoming_fill_st0_st2 
                          || incoming_fill_st1_st2;

    wire send_fill_req_st2 = valid_st2 && miss_st2 
                          && (!force_miss_st2 
                           || (is_mshr_st2 && addr_st2 != addr_st3))
                          && !incoming_fill_st2;

    wire do_writeback_st2  = valid_st2 && dirty_st2 
                          && (is_fill_st2 
                           || (!force_miss_st2 && is_snp_st2));

    wire send_dwb_req_st2 = send_fill_req_st2 || do_writeback_st2;

    wire send_snp_rsp_st2 = valid_st2 && is_snp_st2 && !force_miss_st2;
    
    VX_generic_register #(
        .N(1+ 1+ 1 + 1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `UP(`WORD_SELECT_WIDTH) + `WORD_WIDTH + `WORD_WIDTH + `BANK_LINE_WIDTH + `TAG_SELECT_BITS + 1 + 1 + BANK_LINE_SIZE + `REQ_INST_META_WIDTH)
    ) pipe_reg2 (
        .clk   (clk),
        .reset (reset),
        .stall (pipeline_stall),
        .flush (1'b0),
        .in    ({is_mshr_st2, incoming_fill_st2, send_core_rsp_st2, send_dwb_req_st2, do_writeback_st2, send_snp_rsp_st2, force_miss_st2, is_snp_st2, snp_invalidate_st2, valid_st2, addr_st2, wsel_st2, writeword_st2, readword_st2, readdata_st2, readtag_st2, miss_st2, dirtyb_st2, inst_meta_st2}),
        .out   ({is_mshr_st3, incoming_fill_st3, send_core_rsp_st3, send_dwb_req_st3, do_writeback_st3, send_snp_rsp_st3, force_miss_st3, is_snp_st3, snp_invalidate_st3, valid_st3, addr_st3, wsel_st3, writeword_st3, readword_st3, readdata_st3, readtag_st3, miss_st3, dirtyb_st3, inst_meta_st3})
    );    

`ifdef DBG_CACHE_REQ_INFO
    if (CORE_TAG_WIDTH != CORE_TAG_ID_BITS && CORE_TAG_ID_BITS != 0) begin
        assign {debug_pc_st3, debug_rd_st3, debug_wid_st3, debug_tagid_st3, debug_rw_st3, debug_byteen_st3, debug_tid_st3} = inst_meta_st3;
    end else begin
        assign {debug_pc_st3, debug_rd_st3, debug_wid_st3, debug_tagid_st3, debug_rw_st3, debug_byteen_st3, debug_tid_st3} = 0;
    end
`endif

    assign is_mshr_miss_st3 = (miss_st3 || force_miss_st3) && is_mshr_st3;

    // Enqueue to miss reserv if it's a valid miss

    wire[`REQS_BITS-1:0]        req_tid_st3;
    wire[`REQ_TAG_WIDTH-1:0]    req_tag_st3;
    wire                        req_rw_st3;
    wire[WORD_SIZE-1:0]         req_byteen_st3;

    wire mshr_push_unqual = miss_st3 || force_miss_st3;
    assign mshr_push_stall = mshr_push_unqual && mshr_full;

    wire mshr_push = mshr_push_unqual
                  && !mshr_full 
                  && !cwbq_push_stall 
                  && !dwbq_push_stall
                  && !snpq_push_stall;  

    assign {req_tag_st3, req_rw_st3, req_byteen_st3, req_tid_st3} = inst_meta_st3;

    if (DRAM_ENABLE) begin
        wire mshr_dequeue_st3 = valid_st3 && is_mshr_st3 && !mshr_push_unqual && !pipeline_stall;

        // mark msrq entry that match DRAM fill as 'ready'
        wire update_ready_st0 = dfpq_pop;

        // push missed requests as 'ready' if it was a forced miss but actually had a hit 
        // or the fill request is comming for the missed block
        wire mshr_init_ready_state_st3 = !miss_st3 || incoming_fill_st3; 

        VX_cache_miss_resrv #(
            .BANK_ID            (BANK_ID),
            .CACHE_ID           (CACHE_ID),      
            .CORE_TAG_ID_BITS   (CORE_TAG_ID_BITS),
            .BANK_LINE_SIZE     (BANK_LINE_SIZE),
            .NUM_BANKS          (NUM_BANKS),
            .WORD_SIZE          (WORD_SIZE),
            .NUM_REQUESTS       (NUM_REQUESTS),
            .MSHR_SIZE          (MSHR_SIZE),
            .CORE_TAG_WIDTH     (CORE_TAG_WIDTH),
            .SNP_TAG_WIDTH      (SNP_TAG_WIDTH)
        ) cache_miss_resrv (
            .clk                (clk),
            .reset              (reset),

        `ifdef DBG_CACHE_REQ_INFO
            .debug_pc_st0       (debug_pc_st0),
            .debug_rd_st0       (debug_rd_st0),
            .debug_wid_st0      (debug_wid_st0),
            .debug_tagid_st0    (debug_tagid_st0),
            .debug_pc_st3       (debug_pc_st3),
            .debug_rd_st3       (debug_rd_st3),
            .debug_wid_st3      (debug_wid_st3),
            .debug_tagid_st3    (debug_tagid_st3),
        `endif

            // enqueue
            .enqueue_st3        (mshr_push),        
            .enqueue_addr_st3   (addr_st3),
            .enqueue_wsel_st3   (wsel_st3),
            .enqueue_data_st3   (writeword_st3),
            .enqueue_tid_st3    (req_tid_st3),
            .enqueue_tag_st3    (req_tag_st3),
            .enqueue_rw_st3     (req_rw_st3),
            .enqueue_byteen_st3 (req_byteen_st3),
            .enqueue_is_snp_st3 (is_snp_st3),
            .enqueue_snp_inv_st3(snp_invalidate_st3),
            .enqueue_mshr_st3   (is_mshr_st3),
            .enqueue_ready_st3  (mshr_init_ready_state_st3),
            .enqueue_full       (mshr_full),
            .enqueue_almfull    (mshr_almfull),

            // fill
            .update_ready_st0   (update_ready_st0),
            .addr_st0           (addr_st0),
            .pending_hazard_st0 (mshr_pending_hazard_unqual_st0),
            
            // dequeue
            .schedule_st0       (mshr_pop),        
            .dequeue_valid_st0  (mshr_valid_st0),
            .dequeue_addr_st0   (mshr_addr_st0),
            .dequeue_wsel_st0   (mshr_wsel_st0),
            .dequeue_data_st0   (mshr_writeword_st0),
            .dequeue_tid_st0    (mshr_tid_st0),
            .dequeue_tag_st0    (mshr_tag_st0),
            .dequeue_rw_st0     (mshr_rw_st0),
            .dequeue_byteen_st0 (mshr_byteen_st0),
            .dequeue_is_snp_st0 (mshr_is_snp_st0),
            .dequeue_snp_inv_st0(mshr_snp_invalidate_st0),
            .dequeue_st3        (mshr_dequeue_st3)
        );
    end else begin
        `UNUSED_VAR (valid_st3)        
        `UNUSED_VAR (mshr_push)
        `UNUSED_VAR (wsel_st3)
        `UNUSED_VAR (writeword_st3)
        `UNUSED_VAR (snp_invalidate_st3)
        `UNUSED_VAR (req_byteen_st3)
        `UNUSED_VAR (is_snp_st3)
        `UNUSED_VAR (incoming_fill_st3)
        assign mshr_pending_hazard_unqual_st0 = 0;
        assign mshr_full = 0;
        assign mshr_almfull = 0;
        assign mshr_valid_st0 = 0;
        assign mshr_addr_st0 = 0;
        assign mshr_wsel_st0 = 0;
        assign mshr_writeword_st0 = 0;
        assign mshr_tid_st0 = 0;
        assign mshr_tag_st0 = 0;
        assign mshr_rw_st0 = 0;
        assign mshr_byteen_st0 = 0;
        assign mshr_is_snp_st0 = 0;
        assign mshr_snp_invalidate_st0 = 0;
    end

    // Enqueue core response
     
    wire cwbq_empty, cwbq_full;

    wire cwbq_push_unqual = send_core_rsp_st3 && !req_rw_st3;
    assign cwbq_push_stall = cwbq_push_unqual && cwbq_full;

    wire cwbq_push = cwbq_push_unqual
                  && !cwbq_full
                  && !mshr_push_stall
                  && !dwbq_push_stall
                  && !snpq_push_stall;

    wire cwbq_pop = core_rsp_valid && core_rsp_ready;

    wire [`REQS_BITS-1:0]     cwbq_tid_st3  = req_tid_st3;
    wire [CORE_TAG_WIDTH-1:0] cwbq_tag_st3  = CORE_TAG_WIDTH'(req_tag_st3);
    wire [`WORD_WIDTH-1:0]    cwbq_data_st3 = readword_st3;
  
    VX_generic_queue #(
        .DATAW(`REQS_BITS + CORE_TAG_WIDTH + `WORD_WIDTH), 
        .SIZE(CWBQ_SIZE)
    ) cwb_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (cwbq_push),
        .pop     (cwbq_pop),
        .data_in ({cwbq_tid_st3, cwbq_tag_st3, cwbq_data_st3}),        
        .data_out({core_rsp_tid, core_rsp_tag, core_rsp_data}),
        .empty   (cwbq_empty),
        .full    (cwbq_full),
        `UNUSED_PIN (size)
    );

    assign core_rsp_valid = !cwbq_empty;

    // Enqueue DRAM request

    wire dwbq_empty, dwbq_full;
    
    wire dwbq_push_unqual = send_dwb_req_st3;    

    assign dwbq_push_stall = dwbq_push_unqual && dwbq_full;
    
    wire dwbq_push = dwbq_push_unqual
                  && !dwbq_full
                  && !mshr_push_stall
                  && !cwbq_push_stall
                  && !snpq_push_stall;

    wire dwbq_pop = dram_req_valid && dram_req_ready;

    wire [`LINE_ADDR_WIDTH-1:0] dwbq_addr = do_writeback_st3 ? {readtag_st3, addr_st3[`LINE_SELECT_BITS-1:0]} : 
                                                               addr_st3;

    wire [BANK_LINE_SIZE-1:0] dwbq_byteen = do_writeback_st3 ? dirtyb_st3 : {BANK_LINE_SIZE{1'b1}};

    if (DRAM_ENABLE) begin       
        VX_generic_queue #(
            .DATAW(1 + BANK_LINE_SIZE + `LINE_ADDR_WIDTH + `BANK_LINE_WIDTH), 
            .SIZE(DREQ_SIZE)
        ) dwb_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (dwbq_push),
            .pop     (dwbq_pop),
            .data_in ({do_writeback_st3, dwbq_byteen,     dwbq_addr,     readdata_st3}),        
            .data_out({dram_req_rw,      dram_req_byteen, dram_req_addr, dram_req_data}),
            .empty   (dwbq_empty),
            .full    (dwbq_full),
            `UNUSED_PIN (size)
        );
    end else begin
        `UNUSED_VAR (dwbq_push)
        `UNUSED_VAR (dwbq_pop)
        `UNUSED_VAR (dwbq_addr)
        `UNUSED_VAR (dwbq_byteen)
        `UNUSED_VAR (readtag_st3)
        `UNUSED_VAR (dirtyb_st3)
        `UNUSED_VAR (readdata_st3)         
        `UNUSED_VAR (dram_req_ready)
        assign dwbq_empty = 1;       
        assign dwbq_full = 0;
        assign dram_req_rw = 0;
        assign dram_req_byteen = 0;
        assign dram_req_addr = 0;
        assign dram_req_data = 0;
    end  

    assign dram_req_valid = !dwbq_empty;     

    // Enqueue snoop response

    wire snpq_empty, snpq_full;
    
    wire snpq_push_unqual = send_snp_rsp_st3;    

    assign snpq_push_stall = snpq_push_unqual && snpq_full;

    wire snpq_push = snpq_push_unqual
                  && !snpq_full
                  && !mshr_push_stall
                  && !cwbq_push_stall
                  && !dwbq_push_stall;

    wire snpq_pop = snp_rsp_valid && snp_rsp_ready;

    wire [SNP_TAG_WIDTH-1:0] snpq_tag_st3 = SNP_TAG_WIDTH'(req_tag_st3);

    if (FLUSH_ENABLE) begin
        VX_generic_queue #(
            .DATAW  (SNP_TAG_WIDTH), 
            .SIZE   (SNPQ_SIZE)
        ) snp_rsp_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (snpq_push),
            .pop     (snpq_pop),
            .data_in (snpq_tag_st3),        
            .data_out(snp_rsp_tag),
            .empty   (snpq_empty),
            .full    (snpq_full),
            `UNUSED_PIN (size)
        );
    end else begin
        `UNUSED_VAR (snpq_push) 
        `UNUSED_VAR (snpq_pop) 
        `UNUSED_VAR (snpq_tag_st3)        
        assign snpq_empty = 1;
        assign snpq_full = 0;
        assign snp_rsp_tag = 0;
        `UNUSED_VAR (snp_rsp_ready)
    end 

    assign snp_rsp_valid = !snpq_empty
                        && dwbq_empty; // ensure all writebacks are sent

    // bank pipeline stall
    assign pipeline_stall = mshr_push_stall
                         || cwbq_push_stall 
                         || dwbq_push_stall 
                         || snpq_push_stall;
                         
    `SCOPE_ASSIGN (valid_st0, valid_st0);
    `SCOPE_ASSIGN (valid_st1, valid_st1);
    `SCOPE_ASSIGN (valid_st2, valid_st2);
    `SCOPE_ASSIGN (valid_st3, valid_st3);

    `SCOPE_ASSIGN (is_mshr_st1,    is_mshr_st1);
    `SCOPE_ASSIGN (miss_st1,       miss_st1);
    `SCOPE_ASSIGN (dirty_st1,      dirty_st1);
    `SCOPE_ASSIGN (force_miss_st1, force_miss_st1);
    `SCOPE_ASSIGN (pipeline_stall, pipeline_stall);

    `SCOPE_ASSIGN (addr_st0, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID));
    `SCOPE_ASSIGN (addr_st1, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));
    `SCOPE_ASSIGN (addr_st2, `LINE_TO_BYTE_ADDR(addr_st2, BANK_ID));
    `SCOPE_ASSIGN (addr_st3, `LINE_TO_BYTE_ADDR(addr_st3, BANK_ID));

`ifdef DBG_PRINT_CACHE_BANK
    wire incoming_fill_dfp_st3 = dram_rsp_fire && (addr_st3 == dram_rsp_addr);
    always @(posedge clk) begin        
        if (miss_st3 && (incoming_fill_st3 || incoming_fill_dfp_st3)) begin
            $display("%t: incoming fill - addr=%0h, st3=%b, dfp=%b", $time, `LINE_TO_BYTE_ADDR(addr_st3, BANK_ID), incoming_fill_st3, incoming_fill_dfp_st3);
            assert(!is_mshr_st3);
        end
        if (pipeline_stall) begin
            $display("%t: cache%0d:%0d pipeline-stall: msrq=%b, cwbq=%b, dwbq=%b, snpq=%b", $time, CACHE_ID, BANK_ID, mshr_push_stall, cwbq_push_stall, dwbq_push_stall, snpq_push_stall);
        end
        if (dfpq_pop) begin
            $display("%t: cache%0d:%0d fill-rsp: addr=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), dfpq_filldata_st0);
        end
        if (reqq_pop) begin
            if (reqq_rw_st0)
                $display("%t: cache%0d:%0d core-wr-req: addr=%0h, tag=%0h, tid=%0d, byteen=%b, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), reqq_tag_st0, reqq_tid_st0, reqq_byteen_st0, reqq_writeword_st0, debug_wid_st0, debug_pc_st0);
            else
                $display("%t: cache%0d:%0d core-rd-req: addr=%0h, tag=%0h, tid=%0d, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), reqq_tag_st0, reqq_tid_st0, reqq_byteen_st0, reqq_writeword_st0, debug_wid_st0, debug_pc_st0);
        end
        if (snrq_pop) begin
            $display("%t: cache%0d:%0d snp-req: addr=%0h, tag=%0h, invalidate=%0d", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st0, BANK_ID), snrq_tag_st0, snrq_invalidate_st0);
        end
        if (cwbq_push) begin
            $display("%t: cache%0d:%0d core-rsp: addr=%0h, tag=%0h, tid=%0d, data=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st3, BANK_ID), cwbq_tag_st3, cwbq_tid_st3, cwbq_data_st3, debug_wid_st3, debug_pc_st3);
        end
        if (dwbq_push) begin
            if (do_writeback_st3)
                $display("%t: cache%0d:%0d writeback: addr=%0h, data=%0h, byteen=%b, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dwbq_addr, BANK_ID), readdata_st3, dirtyb_st3, debug_wid_st3, debug_pc_st3);
            else
                $display("%t: cache%0d:%0d fill-req: addr=%0h, wid=%0d, PC=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dwbq_addr, BANK_ID), debug_wid_st3, debug_pc_st3);
        end
        if (snpq_push) begin
            $display("%t: cache%0d:%0d snp-rsp: addr=%0h, tag=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st3, BANK_ID), snpq_tag_st3);
        end
    end    
`endif

endmodule