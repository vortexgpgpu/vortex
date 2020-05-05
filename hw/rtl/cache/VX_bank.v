`include "VX_cache_config.vh"
`include "VX_define.vh"
module VX_bank #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
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
     parameter WRITE_ENABLE                 = 1,

     // Enable dram update
     parameter DRAM_ENABLE                  = 1,

    // Enable snoop forwarding
    parameter SNOOP_FORWARDING_ENABLE       = 0,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    input wire                                    clk,
    input wire                                    reset,

    // Core Request    
    input wire                                    core_req_ready,    
    input wire [NUM_REQUESTS-1:0]                 core_req_valids,        
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0] core_req_read,  
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0] core_req_write,
    input wire [NUM_REQUESTS-1:0][31:0]           core_req_addr,
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0] core_req_data,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire                                   core_req_full,

    // Core Response    
    output wire                                   core_rsp_valid,
    output wire [`LOG2UP(NUM_REQUESTS)-1:0]       core_rsp_tid,
    output wire [`WORD_WIDTH-1:0]                 core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]              core_rsp_tag,
    input  wire                                   core_rsp_pop,

    // Dram Fill Requests
    output wire                                   dram_fill_req_valid,
    output wire[`LINE_ADDR_WIDTH-1:0]             dram_fill_req_addr,
    output wire                                   dram_fill_req_is_snp,
    input  wire                                   dram_fill_req_full,

    // Dram Fill Response
    input  wire                                   dram_fill_rsp_valid,    
    input  wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] dram_fill_rsp_data,
    input  wire [`LINE_ADDR_WIDTH-1:0]            dram_fill_rsp_addr,
    output wire                                   dram_fill_rsp_ready,

    // Dram WB Requests    
    output wire                                   dram_wb_req_valid,
    output wire [`LINE_ADDR_WIDTH-1:0]            dram_wb_req_addr,
    output wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] dram_wb_req_data,
    input  wire                                   dram_wb_req_pop,

    // Snp Request
    input  wire                                   snp_req_valid,
    input  wire [`LINE_ADDR_WIDTH-1:0]            snp_req_addr,
    output wire                                   snp_req_full,

    output wire                                   snp_fwd_valid,
    output wire [`LINE_ADDR_WIDTH-1:0]            snp_fwd_addr,
    input  wire                                   snp_fwd_pop
);

    reg snoop_state = 0;

    always @(posedge clk) begin
        if (reset) begin
            snoop_state <= 0;
        end else begin
            snoop_state <= (snoop_state | snp_req_valid) && SNOOP_FORWARDING_ENABLE;
        end
    end

    wire       snrq_pop;
    wire       snrq_empty;

    wire       snrq_valid_st0;
    wire[`LINE_ADDR_WIDTH-1:0] snrq_addr_st0;

    assign snrq_valid_st0 = !snrq_empty;
    
    VX_generic_queue #(
        .DATAW($bits(snp_req_addr)), 
        .SIZE(SNRQ_SIZE)
    ) snr_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (snp_req_valid),
        .data_in (snp_req_addr),
        .pop     (snrq_pop),
        .data_out(snrq_addr_st0),
        .empty   (snrq_empty),
        .full    (snp_req_full)
    );

    wire                      dfpq_pop;
    wire                      dfpq_empty;
    wire                      dfpq_full;
    wire [`LINE_ADDR_WIDTH-1:0] dfpq_addr_st0;
    wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] dfpq_filldata_st0;

    assign dram_fill_rsp_ready = !dfpq_full;

    VX_generic_queue #(
        .DATAW($bits(dram_fill_rsp_addr) + $bits(dram_fill_rsp_data)), 
        .SIZE(DFPQ_SIZE)
    ) dfp_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (dram_fill_rsp_valid),
        .data_in ({dram_fill_rsp_addr, dram_fill_rsp_data}),
        .pop     (dfpq_pop),
        .data_out({dfpq_addr_st0, dfpq_filldata_st0}),
        .empty   (dfpq_empty),
        .full    (dfpq_full)
    );

    wire                                  reqq_pop;
    wire                                  reqq_push;
    wire                                  reqq_empty;
    wire                                  reqq_req_st0;
    wire[`LOG2UP(NUM_REQUESTS)-1:0]       reqq_req_tid_st0;
`IGNORE_WARNINGS_BEGIN
    wire [31:0]                           reqq_req_addr_st0;
`IGNORE_WARNINGS_END    
    wire [`WORD_WIDTH-1:0]                reqq_req_writeword_st0;
    wire [CORE_TAG_WIDTH-1:0]             reqq_req_tag_st0;
    wire [`BYTE_EN_BITS-1:0]              reqq_req_mem_read_st0;  
    wire [`BYTE_EN_BITS-1:0]              reqq_req_mem_write_st0;

    assign reqq_push = core_req_ready && (|core_req_valids);

    VX_cache_req_queue #(
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
        .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE),
        .CORE_TAG_WIDTH         (CORE_TAG_WIDTH),        
        .CORE_TAG_ID_BITS       (CORE_TAG_ID_BITS)
    ) req_queue (
        .clk                   (clk),
        .reset                 (reset),
        // Enqueue
        .reqq_push             (reqq_push),
        .bank_valids           (core_req_valids),
        .bank_addr             (core_req_addr),
        .bank_writedata        (core_req_data),
        .bank_tag              (core_req_tag),
        .bank_mem_read         (core_req_read),
        .bank_mem_write        (core_req_write),

        // Dequeue
        .reqq_pop              (reqq_pop),
        .reqq_req_st0          (reqq_req_st0),
        .reqq_req_tid_st0      (reqq_req_tid_st0),
        .reqq_req_addr_st0     (reqq_req_addr_st0),
        .reqq_req_writedata_st0(reqq_req_writeword_st0),
        .reqq_req_tag_st0      (reqq_req_tag_st0),
        .reqq_req_mem_read_st0 (reqq_req_mem_read_st0),
        .reqq_req_mem_write_st0(reqq_req_mem_write_st0),
        .reqq_empty            (reqq_empty),
        .reqq_full             (core_req_full)
    );

    wire                                  mrvq_pop;
    wire                                  mrvq_full;
    wire                                  mrvq_stop;
    wire                                  mrvq_valid_st0;
    wire[`LOG2UP(NUM_REQUESTS)-1:0]       mrvq_tid_st0;
    wire [`LINE_ADDR_WIDTH-1:0]           mrvq_addr_st0;
    wire [`BASE_ADDR_BITS-1:0]            mrvq_wsel_st0;
    wire [`WORD_WIDTH-1:0]                mrvq_writeword_st0;
    wire [CORE_TAG_WIDTH-1:0]             mrvq_tag_st0;
    wire [`BYTE_EN_BITS-1:0]              mrvq_mem_read_st0;  
    wire [`BYTE_EN_BITS-1:0]              mrvq_mem_write_st0;
 
    wire                                  miss_add;
    wire[`LINE_ADDR_WIDTH-1:0]            miss_add_addr;
    wire[`BASE_ADDR_BITS-1:0]             miss_add_wsel;
    wire[`WORD_WIDTH-1:0]                 miss_add_data;
    wire[`LOG2UP(NUM_REQUESTS)-1:0]       miss_add_tid;
    wire[CORE_TAG_WIDTH-1:0]              miss_add_tag;
    wire[`BYTE_EN_BITS-1:0]               miss_add_mem_read;
    wire[`BYTE_EN_BITS-1:0]               miss_add_mem_write;

    wire[`LINE_ADDR_WIDTH-1:0]            addr_st2;
    wire                                  is_fill_st2;

    wire stall_bank_pipe;
    reg  is_fill_in_pipe;
    
    wire            is_fill_st1       [STAGE_1_CYCLES-1:0];
`DEBUG_BEGIN
    wire            going_to_write_st1[STAGE_1_CYCLES-1:0];    
`DEBUG_END
    
    integer i;
    always @(*) begin
        is_fill_in_pipe = 0;
        for (i = 0; i < STAGE_1_CYCLES; i=i+1) begin
            if (is_fill_st1[i]) begin
                is_fill_in_pipe = 1;
            end
        end

        if (is_fill_st2) begin
            is_fill_in_pipe = 1;
        end
    end

    assign mrvq_pop = mrvq_valid_st0 && !stall_bank_pipe;
    assign dfpq_pop = !mrvq_pop && !dfpq_empty && !stall_bank_pipe;
    assign reqq_pop = !mrvq_stop && !mrvq_pop && !dfpq_pop && !reqq_empty && reqq_req_st0 && !stall_bank_pipe && !is_fill_st1[0] && !is_fill_in_pipe;
    assign snrq_pop = !reqq_pop && !reqq_pop && !mrvq_pop && !dfpq_pop && snrq_valid_st0 && !stall_bank_pipe;

    wire                                  qual_is_fill_st0;
    wire                                  qual_valid_st0;
    wire [`LINE_ADDR_WIDTH-1:0]           qual_addr_st0;
    wire [`WORD_SELECT_ADDR_END:0]        qual_wsel_st0;

    wire [`WORD_WIDTH-1:0]                qual_writeword_st0;
    wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] qual_writedata_st0;
    wire [`REQ_INST_META_WIDTH-1:0]       qual_inst_meta_st0;
    wire                                  qual_going_to_write_st0;
    wire                                  qual_is_snp;

    wire                                  valid_st1         [STAGE_1_CYCLES-1:0];
    wire [`LINE_ADDR_WIDTH-1:0]           addr_st1          [STAGE_1_CYCLES-1:0];
    wire [`WORD_SELECT_ADDR_END:0]        wsel_st1          [STAGE_1_CYCLES-1:0];
    wire [`WORD_WIDTH-1:0]                writeword_st1     [STAGE_1_CYCLES-1:0];
    wire [`REQ_INST_META_WIDTH-1:0]       inst_meta_st1     [STAGE_1_CYCLES-1:0];    
    wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] writedata_st1[STAGE_1_CYCLES-1:0];
    wire                                  is_snp_st1        [STAGE_1_CYCLES-1:0];

    assign qual_is_fill_st0 = dfpq_pop;

    assign qual_valid_st0   = dfpq_pop || mrvq_pop || reqq_pop || snrq_pop;

    assign qual_addr_st0    = dfpq_pop ? dfpq_addr_st0     :
                              mrvq_pop ? mrvq_addr_st0     :
                              reqq_pop ? reqq_req_addr_st0[31:`LINE_SELECT_ADDR_START] :
                              snrq_pop ? snrq_addr_st0     :
                                         0;

    assign qual_wsel_st0   =  reqq_pop ? reqq_req_addr_st0[`BASE_ADDR_BITS-1:0] :
                              mrvq_pop ? mrvq_wsel_st0 :
                                         0;

    assign qual_writedata_st0 = dfpq_pop ? dfpq_filldata_st0 : 57;

    assign qual_inst_meta_st0 = mrvq_pop ? {mrvq_tag_st0    , mrvq_mem_read_st0,     mrvq_mem_write_st0,     mrvq_tid_st0} :
                                reqq_pop ? {reqq_req_tag_st0, reqq_req_mem_read_st0, reqq_req_mem_write_st0, reqq_req_tid_st0} :
                                           0;

    assign qual_going_to_write_st0 = dfpq_pop ? 1 :
                                        (mrvq_pop && (mrvq_mem_write_st0 != `BYTE_EN_NO)) ? 1 :
                                            (reqq_pop && (reqq_req_mem_write_st0 != `BYTE_EN_NO)) ? 1 :
                                                (snrq_pop) ? 1 :
                                                    0;

     assign qual_is_snp = snrq_pop ? 1 : 0;

    assign qual_writeword_st0 = mrvq_pop ? mrvq_writeword_st0     :
                                reqq_pop ? reqq_req_writeword_st0 :
                                0;

    VX_generic_register #(
        .N(1 + 1 + 1 + `LINE_ADDR_WIDTH + `BASE_ADDR_BITS + `WORD_WIDTH + `REQ_INST_META_WIDTH + 1 + (`BANK_LINE_WORDS*`WORD_WIDTH))
    ) s0_1_c0 (
        .clk   (clk),
        .reset (reset),
        .stall (stall_bank_pipe),
        .flush (0),
        .in    ({qual_is_snp,   qual_going_to_write_st0, qual_valid_st0, qual_addr_st0, qual_wsel_st0, qual_writeword_st0, qual_inst_meta_st0, qual_is_fill_st0, qual_writedata_st0}),
        .out   ({is_snp_st1[0], going_to_write_st1[0],   valid_st1[0],   addr_st1[0],   wsel_st1[0],   writeword_st1[0],   inst_meta_st1[0],   is_fill_st1[0],   writedata_st1[0]})
    );

    genvar stage;
    for (stage = 1; stage < STAGE_1_CYCLES; stage = stage + 1) begin
        VX_generic_register #(
            .N(1 + 1 + 1 + `LINE_ADDR_WIDTH + `BASE_ADDR_BITS + `WORD_WIDTH + `REQ_INST_META_WIDTH + 1 + (`BANK_LINE_WORDS*`WORD_WIDTH))
        ) s0_1_cc (
            .clk  (clk),
            .reset(reset),
            .stall(stall_bank_pipe),
            .flush(0),
            .in  ({is_snp_st1[stage-1], going_to_write_st1[stage-1], valid_st1[stage-1], addr_st1[stage-1],   wsel_st1[stage-1], writeword_st1[stage-1], inst_meta_st1[stage-1], is_fill_st1[stage-1], writedata_st1[stage-1]}),
            .out ({is_snp_st1[stage],   going_to_write_st1[stage],   valid_st1[stage],   addr_st1[stage],     wsel_st1[stage],   writeword_st1[stage],   inst_meta_st1[stage],   is_fill_st1[stage],   writedata_st1[stage]})
        );
    end

    wire[`WORD_WIDTH-1:0]                  readword_st1e;
    wire[`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] readdata_st1e;
    wire[`TAG_SELECT_BITS-1:0]             readtag_st1e;
    wire                                   miss_st1e;
    wire                                   dirty_st1e;
`DEBUG_BEGIN
    wire [CORE_TAG_WIDTH-1:0]              tag_st1e;
    wire [`LOG2UP(NUM_REQUESTS)-1:0]       tid_st1e;
`DEBUG_END
    wire [`BYTE_EN_BITS-1:0]               mem_read_st1e;  
    wire [`BYTE_EN_BITS-1:0]               mem_write_st1e;    
    wire                                   fill_saw_dirty_st1e;
    wire                                   is_snp_st1e;

    assign is_snp_st1e = is_snp_st1[STAGE_1_CYCLES-1];
    assign {tag_st1e, mem_read_st1e, mem_write_st1e, tid_st1e} = inst_meta_st1[STAGE_1_CYCLES-1];

    VX_tag_data_access #(
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
        .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE),
        .DRAM_ENABLE            (DRAM_ENABLE),
        .WRITE_ENABLE           (WRITE_ENABLE)
     ) tag_data_access (
        .clk           (clk),
        .reset         (reset),
        .stall         (stall_bank_pipe),
        .stall_bank_pipe(stall_bank_pipe),

        // Initial Read
        .readaddr_st10 (addr_st1[0][`LINE_SELECT_BITS-1:0]),

        // Actual Read/Write
        .valid_req_st1e(valid_st1[STAGE_1_CYCLES-1]),
        .writefill_st1e(is_fill_st1[STAGE_1_CYCLES-1]),
        .writeaddr_st1e(addr_st1[STAGE_1_CYCLES-1]),
        .writewsel_st1e(wsel_st1[STAGE_1_CYCLES-1]),
        .writeword_st1e(writeword_st1[STAGE_1_CYCLES-1]),
        .writedata_st1e(writedata_st1[STAGE_1_CYCLES-1]),

        .mem_write_st1e(mem_write_st1e),
        .mem_read_st1e (mem_read_st1e), 

        .is_snp_st1e   (is_snp_st1e),

        // Read Data
        .readword_st1e (readword_st1e),
        .readdata_st1e (readdata_st1e),
        .readtag_st1e  (readtag_st1e),
        .miss_st1e     (miss_st1e),
        .dirty_st1e    (dirty_st1e),
        .fill_saw_dirty_st1e(fill_saw_dirty_st1e)
    );

    wire qual_valid_st1e_2 = valid_st1[STAGE_1_CYCLES-1] && !is_fill_st1[STAGE_1_CYCLES-1];

    wire                            valid_st2;    
    wire [`BASE_ADDR_BITS-1:0]      wsel_st2;
    wire [`WORD_WIDTH-1:0]          writeword_st2;
    wire [`WORD_WIDTH-1:0]          readword_st2;
    wire [`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] readdata_st2;
    wire                            miss_st2;
    wire                            dirty_st2;
    wire [`REQ_INST_META_WIDTH-1:0] inst_meta_st2;
    wire [`TAG_SELECT_BITS-1:0]     readtag_st2;    
    wire                            fill_saw_dirty_st2;
    wire                            is_snp_st2;

    VX_generic_register #(
        .N(1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `BASE_ADDR_BITS + `WORD_WIDTH + `WORD_WIDTH + (`BANK_LINE_WORDS * `WORD_WIDTH) + `TAG_SELECT_BITS + 1 + 1 + `REQ_INST_META_WIDTH)
    ) st_1e_2 (
        .clk  (clk),
        .reset(reset),
        .stall(stall_bank_pipe),
        .flush(0),
        .in  ({is_snp_st1e, fill_saw_dirty_st1e, is_fill_st1[STAGE_1_CYCLES-1] , qual_valid_st1e_2, addr_st1[STAGE_1_CYCLES-1], wsel_st1[STAGE_1_CYCLES-1], writeword_st1[STAGE_1_CYCLES-1], readword_st1e, readdata_st1e, readtag_st1e, miss_st1e, dirty_st1e, inst_meta_st1[STAGE_1_CYCLES-1]}),
        .out ({is_snp_st2 , fill_saw_dirty_st2 , is_fill_st2                   , valid_st2        , addr_st2                  , wsel_st2,                   writeword_st2                  , readword_st2 , readdata_st2 , readtag_st2 , miss_st2 , dirty_st2 , inst_meta_st2                  })
    );

    wire should_flush;
    wire dwbq_push;

    wire cwbq_full;
    wire dwbq_full;
    wire ffsq_full;
    wire invalidate_fill;

    // Enqueue to miss reserv if it's a valid miss
    assign miss_add = valid_st2 
                   && !is_snp_st2 
                   && miss_st2 
                   && !mrvq_full 
                   && !(should_flush && dwbq_push) 
                   && !((is_snp_st2 && valid_st2 && ffsq_full) 
                     || ((valid_st2 && !miss_st2) && cwbq_full) 
                     || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) 
                     || (valid_st2 && miss_st2 && mrvq_full) 
                     || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_full));

    assign miss_add_addr  = addr_st2;
    assign miss_add_wsel  = wsel_st2;
    assign miss_add_data  = writeword_st2;
    assign {miss_add_tag, miss_add_mem_read, miss_add_mem_write, miss_add_tid} = inst_meta_st2;

    VX_cache_miss_resrv #(
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
        .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE),
        .CORE_TAG_WIDTH         (CORE_TAG_WIDTH)
    ) cache_miss_resrv (
        .clk                     (clk),
        .reset                   (reset),
        // Enqueue
        .miss_add                (miss_add), // Need to do all 
        .miss_add_addr           (miss_add_addr),
        .miss_add_wsel           (miss_add_wsel),
        .miss_add_data           (miss_add_data),
        .miss_add_tid            (miss_add_tid),
        .miss_add_tag            (miss_add_tag),
        .miss_add_mem_read       (miss_add_mem_read),
        .miss_add_mem_write      (miss_add_mem_write),
        .miss_resrv_full         (mrvq_full),
        .miss_resrv_stop         (mrvq_stop),

        // Broadcast
        .is_fill_st1             (is_fill_st2),
        .fill_addr_st1           (addr_st2),

        // Dequeue
        .miss_resrv_pop          (mrvq_pop),
        .miss_resrv_valid_st0    (mrvq_valid_st0),
        .miss_resrv_addr_st0     (mrvq_addr_st0),
        .miss_resrv_wsel_st0     (mrvq_wsel_st0),
        .miss_resrv_data_st0     (mrvq_writeword_st0),
        .miss_resrv_tid_st0      (mrvq_tid_st0),
        .miss_resrv_tag_st0      (mrvq_tag_st0),
        .miss_resrv_mem_read_st0 (mrvq_mem_read_st0),
        .miss_resrv_mem_write_st0(mrvq_mem_write_st0)
    );

    // Enqueue to CWB Queue
    wire cwbq_push = (valid_st2 && !miss_st2) 
                   && !cwbq_full 
                   && !(SNOOP_FORWARDING_ENABLE && (miss_add_mem_write == `BYTE_EN_NO)) 
                   && !((is_snp_st2 && valid_st2 && ffsq_full) 
                     || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) 
                     || (valid_st2 && miss_st2 && mrvq_full) 
                     || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_full));

    wire [`WORD_WIDTH-1:0]            cwbq_data = readword_st2;
    wire [`LOG2UP(NUM_REQUESTS)-1:0]  cwbq_tid  = miss_add_tid;
    wire [CORE_TAG_WIDTH-1:0]         cwbq_tag  = miss_add_tag;
    
    wire                              cwbq_empty;
    assign core_rsp_valid = !cwbq_empty;
    VX_generic_queue #(
        .DATAW(`LOG2UP(NUM_REQUESTS) + CORE_TAG_WIDTH + `WORD_WIDTH), 
        .SIZE(CWBQ_SIZE)
    ) cwb_queue (
        .clk     (clk),
        .reset   (reset),

        .push    (cwbq_push),
        .data_in ({cwbq_tid, cwbq_tag, cwbq_data}),

        .pop     (core_rsp_pop),
        .data_out({core_rsp_tid, core_rsp_tag, core_rsp_data}),
        .empty   (cwbq_empty),
        .full    (cwbq_full)
    );

    assign should_flush = snoop_state 
                       && valid_st2 
                       && (miss_add_mem_write != `BYTE_EN_NO) 
                       && !is_snp_st2 && !is_fill_st2;

    // Enqueue to DWB Queue
    assign dwbq_push = ((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2 || should_flush) 
                    && !dwbq_full 
                    && !((is_snp_st2 && valid_st2 && ffsq_full) 
                      || ((valid_st2 && !miss_st2) && cwbq_full) 
                      || (valid_st2 && miss_st2 && mrvq_full) 
                      || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_full));

    wire[`LINE_ADDR_WIDTH-1:0] dwbq_req_addr;
    wire       dwbq_empty;

    wire[`BANK_LINE_WORDS-1:0][`WORD_WIDTH-1:0] dwbq_req_data;

    if (SNOOP_FORWARDING_ENABLE) begin
        assign dwbq_req_data = (should_flush && dwbq_push) ? writeword_st2 : readdata_st2;
        assign dwbq_req_addr = (should_flush && dwbq_push) ? addr_st2 : {readtag_st2, addr_st2[`LINE_SELECT_BITS-1:0]};
    end else begin
        assign dwbq_req_data = readdata_st2;
        assign dwbq_req_addr = {readtag_st2, addr_st2[`LINE_SELECT_BITS-1:0]};
    end

    wire possible_fill = valid_st2 && miss_st2 && !dram_fill_req_full && !is_snp_st2;
    wire [`LINE_ADDR_WIDTH-1:0] fill_invalidator_addr = addr_st2;

    VX_fill_invalidator #(
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
        .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE)
    ) fill_invalidator (
        .clk               (clk),
        .reset             (reset),
        .possible_fill     (possible_fill),
        .success_fill      (is_fill_st2),
        .fill_addr         (fill_invalidator_addr),

        .invalidate_fill   (invalidate_fill)
    );

    // Enqueue in dram_fill_req
    assign dram_fill_req_valid  = possible_fill && !invalidate_fill;
    assign dram_fill_req_is_snp = is_snp_st2 && valid_st2 && miss_st2;
    assign dram_fill_req_addr   = addr_st2;

    assign dram_wb_req_valid = !dwbq_empty;

    VX_generic_queue #(
        .DATAW(`LINE_ADDR_WIDTH + (`BANK_LINE_WORDS * `WORD_WIDTH)), 
        .SIZE(DWBQ_SIZE)
    ) dwb_queue (
        .clk     (clk),
        .reset   (reset),

        .push    (dwbq_push),
        .data_in ({dwbq_req_addr, dwbq_req_data}),

        .pop     (dram_wb_req_pop),
        .data_out({dram_wb_req_addr, dram_wb_req_data}),
        .empty   (dwbq_empty),
        .full    (dwbq_full)
    );

    wire snp_fwd_push;    
    wire ffsq_empty;

    assign snp_fwd_push = is_snp_st2 
                       && valid_st2 
                       && !ffsq_full 
                       && !(((valid_st2 && !miss_st2) && cwbq_full) 
                         || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) 
                         || (valid_st2 && miss_st2 && mrvq_full) 
                         || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_full));
    
    assign snp_fwd_valid = !ffsq_empty;

    VX_generic_queue #(
        .DATAW(`LINE_ADDR_WIDTH), 
        .SIZE(FFSQ_SIZE)
    ) ffs_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (snp_fwd_push),
        .data_in (addr_st2),
        .pop     (snp_fwd_pop),
        .data_out(snp_fwd_addr),
        .empty   (ffsq_empty),
        .full    (ffsq_full)
    );

    assign stall_bank_pipe = (is_snp_st2 && valid_st2 && ffsq_full) 
                          || ((valid_st2 && !miss_st2) && cwbq_full) 
                          || (((valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2) && dwbq_full) 
                          || (valid_st2 && miss_st2 && mrvq_full) 
                          || (valid_st2 && miss_st2 && !invalidate_fill && dram_fill_req_full);

endmodule : VX_bank