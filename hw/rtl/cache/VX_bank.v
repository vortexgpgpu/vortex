`include "VX_cache_config.vh"

module VX_bank #(
    parameter CACHE_ID                      = 0,
    parameter BANK_ID                       = 0,    
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 0, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 0, 
    // Number of cycles to complete i 1 (read from memory)
    parameter STAGE_1_CYCLES                = 0,

    // Queues feeding into banks Knobs {1, 2, 4, 8, ...}
    // Core Request Queue Size
    parameter REQQ_SIZE                     = 0, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 0, 
    // Dram Fill Rsp Queue Size
    parameter DFPQ_SIZE                     = 0, 
    // Snoop Req Queue Size
    parameter SNRQ_SIZE                     = 0, 

    // Queues for writebacks Knobs {1, 2, 4, 8, ...}
    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 0, 
    // Dram Writeback Queue Size
    parameter DWBQ_SIZE                     = 0, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 0, 

    // Enable cache writeable
     parameter WRITE_ENABLE                 = 0,

     // Enable dram update
     parameter DRAM_ENABLE                  = 0,

    // Enable snoop forwarding
    parameter SNOOP_FORWARDING              = 0,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 0,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0,

    // Snooping request tag width
    parameter SNP_REQ_TAG_WIDTH             = 0
) (
    input wire clk,
    input wire reset,

    // Core Request    
    input wire [NUM_REQUESTS-1:0]                 core_req_valids,        
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0] core_req_read,  
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0] core_req_write,
    input wire [NUM_REQUESTS-1:0][31:0]           core_req_addr,
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0] core_req_data,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] core_req_tag,
    output wire                                   core_req_ready,
    
    // Core Response    
    output wire                                   core_rsp_valid,
    output wire [`REQS_BITS-1:0]                  core_rsp_tid,
    output wire [`WORD_WIDTH-1:0]                 core_rsp_data,
    output wire [CORE_TAG_WIDTH-1:0]              core_rsp_tag,
    input  wire                                   core_rsp_ready,

    // Dram Fill Requests
    output wire                                   dram_fill_req_valid,
    output wire[`LINE_ADDR_WIDTH-1:0]             dram_fill_req_addr,
    input  wire                                   dram_fill_req_ready,

    // Dram Fill Response
    input  wire                                   dram_fill_rsp_valid,    
    input  wire [`BANK_LINE_WIDTH-1:0]            dram_fill_rsp_data,
    input  wire [`LINE_ADDR_WIDTH-1:0]            dram_fill_rsp_addr,
    output wire                                   dram_fill_rsp_ready,

    // Dram WB Requests    
    output wire                                   dram_wb_req_valid,
    output wire [`LINE_ADDR_WIDTH-1:0]            dram_wb_req_addr,
    output wire [`BANK_LINE_WIDTH-1:0]            dram_wb_req_data,
    input  wire                                   dram_wb_req_ready,

    // Snp Request
    input  wire                                   snp_req_valid,
    input  wire [`LINE_ADDR_WIDTH-1:0]            snp_req_addr,
    input  wire [SNP_REQ_TAG_WIDTH-1:0]           snp_req_tag,
    output wire                                   snp_req_ready,

    output wire                                   snp_rsp_valid,
    output wire [SNP_REQ_TAG_WIDTH-1:0]           snp_rsp_tag,
    input  wire                                   snp_rsp_ready
);

`DEBUG_BEGIN
    wire[31:0]           debug_use_pc_st0;
    wire[1:0]            debug_wb_st0;
    wire[4:0]            debug_rd_st0;
    wire[`NW_BITS-1:0]   debug_warp_num_st0;
    wire[2:0]            debug_mem_read_st0;    
    wire[2:0]            debug_mem_write_st0;
    wire[`REQS_BITS-1:0] debug_tid_st0;

    wire[31:0]           debug_use_pc_st1e;
    wire[1:0]            debug_wb_st1e;
    wire[4:0]            debug_rd_st1e;
    wire[`NW_BITS-1:0]   debug_warp_num_st1e;
    wire[2:0]            debug_mem_read_st1e;    
    wire[2:0]            debug_mem_write_st1e;
    wire[`REQS_BITS-1:0] debug_tid_st1e;

    wire[31:0]           debug_use_pc_st2;
    wire[1:0]            debug_wb_st2;
    wire[4:0]            debug_rd_st2;
    wire[`NW_BITS-1:0]   debug_warp_num_st2;
    wire[2:0]            debug_mem_read_st2;    
    wire[2:0]            debug_mem_write_st2;
    wire[`REQS_BITS-1:0] debug_tid_st2;
`DEBUG_END

    wire snrq_pop;
    wire snrq_empty;
    wire snrq_full;
    
    wire [`LINE_ADDR_WIDTH-1:0] snrq_addr_st0;
    wire [SNP_REQ_TAG_WIDTH-1:0] snrq_tag_st0;

    VX_generic_queue #(
        .DATAW(`LINE_ADDR_WIDTH + SNP_REQ_TAG_WIDTH), 
        .SIZE(SNRQ_SIZE)
    ) snp_req_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (snp_req_valid),
        .data_in ({snp_req_addr, snp_req_tag}),
        .pop     (snrq_pop),
        .data_out({snrq_addr_st0, snrq_tag_st0}),
        .empty   (snrq_empty),
        .full    (snrq_full),
        `UNUSED_PIN (size)
    );

    assign snp_req_ready = ~snrq_full;

    wire dfpq_pop;
    wire dfpq_empty;
    wire dfpq_full;
    wire [`LINE_ADDR_WIDTH-1:0] dfpq_addr_st0;
    wire [`BANK_LINE_WIDTH-1:0] dfpq_filldata_st0;    

    VX_generic_queue #(
        .DATAW(`LINE_ADDR_WIDTH + $bits(dram_fill_rsp_data)), 
        .SIZE(DFPQ_SIZE)
    ) dfp_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (dram_fill_rsp_valid),
        .data_in ({dram_fill_rsp_addr, dram_fill_rsp_data}),
        .pop     (dfpq_pop),
        .data_out({dfpq_addr_st0, dfpq_filldata_st0}),
        .empty   (dfpq_empty),
        .full    (dfpq_full),
        `UNUSED_PIN (size)
    );

    assign dram_fill_rsp_ready = !dfpq_full;

    wire                        reqq_pop;
    wire                        reqq_push;
    wire                        reqq_empty;
    wire                        reqq_full;
    wire                        reqq_req_st0;
    wire[`REQS_BITS-1:0]        reqq_req_tid_st0;
`IGNORE_WARNINGS_BEGIN
    wire [31:0]                 reqq_req_addr_st0;
`IGNORE_WARNINGS_END    
    wire [`WORD_WIDTH-1:0]      reqq_req_writeword_st0;
    wire [CORE_TAG_WIDTH-1:0]   reqq_req_tag_st0;
    wire [`BYTE_EN_BITS-1:0]    reqq_req_mem_read_st0;  
    wire [`BYTE_EN_BITS-1:0]    reqq_req_mem_write_st0;

    VX_cache_req_queue #(
        .WORD_SIZE        (WORD_SIZE),
        .NUM_REQUESTS     (NUM_REQUESTS),
        .REQQ_SIZE        (REQQ_SIZE),
        .CORE_TAG_WIDTH   (CORE_TAG_WIDTH),        
        .CORE_TAG_ID_BITS (CORE_TAG_ID_BITS)
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
        .reqq_full             (reqq_full)
    );    

    assign core_req_ready = ~reqq_full;
    assign reqq_push = (| core_req_valids) && core_req_ready;

    wire                                  mrvq_pop;
    wire                                  mrvq_full;
    wire                                  mrvq_stop;
    wire                                  mrvq_valid_st0;
    wire[`REQS_BITS-1:0]                  mrvq_tid_st0;
    wire [`LINE_ADDR_WIDTH-1:0]           mrvq_addr_st0;
    wire [`BASE_ADDR_BITS-1:0]            mrvq_wsel_st0;
    wire [`WORD_WIDTH-1:0]                mrvq_writeword_st0;
    wire [`REQ_TAG_WIDTH-1:0]             mrvq_tag_st0;
    wire [`BYTE_EN_BITS-1:0]              mrvq_mem_read_st0;  
    wire [`BYTE_EN_BITS-1:0]              mrvq_mem_write_st0;
    wire                                  mrvq_is_snp_st0;

    wire                                  mrvq_pending_hazard_st1e;
    wire                                  st2_pending_hazard_st1e;
    wire                                  force_request_miss_st1e;

    wire[`REQS_BITS-1:0]                  miss_add_tid;
    wire[`REQ_TAG_WIDTH-1:0]              miss_add_tag;
    wire[`BYTE_EN_BITS-1:0]               miss_add_mem_read;
    wire[`BYTE_EN_BITS-1:0]               miss_add_mem_write;

    wire[`LINE_ADDR_WIDTH-1:0]            addr_st2;
    wire                                  is_fill_st2;

    wire mrvq_push_stall;
    wire cwbq_push_stall;    
    wire dwbq_push_stall;    
    wire dram_fill_req_stall;    
    wire stall_bank_pipe;

    reg  is_fill_in_pipe;
    
    wire is_fill_st1 [STAGE_1_CYCLES-1:0];
`DEBUG_BEGIN
    wire going_to_write_st1 [STAGE_1_CYCLES-1:0];    
`DEBUG_END
    
    integer j;
    always @(*) begin
        is_fill_in_pipe = 0;
        for (j = 0; j < STAGE_1_CYCLES; j++) begin
            if (is_fill_st1[j]) begin
                is_fill_in_pipe = 1;
            end
        end
    end


    wire mrvq_pop_unqual = mrvq_valid_st0;
    wire dfpq_pop_unqual = !mrvq_pop_unqual && !dfpq_empty;
    wire reqq_pop_unqual = !mrvq_stop && !mrvq_pop_unqual && !dfpq_pop_unqual && !reqq_empty && reqq_req_st0 && !is_fill_st1[0] && !is_fill_in_pipe;
    wire snrq_pop_unqual = !mrvq_stop && !reqq_pop_unqual && !reqq_pop_unqual && !mrvq_pop_unqual && !dfpq_pop_unqual && !snrq_empty;


    assign mrvq_pop = mrvq_pop_unqual && !stall_bank_pipe && !recover_mrvq_state_st2;
    assign dfpq_pop = dfpq_pop_unqual && !stall_bank_pipe;
    assign reqq_pop = reqq_pop_unqual && !stall_bank_pipe;
    assign snrq_pop = snrq_pop_unqual && !stall_bank_pipe;

    wire                                  qual_is_fill_st0;
    wire                                  qual_valid_st0;
    wire [`LINE_ADDR_WIDTH-1:0]           qual_addr_st0;
    wire [`WORD_SELECT_ADDR_END:0]        qual_wsel_st0;
    wire                                  qual_from_mrvq_st0;

    wire [`WORD_WIDTH-1:0]                qual_writeword_st0;
    wire [`BANK_LINE_WIDTH-1:0]           qual_writedata_st0;
    wire [`REQ_INST_META_WIDTH-1:0]       qual_inst_meta_st0;
    wire                                  qual_going_to_write_st0;
    wire                                  qual_is_snp_st0;

    wire                                  valid_st1     [STAGE_1_CYCLES-1:0];
    wire [`LINE_ADDR_WIDTH-1:0]           addr_st1      [STAGE_1_CYCLES-1:0];
    wire [`WORD_SELECT_ADDR_END:0]        wsel_st1      [STAGE_1_CYCLES-1:0];
    wire [`WORD_WIDTH-1:0]                writeword_st1 [STAGE_1_CYCLES-1:0];
    wire [`REQ_INST_META_WIDTH-1:0]       inst_meta_st1 [STAGE_1_CYCLES-1:0];    
    wire [`BANK_LINE_WIDTH-1:0]           writedata_st1 [STAGE_1_CYCLES-1:0];
    wire                                  is_snp_st1    [STAGE_1_CYCLES-1:0];
    wire                                  from_mrvq_st1 [STAGE_1_CYCLES-1:0];

    assign qual_is_fill_st0 = dfpq_pop_unqual;

    assign qual_valid_st0   = dfpq_pop || mrvq_pop || reqq_pop || snrq_pop;

    assign qual_addr_st0    = dfpq_pop_unqual ? dfpq_addr_st0 :
                              mrvq_pop_unqual ? mrvq_addr_st0 :
                              reqq_pop_unqual ? reqq_req_addr_st0[31:`LINE_SELECT_ADDR_START] :
                              snrq_pop_unqual ? snrq_addr_st0 :
                                         0;

    assign qual_wsel_st0   =  reqq_pop_unqual ? reqq_req_addr_st0[`BASE_ADDR_BITS-1:0] :
                              mrvq_pop_unqual ? mrvq_wsel_st0 :
                                         0;

    assign qual_writedata_st0 = dfpq_pop_unqual ? dfpq_filldata_st0 : 57;

    assign qual_inst_meta_st0 = mrvq_pop_unqual ? {`REQ_TAG_WIDTH'(mrvq_tag_st0)    , mrvq_mem_read_st0,     mrvq_mem_write_st0,     mrvq_tid_st0} :
                                reqq_pop_unqual ? {`REQ_TAG_WIDTH'(reqq_req_tag_st0), reqq_req_mem_read_st0, reqq_req_mem_write_st0, reqq_req_tid_st0} :
                                snrq_pop_unqual ? {`REQ_TAG_WIDTH'(snrq_tag_st0),     `BYTE_EN_BITS'(0),     `BYTE_EN_BITS'(0),      `REQS_BITS'(0)} :
                                           0;

    assign qual_going_to_write_st0 = dfpq_pop_unqual ? 1 :
                                        (mrvq_pop_unqual && (mrvq_mem_write_st0 != `BYTE_EN_NO)) ? 1 :
                                            (reqq_pop_unqual && (reqq_req_mem_write_st0 != `BYTE_EN_NO)) ? 1 :
                                                0;

    assign qual_is_snp_st0 = mrvq_pop_unqual ? mrvq_is_snp_st0 :
                             snrq_pop_unqual ? 1 :
                                        0;

    assign qual_writeword_st0 = mrvq_pop_unqual ? mrvq_writeword_st0     :
                                reqq_pop_unqual ? reqq_req_writeword_st0 :
                                           0;

    assign qual_from_mrvq_st0 = mrvq_pop_unqual;

`DEBUG_BEGIN
    if (WORD_SIZE != `GLOBAL_BLOCK_SIZE) begin
        assign {debug_use_pc_st0, debug_wb_st0, debug_rd_st0, debug_warp_num_st0, debug_mem_read_st0, debug_mem_write_st0, debug_tid_st0} = qual_inst_meta_st0;
    end
`DEBUG_END

    VX_generic_register #(
        .N(1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `BASE_ADDR_BITS + `WORD_WIDTH + `REQ_INST_META_WIDTH + 1 + `BANK_LINE_WIDTH)
    ) s0_1_c0 (
        .clk   (clk),
        .reset (reset),
        .stall (stall_bank_pipe),
        .flush (1'b0),
        .in    ({qual_from_mrvq_st0, qual_is_snp_st0, qual_going_to_write_st0, qual_valid_st0, qual_addr_st0, qual_wsel_st0, qual_writeword_st0, qual_inst_meta_st0, qual_is_fill_st0, qual_writedata_st0}),
        .out   ({from_mrvq_st1[0]  , is_snp_st1[0],   going_to_write_st1[0],   valid_st1[0],   addr_st1[0],   wsel_st1[0],   writeword_st1[0],   inst_meta_st1[0],   is_fill_st1[0],   writedata_st1[0]})
    );

    genvar i;
    for (i = 1; i < STAGE_1_CYCLES; i++) begin
        VX_generic_register #(
            .N(1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `BASE_ADDR_BITS + `WORD_WIDTH + `REQ_INST_META_WIDTH + 1 + `BANK_LINE_WIDTH)
        ) s0_1_cc (
            .clk  (clk),
            .reset(reset),
            .stall(stall_bank_pipe),
            .flush(1'b0),
            .in  ({from_mrvq_st1[i-1], is_snp_st1[i-1], going_to_write_st1[i-1], valid_st1[i-1], addr_st1[i-1],   wsel_st1[i-1], writeword_st1[i-1], inst_meta_st1[i-1], is_fill_st1[i-1], writedata_st1[i-1]}),
            .out ({from_mrvq_st1[i]  , is_snp_st1[i],   going_to_write_st1[i],   valid_st1[i],   addr_st1[i],     wsel_st1[i],   writeword_st1[i],   inst_meta_st1[i],   is_fill_st1[i],   writedata_st1[i]})
        );
    end

    wire[`WORD_WIDTH-1:0]       readword_st1e;
    wire[`BANK_LINE_WIDTH-1:0]  readdata_st1e;
    wire[`TAG_SELECT_BITS-1:0]  readtag_st1e;
    wire                        miss_st1e;
    wire                        dirty_st1e;
`DEBUG_BEGIN
    wire [`REQ_TAG_WIDTH-1:0]   tag_st1e;
    wire [`REQS_BITS-1:0]       tid_st1e;
`DEBUG_END
    wire [`BYTE_EN_BITS-1:0]    mem_read_st1e;  
    wire [`BYTE_EN_BITS-1:0]    mem_write_st1e;    
    wire                        fill_saw_dirty_st1e;
    wire                        is_snp_st1e;
    wire                        snp_to_mrvq_st1e;
    wire                        mrvq_init_ready_state_st1e;
    wire                        miss_add_because_miss;
    wire                        valid_st1e;
    wire                        from_mrvq_st1e;

    assign from_mrvq_st1e = from_mrvq_st1[STAGE_1_CYCLES-1];
    assign valid_st1e     = valid_st1    [STAGE_1_CYCLES-1];
    assign is_snp_st1e    = is_snp_st1   [STAGE_1_CYCLES-1];

    assign {tag_st1e, mem_read_st1e, mem_write_st1e, tid_st1e} = inst_meta_st1[STAGE_1_CYCLES-1];

    assign st2_pending_hazard_st1e = (miss_add_because_miss) && ((addr_st2 == addr_st1[STAGE_1_CYCLES-1]) && !is_fill_st2);

    assign force_request_miss_st1e  = (valid_st1e && !from_mrvq_st1e && (mrvq_pending_hazard_st1e || st2_pending_hazard_st1e)) || (valid_st1e && from_mrvq_st1e && recover_mrvq_state_st2);

    VX_tag_data_access #(
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .STAGE_1_CYCLES (STAGE_1_CYCLES),
        .DRAM_ENABLE    (DRAM_ENABLE),
        .WRITE_ENABLE   (WRITE_ENABLE)
     ) tag_data_access (
        .clk            (clk),
        .reset          (reset),
        .stall          (stall_bank_pipe),
        .stall_bank_pipe(stall_bank_pipe),

        .force_request_miss_st1e(force_request_miss_st1e),

        // Initial Read
        .readaddr_st10(addr_st1[0][`LINE_SELECT_BITS-1:0]),

        // Actual Read/Write
        .valid_req_st1e(valid_st1e),
        .writefill_st1e(is_fill_st1[STAGE_1_CYCLES-1]),
        .writeaddr_st1e(addr_st1[STAGE_1_CYCLES-1]),
        .writewsel_st1e(wsel_st1[STAGE_1_CYCLES-1]),
        .writeword_st1e(writeword_st1[STAGE_1_CYCLES-1]),
        .writedata_st1e(writedata_st1[STAGE_1_CYCLES-1]),

        .mem_write_st1e(mem_write_st1e),
        .mem_read_st1e (mem_read_st1e), 

        .is_snp_st1e   (is_snp_st1e),

        // Read Data
        .readword_st1e             (readword_st1e),
        .readdata_st1e             (readdata_st1e),
        .readtag_st1e              (readtag_st1e),
        .miss_st1e                 (miss_st1e),
        .dirty_st1e                (dirty_st1e),
        .fill_saw_dirty_st1e       (fill_saw_dirty_st1e),
        .snp_to_mrvq_st1e          (snp_to_mrvq_st1e),
        .mrvq_init_ready_state_st1e(mrvq_init_ready_state_st1e)
    );

`DEBUG_BEGIN
    if (WORD_SIZE != `GLOBAL_BLOCK_SIZE) begin
        assign {debug_use_pc_st1e, debug_wb_st1e, debug_rd_st1e, debug_warp_num_st1e, debug_mem_read_st1e, debug_mem_write_st1e, debug_tid_st1e} = inst_meta_st1[STAGE_1_CYCLES-1];
    end
`DEBUG_END

    wire qual_valid_st1e_2  = valid_st1e && !is_fill_st1[STAGE_1_CYCLES-1];
    wire from_mrvq_st1e_st2 = from_mrvq_st1e && !is_snp_st1e;

    wire                            valid_st2;    
    wire [`BASE_ADDR_BITS-1:0]      wsel_st2;
    wire [`WORD_WIDTH-1:0]          writeword_st2;
    wire [`WORD_WIDTH-1:0]          readword_st2;
    wire [`BANK_LINE_WIDTH-1:0]     readdata_st2;
    wire                            miss_st2;
    wire                            dirty_st2;
    wire [`REQ_INST_META_WIDTH-1:0] inst_meta_st2;
    wire [`TAG_SELECT_BITS-1:0]     readtag_st2;    
    wire                            fill_saw_dirty_st2;
    wire                            is_snp_st2;
    wire                            snp_to_mrvq_st2;
    wire                            from_mrvq_st2;
    wire                            mrvq_init_ready_state_st2;
    wire                            mrvq_init_ready_state_unqual_st2;
    wire                            mrvq_init_ready_state_hazard_st0_st1;
    wire                            mrvq_init_ready_state_hazard_st1e_st1;
    wire                            recover_mrvq_state_st2;

    VX_generic_register #(
        .N(1+ 1 + 1 + 1 + 1 + 1 + 1 + `LINE_ADDR_WIDTH + `BASE_ADDR_BITS + `WORD_WIDTH + `WORD_WIDTH + `BANK_LINE_WIDTH + `TAG_SELECT_BITS + 1 + 1 + `REQ_INST_META_WIDTH)
    ) st_1e_2 (
        .clk  (clk),
        .reset(reset),
        .stall(stall_bank_pipe),
        .flush(1'b0),
        .in  ({from_mrvq_st1e_st2, mrvq_init_ready_state_st1e      ,  snp_to_mrvq_st1e, is_snp_st1e, fill_saw_dirty_st1e, is_fill_st1[STAGE_1_CYCLES-1] , qual_valid_st1e_2, addr_st1[STAGE_1_CYCLES-1], wsel_st1[STAGE_1_CYCLES-1], writeword_st1[STAGE_1_CYCLES-1], readword_st1e, readdata_st1e, readtag_st1e, miss_st1e, dirty_st1e, inst_meta_st1[STAGE_1_CYCLES-1]}),
        .out ({from_mrvq_st2     , mrvq_init_ready_state_unqual_st2,  snp_to_mrvq_st2 , is_snp_st2 , fill_saw_dirty_st2 , is_fill_st2                   , valid_st2        , addr_st2                  , wsel_st2,                   writeword_st2                  , readword_st2 , readdata_st2 , readtag_st2 , miss_st2 , dirty_st2 , inst_meta_st2           })
    );    


`DEBUG_BEGIN
    if (WORD_SIZE != `GLOBAL_BLOCK_SIZE) begin
        assign {debug_use_pc_st2, debug_wb_st2, debug_rd_st2, debug_warp_num_st2, debug_mem_read_st2, debug_mem_write_st2, debug_tid_st2} = inst_meta_st2;
    end
`DEBUG_END

    // Enqueue to miss reserv if it's a valid miss
    assign miss_add_because_miss  = valid_st2 && !is_snp_st2 && miss_st2;
    wire miss_add_because_pending = snp_to_mrvq_st2;

    wire miss_add_unqual = (miss_add_because_miss || miss_add_because_pending);
    assign mrvq_push_stall = miss_add_unqual && mrvq_full;


    wire miss_add = miss_add_unqual
                   && !mrvq_full 
                   && !(cwbq_push_stall 
                     || dwbq_push_stall 
                     || dram_fill_req_stall);  


    assign recover_mrvq_state_st2 = miss_add && from_mrvq_st2;  

    wire [`LINE_ADDR_WIDTH-1:0] miss_add_addr  = addr_st2;
    wire [`BASE_ADDR_BITS-1:0] miss_add_wsel  = wsel_st2;
    wire [`WORD_WIDTH-1:0] miss_add_data  = writeword_st2;
    assign {miss_add_tag, miss_add_mem_read, miss_add_mem_write, miss_add_tid} = inst_meta_st2;
    wire miss_add_is_snp = is_snp_st2;

    wire miss_add_from_mrvq = valid_st2 && from_mrvq_st2 && !stall_bank_pipe;


    assign mrvq_init_ready_state_hazard_st0_st1  = miss_add_unqual && qual_is_fill_st0              && (miss_add_addr == qual_addr_st0             );
    assign mrvq_init_ready_state_hazard_st1e_st1 = miss_add_unqual && is_fill_st1[STAGE_1_CYCLES-1] && (miss_add_addr == addr_st1[STAGE_1_CYCLES-1]);

    assign mrvq_init_ready_state_st2 = mrvq_init_ready_state_unqual_st2 || mrvq_init_ready_state_hazard_st0_st1 || mrvq_init_ready_state_hazard_st1e_st1;

    VX_cache_miss_resrv #(
        .BANK_ID                (BANK_ID),
        .CACHE_ID               (CACHE_ID),
        .BANK_LINE_SIZE         (BANK_LINE_SIZE),
        .NUM_BANKS              (NUM_BANKS),
        .WORD_SIZE              (WORD_SIZE),
        .NUM_REQUESTS           (NUM_REQUESTS),
        .MRVQ_SIZE              (MRVQ_SIZE),
        .CORE_TAG_WIDTH         (CORE_TAG_WIDTH),
        .SNP_REQ_TAG_WIDTH      (SNP_REQ_TAG_WIDTH)
    ) cache_miss_resrv (
        .clk                     (clk),
        .reset                   (reset),

        // Enqueue
        .miss_add                (miss_add),
        .from_mrvq               (miss_add_from_mrvq),
        .miss_add_addr           (miss_add_addr),
        .miss_add_wsel           (miss_add_wsel),
        .miss_add_data           (miss_add_data),
        .miss_add_tid            (miss_add_tid),
        .miss_add_tag            (miss_add_tag),
        .miss_add_mem_read       (miss_add_mem_read),
        .miss_add_mem_write      (miss_add_mem_write),
        .miss_add_is_snp         (miss_add_is_snp),
        .miss_resrv_full         (mrvq_full),
        .miss_resrv_stop         (mrvq_stop),
        .mrvq_init_ready_state   (mrvq_init_ready_state_st2),

        // Broadcast
        .is_fill_st1             (is_fill_st1[STAGE_1_CYCLES-1]),
        .fill_addr_st1           (addr_st1[STAGE_1_CYCLES-1]),
        .pending_hazard          (mrvq_pending_hazard_st1e),

        // Dequeue
        .miss_resrv_pop          (mrvq_pop),
        .miss_resrv_valid_st0    (mrvq_valid_st0),
        .miss_resrv_addr_st0     (mrvq_addr_st0),
        .miss_resrv_wsel_st0     (mrvq_wsel_st0),
        .miss_resrv_data_st0     (mrvq_writeword_st0),
        .miss_resrv_tid_st0      (mrvq_tid_st0),
        .miss_resrv_tag_st0      (mrvq_tag_st0),
        .miss_resrv_mem_read_st0 (mrvq_mem_read_st0),
        .miss_resrv_mem_write_st0(mrvq_mem_write_st0),
        .miss_resrv_is_snp_st0   (mrvq_is_snp_st0)
    );

    // Enqueue core response
     
    wire cwbq_push;
    wire cwbq_pop;
    wire cwbq_empty;
    wire cwbq_full;

    wire cwbq_push_unqual = valid_st2 && !miss_st2 && !is_fill_st2 && !is_snp_st2;
    assign cwbq_push_stall = cwbq_push_unqual && cwbq_full;

    assign cwbq_push = cwbq_push_unqual
                    && !cwbq_full 
                    && (miss_add_mem_write == `BYTE_EN_NO) 
                    && !(dwbq_push_stall 
                      || mrvq_push_stall 
                      || dram_fill_req_stall);

    wire [`WORD_WIDTH-1:0]    cwbq_data = readword_st2;
    wire [`REQS_BITS-1:0]     cwbq_tid  = miss_add_tid;
    wire [CORE_TAG_WIDTH-1:0] cwbq_tag  = CORE_TAG_WIDTH'(miss_add_tag);

    VX_generic_queue #(
        .DATAW(`REQS_BITS + CORE_TAG_WIDTH + `WORD_WIDTH), 
        .SIZE(CWBQ_SIZE)
    ) cwb_queue (
        .clk     (clk),
        .reset   (reset),

        .push    (cwbq_push),
        .data_in ({cwbq_tid, cwbq_tag, cwbq_data}),

        .pop     (cwbq_pop),
        .data_out({core_rsp_tid, core_rsp_tag, core_rsp_data}),
        .empty   (cwbq_empty),
        .full    (cwbq_full),
        `UNUSED_PIN (size)
    );    

    assign core_rsp_valid = !cwbq_empty;
    assign cwbq_pop = core_rsp_valid && core_rsp_ready;

    // Enqueue DRAM fill request

// `IGNORE_WARNINGS_BEGIN
//     wire invalidate_fill;
// `IGNORE_WARNINGS_END
//     wire possible_fill = valid_st2 && miss_st2 && dram_fill_req_ready && ~is_snp_st2;
//     wire [`LINE_ADDR_WIDTH-1:0] fill_invalidator_addr = addr_st2;

//     VX_fill_invalidator #(
//         .BANK_LINE_SIZE         (BANK_LINE_SIZE),
//         .NUM_BANKS              (NUM_BANKS),
//         .FILL_INVALIDAOR_SIZE   (FILL_INVALIDAOR_SIZE)
//     ) fill_invalidator (
//         .clk               (clk),
//         .reset             (reset),
//         .possible_fill     (possible_fill),
//         .success_fill      (is_fill_st2),
//         .fill_addr         (fill_invalidator_addr),
//         .invalidate_fill   (invalidate_fill)
//     );    

    wire dram_fill_req_unqual  = miss_add_unqual && (!mrvq_init_ready_state_st2 || from_mrvq_st2);

    assign dram_fill_req_valid =    dram_fill_req_unqual 
                                &&  dram_fill_req_ready
                                && !(  dwbq_push_stall
                                    || mrvq_push_stall
                                    || cwbq_push_stall);

    assign dram_fill_req_addr  = addr_st2;
    assign dram_fill_req_stall = dram_fill_req_unqual && ~dram_fill_req_ready;

    // Enqueue DRAM writeback request

    wire dwbq_push;
    wire dwbq_pop;
    wire dwbq_empty;   
    wire dwbq_full;

    wire dwbq_is_dwb_in, dwbq_is_snp_in;
    wire dwbq_is_dwb_out, dwbq_is_snp_out; 

    assign dwbq_is_snp_in = is_snp_st2 && valid_st2 && !snp_to_mrvq_st2;    
    assign dwbq_is_dwb_in = (valid_st2 && miss_st2 && dirty_st2) || fill_saw_dirty_st2;
    wire dwbq_push_unqual = dwbq_is_dwb_in || dwbq_is_snp_in;    

    assign dwbq_push_stall = dwbq_push_unqual && dwbq_full;

    assign dwbq_push = dwbq_push_unqual
                    && !dwbq_full 
                    && !(cwbq_push_stall 
                      || mrvq_push_stall 
                      || dram_fill_req_stall);

    wire [`BANK_LINE_WIDTH-1:0] dwbq_req_data = readdata_st2;
    wire [`LINE_ADDR_WIDTH-1:0] dwbq_req_addr = {readtag_st2, addr_st2[`LINE_SELECT_BITS-1:0]};

    wire [SNP_REQ_TAG_WIDTH-1:0] snrq_tag_st2 = SNP_REQ_TAG_WIDTH'(miss_add_tag);
    
    VX_generic_queue #(
        .DATAW(1 + 1 + `LINE_ADDR_WIDTH + `BANK_LINE_WIDTH + SNP_REQ_TAG_WIDTH), 
        .SIZE(DWBQ_SIZE)
    ) dwb_queue (
        .clk     (clk),
        .reset   (reset),

        .push    (dwbq_push),
        .data_in ({dwbq_is_dwb_in, dwbq_is_snp_in, dwbq_req_addr, dwbq_req_data, snrq_tag_st2}),

        .pop     (dwbq_pop),
        .data_out({dwbq_is_dwb_out, dwbq_is_snp_out, dram_wb_req_addr, dram_wb_req_data, snp_rsp_tag}),
        .empty   (dwbq_empty),
        .full    (dwbq_full),
        `UNUSED_PIN (size)
    );       

    wire dram_wb_req_fire = dram_wb_req_valid && dram_wb_req_ready;
    wire snp_rsp_fire     = snp_rsp_valid && snp_rsp_ready;

    reg dwbq_dual_valid_sel;

    always @(posedge clk) begin
        if (reset) begin
            dwbq_dual_valid_sel <= 0;
        end else if (dwbq_is_dwb_out && dwbq_is_snp_out && (dram_wb_req_fire || snp_rsp_fire)) begin
            dwbq_dual_valid_sel <= ~dwbq_dual_valid_sel;
        end
    end

    // when both dwb and snp are asserted, first release the cwb, then release the snp.
    assign dram_wb_req_valid = ~dwbq_empty && dwbq_is_dwb_out && (~dwbq_is_snp_out || dwbq_dual_valid_sel == 0);
    assign snp_rsp_valid = ~dwbq_empty && dwbq_is_snp_out && (~dwbq_is_dwb_out || dwbq_dual_valid_sel == 1);  
    
    assign dwbq_pop =  (dwbq_is_dwb_out && ~dwbq_is_snp_out && dram_wb_req_fire)
                    || (dwbq_is_snp_out && ~dwbq_is_dwb_out && snp_rsp_fire)
                    || (dwbq_is_dwb_out && dwbq_is_snp_out && snp_rsp_fire);

    // bank pipeline stall
    assign stall_bank_pipe = cwbq_push_stall 
                          || dwbq_push_stall 
                          || mrvq_push_stall 
                          || dram_fill_req_stall;

`ifdef DBG_PRINT_CACHE_BANK
    always_ff @(posedge clk) begin
        if (dram_fill_req_valid && dram_fill_req_ready) begin
            $display("%t: bank%02d:%01d dram_fill req: addr=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dram_fill_req_addr, BANK_ID));
        end
        if (dram_wb_req_valid && dram_wb_req_ready) begin
            $display("%t: bank%02d:%01d dram_wb req: addr=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dram_wb_req_addr, BANK_ID), dram_wb_req_data);
        end
        if (dram_fill_rsp_valid && dram_fill_rsp_ready) begin
            $display("%t: bank%02d:%01d dram_fill rsp: addr=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(dram_fill_rsp_addr, BANK_ID), dram_fill_rsp_data);
        end
    end
`endif

endmodule : VX_bank