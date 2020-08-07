`include "VX_cache_config.vh"

module VX_tag_data_access #(
    parameter CACHE_ID          = 0,
    parameter BANK_ID           = 0,   
    parameter CORE_TAG_ID_BITS  = 0, 
    // Size of cache in bytes
    parameter CACHE_SIZE        = 0, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE    = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS         = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 0, 

    // Number of cycles to complete stage 1 (read from memory)
    parameter STAGE_1_CYCLES    = 0, 

     // Enable cache writeable
     parameter WRITE_ENABLE     = 0,

     // Enable dram update
     parameter DRAM_ENABLE      = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CORE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_st1e,
    input wire                          debug_wb_st1e,
    input wire[`NR_BITS-1:0]            debug_rd_st1e,
    input wire[`NW_BITS-1:0]            debug_warp_num_st1e,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st1e,
`IGNORE_WARNINGS_END
`endif

    input wire                          stall,
    input wire                          is_snp_st1e,
    input wire                          snp_invalidate_st1e,
    input wire                          stall_bank_pipe,

    input wire                          force_request_miss_st1e,

    input wire[`LINE_SELECT_BITS-1:0]   readaddr_st10, 
    input wire[`LINE_ADDR_WIDTH-1:0]    writeaddr_st1e,
    
    input wire                          valid_req_st1e,
    input wire                          writefill_st1e,
    input wire[`WORD_WIDTH-1:0]         writeword_st1e,
    input wire[`BANK_LINE_WIDTH-1:0]    writedata_st1e,

`IGNORE_WARNINGS_BEGIN    
    input wire                          mem_rw_st1e,
    input wire[WORD_SIZE-1:0]           mem_byteen_st1e, 
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] wordsel_st1e,
`IGNORE_WARNINGS_END

    output wire[`WORD_WIDTH-1:0]        readword_st1e,
    output wire[`BANK_LINE_WIDTH-1:0]   readdata_st1e,
    output wire[`TAG_SELECT_BITS-1:0]   readtag_st1e,
    output wire                         miss_st1e,
    output wire                         dirty_st1e,
    output wire[BANK_LINE_SIZE-1:0]     dirtyb_st1e,
    output wire                         fill_saw_dirty_st1e,
    output wire                         snp_to_mrvq_st1e,
    output wire                         mrvq_init_ready_state_st1e
);

    wire                        read_valid_st1c[STAGE_1_CYCLES-1:0];
    wire                        read_dirty_st1c[STAGE_1_CYCLES-1:0];
    wire[BANK_LINE_SIZE-1:0]    read_dirtyb_st1c[STAGE_1_CYCLES-1:0];
    wire[`TAG_SELECT_BITS-1:0]  read_tag_st1c  [STAGE_1_CYCLES-1:0];
    wire[`BANK_LINE_WIDTH-1:0]  read_data_st1c [STAGE_1_CYCLES-1:0];

    wire                        qual_read_valid_st1;
    wire                        qual_read_dirty_st1;
    wire[BANK_LINE_SIZE-1:0]    qual_read_dirtyb_st1;
    wire[`TAG_SELECT_BITS-1:0]  qual_read_tag_st1;
    wire[`BANK_LINE_WIDTH-1:0]  qual_read_data_st1;

    wire                        use_read_valid_st1e;
    wire                        use_read_dirty_st1e;
    wire[BANK_LINE_SIZE-1:0]    use_read_dirtyb_st1e;
    wire[`TAG_SELECT_BITS-1:0]  use_read_tag_st1e;
    wire[`BANK_LINE_WIDTH-1:0]  use_read_data_st1e;
    wire[`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] use_write_enable;
    wire[`BANK_LINE_WIDTH-1:0]  use_write_data;

    wire fill_sent;
    wire invalidate_line;
    wire tags_match;

    wire real_writefill = valid_req_st1e && writefill_st1e
                       && ((~use_read_valid_st1e) || (use_read_valid_st1e && ~tags_match)); 

    wire[`TAG_SELECT_BITS-1:0] writetag_st1e = writeaddr_st1e[`TAG_LINE_ADDR_RNG];
    wire[`LINE_SELECT_BITS-1:0] writeladdr_st1e = writeaddr_st1e[`LINE_SELECT_BITS-1:0];

    VX_tag_data_store #(
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE)
    ) tag_data_store (
        .clk         (clk),
        .reset       (reset),
        .stall_bank_pipe(stall_bank_pipe),

        .read_addr   (readaddr_st10),
        .read_valid  (qual_read_valid_st1),        
        .read_dirty  (qual_read_dirty_st1),
        .read_dirtyb (qual_read_dirtyb_st1),
        .read_tag    (qual_read_tag_st1),
        .read_data   (qual_read_data_st1),

        .invalidate  (invalidate_line),
        .write_enable(use_write_enable),
        .write_fill  (real_writefill),
        .write_addr  (writeladdr_st1e),
        .tag_index   (writetag_st1e),
        .write_data  (use_write_data),
        .fill_sent   (fill_sent)
    );

    VX_generic_register #(
        .N(1 + 1 + BANK_LINE_SIZE + `TAG_SELECT_BITS + `BANK_LINE_WIDTH), 
        .PASSTHRU(1)
    ) s0_1_c0 (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({qual_read_valid_st1, qual_read_dirty_st1, qual_read_dirtyb_st1, qual_read_tag_st1, qual_read_data_st1}),
        .out   ({read_valid_st1c[0],  read_dirty_st1c[0],  read_dirtyb_st1c[0],  read_tag_st1c[0],  read_data_st1c[0]})
    );

    genvar i;
    for (i = 1; i < STAGE_1_CYCLES-1; i++) begin
        VX_generic_register #(
            .N(1 + 1 + BANK_LINE_SIZE + `TAG_SELECT_BITS + `BANK_LINE_WIDTH)
        ) s0_1_cc (
            .clk   (clk),
            .reset (reset),
            .stall (stall),
            .flush (0),
            .in    ({read_valid_st1c[i-1], read_dirty_st1c[i-1], read_dirtyb_st1c[i-1], read_tag_st1c[i-1], read_data_st1c[i-1]}),
            .out   ({read_valid_st1c[i],   read_dirty_st1c[i],   read_dirtyb_st1c[i],   read_tag_st1c[i],   read_data_st1c[i]})
        );
    end

    assign use_read_valid_st1e = read_valid_st1c[STAGE_1_CYCLES-1] || !DRAM_ENABLE; // If shared memory, always valid
    assign use_read_dirty_st1e = read_dirty_st1c[STAGE_1_CYCLES-1] && DRAM_ENABLE && WRITE_ENABLE; // Dirty only applies in Dcache
    assign use_read_tag_st1e   = DRAM_ENABLE ? read_tag_st1c[STAGE_1_CYCLES-1] : writetag_st1e; // Tag is always the same in SM
    assign use_read_dirtyb_st1e= read_dirtyb_st1c[STAGE_1_CYCLES-1];
    assign use_read_data_st1e  = read_data_st1c[STAGE_1_CYCLES-1];
    
    if (`WORD_SELECT_WIDTH != 0) begin
        wire [`WORD_WIDTH-1:0] readword = use_read_data_st1e[wordsel_st1e * `WORD_WIDTH +: `WORD_WIDTH];
        for (i = 0; i < WORD_SIZE; i++) begin
            assign readword_st1e[i * 8 +: 8] = readword[i * 8 +: 8] & {8{mem_byteen_st1e[i]}};
        end
    end else begin
        for (i = 0; i < WORD_SIZE; i++) begin
            assign readword_st1e[i * 8 +: 8] = use_read_data_st1e[i * 8 +: 8] & {8{mem_byteen_st1e[i]}};
        end
    end

    wire [`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] we;
    wire [`BANK_LINE_WIDTH-1:0] data_write;

    wire should_write = mem_rw_st1e 
                     && valid_req_st1e 
                     && use_read_valid_st1e 
                     && ~miss_st1e 
                     && ~is_snp_st1e
                     && ~real_writefill;

    for (i = 0; i < `BANK_LINE_WORDS; i++) begin
        wire normal_write = ((`WORD_SELECT_WIDTH == 0) || (wordsel_st1e == `UP(`WORD_SELECT_WIDTH)'(i))) 
                         && should_write;

        assign we[i] = real_writefill ? {WORD_SIZE{1'b1}} : 
                         normal_write ? mem_byteen_st1e :
                                        {WORD_SIZE{1'b0}};

        assign data_write[i * `WORD_WIDTH +: `WORD_WIDTH] = real_writefill ? writedata_st1e[i * `WORD_WIDTH +: `WORD_WIDTH] : writeword_st1e;
    end

    assign use_write_enable = (writefill_st1e && ~real_writefill) ? 0 : we;
    assign use_write_data   = data_write;

    // use "case equality" to handle uninitialized tag when block entry is not valid
    assign tags_match = (writetag_st1e === use_read_tag_st1e);

    wire snoop_hit_no_pending = valid_req_st1e &&  is_snp_st1e &&  use_read_valid_st1e && tags_match && (use_read_dirty_st1e || snp_invalidate_st1e) && ~force_request_miss_st1e;
    wire req_invalid          = valid_req_st1e && ~is_snp_st1e && ~use_read_valid_st1e && ~writefill_st1e;
    wire req_miss             = valid_req_st1e && ~is_snp_st1e &&  use_read_valid_st1e && ~writefill_st1e && ~tags_match;
    wire real_miss            = req_invalid || req_miss;
    wire force_core_miss      = (force_request_miss_st1e && ~is_snp_st1e && ~writefill_st1e && valid_req_st1e && ~real_miss);    
    assign snp_to_mrvq_st1e   = valid_req_st1e && is_snp_st1e && force_request_miss_st1e;
    
    // The second term is basically saying always make an entry ready if there's already antoher entry waiting, even if you yourself see a miss
    assign mrvq_init_ready_state_st1e = snp_to_mrvq_st1e 
                                     || (force_request_miss_st1e && ~is_snp_st1e && ~writefill_st1e && valid_req_st1e);

    assign miss_st1e           = real_miss || snoop_hit_no_pending || force_core_miss;
    assign dirty_st1e          = valid_req_st1e && use_read_valid_st1e && use_read_dirty_st1e;
    assign dirtyb_st1e         = use_read_dirtyb_st1e;
    assign readdata_st1e       = use_read_data_st1e;
    assign readtag_st1e        = use_read_tag_st1e;
    assign fill_sent           = miss_st1e;
    assign fill_saw_dirty_st1e = real_writefill && dirty_st1e;
    assign invalidate_line     = snoop_hit_no_pending;

`ifdef DBG_PRINT_CACHE_BANK
    always @(posedge clk) begin        
        if (valid_req_st1e) begin             
            if ((| use_write_enable)) begin
                if (writefill_st1e) begin
                    $display("%t: bank%0d:%0d store-fill: warp=%0d, PC=%0h, tag=%0h, wb=%b, rd=%0d, dirty=%b, blk_addr=%0d, tag_id=%0h, data=%0h", $time, CACHE_ID, BANK_ID, debug_warp_num_st1e, debug_pc_st1e, debug_tagid_st1e, debug_wb_st1e, debug_rd_st1e, dirty_st1e, writeladdr_st1e, writetag_st1e, use_write_data);
                end else begin
                    $display("%t: bank%0d:%0d store-write: warp=%0d, PC=%0h, tag=%0h, wb=%b, rd=%0d, dirty=%b, blk_addr=%0d, tag_id=%0h, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, debug_warp_num_st1e, debug_pc_st1e, debug_tagid_st1e, debug_wb_st1e, debug_rd_st1e, dirty_st1e, writeladdr_st1e, writetag_st1e, wordsel_st1e, writeword_st1e);
                end
            end else
            if (miss_st1e) begin
                $display("%t: bank%0d:%0d store-miss: warp=%0d, PC=%0h, tag=%0h, wb=%b, rd=%0d, dirty=%b", $time, CACHE_ID, BANK_ID, debug_warp_num_st1e, debug_pc_st1e, debug_tagid_st1e, debug_wb_st1e, debug_rd_st1e, dirty_st1e);
            end else begin
                $display("%t: bank%0d:%0d store-read: warp=%0d, PC=%0h, tag=%0h, wb=%b, rd=%0d, dirty=%b, blk_addr=%0d, tag_id=%0h, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, debug_warp_num_st1e, debug_pc_st1e, debug_tagid_st1e, debug_wb_st1e, debug_rd_st1e, dirty_st1e, readaddr_st10, qual_read_tag_st1, wordsel_st1e, qual_read_data_st1);
            end            
        end
    end    
`endif

endmodule