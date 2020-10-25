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

     // Enable cache writeable
     parameter WRITE_ENABLE     = 0,

     // Enable dram update
     parameter DRAM_ENABLE      = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CORE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_st1,
    input wire[`NR_BITS-1:0]            debug_rd_st1,
    input wire[`NW_BITS-1:0]            debug_wid_st1,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st1,
`IGNORE_WARNINGS_END
`endif

    input wire                          stall,
    input wire                          is_snp_st1,
    input wire                          snp_invalidate_st1,
    input wire                          stall_bank_pipe,

    input wire                          force_request_miss_st1,

    input wire[`LINE_SELECT_BITS-1:0]   readaddr_st1, 
    input wire[`LINE_ADDR_WIDTH-1:0]    writeaddr_st1,
    
    input wire                          valid_req_st1,
    input wire                          writefill_st1,
    input wire[`WORD_WIDTH-1:0]         writeword_st1,
    input wire[`BANK_LINE_WIDTH-1:0]    writedata_st1,

`IGNORE_WARNINGS_BEGIN    
    input wire                          mem_rw_st1,
    input wire[WORD_SIZE-1:0]           mem_byteen_st1, 
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] wordsel_st1,
`IGNORE_WARNINGS_END

    output wire[`WORD_WIDTH-1:0]        readword_st1,
    output wire[`BANK_LINE_WIDTH-1:0]   readdata_st1,
    output wire[`TAG_SELECT_BITS-1:0]   readtag_st1,
    output wire                         miss_st1,
    output wire                         dirty_st1,
    output wire[BANK_LINE_SIZE-1:0]     dirtyb_st1,
    output wire                         fill_saw_dirty_st1,
    output wire                         snp_to_mrvq_st1,
    output wire                         mrvq_init_ready_state_st1
);
    `UNUSED_VAR (stall)
    
    wire                        qual_read_valid_st1;
    wire                        qual_read_dirty_st1;
    wire[BANK_LINE_SIZE-1:0]    qual_read_dirtyb_st1;
    wire[`TAG_SELECT_BITS-1:0]  qual_read_tag_st1;
    wire[`BANK_LINE_WIDTH-1:0]  qual_read_data_st1;

    wire                        use_read_valid_st1;
    wire                        use_read_dirty_st1;
    wire[BANK_LINE_SIZE-1:0]    use_read_dirtyb_st1;
    wire[`TAG_SELECT_BITS-1:0]  use_read_tag_st1;
    wire[`BANK_LINE_WIDTH-1:0]  use_read_data_st1;
    wire[`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] use_write_enable;
    wire[`BANK_LINE_WIDTH-1:0]  use_write_data;

    wire fill_sent;
    wire invalidate_line;
    wire tags_match;

    wire real_writefill = valid_req_st1 && writefill_st1
                       && ((~use_read_valid_st1) || (use_read_valid_st1 && ~tags_match)); 

    wire[`TAG_SELECT_BITS-1:0] writetag_st1 = writeaddr_st1[`TAG_LINE_ADDR_RNG];
    wire[`LINE_SELECT_BITS-1:0] writeladdr_st1 = writeaddr_st1[`LINE_SELECT_BITS-1:0];

    VX_tag_data_store #(
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE)
    ) tag_data_store (
        .clk         (clk),
        .reset       (reset),
        .stall_bank_pipe(stall_bank_pipe),

        .read_addr   (readaddr_st1),
        .read_valid  (qual_read_valid_st1),        
        .read_dirty  (qual_read_dirty_st1),
        .read_dirtyb (qual_read_dirtyb_st1),
        .read_tag    (qual_read_tag_st1),
        .read_data   (qual_read_data_st1),

        .invalidate  (invalidate_line),
        .write_enable(use_write_enable),
        .write_fill  (real_writefill),
        .write_addr  (writeladdr_st1),
        .tag_index   (writetag_st1),
        .write_data  (use_write_data),
        .fill_sent   (fill_sent)
    );

    assign use_read_valid_st1 = qual_read_valid_st1 || !DRAM_ENABLE; // If shared memory, always valid
    assign use_read_dirty_st1 = qual_read_dirty_st1 && DRAM_ENABLE && WRITE_ENABLE; // Dirty only applies in Dcache
    assign use_read_tag_st1   = DRAM_ENABLE ? qual_read_tag_st1 : writetag_st1; // Tag is always the same in SM
    assign use_read_dirtyb_st1= qual_read_dirtyb_st1;
    assign use_read_data_st1  = qual_read_data_st1;
    
    if (`WORD_SELECT_WIDTH != 0) begin
        wire [`WORD_WIDTH-1:0] readword = use_read_data_st1[wordsel_st1 * `WORD_WIDTH +: `WORD_WIDTH];
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_st1[i * 8 +: 8] = readword[i * 8 +: 8] & {8{mem_byteen_st1[i]}};
        end
    end else begin
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_st1[i * 8 +: 8] = use_read_data_st1[i * 8 +: 8] & {8{mem_byteen_st1[i]}};
        end
    end

    wire [`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] we;
    wire [`BANK_LINE_WIDTH-1:0] data_write;

    wire should_write = mem_rw_st1 
                     && valid_req_st1 
                     && use_read_valid_st1 
                     && ~miss_st1 
                     && ~is_snp_st1
                     && ~real_writefill;

    for (genvar i = 0; i < `BANK_LINE_WORDS; i++) begin
        wire normal_write = ((`WORD_SELECT_WIDTH == 0) || (wordsel_st1 == `UP(`WORD_SELECT_WIDTH)'(i))) 
                         && should_write;

        assign we[i] = real_writefill ? {WORD_SIZE{1'b1}} : 
                         normal_write ? mem_byteen_st1 :
                                        {WORD_SIZE{1'b0}};

        assign data_write[i * `WORD_WIDTH +: `WORD_WIDTH] = real_writefill ? writedata_st1[i * `WORD_WIDTH +: `WORD_WIDTH] : writeword_st1;
    end

    assign use_write_enable = (writefill_st1 && ~real_writefill) ? 0 : we;
    assign use_write_data   = data_write;

    // use "case equality" to handle uninitialized tag when block entry is not valid
    assign tags_match = (writetag_st1 === use_read_tag_st1);

    wire snoop_hit_no_pending = valid_req_st1 &&  is_snp_st1 &&  use_read_valid_st1 && tags_match && (use_read_dirty_st1 || snp_invalidate_st1) && ~force_request_miss_st1;
    wire req_invalid          = valid_req_st1 && ~is_snp_st1 && ~use_read_valid_st1 && ~writefill_st1;
    wire req_miss             = valid_req_st1 && ~is_snp_st1 &&  use_read_valid_st1 && ~writefill_st1 && ~tags_match;
    wire real_miss            = req_invalid || req_miss;
    wire force_core_miss      = (force_request_miss_st1 && ~is_snp_st1 && ~writefill_st1 && valid_req_st1 && ~real_miss);    
    assign snp_to_mrvq_st1    = valid_req_st1 && is_snp_st1 && force_request_miss_st1;
    
    // The second term is basically saying always make an entry ready if there's already antoher entry waiting, even if you yourself see a miss
    assign mrvq_init_ready_state_st1 = snp_to_mrvq_st1 
                                     || (force_request_miss_st1 && ~is_snp_st1 && ~writefill_st1 && valid_req_st1);

    assign miss_st1           = real_miss || snoop_hit_no_pending || force_core_miss;
    assign dirty_st1          = valid_req_st1 && use_read_valid_st1 && use_read_dirty_st1;
    assign dirtyb_st1         = use_read_dirtyb_st1;
    assign readdata_st1       = use_read_data_st1;
    assign readtag_st1        = use_read_tag_st1;
    assign fill_sent          = miss_st1;
    assign fill_saw_dirty_st1 = real_writefill && dirty_st1;
    assign invalidate_line    = snoop_hit_no_pending;

`ifdef DBG_PRINT_CACHE_BANK
    always @(posedge clk) begin        
        if (valid_req_st1) begin             
            if ((| use_write_enable)) begin
                if (writefill_st1) begin
                    $display("%t: cache%0d:%0d data-fill: wid=%0d, PC=%0h, tag=%0h, rd=%0d, dirty=%b, blk_addr=%0d, tag_id=%0h, data=%0h", $time, CACHE_ID, BANK_ID, debug_wid_st1, debug_pc_st1, debug_tagid_st1, debug_rd_st1, dirty_st1, writeladdr_st1, writetag_st1, use_write_data);
                end else begin
                    $display("%t: cache%0d:%0d data-write: wid=%0d, PC=%0h, tag=%0h, rd=%0d, dirty=%b, blk_addr=%0d, tag_id=%0h, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, debug_wid_st1, debug_pc_st1, debug_tagid_st1, debug_rd_st1, dirty_st1, writeladdr_st1, writetag_st1, wordsel_st1, writeword_st1);
                end
            end else
            if (miss_st1) begin
                $display("%t: cache%0d:%0d data-miss: wid=%0d, PC=%0h, tag=%0h, rd=%0d, dirty=%b", $time, CACHE_ID, BANK_ID, debug_wid_st1, debug_pc_st1, debug_tagid_st1, debug_rd_st1, dirty_st1);
            end else begin
                $display("%t: cache%0d:%0d data-read: wid=%0d, PC=%0h, tag=%0h, rd=%0d, dirty=%b, blk_addr=%0d, tag_id=%0h, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, debug_wid_st1, debug_pc_st1, debug_tagid_st1, debug_rd_st1, dirty_st1, readaddr_st1, qual_read_tag_st1, wordsel_st1, qual_read_data_st1);
            end            
        end
    end    
`endif

endmodule