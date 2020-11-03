`include "VX_cache_config.vh"

module VX_tag_data_access #(
    parameter CACHE_ID          = 0,
    parameter BANK_ID           = 0,   

    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE    = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 

    // Enable cache writeable
    parameter WRITE_ENABLE      = 0,

    // Enable dram update
    parameter DRAM_ENABLE       = 0,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire                          clk,
    input wire                          reset,

    input wire                          stall,

`ifdef DBG_CORE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_st1,
    input wire[`NR_BITS-1:0]            debug_rd_st1,
    input wire[`NW_BITS-1:0]            debug_wid_st1,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st1,
`IGNORE_WARNINGS_END
`endif

    input wire                          is_snp_st1,
    input wire                          snp_invalidate_st1,

    input wire[`LINE_ADDR_WIDTH-1:0]    addr_st1,
    
    input wire                          valid_req_st1,
    input wire                          writefill_st1,
    input wire[`WORD_WIDTH-1:0]         writeword_st1,
    input wire[`BANK_LINE_WIDTH-1:0]    writedata_st1,

`IGNORE_WARNINGS_BEGIN    
    input wire                          mem_rw_st1,
    input wire[WORD_SIZE-1:0]           mem_byteen_st1, 
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] wordsel_st1,
`IGNORE_WARNINGS_END

    input wire                          force_miss_st1,

    output wire[`WORD_WIDTH-1:0]        readword_st1,
    output wire[`BANK_LINE_WIDTH-1:0]   readdata_st1,
    output wire[`TAG_SELECT_BITS-1:0]   readtag_st1,
    output wire                         miss_st1,
    output wire                         dirty_st1,
    output wire[BANK_LINE_SIZE-1:0]     dirtyb_st1
);

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
    wire                        use_invalidate;

    wire[`TAG_SELECT_BITS-1:0] addrtag_st1 = addr_st1[`TAG_LINE_ADDR_RNG];
    wire[`LINE_SELECT_BITS-1:0] addrline_st1 = addr_st1[`LINE_SELECT_BITS-1:0];

    VX_tag_data_store #(
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE)
    ) tag_data_store (
        .clk         (clk),
        .reset       (reset),

        .read_addr   (addrline_st1),
        .read_valid  (qual_read_valid_st1),        
        .read_dirty  (qual_read_dirty_st1),
        .read_dirtyb (qual_read_dirtyb_st1),
        .read_tag    (qual_read_tag_st1),
        .read_data   (qual_read_data_st1),

        .invalidate  (use_invalidate),
        .write_enable(use_write_enable),
        .write_fill  (writefill_st1),
        .write_addr  (addrline_st1),
        .tag_index   (addrtag_st1),
        .write_data  (use_write_data)
    );

    assign use_read_valid_st1 = qual_read_valid_st1 || !DRAM_ENABLE; // If shared memory, always valid
    assign use_read_dirty_st1 = qual_read_dirty_st1 && DRAM_ENABLE && WRITE_ENABLE; // Dirty only applies in Dcache
    assign use_read_tag_st1   = DRAM_ENABLE ? qual_read_tag_st1 : addrtag_st1; // Tag is always the same in SM
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

    // use "case equality" to handle uninitialized tag when block entry is not valid
    wire tags_match = use_read_valid_st1 && (addrtag_st1 === use_read_tag_st1);

    wire [`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] write_enable;
    wire [`BANK_LINE_WIDTH-1:0] data_write;

    wire normal_write = valid_req_st1
                     && !writefill_st1  
                     && !is_snp_st1 
                     && !miss_st1
                     && !force_miss_st1
                     && mem_rw_st1  
                     && use_read_valid_st1;

    wire fill_write = valid_req_st1 && writefill_st1 
                   && !tags_match; // disable redundant fills because the block could be dirty

    for (genvar i = 0; i < `BANK_LINE_WORDS; i++) begin
        wire normal_write_w = ((`WORD_SELECT_WIDTH == 0) || (wordsel_st1 == `UP(`WORD_SELECT_WIDTH)'(i))) 
                           && normal_write;

        assign write_enable[i] = fill_write ? {WORD_SIZE{1'b1}} : 
                                    normal_write_w ? mem_byteen_st1 :
                                        {WORD_SIZE{1'b0}};

        assign data_write[i * `WORD_WIDTH +: `WORD_WIDTH] = writefill_st1 ? writedata_st1[i * `WORD_WIDTH +: `WORD_WIDTH] : writeword_st1;
    end

    assign use_write_enable = write_enable;
    assign use_write_data   = data_write;
    assign use_invalidate   = valid_req_st1 && is_snp_st1 && tags_match 
                           && (use_read_dirty_st1 || snp_invalidate_st1)  // block is dirty or need to force invalidation
                           && !force_miss_st1
                           && !stall; // do not invalidate the cache on stalls
    
    wire core_req_miss = valid_req_st1 && !is_snp_st1 && !writefill_st1 // is core request
                      && (!use_read_valid_st1 || !tags_match);   // block missing or has wrong tag

    assign miss_st1     = core_req_miss;
    assign dirty_st1    = valid_req_st1 && use_read_valid_st1 && use_read_dirty_st1;
    assign dirtyb_st1   = use_read_dirtyb_st1;
    assign readdata_st1 = use_read_data_st1;
    assign readtag_st1  = use_read_tag_st1;    

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin            
        if (valid_req_st1) begin
            if (writefill_st1 && use_read_valid_st1 && tags_match) begin
                $display("%t: warning: redundant fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));
            end    
            if (miss_st1) begin
                $display("%t: cache%0d:%0d data-miss: addr=%0h, wid=%0d, PC=%0h, valid=%b, tagmatch=%b, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), debug_wid_st1, debug_pc_st1, use_read_dirty_st1, tags_match, addrline_st1, addrtag_st1);
            end else if ((| use_write_enable)) begin
                if (writefill_st1) begin
                    $display("%t: cache%0d:%0d data-fill: addr=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), dirtyb_st1, addrline_st1, addrtag_st1, use_write_data);
                end else begin
                    $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), debug_wid_st1, debug_pc_st1, dirtyb_st1, addrline_st1, addrtag_st1, wordsel_st1, writeword_st1);
                end
            end else begin
                $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), debug_wid_st1, debug_pc_st1, dirtyb_st1, addrline_st1, qual_read_tag_st1, wordsel_st1, qual_read_data_st1);
            end            
        end
    end    
`endif

endmodule