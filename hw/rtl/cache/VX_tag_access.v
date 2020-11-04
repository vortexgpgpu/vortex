`include "VX_cache_config.vh"

module VX_tag_access #(
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

    input wire[`LINE_ADDR_WIDTH-1:0]    addr_st1,
    
    input wire                          valid_req_st1,
    input wire                          writefill_st1,

    input wire                          mem_rw_st1,

    input wire                          force_miss_st1,

    output wire[`TAG_SELECT_BITS-1:0]   readtag_st1,
    output wire                         miss_st1,
    output wire                         dirty_st1,
    output wire                         writeen_st1
);

    wire                        qual_read_valid_st1;
    wire                        qual_read_dirty_st1;
    wire[`TAG_SELECT_BITS-1:0]  qual_read_tag_st1;

    wire                        use_read_valid_st1;
    wire                        use_read_dirty_st1;
    wire[`TAG_SELECT_BITS-1:0]  use_read_tag_st1;

    wire                        use_write_enable;
    wire                        use_invalidate;  
    
    wire[`TAG_SELECT_BITS-1:0] addrtag_st1 = addr_st1[`TAG_LINE_ADDR_RNG];
    wire[`LINE_SELECT_BITS-1:0] addrline_st1 = addr_st1[`LINE_SELECT_BITS-1:0];

    VX_tag_store #(
        .CACHE_SIZE (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .WORD_SIZE  (WORD_SIZE)
    ) tag_store (
        .clk         (clk),
        .reset       (reset),

        .stall       (stall),

        .read_addr   (addrline_st1),
        .read_valid  (qual_read_valid_st1),        
        .read_dirty  (qual_read_dirty_st1),
        .read_tag    (qual_read_tag_st1),

        .invalidate  (use_invalidate),
        .write_enable(use_write_enable),
        .write_fill  (writefill_st1),
        .write_addr  (addrline_st1),
        .write_tag   (addrtag_st1)
    );

    assign use_read_valid_st1 = qual_read_valid_st1 || !DRAM_ENABLE; // If shared memory, always valid
    assign use_read_dirty_st1 = qual_read_dirty_st1 && DRAM_ENABLE && WRITE_ENABLE; // Dirty only applies in Dcache
    assign use_read_tag_st1   = DRAM_ENABLE ? qual_read_tag_st1 : addrtag_st1; // Tag is always the same in SM

    // use "case equality" to handle uninitialized tag when block entry is not valid
    wire tags_match = use_read_valid_st1 && (addrtag_st1 === use_read_tag_st1);

    wire normal_write = valid_req_st1
                     && mem_rw_st1  
                     && use_read_valid_st1
                     && !writefill_st1  
                     && !is_snp_st1 
                     && !miss_st1
                     && !force_miss_st1;

    wire fill_write = valid_req_st1 && writefill_st1 
                   && !tags_match;  // discard redundant fills because the block could be dirty

    assign use_write_enable = normal_write || fill_write;

    assign use_invalidate   = valid_req_st1 && is_snp_st1 && tags_match 
                           && (use_read_dirty_st1 || snp_invalidate_st1)  // block is dirty or need to force invalidation
                           && !force_miss_st1;
    
    wire core_req_miss = valid_req_st1 && !is_snp_st1 && !writefill_st1 // is core request
                      && (!use_read_valid_st1 || !tags_match);   // block missing or has wrong tag

    assign miss_st1    = core_req_miss;
    assign dirty_st1   = valid_req_st1 && use_read_valid_st1 && use_read_dirty_st1;
    assign readtag_st1 = use_read_tag_st1;
    assign writeen_st1 = use_write_enable;    

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin            
        if (valid_req_st1 && !stall) begin
            if (writefill_st1 && use_read_valid_st1 && tags_match) begin
                $display("%t: warning: redundant fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID));
            end    
            if (miss_st1) begin
                $display("%t: cache%0d:%0d data-miss: addr=%0h, wid=%0d, PC=%0h, valid=%b, tagmatch=%b, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), debug_wid_st1, debug_pc_st1, use_read_dirty_st1, tags_match, addrline_st1, addrtag_st1);
            end else if ((| use_write_enable)) begin
                if (writefill_st1) begin
                    $display("%t: cache%0d:%0d data-fill: addr=%0h, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), addrline_st1, addrtag_st1);
                end else begin
                    $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), debug_wid_st1, debug_pc_st1, addrline_st1, addrtag_st1);
                end
            end else begin
                $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st1, BANK_ID), debug_wid_st1, debug_pc_st1, addrline_st1, qual_read_tag_st1);
            end            
        end
    end    
`endif

endmodule