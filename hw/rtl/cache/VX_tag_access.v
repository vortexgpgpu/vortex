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

    // Enable cache flush
    parameter FLUSH_ENABLE      = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc,
    input wire[`NR_BITS-1:0]            debug_rd,
    input wire[`NW_BITS-1:0]            debug_wid,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid,
`IGNORE_WARNINGS_END
`endif

    input wire                          stall,

    // Inputs
    input wire                          valid_in,
    input wire[`LINE_ADDR_WIDTH-1:0]    addr_in,   
    input wire                          is_write_in,
    input wire                          is_fill_in,
    input wire                          is_snp_in,
    input wire                          snp_inv_in,
    input wire                          force_miss_in,

    // Outputs
    output wire[`TAG_SELECT_BITS-1:0]   readtag_out,
    output wire                         miss_out,
    output wire                         dirty_out,
    output wire                         writeen_out
);

    wire                        read_valid;
    wire                        read_dirty;
    wire[`TAG_SELECT_BITS-1:0]  read_tag;

    wire                        do_fill;
    wire                        do_write;
    wire                        do_invalidate;  
    
    wire [`TAG_SELECT_BITS-1:0] addrtag = addr_in [`TAG_LINE_ADDR_RNG];
    wire [`LINE_SELECT_BITS-1:0] addrline = addr_in [`LINE_SELECT_BITS-1:0];

    VX_tag_store #(
        .CACHE_SIZE (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .WORD_SIZE  (WORD_SIZE)
    ) tag_store (
        .clk         (clk),
        .reset       (reset),

        .read_addr   (addrline),
        .read_valid  (read_valid),        
        .read_dirty  (read_dirty),
        .read_tag    (read_tag),

        .do_fill     (do_fill),
        .do_write    (do_write),
        .invalidate  (do_invalidate),
        .write_addr  (addrline),
        .write_tag   (addrtag)
    );

    // use "case equality" to handle uninitialized tag when block entry is not valid
    wire tags_match = read_valid && (addrtag == read_tag);  
                      
    assign do_write = WRITE_ENABLE
                   && valid_in                        
                   && tags_match
                   && !is_snp_in
                   && !is_fill_in
                   && is_write_in                       
                   && !force_miss_in
                   && !stall;

    assign do_fill = valid_in 
                  && is_fill_in
                  && !stall;

    assign do_invalidate = FLUSH_ENABLE
                        && valid_in
                        && tags_match
                        && is_snp_in                         
                        && (read_dirty || snp_inv_in)
                        && !force_miss_in
                        && !stall;

    assign miss_out = valid_in 
                   && !tags_match
                   && !is_snp_in 
                   && !is_fill_in;

    assign dirty_out = WRITE_ENABLE 
                    && valid_in 
                    && read_valid 
                    && read_dirty 
                    && !(is_fill_in && tags_match);  // discard writeback for redundant fills

    assign readtag_out = read_tag;

    assign writeen_out = do_write || (do_fill 
                                   && !tags_match); // discard data update for redundant fills

`ifdef DBG_PRINT_CACHE_TAG
    always @(posedge clk) begin            
        if (valid_in && !stall) begin
            if (do_fill) begin
                $display("%t: cache%0d:%0d tag-fill: addr=%0h, blk_addr=%0d, tag_id=%0h, old_tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), addrline, addrtag, read_tag);   
                if (tags_match) begin
                    $display("%t: warning: redundant fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID));
                end
            end else if (tags_match) begin
                $display("%t: cache%0d:%0d tag-hit: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), debug_wid, debug_pc, read_dirty, addrline, addrtag);                
            end else begin
                $display("%t: cache%0d:%0d tag-miss: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h, old_tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), debug_wid, debug_pc, read_dirty, addrline, addrtag, read_tag);
            end            
        end
    end    
`endif

endmodule