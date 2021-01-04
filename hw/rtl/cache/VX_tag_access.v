`include "VX_cache_config.vh"

module VX_tag_access #(
    parameter CACHE_ID          = 0,
    parameter BANK_ID           = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1,
    // Enable cache writeable
    parameter WRITE_ENABLE      = 0,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0,
    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET  = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc,
    input wire[`NW_BITS-1:0]            debug_wid,
`IGNORE_WARNINGS_END
`endif

    input wire                          stall,

    // read/fill
    input wire                          lookup_in,
    input wire[`LINE_ADDR_WIDTH-1:0]    raddr_in,   
    input wire                          do_fill_in,    
    output wire                         miss_out,
    output wire[`TAG_SELECT_BITS-1:0]   readtag_out,    
    output wire                         dirty_out,

    // write
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    waddr_in,   
`IGNORE_WARNINGS_END
    input wire                          writeen_in
);

    wire                        read_valid;
    wire                        read_dirty;
    wire [`TAG_SELECT_BITS-1:0] read_tag;

    wire                        do_fill;
    wire                        do_write;
    
    wire [`TAG_SELECT_BITS-1:0] raddr_tag = `LINE_TAG_ADDR(raddr_in);
    wire [`LINE_SELECT_BITS-1:0] raddr = raddr_in [`LINE_SELECT_BITS-1:0];
    wire [`LINE_SELECT_BITS-1:0] waddr = waddr_in [`LINE_SELECT_BITS-1:0];

    VX_tag_store #(
        .CACHE_SIZE (CACHE_SIZE),
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .WORD_SIZE  (WORD_SIZE),
        .BANK_ADDR_OFFSET (BANK_ADDR_OFFSET)
    ) tag_store (
        .clk         (clk),
        .reset       (reset),

        .raddr       (raddr),
        .read_valid  (read_valid),        
        .read_dirty  (read_dirty),
        .read_tag    (read_tag),
        .do_fill     (do_fill),
        .fill_tag    (raddr_tag),

        .waddr       (waddr),
        .do_write    (do_write)        
    );

    // read/fill stage

    wire tags_match = read_valid && (raddr_tag == read_tag);

    assign do_fill = do_fill_in && !stall;

    assign readtag_out = read_tag; 

    assign miss_out = !tags_match;

    assign dirty_out = read_dirty || ((raddr == waddr) && writeen_in);   

    // write stage
                      
    assign do_write = WRITE_ENABLE && writeen_in && !stall;

    wire do_lookup = lookup_in && !stall;
    `UNUSED_VAR (do_lookup)

`ifdef DBG_PRINT_CACHE_TAG
    always @(posedge clk) begin          
        if (do_fill) begin
            $display("%t: cache%0d:%0d tag-fill: addr=%0h, blk_addr=%0d, tag_id=%0h, old_tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(raddr_in, BANK_ID), raddr, raddr_tag, read_tag);   
            if (tags_match) begin
                $display("%t: warning: redundant fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(raddr_in, BANK_ID));
            end
        end else if (do_lookup) begin                
            if (tags_match) begin
                $display("%t: cache%0d:%0d tag-hit: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(raddr_in, BANK_ID), debug_wid, debug_pc, read_dirty, raddr, raddr_tag);                
            end else begin
                $display("%t: cache%0d:%0d tag-miss: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, tag_id=%0h, old_tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(raddr_in, BANK_ID), debug_wid, debug_pc, read_dirty, raddr, raddr_tag, read_tag);
            end  
        end          
    end    
`endif

endmodule