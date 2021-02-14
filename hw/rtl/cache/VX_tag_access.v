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
    
    // read/fill
    input wire                          lookup,
    input wire[`LINE_ADDR_WIDTH-1:0]    addr,   
    input wire                          fill,    
    input wire                          is_flush,
    output wire                         tag_match
);
    `UNUSED_VAR (reset)
    `UNUSED_VAR (lookup)

    wire                        read_valid;
    wire [`TAG_SELECT_BITS-1:0] read_tag;
    
    wire [`TAG_SELECT_BITS-1:0] line_tag = `LINE_TAG_ADDR(addr);
    wire [`LINE_SELECT_BITS-1:0] line_addr = addr [`LINE_SELECT_BITS-1:0];

    VX_sp_ram #(
        .DATAW(`TAG_SELECT_BITS + 1),
        .SIZE(`LINES_PER_BANK),
        .INITZERO(1),
        .RWCHECK(1)
    ) tag_store (
        .clk(clk),                 
        .addr(line_addr),   
        .wren(fill),
        .byteen(1'b1),
        .rden(1'b1),
        .din({!is_flush, line_tag}),
        .dout({read_valid, read_tag})
    );

    assign tag_match = read_valid && (line_tag == read_tag);

`ifdef DBG_PRINT_CACHE_TAG
    always @(posedge clk) begin          
        if (fill) begin
            if (is_flush) begin
                $display("%t: cache%0d:%0d tag-flush: addr=%0h, blk_addr=%0d", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr);
            end else begin
                $display("%t: cache%0d:%0d tag-fill: addr=%0h, blk_addr=%0d, tag_id=%0h, old_tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag, read_tag);
                if (tag_match) begin
                    $display("%t: warning: redundant fill - addr=%0h", $time, `LINE_TO_BYTE_ADDR(addr, BANK_ID));
                end
            end
        end else if (lookup) begin                
            if (tag_match) begin
                $display("%t: cache%0d:%0d tag-hit: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, line_addr, line_tag);                
            end else begin
                $display("%t: cache%0d:%0d tag-miss: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, tag_id=%0h, old_tag_id=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, line_addr, line_tag, read_tag);
            end  
        end          
    end    
`endif

endmodule