`include "VX_cache_define.vh"

module VX_tag_access #(
    parameter CACHE_ID          = 0,
    parameter BANK_ID           = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Number of associative ways
    parameter NUM_WAYS          = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // bank offset from beginning of index range
    parameter BANK_ADDR_OFFSET  = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire[`DBG_CACHE_REQ_IDW-1:0]  req_id,
`IGNORE_UNUSED_END

    input wire                          stall,

    // read/fill
    input wire                          lookup,
    input wire[`LINE_ADDR_WIDTH-1:0]    addr,
    input wire                          fill,    
    input wire                          flush,
    output wire[NUM_WAYS-1:0]              select_way,
    output wire                         tag_match
);

    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (lookup)

    wire [`TAG_SELECT_BITS-1:0] read_tag;
    wire read_valid;
    wire[NUM_WAYS-1:0]              tag_match_way;
    wire [`LINE_SELECT_BITS-1:0] line_addr = addr[`LINE_SELECT_BITS-1:0];
    wire [`TAG_SELECT_BITS-1:0] line_tag = `LINE_TAG_ADDR(addr);
    logic [NUM_WAYS-1:0] repl_way;
    wire fill_local[NUM_WAYS-1:0];

    //cyclic assignment of replacement way
    initial begin
        if (reset)
            repl_way =1;
        else 
            repl_way = {repl_way[NUM_WAYS-2:0],repl_way[NUM_WAYS-1]};//rotate left     
    end
    generate
    genvar g;
    for (g = 0; g < NUM_WAYS; g = g+1) begin
        assign fill_local[g] = (fill && repl_way[g]) ? 1 : 0;
    end 
    endgenerate

    //We use a tag match array to check if each of the arrays has a match 
    //assign the output wire to the ANDed result of tag_match_array

    wire fill_local[NUM_WAYS-1:0];
    for (i = 0; i < NUM_WAYS; i = i+1) begin
        assign fill_local[i] = (fill && repl_way[i]) ? 1 : 0;
    end

    generate 
        genvar i;
        for (i = 0; i < NUM_WAYS; i = i+1) begin
            VX_sp_ram #(
            .DATAW      (`TAG_SELECT_BITS + 1),
            .SIZE       (`LINES_PER_BANK),
            .NO_RWCHECK (1)
            ) tag_store (
                .clk(  clk),                 
                .addr  (line_addr),   
                .wren  (fill_local[i] || flush),
                .wdata ({!flush, line_tag}), 
                .rdata ({read_valid, read_tag})
            );
            assign tag_match_way[i] = read_valid && (line_tag == read_tag);
        end
    endgenerate
    //Check if any of the ways have tag match
    assign tag_match = |tag_match_way;

    //select_ways is passed to VX_bank and VX_data_access
    //select_ways = tag_match_way if the access is with fill = 0, else select_ways = replacement way
    generate
        genvar m;
        for (m = 0; m < NUM_WAYS; m = m+1) begin
            assign select_way[m] = ((fill & repl_way[m]) || (!fill & tag_match_way[m])) ? 1: 0; 
        end
    endgenerate
    
    `UNUSED_VAR (stall)
    
`ifdef DBG_TRACE_CACHE_TAG
    always @(posedge clk) begin
        if (fill && ~stall) begin
            dpi_trace("%d: cache%0d:%0d tag-fill: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag);
        end
        if (flush) begin
            dpi_trace("%d: cache%0d:%0d tag-flush: addr=0x%0h, blk_addr=%0d\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr);
        end
        if (lookup && ~stall) begin                
            if (tag_match) begin
                dpi_trace("%d: cache%0d:%0d tag-hit: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag, req_id);
            end else begin
                dpi_trace("%d: cache%0d:%0d tag-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, old_tag_id=0x%0h (#%0d)\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag, read_tag, req_id);
            end
        end          
    end    
`endif

endmodule