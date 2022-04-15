`include "VX_cache_define.vh"

module VX_tag_access #(
    parameter string CACHE_ID   = "",
    parameter BANK_ID           = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Number of associative ways
    parameter NUM_WAYS          = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // Request debug identifier
    parameter REQ_DBG_IDW       = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire [`DBG_CACHE_REQ_IDW-1:0] req_id,
`IGNORE_UNUSED_END

    input wire                          stall,

    // read/fill
    input wire                          lookup,
    input wire [`LINE_ADDR_WIDTH-1:0]   addr,
    input wire                          fill,    
    input wire                          flush,
    output wire [NUM_WAYS-1:0]          way_sel,
    output wire                         tag_match
);

    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (lookup)

    wire [NUM_WAYS-1:0] tag_match_way;
    wire [`LINE_SEL_BITS-1:0] line_addr = addr[`LINE_SEL_BITS-1:0];
    wire [`TAG_SEL_BITS-1:0] line_tag = `LINE_TAG_ADDR(addr);    
    wire [NUM_WAYS-1:0] fill_way;

    if (NUM_WAYS > 1)  begin
        reg [NUM_WAYS-1:0] repl_way;
        // cyclic assignment of replacement way
        always @(posedge clk) begin
            if (reset)
                repl_way <= 1;
            else 
                if (!stall) begin // hold the value on stalls prevent filling different slots twice
                    repl_way <= {repl_way[NUM_WAYS-2:0], repl_way[NUM_WAYS-1]};
                end
        end        
        for (genvar i = 0; i < NUM_WAYS; ++i) begin
            assign fill_way[i] = fill & repl_way[i];
        end
    end else begin
        assign fill_way = fill;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin
        wire [`TAG_SEL_BITS-1:0] read_tag;
        wire read_valid;

        VX_sp_ram #(
            .DATAW      (`TAG_SEL_BITS + 1),
            .SIZE       (`LINES_PER_BANK),
            .NO_RWCHECK (1)
        ) tag_store (
            .clk(  clk),                 
            .addr  (line_addr),   
            .wren  (fill_way[i] || flush),
            .wdata ({!flush,     line_tag}), 
            .rdata ({read_valid, read_tag})
        );
        
        assign tag_match_way[i] = read_valid && (line_tag == read_tag);
    end

    // Check if any of the ways have tag match
    assign tag_match = (| tag_match_way);

    // return the selected way
    assign way_sel = fill_way | tag_match_way;
    
`ifdef DBG_TRACE_CACHE_TAG
    always @(posedge clk) begin
        if (fill && ~stall) begin
            dpi_trace(3, "%d: %s:%0d tag-fill: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, line_tag);
        end
        if (flush) begin
            dpi_trace(3, "%d: %s:%0d tag-flush: addr=0x%0h, blk_addr=%0d\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr);
        end
        if (lookup && ~stall) begin
            if (tag_match) begin
                dpi_trace(3, "%d: %s:%0d tag-hit: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, line_tag, req_id);
            end else begin
                dpi_trace(3, "%d: %s:%0d tag-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, (#%0d)\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag, req_id);
            end
        end          
    end    
`endif

endmodule
