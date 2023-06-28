`include "VX_cache_define.vh"

module VX_cache_tags #(
    parameter `STRING INSTANCE_ID = "",
    parameter BANK_ID       = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE    = 1, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE     = 1, 
    // Number of banks
    parameter NUM_BANKS     = 1, 
    // Number of associative ways
    parameter NUM_WAYS      = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE     = 1, 
    // Request debug identifier
    parameter UUID_WIDTH    = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire [`UP(UUID_WIDTH)-1:0]    req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

    // read/fill
    input wire                          lookup,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] addr,
    input wire                          fill,    
    input wire                          init,
    output wire [NUM_WAYS-1:0]          way_sel,
    output wire [NUM_WAYS-1:0]          tag_matches
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (lookup)

    localparam TAG_WIDTH = 1 + `CS_TAG_SEL_BITS;

    wire [`CS_LINE_SEL_BITS-1:0] line_addr = addr[`CS_LINE_SEL_BITS-1:0];
    wire [`CS_TAG_SEL_BITS-1:0] line_tag = `CS_LINE_TAG_ADDR(addr);

    if (NUM_WAYS > 1)  begin
        reg [NUM_WAYS-1:0] repl_way;
        // cyclic assignment of replacement way
        always @(posedge clk) begin
            if (reset) begin
                repl_way <= 1;
            end else if (~stall) begin // hold the value on stalls prevent filling different slots twice
                repl_way <= {repl_way[NUM_WAYS-2:0], repl_way[NUM_WAYS-1]};
            end
        end        
        for (genvar i = 0; i < NUM_WAYS; ++i) begin
            assign way_sel[i] = fill && repl_way[i];
        end
    end else begin
        `UNUSED_VAR (stall)
        assign way_sel = fill;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin
        wire [`CS_TAG_SEL_BITS-1:0] read_tag;
        wire read_valid;

        VX_sp_ram #(
            .DATAW (TAG_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .NO_RWCHECK (1)
        ) tag_store (
            .clk   (clk), 
            .write (way_sel[i] || init),
            `UNUSED_PIN (wren),                
            .addr  (line_addr),
            .wdata ({~init, line_tag}), 
            .rdata ({read_valid, read_tag})
        );
        
        assign tag_matches[i] = read_valid && (line_tag == read_tag);
    end
    
`ifdef DBG_TRACE_CACHE_TAG
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s:%0d tag-fill: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, line_tag));
        end
        if (init) begin
            `TRACE(3, ("%d: %s:%0d tag-init: addr=0x%0h, blk_addr=%0d\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr));
        end
        if (lookup && ~stall) begin
            if (tag_matches) begin
                `TRACE(3, ("%d: %s:%0d tag-hit: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, line_tag, req_uuid));
            end else begin
                `TRACE(3, ("%d: %s:%0d tag-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag, req_uuid));
            end
        end          
    end    
`endif

endmodule
