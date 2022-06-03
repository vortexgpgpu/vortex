`include "VX_cache_define.vh"

module VX_tag_access #(
    parameter string INSTANCE_ID= "",
    parameter BANK_ID           = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE         = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Number of associative ways
    parameter NUM_WAYS          = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 
    // Request debug identifier
    parameter REQ_UUID_BITS     = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire [`UP(REQ_UUID_BITS)-1:0] req_uuid,
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

    `UNUSED_PARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (lookup)

    localparam TAG_WIDTH = `TAG_SEL_BITS + 1;
    localparam TAG_BYTES = ((TAG_WIDTH + 7) / 8);

    wire [NUM_WAYS-1:0] tag_matches;
    wire [`LINE_SEL_BITS-1:0] line_addr = addr[`LINE_SEL_BITS-1:0];
    wire [`TAG_SEL_BITS-1:0]  line_tag = `LINE_TAG_ADDR(addr);    
    wire [NUM_WAYS-1:0] fill_way;

    if (NUM_WAYS > 1)  begin
        reg [NUM_WAYS-1:0] repl_way;
        // cyclic assignment of replacement way
        always @(posedge clk) begin
            if (reset) begin
                repl_way <= 1;
            end else if (!stall) begin // hold the value on stalls prevent filling different slots twice
                repl_way <= {repl_way[NUM_WAYS-2:0], repl_way[NUM_WAYS-1]};
            end
        end        
        for (genvar i = 0; i < NUM_WAYS; ++i) begin
            assign fill_way[i] = fill & repl_way[i];
        end
    end else begin
        `UNUSED_VAR (stall)
        assign fill_way = fill;
    end

    wire [NUM_WAYS-1:0][(8 * TAG_BYTES)-1:0] wdata, rdata;
    wire [NUM_WAYS-1:0][TAG_BYTES-1:0] byteen;

    for (genvar i = 0; i < NUM_WAYS; ++i) begin
        wire [`TAG_SEL_BITS-1:0] read_tag;
        wire read_valid;

        assign wdata[i]  = (8 * TAG_BYTES)'({!flush, line_tag});
        assign byteen[i] = {TAG_BYTES{(fill_way[i] || flush)}};
        
        assign {read_valid, read_tag} = rdata[i][(`TAG_SEL_BITS+1)-1:0];
        assign tag_matches[i] = read_valid && (line_tag == read_tag);
    end

    VX_sp_ram #(
        .DATAW      (8 * TAG_BYTES * NUM_WAYS),
        .SIZE       (`LINES_PER_BANK),
        .BYTEENW    (TAG_BYTES * NUM_WAYS),
        .NO_RWCHECK (1)
    ) tag_store (
        .clk   (clk),                 
        .addr  (line_addr),
        .wren  (byteen),
        .wdata (wdata), 
        .rdata (rdata)
    );

    // found a tag match?
    assign tag_match = (| tag_matches);

    // return the selected way
    assign way_sel = fill_way | tag_matches;
    
`ifdef DBG_TRACE_CACHE_TAG
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s:%0d tag-fill: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, line_tag));
        end
        if (flush) begin
            `TRACE(3, ("%d: %s:%0d tag-flush: addr=0x%0h, blk_addr=%0d\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr));
        end
        if (lookup && ~stall) begin
            if (tag_match) begin
                `TRACE(3, ("%d: %s:%0d tag-hit: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, line_tag, req_uuid));
            end else begin
                `TRACE(3, ("%d: %s:%0d tag-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, line_tag, req_uuid));
            end
        end          
    end    
`endif

endmodule
