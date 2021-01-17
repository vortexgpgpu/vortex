`include "VX_cache_config.vh"

module VX_data_access #(
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
    parameter WRITE_ENABLE      = 1,
    // Enable write-through
    parameter WRITE_THROUGH     = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc,
    input wire[`NW_BITS-1:0]            debug_wid,
`IGNORE_WARNINGS_END
`endif

    input  wire                         stall,

`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    addr_in,    
`IGNORE_WARNINGS_END

    // reading
    input wire                          readen_in,
    output wire [`CACHE_LINE_WIDTH-1:0] readdata_out,

    // writing
    input wire                          writeen_in,
    input wire [`UP(`WORD_SELECT_BITS)-1:0] wwsel_in,
    input wire [WORD_SIZE-1:0]          wbyteen_in,    
    input wire                          wfill_in,
    input wire [`WORD_WIDTH-1:0]        writeword_in,
    input wire [`CACHE_LINE_WIDTH-1:0]  filldata_in
);
    `UNUSED_VAR (reset)

    wire [`CACHE_LINE_WIDTH-1:0] read_data;
    wire [CACHE_LINE_SIZE-1:0]   byte_enable; 
    wire [`CACHE_LINE_WIDTH-1:0] write_data;   
    wire                         write_enable;

    wire [`LINE_SELECT_BITS-1:0] line_addr = addr_in[`LINE_SELECT_BITS-1:0];

    VX_sp_ram #(
        .DATAW(CACHE_LINE_SIZE * 8),
        .SIZE(`LINES_PER_BANK),
        .BYTEENW(CACHE_LINE_SIZE),
        .RWCHECK(1)
    ) data_store (
        .clk(clk),
        .addr(line_addr),             
        .wren(write_enable),  
        .byteen(byte_enable),
        .rden(1'b1),
        .din(write_data),
        .dout(read_data)
    );

    wire [`WORDS_PER_LINE-1:0][WORD_SIZE-1:0] wbyteen_qual;
    wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] writedata_qual;

    if (`WORD_SELECT_BITS != 0) begin
        for (genvar i = 0; i < `WORDS_PER_LINE; i++) begin
            assign wbyteen_qual[i]   = (wwsel_in == `WORD_SELECT_BITS'(i)) ? wbyteen_in : {WORD_SIZE{1'b0}};
            assign writedata_qual[i] = writeword_in;
        end
    end else begin
        `UNUSED_VAR (wwsel_in)
        assign wbyteen_qual   = wbyteen_in;
        assign writedata_qual = writeword_in;
    end
    
    assign write_enable = writeen_in && !stall;
    assign byte_enable  = wfill_in ? {CACHE_LINE_SIZE{1'b1}} : wbyteen_qual;
    assign write_data   = wfill_in ? filldata_in : writedata_qual;
    assign readdata_out = read_data;

    `UNUSED_VAR (readen_in)

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin            
        if (!stall) begin
            if (writeen_in) begin
                if (wfill_in) begin
                    $display("%t: cache%0d:%0d data-fill: addr=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), line_addr, write_data);
                end else begin
                    $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), debug_wid, debug_pc, byte_enable, line_addr, wwsel_in, writeword_in);
                end
            end 
            if (readen_in) begin
                $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), debug_wid, debug_pc, line_addr, read_data);
            end            
        end
    end    
`endif

endmodule