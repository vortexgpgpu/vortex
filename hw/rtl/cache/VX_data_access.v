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
    parameter WRITE_THROUGH     = 1
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc,
    input wire[`NW_BITS-1:0]            debug_wid,
`IGNORE_WARNINGS_END
`endif

`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    addr,    
`IGNORE_WARNINGS_END

    // reading
    input wire                          readen,
    output wire [`CACHE_LINE_WIDTH-1:0] rddata,

    // writing
    input wire                          writeen,
    input wire                          is_fill,
    input wire [CACHE_LINE_SIZE-1:0]    byteen,        
    input wire [`CACHE_LINE_WIDTH-1:0]  wrdata
);
    `UNUSED_VAR (reset)

    wire [CACHE_LINE_SIZE-1:0] byte_enable;

    wire [`LINE_SELECT_BITS-1:0] line_addr = addr[`LINE_SELECT_BITS-1:0];

    VX_sp_ram #(
        .DATAW(CACHE_LINE_SIZE * 8),
        .SIZE(`LINES_PER_BANK),
        .BYTEENW(CACHE_LINE_SIZE),
        .RWCHECK(1)
    ) data_store (
        .clk(clk),
        .addr(line_addr),             
        .wren(writeen),  
        .byteen(byte_enable),
        .rden(1'b1),
        .din(wrdata),
        .dout(rddata)
    );
    
    assign byte_enable = is_fill ? {CACHE_LINE_SIZE{1'b1}} : byteen;

    `UNUSED_VAR (readen)

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin 
        if (writeen) begin
            if (is_fill) begin
                $display("%t: cache%0d:%0d data-fill: addr=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, wrdata);
            end else begin
                $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, byte_enable, line_addr, wrdata);
            end
        end 
        if (readen) begin
            $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, line_addr, rddata);
        end            
    end    
`endif

endmodule