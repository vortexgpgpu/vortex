`include "VX_cache_define.vh"

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
    parameter WRITE_ENABLE      = 1
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
    output wire [`CACHE_LINE_WIDTH-1:0] rdata,

    // writing
    input wire                          writeen,
    input wire                          is_fill,
    input wire [CACHE_LINE_SIZE-1:0]    byteen,
    input wire [`CACHE_LINE_WIDTH-1:0]  wdata
);

    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (readen)

    localparam BYTEENW = WRITE_ENABLE ? CACHE_LINE_SIZE : 1;

    wire [`LINE_SELECT_BITS-1:0] line_addr;
    wire [BYTEENW-1:0] byte_enable;
    
    assign line_addr = addr[`LINE_SELECT_BITS-1:0];

    if (WRITE_ENABLE) begin
        assign byte_enable = is_fill ? {BYTEENW{1'b1}} : byteen;
    end else begin
        `UNUSED_VAR (byteen)
        `UNUSED_VAR (is_fill)
        assign byte_enable = 1'b1;
    end

    VX_sp_ram #(
        .DATAW   (CACHE_LINE_SIZE * 8),
        .SIZE    (`LINES_PER_BANK),
        .BYTEENW (BYTEENW),
        .RWCHECK (1)
    ) data_store (
        .clk(clk),        
        .addr(line_addr),
        .wren(writeen),
        .byteen(byte_enable),
        .rden(1'b1),
        .din(wdata),
        .dout(rdata)
    );

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin 
        if (writeen) begin
            if (is_fill) begin
                $display("%t: cache%0d:%0d data-fill: addr=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, wdata);
            end else begin
                $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, byte_enable, line_addr, wdata);
            end
        end 
        if (readen) begin
            $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, line_addr, rdata);
        end            
    end    
`endif

endmodule