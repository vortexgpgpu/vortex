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
    parameter WRITE_ENABLE      = 1
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_r,
    input wire[`NW_BITS-1:0]            debug_wid_r,
    input wire[31:0]                    debug_pc_w,
    input wire[`NW_BITS-1:0]            debug_wid_w,
`IGNORE_WARNINGS_END
`endif

    // reading
    input wire                          readen,
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    raddr,
`IGNORE_WARNINGS_END
    output wire [`CACHE_LINE_WIDTH-1:0] rdata,

    // writing
    input wire                          writeen,
    input wire                          is_fill,
    input wire [CACHE_LINE_SIZE-1:0]    byteen,        
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    waddr,
`IGNORE_WARNINGS_END
    input wire [`CACHE_LINE_WIDTH-1:0]  wdata
);
    `UNUSED_VAR (reset)
    `UNUSED_VAR (readen)

    wire [`LINE_SELECT_BITS-1:0] line_raddr, line_waddr;
    wire [CACHE_LINE_SIZE-1:0] byte_enable;
    
    assign line_raddr = raddr[`LINE_SELECT_BITS-1:0];
    assign line_waddr = waddr[`LINE_SELECT_BITS-1:0];
    assign byte_enable = (WRITE_ENABLE && !is_fill) ? byteen : {CACHE_LINE_SIZE{1'b1}};

    VX_dp_ram #(
        .DATAW(CACHE_LINE_SIZE * 8),
        .SIZE(`LINES_PER_BANK),
        .BYTEENW(CACHE_LINE_SIZE),
        .RWCHECK(1)
    ) data_store (
        .clk(clk),        
        .raddr(line_raddr),
        .waddr(line_waddr),
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
                $display("%t: cache%0d:%0d data-fill: addr=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(waddr, BANK_ID), line_waddr, wdata);
            end else begin
                $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(waddr, BANK_ID), debug_wid_w, debug_pc_w, byte_enable, line_waddr, wdata);
            end
        end 
        if (readen) begin
            $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(raddr, BANK_ID), debug_wid_r, debug_pc_r, line_raddr, rdata);
        end            
    end    
`endif

endmodule