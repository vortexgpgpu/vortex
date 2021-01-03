`include "VX_cache_config.vh"

module VX_tag_store #(
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
    input  wire                             clk,
    input  wire                             reset,  

    input  wire[`LINE_SELECT_BITS-1:0]      addr,
    
    input  wire                             do_fill,
    input  wire                             do_write,
    input  wire                             invalidate,
    input  wire[`TAG_SELECT_BITS-1:0]       write_tag,   
    
    output wire[`TAG_SELECT_BITS-1:0]       read_tag,
    output wire                             read_valid,
    output wire                             read_dirty    
);
    reg [`LINES_PER_BANK-1:0] dirty;   
    reg [`LINES_PER_BANK-1:0] valid;  

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < `LINES_PER_BANK; i++) begin
                valid[i] <= 0;
                dirty[i] <= 0;
            end
        end else begin
            if (do_fill) begin
                valid[addr] <= 1;
                dirty[addr] <= 0;
            end else if (do_write) begin
                dirty[addr] <= 1;
            end else if (invalidate) begin
                valid[addr] <= 0;
            end
        end
    end

    VX_sp_ram #(
        .DATAW(`TAG_SELECT_BITS),
        .SIZE(`LINES_PER_BANK),
        .RWCHECK(1)
    ) tags (
        .clk(clk),                 
        .addr(addr),   
        .wren(do_fill),
        .byteen(1'b1),
        .rden(1'b1),
        .din(write_tag),
        .dout(read_tag)
    );  
    
    assign read_valid = valid[addr];
    assign read_dirty = dirty[addr];

endmodule
