`include "VX_cache_config.vh"

module VX_tag_store #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 1, 
    // Number of banks
    parameter NUM_BANKS                     = 1,
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1
) (
    input  wire                             clk,
    input  wire                             reset,  

    input  wire                             do_fill,
    input  wire                             do_write,
    input  wire                             invalidate, 
    input  wire[`LINE_SELECT_BITS-1:0]      write_addr,
    input  wire[`TAG_SELECT_BITS-1:0]       write_tag,   
    
    input  wire[`LINE_SELECT_BITS-1:0]      read_addr,
    output wire[`TAG_SELECT_BITS-1:0]       read_tag,
    output wire                             read_valid,
    output wire                             read_dirty    
);
    reg [`BANK_LINE_COUNT-1:0] dirty;   
    reg [`BANK_LINE_COUNT-1:0] valid;  

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < `BANK_LINE_COUNT; i++) begin
                valid[i] <= 0;
                dirty[i] <= 0;
            end
        end else begin
            if (do_fill) begin
                valid[write_addr] <= 1;
                dirty[write_addr] <= 0;
            end else if (do_write) begin
                dirty[write_addr] <= 1;
            end else if (invalidate) begin
                valid[write_addr] <= 0;
            end
        end
    end

    VX_dp_ram #(
        .DATAW(`TAG_SELECT_BITS),
        .SIZE(`BANK_LINE_COUNT),
        .BYTEENW(1),
        .BUFFERED(0),
        .RWCHECK(1)
    ) tags (
        .clk(clk),                 
        .waddr(write_addr),                                
        .raddr(read_addr),                
        .wren(do_fill),
        .byteen(1'b1),
        .rden(1'b1),
        .din(write_tag),
        .dout(read_tag)
    );  
    
    assign read_valid = valid[read_addr];
    assign read_dirty = dirty[read_addr];

endmodule
