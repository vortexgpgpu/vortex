`include "VX_cache_config.vh"

module VX_data_store #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE               = 1, 
    // Number of banks
    parameter NUM_BANKS                     = 1,
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1,

    // Enable cache writeable
    parameter WRITE_ENABLE                  = 0
) (
    input  wire                             clk,
    input  wire                             reset,

    input  wire                             write_enable,
    input  wire                             write_fill,
    input  wire[CACHE_LINE_SIZE-1:0]         byte_enable,
    input  wire[`LINE_SELECT_BITS-1:0]      write_addr,
    input  wire[`CACHE_LINE_WIDTH-1:0]       write_data,

    input  wire[`LINE_SELECT_BITS-1:0]      read_addr,
    output wire[`WORDS_PER_LINE-1:0][WORD_SIZE-1:0] read_dirtyb,
    output wire[`CACHE_LINE_WIDTH-1:0]       read_data
);
    `UNUSED_VAR (reset)

    if (WRITE_ENABLE) begin
        reg [`WORDS_PER_LINE-1:0][WORD_SIZE-1:0] dirtyb[`LINES_PER_BANK-1:0];
        always @(posedge clk) begin
            if (write_enable) begin
                dirtyb[write_addr] <= write_fill ? 0 : (dirtyb[write_addr] | byte_enable);
            end
        end
        assign read_dirtyb = dirtyb [read_addr];
    end else begin
        `UNUSED_VAR (write_fill)
        `UNUSED_VAR (byte_enable)
        assign read_dirtyb = 0;
    end

    VX_dp_ram #(
        .DATAW(CACHE_LINE_SIZE * 8),
        .SIZE(`LINES_PER_BANK),
        .RWCHECK(1)
    ) data (
        .clk(clk),
        .waddr(write_addr),                                
        .raddr(read_addr),                
        .wren(write_enable),  
        .byteen(1'b1),
        .rden(1'b1),
        .din(write_data),
        .dout(read_data)
    );

endmodule
