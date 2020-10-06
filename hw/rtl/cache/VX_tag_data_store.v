`include "VX_cache_config.vh"

module VX_tag_data_store #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 0, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...} 
    parameter NUM_BANKS                     = 0, //unused parameter?
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0
) (
    input  wire                             clk,
    input  wire                             reset,
    input  wire                             stall_bank_pipe,

    input  wire[`LINE_SELECT_BITS-1:0]      read_addr,
    output wire                             read_valid,
    output wire                             read_dirty,
    output wire[`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] read_dirtyb,
    output wire[`TAG_SELECT_BITS-1:0]       read_tag,
    output wire[`BANK_LINE_WIDTH-1:0]       read_data,

    input  wire                             invalidate,
    input  wire[`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] write_enable,
    input  wire                             write_fill,
    input  wire[`LINE_SELECT_BITS-1:0]      write_addr,
    input  wire[`TAG_SELECT_BITS-1:0]       tag_index,
    input  wire[`BANK_LINE_WIDTH-1:0]       write_data,
    input  wire                             fill_sent    
);

    reg [`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0][7:0] data [`BANK_LINE_COUNT-1:0];    
    reg [`TAG_SELECT_BITS-1:0]                      tag [`BANK_LINE_COUNT-1:0];
    reg [`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0]     dirtyb[`BANK_LINE_COUNT-1:0];
    reg [`BANK_LINE_COUNT-1:0]                     dirty;   
    reg [`BANK_LINE_COUNT-1:0]                     valid;    
    
    assign read_valid  = valid  [read_addr];
    assign read_dirty  = dirty  [read_addr];
    assign read_dirtyb = dirtyb [read_addr];
    assign read_tag    = tag    [read_addr];
    assign read_data   = data   [read_addr];

    wire do_write = (| write_enable);

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < `BANK_LINE_COUNT; i++) begin
                valid[i] <= 0;
                dirty[i] <= 0;
            end
        end else if (!stall_bank_pipe) begin
            if (do_write) begin
                valid[write_addr] <= 1;
                tag  [write_addr] <= tag_index;
                if (write_fill) begin
                    dirty[write_addr]  <= 0;
                    dirtyb[write_addr] <= 0;
                end else begin
                    dirty[write_addr]  <= 1;
                    dirtyb[write_addr] <= dirtyb[write_addr] | write_enable;
                end
            end else if (fill_sent) begin
                dirty[write_addr]  <= 0;
                dirtyb[write_addr] <= 0;
            end

            if (invalidate) begin
                valid[write_addr] <= 0;
            end

            for (integer j = 0; j < `BANK_LINE_WORDS; j++) begin
                for (integer i = 0; i < WORD_SIZE; i++) begin
                    if (write_enable[j][i]) begin
                        data[write_addr][j][i] <= write_data[j * `WORD_WIDTH + i * 8 +: 8];
                    end
                end
            end
        end
    end

endmodule
