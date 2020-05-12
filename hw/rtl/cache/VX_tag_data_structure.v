`include "VX_cache_config.vh"

module VX_tag_data_structure #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 0, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...} 
    parameter NUM_BANKS                     = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0
) (
    input  wire                             clk,
    input  wire                             reset,
    input  wire                             stall_bank_pipe,

    input  wire[`LINE_SELECT_BITS-1:0]      read_addr,
    output wire                             read_valid,
    output wire                             read_dirty,
    output wire[`TAG_SELECT_BITS-1:0]       read_tag,
    output wire[`BANK_LINE_WIDTH-1:0]       read_data,

    input  wire                             invalidate,
    input  wire[`BANK_LINE_WORDS-1:0][3:0]  write_enable,
    input  wire                             write_fill,
    input  wire[`LINE_SELECT_BITS-1:0]      write_addr,
    input  wire[`TAG_SELECT_BITS-1:0]       tag_index,
    input  wire[`BANK_LINE_WIDTH-1:0]       write_data,
    input  wire                             fill_sent    
);

    reg [`BANK_LINE_WORDS-1:0][3:0][`BYTE_WIDTH-1:0] data [`BANK_LINE_COUNT-1:0];
    reg [`TAG_SELECT_BITS-1:0]                        tag [`BANK_LINE_COUNT-1:0];
    reg                                             valid [`BANK_LINE_COUNT-1:0];
    reg                                             dirty [`BANK_LINE_COUNT-1:0];   

    assign read_valid = valid [read_addr];
    assign read_dirty = dirty [read_addr];
    assign read_tag   = tag   [read_addr];
    assign read_data  = data  [read_addr];

    wire going_to_write = (| write_enable);

    integer i;
    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < `BANK_LINE_COUNT; i++) begin
                valid[i] <= 0;
                dirty[i] <= 0;
            end
        end else if (!stall_bank_pipe) begin
            if (going_to_write) begin
                valid[write_addr] <= 1;
                tag  [write_addr] <= tag_index;
                if (write_fill) begin
                    dirty[write_addr] <= 0;
                end else begin
                    dirty[write_addr] <= 1;
                end
            end else if (fill_sent) begin
                dirty[write_addr] <= 0;
            end

            if (invalidate) begin
                valid[write_addr] <= 0;
            end

            for (i = 0; i < `BANK_LINE_WORDS; i++) begin
                if (write_enable[i][0]) data[write_addr][i][0] <= write_data[i * `WORD_WIDTH + 0 * `BYTE_WIDTH +: `BYTE_WIDTH];
                if (write_enable[i][1]) data[write_addr][i][1] <= write_data[i * `WORD_WIDTH + 1 * `BYTE_WIDTH +: `BYTE_WIDTH];
                if (write_enable[i][2]) data[write_addr][i][2] <= write_data[i * `WORD_WIDTH + 2 * `BYTE_WIDTH +: `BYTE_WIDTH];
                if (write_enable[i][3]) data[write_addr][i][3] <= write_data[i * `WORD_WIDTH + 3 * `BYTE_WIDTH +: `BYTE_WIDTH];
            end
        end
    end

endmodule