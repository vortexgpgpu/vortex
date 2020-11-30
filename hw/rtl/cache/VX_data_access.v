`include "VX_cache_config.vh"

module VX_data_access #(
    parameter CACHE_ID          = 0,
    parameter BANK_ID           = 0,   

    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE    = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE         = 1, 

    // Enable cache writeable
    parameter WRITE_ENABLE      = 0,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc,
    input wire[`NR_BITS-1:0]            debug_rd,
    input wire[`NW_BITS-1:0]            debug_wid,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid,
`IGNORE_WARNINGS_END
`endif

    input  wire                         stall,

    // Inputs
    input wire                          valid_in,
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    addr_in,    
`IGNORE_WARNINGS_END    
    input wire                          writeen_in,
    input wire                          is_fill_in,
    input wire[`WORD_WIDTH-1:0]         writeword_in,
    input wire[`BANK_LINE_WIDTH-1:0]    writedata_in,
    input wire[WORD_SIZE-1:0]           byteen_in, 
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] wordsel_in,

    // Outputs
    output wire[`WORD_WIDTH-1:0]        readword_out,
    output wire[`BANK_LINE_WIDTH-1:0]   readdata_out,
    output wire[BANK_LINE_SIZE-1:0]     dirtyb_out
);

    wire[BANK_LINE_SIZE-1:0]    read_dirtyb_out;
    wire[`BANK_LINE_WIDTH-1:0]  read_data;

    wire[`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] byte_enable;    
    wire                        write_enable;
    wire[`BANK_LINE_WIDTH-1:0]  write_data;

    wire[`LINE_SELECT_BITS-1:0] addrline = addr_in[`LINE_SELECT_BITS-1:0];

    VX_data_store #(
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .WRITE_ENABLE   (WRITE_ENABLE)
    ) data_store (
        .clk         (clk),

        .reset       (reset),

        .read_addr   (addrline),
        .read_dirtyb (read_dirtyb_out),
        .read_data   (read_data),

        .write_enable(write_enable),
        .write_fill  (is_fill_in),
        .byte_enable (byte_enable),
        .write_addr  (addrline),        
        .write_data  (write_data)
    );
    
    if (`WORD_SELECT_WIDTH != 0) begin
        wire [`WORD_WIDTH-1:0] readword = read_data[wordsel_in * `WORD_WIDTH +: `WORD_WIDTH];
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_out[i * 8 +: 8] = readword[i * 8 +: 8] & {8{byteen_in[i]}};
        end
    end else begin
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_out[i * 8 +: 8] = read_data[i * 8 +: 8] & {8{byteen_in[i]}};
        end
    end

    for (genvar i = 0; i < `BANK_LINE_WORDS; i++) begin
        wire word_sel = (`WORD_SELECT_WIDTH == 0) || (wordsel_in == `UP(`WORD_SELECT_WIDTH)'(i));
        
        assign byte_enable[i] = is_fill_in ? {WORD_SIZE{1'b1}} : 
                                        word_sel ? byteen_in :
                                            {WORD_SIZE{1'b0}};

        assign write_data[i * `WORD_WIDTH +: `WORD_WIDTH] = is_fill_in ? writedata_in[i * `WORD_WIDTH +: `WORD_WIDTH] : writeword_in;
    end

    assign write_enable = valid_in 
                       && writeen_in 
                       && !stall;

    assign dirtyb_out   = read_dirtyb_out;
    assign readdata_out = read_data;

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin            
        if (valid_in && !stall) begin
            if (write_enable) begin
                if (is_fill_in) begin
                    $display("%t: cache%0d:%0d data-fill: addr=%0h, dirty=%b, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), dirtyb_out, addrline, write_data);
                end else begin
                    $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, dirty=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), debug_wid, debug_pc, byte_enable, dirtyb_out, addrline, wordsel_in, writeword_in);
                end
            end else begin
                $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_in, BANK_ID), debug_wid, debug_pc, dirtyb_out, addrline, wordsel_in, read_data);
            end            
        end
    end    
`endif

endmodule