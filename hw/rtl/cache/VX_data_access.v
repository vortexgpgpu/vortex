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

    // Enable dram update
    parameter DRAM_ENABLE       = 0,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CORE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    debug_pc_st2,
    input wire[`NR_BITS-1:0]            debug_rd_st2,
    input wire[`NW_BITS-1:0]            debug_wid_st2,
    input wire[`UP(CORE_TAG_ID_BITS)-1:0] debug_tagid_st2,
`IGNORE_WARNINGS_END
`endif

    input  wire                         stall,

    input wire                          valid_req_st2,
    input wire                          writeen_st2,
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    addr_st2,    
`IGNORE_WARNINGS_END    
    input wire                          writefill_st2,
    input wire[`WORD_WIDTH-1:0]         writeword_st2,
    input wire[`BANK_LINE_WIDTH-1:0]    writedata_st2,

    input wire[WORD_SIZE-1:0]           mem_byteen_st2, 
    input wire[`UP(`WORD_SELECT_WIDTH)-1:0] wordsel_st2,

    output wire[`WORD_WIDTH-1:0]        readword_st2,
    output wire[`BANK_LINE_WIDTH-1:0]   readdata_st2,
    output wire[BANK_LINE_SIZE-1:0]     dirtyb_st2
);

    wire[BANK_LINE_SIZE-1:0]    qual_read_dirtyb_st2;
    wire[`BANK_LINE_WIDTH-1:0]  qual_read_data_st2;

    wire[BANK_LINE_SIZE-1:0]    use_read_dirtyb_st2;
    wire[`BANK_LINE_WIDTH-1:0]  use_read_data_st2;
    wire[`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] use_byte_enable;
    wire[`BANK_LINE_WIDTH-1:0]  use_write_data;
    wire                        use_write_enable;

    wire[`LINE_SELECT_BITS-1:0] addrline_st2 = addr_st2[`LINE_SELECT_BITS-1:0];

    VX_data_store #(
        .CACHE_SIZE     (CACHE_SIZE),
        .BANK_LINE_SIZE (BANK_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE)
    ) data_store (
        .clk         (clk),

        .reset       (reset),

        .read_addr   (addrline_st2),
        .read_dirtyb (qual_read_dirtyb_st2),
        .read_data   (qual_read_data_st2),

        .write_enable(use_write_enable),
        .write_fill  (writefill_st2),
        .byte_enable (use_byte_enable),
        .write_addr  (addrline_st2),        
        .write_data  (use_write_data)
    );

    assign use_read_dirtyb_st2= qual_read_dirtyb_st2;
    assign use_read_data_st2  = qual_read_data_st2;
    
    if (`WORD_SELECT_WIDTH != 0) begin
        wire [`WORD_WIDTH-1:0] readword = use_read_data_st2[wordsel_st2 * `WORD_WIDTH +: `WORD_WIDTH];
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_st2[i * 8 +: 8] = readword[i * 8 +: 8] & {8{mem_byteen_st2[i]}};
        end
    end else begin
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_st2[i * 8 +: 8] = use_read_data_st2[i * 8 +: 8] & {8{mem_byteen_st2[i]}};
        end
    end

    wire [`BANK_LINE_WORDS-1:0][WORD_SIZE-1:0] byte_enable;
    wire [`BANK_LINE_WIDTH-1:0] data_write;

    for (genvar i = 0; i < `BANK_LINE_WORDS; i++) begin
        wire word_sel = ((`WORD_SELECT_WIDTH == 0) || (wordsel_st2 == `UP(`WORD_SELECT_WIDTH)'(i)));
        
        assign byte_enable[i] = writefill_st2 ? {WORD_SIZE{1'b1}} : 
                                    word_sel ? mem_byteen_st2 :
                                        {WORD_SIZE{1'b0}};

        assign data_write[i * `WORD_WIDTH +: `WORD_WIDTH] = writefill_st2 ? writedata_st2[i * `WORD_WIDTH +: `WORD_WIDTH] : writeword_st2;
    end

    assign use_write_enable = valid_req_st2 && writeen_st2 && !stall;
    assign use_byte_enable  = byte_enable;
    assign use_write_data   = data_write;

    assign dirtyb_st2   = use_read_dirtyb_st2;
    assign readdata_st2 = use_read_data_st2;

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin            
        if (valid_req_st2 && !stall) begin
            if (use_write_enable) begin
                if (writefill_st2) begin
                    $display("%t: cache%0d:%0d data-fill: addr=%0h, dirty=%b, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st2, BANK_ID), dirtyb_st2, addrline_st2, use_write_data);
                end else begin
                    $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st2, BANK_ID), debug_wid_st2, debug_pc_st2, dirtyb_st2, addrline_st2, wordsel_st2, writeword_st2);
                end
            end else begin
                $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr_st2, BANK_ID), debug_wid_st2, debug_pc_st2, dirtyb_st2, addrline_st2, wordsel_st2, qual_read_data_st2);
            end            
        end
    end    
`endif

endmodule