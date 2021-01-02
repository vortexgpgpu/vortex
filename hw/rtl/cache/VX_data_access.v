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
    
    // Enable dram update
    parameter DRAM_ENABLE       = 1,

    // Enable cache writeable
    parameter WRITE_ENABLE      = 1,

    // Enable write-through
    parameter WRITE_THROUGH     = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS  = 0
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_WARNINGS_BEGIN
    input wire[31:0]                    rdebug_pc,
    input wire[`NW_BITS-1:0]            rdebug_wid,
    input wire[31:0]                    wdebug_pc,
    input wire[`NW_BITS-1:0]            wdebug_wid,
`IGNORE_WARNINGS_END
`endif

    input  wire                         stall,

    // reading
    input wire                          readen_in,
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    raddr_in,    
`IGNORE_WARNINGS_END
    input wire [`UP(`WORD_SELECT_BITS)-1:0] rwsel_in,
    input wire [WORD_SIZE-1:0]          rbyteen_in,
    output wire[`WORD_WIDTH-1:0]        readword_out,
    output wire [`CACHE_LINE_WIDTH-1:0]  readdata_out,
    output wire [CACHE_LINE_SIZE-1:0]    dirtyb_out,

    // writing
    input wire                          writeen_in,
`IGNORE_WARNINGS_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    waddr_in,    
`IGNORE_WARNINGS_END
    input wire [`UP(`WORD_SELECT_BITS)-1:0] wwsel_in,
    input wire [WORD_SIZE-1:0]          wbyteen_in,    
    input wire                          wfill_in,
    input wire [`WORD_WIDTH-1:0]        writeword_in,
    input wire [`CACHE_LINE_WIDTH-1:0]   writedata_in
);

    wire [CACHE_LINE_SIZE-1:0]   read_dirtyb, dirtyb_qual;
    wire [`CACHE_LINE_WIDTH-1:0] read_data, readdata_qual;

    wire [CACHE_LINE_SIZE-1:0]   byte_enable; 
    wire [`CACHE_LINE_WIDTH-1:0] write_data;   
    wire                        write_enable;

    wire [`LINE_SELECT_BITS-1:0] raddr = raddr_in[`LINE_SELECT_BITS-1:0];
    wire [`LINE_SELECT_BITS-1:0] waddr = waddr_in[`LINE_SELECT_BITS-1:0];

    `UNUSED_VAR (readen_in)

    VX_data_store #(
        .CACHE_SIZE     (CACHE_SIZE),
        .CACHE_LINE_SIZE (CACHE_LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .WORD_SIZE      (WORD_SIZE),
        .WRITE_ENABLE   (WRITE_ENABLE)
    ) data_store (
        .clk         (clk),
        .reset       (reset),

        .read_addr   (raddr),
        .read_data   (read_data),
        .read_dirtyb (read_dirtyb),

        .write_enable(write_enable),
        .write_fill  (wfill_in),        
        .write_addr  (waddr),       
        .byte_enable (byte_enable),     
        .write_data  (write_data)
    );

    wire [`WORDS_PER_LINE-1:0][WORD_SIZE-1:0] wbyteen_qual; 
    wire [`CACHE_LINE_WIDTH-1:0] writeword_qual;   

    if (`WORD_SELECT_BITS != 0) begin
        for (genvar i = 0; i < `WORDS_PER_LINE; i++) begin
            assign wbyteen_qual[i] = (wwsel_in == `WORD_SELECT_BITS'(i)) ? wbyteen_in : {WORD_SIZE{1'b0}};
            assign writeword_qual[i * `WORD_WIDTH +: `WORD_WIDTH] = writeword_in;
        end
    end else begin
        `UNUSED_VAR (wwsel_in)
        assign wbyteen_qual   = wbyteen_in;
        assign writeword_qual = writeword_in;
    end    
    
    assign byte_enable = wfill_in ? {CACHE_LINE_SIZE{1'b1}} : wbyteen_qual;
    assign write_data  = wfill_in ? writedata_in : writeword_qual;

    assign write_enable = writeen_in && !stall;

    wire rw_hazard = DRAM_ENABLE && (raddr == waddr) && writeen_in;
    for (genvar i = 0; i < CACHE_LINE_SIZE; i++) begin
        assign dirtyb_qual[i]            = rw_hazard ? byte_enable[i] : read_dirtyb[i];
        assign readdata_qual[i * 8 +: 8] = (rw_hazard && byte_enable[i]) ? write_data[i * 8 +: 8] : read_data[i * 8 +: 8];
    end

    if (WRITE_THROUGH) begin
        `UNUSED_VAR (dirtyb_qual)
        assign dirtyb_out   = wbyteen_qual;
        assign readdata_out = writeword_qual;
    end else begin
        assign dirtyb_out   = dirtyb_qual;
        assign readdata_out = readdata_qual;
    end

     if (`WORD_SELECT_BITS != 0) begin
        wire [`WORD_WIDTH-1:0] readword = readdata_qual[rwsel_in * `WORD_WIDTH +: `WORD_WIDTH];
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_out[i * 8 +: 8] = readword[i * 8 +: 8] & {8{rbyteen_in[i]}};
        end
    end else begin
        `UNUSED_VAR (rwsel_in)
        for (genvar i = 0; i < WORD_SIZE; i++) begin
            assign readword_out[i * 8 +: 8] = readdata_qual[i * 8 +: 8] & {8{rbyteen_in[i]}};
        end
    end

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin            
        if (!stall) begin
            if (writeen_in) begin
                if (wfill_in) begin
                    $display("%t: cache%0d:%0d data-fill: addr=%0h, dirty=%b, blk_addr=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(waddr_in, BANK_ID), dirtyb_out, waddr, write_data);
                end else begin
                    $display("%t: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, dirty=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(waddr_in, BANK_ID), rdebug_wid, rdebug_pc, byte_enable, dirtyb_out, waddr, wwsel_in, writeword_in);
                end
            end 
            if (readen_in) begin
                $display("%t: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, dirty=%b, blk_addr=%0d, wsel=%0d, data=%0h", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(raddr_in, BANK_ID), rdebug_wid, rdebug_pc, dirtyb_out, raddr, rwsel_in, read_data);
            end            
        end
    end    
`endif

endmodule