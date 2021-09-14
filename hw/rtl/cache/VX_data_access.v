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
    // Number of ports per banks
    parameter NUM_PORTS         = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 1,
    // Enable cache writeable
    parameter WRITE_ENABLE      = 1,

    localparam WORD_SELECT_BITS = `UP(`WORD_SELECT_BITS)
) (
    input wire                          clk,
    input wire                          reset,

`ifdef DBG_CACHE_REQ_INFO
`IGNORE_UNUSED_BEGIN
    input wire[31:0]                    debug_pc,
    input wire[`NW_BITS-1:0]            debug_wid,
`IGNORE_UNUSED_END
`endif

    input wire                          stall,

`IGNORE_UNUSED_BEGIN
    input wire[`LINE_ADDR_WIDTH-1:0]    addr,
`IGNORE_UNUSED_END

    input wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] wsel,
    input wire [NUM_PORTS-1:0]          pmask,

    // reading
    input wire                          readen,
    output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] read_data,

    // writing
    input wire                          writeen,
    input wire                          is_fill,
    input wire [NUM_PORTS-1:0][WORD_SIZE-1:0] byteen,
    input wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] write_data,
    input wire [`CACHE_LINE_WIDTH-1:0]  fill_data
);

    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (readen)

    localparam BYTEENW = WRITE_ENABLE ? CACHE_LINE_SIZE : 1;

    wire [`CACHE_LINE_WIDTH-1:0] rdata;
    wire [`CACHE_LINE_WIDTH-1:0] wdata;
    wire [BYTEENW-1:0] wren;

    wire [`LINE_SELECT_BITS-1:0]  line_addr = addr[`LINE_SELECT_BITS-1:0];

    if (WRITE_ENABLE) begin
        wire [`CACHE_LINE_WIDTH-1:0] line_wdata;
        wire [CACHE_LINE_SIZE-1:0] line_byteen;
        if (`WORDS_PER_LINE > 1) begin
            reg [`CACHE_LINE_WIDTH-1:0] line_wdata_r;
            reg [CACHE_LINE_SIZE-1:0] line_byteen_r;
            if (NUM_PORTS > 1) begin
                always @(*) begin
                    line_wdata_r  = 'x;
                    line_byteen_r = 0;
                    for (integer i = 0; i < NUM_PORTS; ++i) begin
                        if (pmask[i]) begin
                            line_wdata_r[wsel[i] * `WORD_WIDTH +: `WORD_WIDTH] = write_data[i];
                            line_byteen_r[wsel[i] * WORD_SIZE +: WORD_SIZE] = byteen[i];
                        end
                    end
                end
            end else begin
                `UNUSED_VAR (pmask)
                always @(*) begin                
                    line_wdata_r = {`WORDS_PER_LINE{write_data}};
                    line_byteen_r = 0;
                    line_byteen_r[wsel * WORD_SIZE +: WORD_SIZE] = byteen;
                end
            end
            assign line_wdata  = line_wdata_r;
            assign line_byteen = line_byteen_r;
        end else begin
            `UNUSED_VAR (wsel)
            `UNUSED_VAR (pmask)
            assign line_wdata  = write_data;
            assign line_byteen = byteen;
        end
        assign wren  = is_fill ? {BYTEENW{writeen}} : ({BYTEENW{writeen}} & line_byteen);
        assign wdata = is_fill ? fill_data : line_wdata;
    end else begin        
        `UNUSED_VAR (is_fill)
        `UNUSED_VAR (byteen)
        `UNUSED_VAR (pmask)
        `UNUSED_VAR (write_data)
        assign wren  = writeen;
        assign wdata = fill_data;
    end

    VX_sp_ram #(
        .DATAW      (`CACHE_LINE_WIDTH),
        .SIZE       (`LINES_PER_BANK),
        .BYTEENW    (BYTEENW),
        .NO_RWCHECK (1)
    ) data_store (
        .clk   (clk),
        .addr  (line_addr),
        .wren  (wren),
        .wdata (wdata),
        .rdata (rdata)
    );

    if (`WORDS_PER_LINE > 1) begin
        for (genvar i = 0; i < NUM_PORTS; ++i) begin
            assign read_data[i] = rdata[wsel[i] * `WORD_WIDTH +: `WORD_WIDTH];
        end
    end else begin
        assign read_data = rdata;
    end

    `UNUSED_VAR (stall)

`ifdef DBG_PRINT_CACHE_DATA
    always @(posedge clk) begin 
        if (writeen && ~stall) begin
            if (is_fill) begin
                dpi_trace("%d: cache%0d:%0d data-fill: addr=%0h, blk_addr=%0d, data=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, fill_data);
            end else begin
                dpi_trace("%d: cache%0d:%0d data-write: addr=%0h, wid=%0d, PC=%0h, byteen=%b, blk_addr=%0d, data=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, wren, line_addr, write_data);
            end
        end 
        if (readen && ~stall) begin
            dpi_trace("%d: cache%0d:%0d data-read: addr=%0h, wid=%0d, PC=%0h, blk_addr=%0d, data=%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), debug_wid, debug_pc, line_addr, read_data);
        end            
    end    
`endif

endmodule