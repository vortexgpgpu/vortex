`include "VX_cache_define.vh"

module VX_data_access #(
    parameter string INSTANCE_ID= "",
    parameter BANK_ID           = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter CACHE_LINE_SIZE   = 1, 
    // Number of banks
    parameter NUM_BANKS         = 1, 
    // Number of ports per banks
    parameter NUM_PORTS         = 1,
    // Number of associative ways
    parameter NUM_WAYS          = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 1,
    // Enable cache writeable
    parameter WRITE_ENABLE      = 1,
    // Request debug identifier
    parameter REQ_UUID_BITS     = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(REQ_UUID_BITS)-1:0]  req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

    input wire                          read,
    input wire                          fill, 
    input wire                          write,
    input wire [`LINE_ADDR_WIDTH-1:0]   addr,
    input wire [NUM_PORTS-1:0][`UP(`WORD_SEL_BITS)-1:0] wsel,
    input wire [NUM_PORTS-1:0]          pmask,
    input wire [NUM_PORTS-1:0][WORD_SIZE-1:0] byteen,
    input wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] fill_data,
    input wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] write_data,
    input wire [`WAY_SEL_BITS-1:0]      way_sel,

    output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] read_data
);
    `UNUSED_PARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (addr)
    `UNUSED_VAR (read)

    localparam BYTEENW = WRITE_ENABLE ? CACHE_LINE_SIZE : 1;

    wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] rdata;
    wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] wdata;
    wire [BYTEENW-1:0] wren;
    wire [`LINE_SEL_BITS-1:0] line_addr = addr[`LINE_SEL_BITS-1:0];

    if (WRITE_ENABLE) begin
        if (`WORDS_PER_LINE > 1) begin
            reg [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] wdata_r;
            reg [`WORDS_PER_LINE-1:0][WORD_SIZE-1:0] wren_r;
            if (NUM_PORTS > 1) begin
                always @(*) begin
                    wdata_r = 'x;
                    wren_r  = 0;
                    for (integer i = 0; i < NUM_PORTS; ++i) begin
                        if (pmask[i]) begin
                            wdata_r[wsel[i]] = write_data[i];
                            wren_r[wsel[i]] = byteen[i];
                        end
                    end
                end
            end else begin
                `UNUSED_VAR (pmask)
                always @(*) begin                
                    wdata_r = {`WORDS_PER_LINE{write_data}};
                    wren_r  = 0;
                    wren_r[wsel] = byteen;
                end
            end
            assign wdata = write ? wdata_r : fill_data;
            assign wren  = write ? wren_r : {BYTEENW{fill}};
        end else begin
            `UNUSED_VAR (wsel)
            `UNUSED_VAR (pmask)
            assign wdata = write ? write_data : fill_data;
            assign wren  = write ? byteen : {BYTEENW{fill}};
        end
    end else begin
        `UNUSED_VAR (write)
        `UNUSED_VAR (byteen)
        `UNUSED_VAR (pmask)
        `UNUSED_VAR (write_data)
        assign wdata = fill_data;
        assign wren  = fill;
    end

    VX_sp_ram #(
        .DATAW      (`CACHE_LINE_WIDTH),
        .SIZE       (`LINES_PER_BANK * NUM_WAYS),
        .BYTEENW    (BYTEENW),
        .NO_RWCHECK (1)
    ) data_store (
        .clk   (clk),
        .addr  ({way_sel, line_addr}),
        .wren  (wren),
        .wdata (wdata),
        .rdata (rdata) 
    );

    if (`WORDS_PER_LINE > 1) begin
        for (genvar i = 0; i < NUM_PORTS; ++i) begin
            assign read_data[i] = rdata[wsel[i]];
        end
    end else begin
        `UNUSED_VAR (wsel)
        assign read_data = rdata;
    end

     `UNUSED_VAR (stall)

`ifdef DBG_TRACE_CACHE_DATA
    always @(posedge clk) begin 
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s:%0d data-fill: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, fill_data));
        end
        if (read && ~stall) begin
            `TRACE(3, ("%d: %s:%0d data-read: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, read_data, req_uuid));
        end 
        if (write && ~stall) begin
            `TRACE(3, ("%d: %s:%0d data-write: addr=0x%0h, way=%b, blk_addr=%0d, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, byteen, write_data, req_uuid));
        end      
    end    
`endif

endmodule
