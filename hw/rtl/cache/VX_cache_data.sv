`include "VX_cache_define.vh"

module VX_cache_data #(
    parameter `STRING INSTANCE_ID= "",
    parameter BANK_ID           = 0,
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE         = 1, 
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
    parameter UUID_WIDTH        = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(UUID_WIDTH)-1:0]     req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

    input wire                          read,
    input wire                          fill, 
    input wire                          write,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] addr,
    input wire [NUM_PORTS-1:0][`UP(`CS_WORD_SEL_BITS)-1:0] wsel,
    input wire [NUM_PORTS-1:0]          pmask,
    input wire [NUM_PORTS-1:0][WORD_SIZE-1:0] byteen,
    input wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] fill_data,
    input wire [NUM_PORTS-1:0][`CS_WORD_WIDTH-1:0] write_data,
    input wire [NUM_WAYS-1:0]           way_sel,

    output wire [NUM_PORTS-1:0][`CS_WORD_WIDTH-1:0] read_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (addr)
    `UNUSED_VAR (read)

    localparam BYTEENW = (WRITE_ENABLE != 0 || (NUM_WAYS > 1)) ? (LINE_SIZE * NUM_WAYS) : 1;

    wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] wdata;
    wire [BYTEENW-1:0] wren;

    if (WRITE_ENABLE != 0 || (NUM_WAYS > 1)) begin
        reg [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] wdata_r;
        reg [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] wren_r;

        if (NUM_PORTS > 1) begin
            always @(*) begin
                wdata_r = 'x;
                wren_r  = '0;
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
                wdata_r = {`CS_WORDS_PER_LINE{write_data}};
                wren_r  = '0;
                wren_r[wsel] = byteen;
            end
        end
        
        // order the data layout to perform ways multiplexing last 
        // this allows performing onehot encoding of the way index in parallel with BRAM read.
        wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][WORD_SIZE-1:0] wren_w;
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin
            assign wdata[i] = fill ? {NUM_WAYS{fill_data[i]}} : {NUM_WAYS{wdata_r[i]}};            
            for (genvar j = 0; j < NUM_WAYS; ++j) begin
                assign wren_w[i][j] = (fill ? {WORD_SIZE{1'b1}} : wren_r[i])
                                    & {WORD_SIZE{((NUM_WAYS == 1) || way_sel[j])}};
            end
        end
        assign wren = wren_w;
    end else begin
        `UNUSED_VAR (write)
        `UNUSED_VAR (byteen)
        `UNUSED_VAR (pmask)
        `UNUSED_VAR (write_data)
        assign wdata = fill_data;
        assign wren  = fill;
    end
    
    wire [`CLOG2(NUM_WAYS)-1:0] way_idx;

    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) way_enc (
        .data_in  (way_sel),
        .data_out (way_idx),
        `UNUSED_PIN (valid_out)
    );

    wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] rdata;

    wire [`CS_LINE_SEL_BITS-1:0] line_addr = addr[`CS_LINE_SEL_BITS-1:0];
    
    VX_sp_ram #(
        .DATAW (`CS_LINE_WIDTH * NUM_WAYS),
        .SIZE  (`CS_LINES_PER_BANK),
        .WRENW (BYTEENW),
        .NO_RWCHECK (1)
    ) data_store (
        .clk   (clk),
        .write (write || fill),
        .wren  (wren),
        .addr  (line_addr),
        .wdata (wdata),
        .rdata (rdata) 
    );

    wire [NUM_PORTS-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] per_way_rdata;

    if (`CS_WORDS_PER_LINE > 1) begin
        for (genvar i = 0; i < NUM_PORTS; ++i) begin
            assign per_way_rdata[i] = rdata[wsel[i]];
        end
    end else begin
        `UNUSED_VAR (wsel)
        assign per_way_rdata = rdata;
    end    

    for (genvar i = 0; i < NUM_PORTS; ++i) begin
        assign read_data[i] = per_way_rdata[i][way_idx];
    end

    `UNUSED_VAR (stall)

`ifdef DBG_TRACE_CACHE_DATA
    always @(posedge clk) begin 
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s:%0d data-fill: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, fill_data));
        end
        if (read && ~stall) begin
            `TRACE(3, ("%d: %s:%0d data-read: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, read_data, req_uuid));
        end 
        if (write && ~stall) begin
            `TRACE(3, ("%d: %s:%0d data-write: addr=0x%0h, way=%b, blk_addr=%0d, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_BYTE_ADDR(addr, BANK_ID), way_sel, line_addr, byteen, write_data, req_uuid));
        end      
    end    
`endif

endmodule
