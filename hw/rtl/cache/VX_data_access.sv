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

    //Swetha: added ways 
    parameter WAYS                          = 8, //dummy value - change this to 1 later

    parameter WORD_SELECT_BITS  = `UP(`WORD_SELECT_BITS)
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire[`DBG_CACHE_REQ_IDW-1:0]  req_id,
`IGNORE_UNUSED_END

    input wire                          stall,

    input wire                          read,
    input wire                          fill, 
    input wire                          write,
    input wire[`LINE_ADDR_WIDTH-1:0]    addr,
    input wire [NUM_PORTS-1:0][WORD_SELECT_BITS-1:0] wsel,
    input wire [NUM_PORTS-1:0]          pmask,
    input wire [NUM_PORTS-1:0][WORD_SIZE-1:0] byteen,
    input wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] fill_data,
    input wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] write_data,
    //Swetha: added for associativity
    input wire[WAYS-1:0]              tag_match_way,
    //input wire[$clog2(WAYS)-1:0]       tag_match_way_num,

    output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] read_data
);
    `UNUSED_PARAM (CACHE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (addr)
    `UNUSED_VAR (read)

    localparam BYTEENW = WRITE_ENABLE ? CACHE_LINE_SIZE : 1;
    //localparam n_BYTEENW = 1;

    wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] rdata;
    wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] wdata;
    wire [BYTEENW-1:0] wren;
    //wire [n_BYTEENW-1:0] n_wren;

    wire [`LINE_SELECT_BITS-1:0] line_addr = addr[`LINE_SELECT_BITS-1:0];

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
        //assign n_wren = fill;
    end

    //Swetha: adding associativity to data access
    /* CHANGES START HERE */

    //Swetha: Local variable to capture data from all ways before assigning to output wire
    wire [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] read_data_local [WAYS-1:0];
    localparam [`WAY_SEL_WIDTH-1:0] which_way = 0; //dummy assignment  

    generate
        genvar m;
        for (m = 0; m < WAYS; m = m+1) begin
            assign which_way = tag_match_way[m] ? m : 'z; 
        end
    endgenerate

    generate
        genvar j;
        for (j = 0; j < which_way; j = j+1) begin
            //assign which_way = tag_match_way[j] ? j : 'z; 
            //assign wren = (tag_match_way[j] == 1'b0) ? {BYTEENW{1'b0}} : wren;
            VX_sp_ram #(
                .DATAW      (`CACHE_LINE_WIDTH),
                .SIZE       (`LINES_PER_BANK),
                .BYTEENW    (1),
                .NO_RWCHECK (1)
            ) data_store (
                .clk   (clk),
                .addr  (line_addr),
                //Swetha: wren is disabled so that data is not written into spram
                .wren  (0),  //& {BYTEENW{tag_match_way[j]}}
                .wdata (wdata),
                //Swetha: modified this for associativity 
                .rdata (read_data_local[j]) 
            );
        end
    endgenerate

     VX_sp_ram #(
                .DATAW      (`CACHE_LINE_WIDTH),
                .SIZE       (`LINES_PER_BANK),
                .BYTEENW    (BYTEENW),
                .NO_RWCHECK (1)
            ) data_store (
                .clk   (clk),
                .addr  (line_addr),
                //Swetha: wren is disabled so that data is not written into spram
                .wren  (wren),  //& {BYTEENW{tag_match_way[j]}}
                .wdata (wdata),
                //Swetha: modified this for associativity 
                .rdata (read_data_local[which_way]) 
            );

localparam temp = which_way + 1; 
    generate
        genvar k;
        for (k = temp; k < WAYS; k = k+1) begin
            //assign which_way = tag_match_way[j] ? j : 'z; 
            //assign wren = (tag_match_way[j] == 1'b0) ? {BYTEENW{1'b0}} : wren;
            VX_sp_ram #(
                .DATAW      (`CACHE_LINE_WIDTH),
                .SIZE       (`LINES_PER_BANK),
                .BYTEENW    (1),
                .NO_RWCHECK (1)
            ) data_store (
                .clk   (clk),
                .addr  (line_addr),
                //Swetha: wren is disabled so that data is not written into spram
                .wren  (0),  //& {BYTEENW{tag_match_way[j]}}
                .wdata (wdata),
                //Swetha: modified this for associativity 
                .rdata (read_data_local[k]) 
            );
        end
    endgenerate
   

    //Approach 1: 
    //assign rdata = read_data_local[which_way];

    //Approach 2:  

    //reg [`WORDS_PER_LINE-1:0][`WORD_WIDTH-1:0] read_data_local[WAYS-1:0];
    //localparam [`WAY_SEL_WIDTH-1:0] which_way = 0;
    //output wire [NUM_PORTS-1:0][`WORD_WIDTH-1:0] read_data

    if (WAYS > 1) begin
        assign rdata = read_data_local[which_way];
    end else begin
        //`UNUSED_VAR (sel_in)
        assign rdata = read_data_local;
    end

    // wire 

    // VX_mux #(
    //     .DATAW    (`WORDS_PER_LINE + `WORD_WIDTH),
    //     .N        (WAYS),  
    // ) find_read_data (
    //     read_data_local,
    //     which_way,
    //     rdata
    // );

    if (`WORDS_PER_LINE > 1) begin
        for (genvar i = 0; i < NUM_PORTS; ++i) begin
            assign read_data[i] = rdata[wsel[i]];
        end
    end else begin
        assign read_data = rdata;
    end

    //Approach 3: 
    // if (`WORDS_PER_LINE > 1) begin
    //     for (genvar i = 0; i < NUM_PORTS; ++i) begin
    //             assign read_data[i] = rdata[which_way][wsel[i]]; 
    //     end
    // end else begin
    //     assign read_data = rdata[which_way]; 
    // end
    /* CHANGES END HERE */
    
    
    // VX_sp_ram #(
    //     .DATAW      (`CACHE_LINE_WIDTH),
    //     .SIZE       (`LINES_PER_BANK),
    //     .BYTEENW    (BYTEENW),
    //     .NO_RWCHECK (1)
    // ) data_store (
    //     .clk   (clk),
    //     .addr  (line_addr),
    //     .wren  (wren),
    //     .wdata (wdata),
    //     .rdata (rdata)
    // );

    // if (`WORDS_PER_LINE > 1) begin
    //     for (genvar i = 0; i < NUM_PORTS; ++i) begin
    //         assign read_data[i] = rdata[wsel[i]];
    //     end
    // end else begin
    //     assign read_data = rdata;
    // end

     `UNUSED_VAR (stall)

`ifdef DBG_TRACE_CACHE_DATA
    always @(posedge clk) begin 
        if (fill && ~stall) begin
            dpi_trace("%d: cache%0d:%0d data-fill: addr=0x%0h, blk_addr=%0d, data=0x%0h\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, fill_data);
        end
        if (read && ~stall) begin
            dpi_trace("%d: cache%0d:%0d data-read: addr=0x%0h, blk_addr=%0d, data=0x%0h (#%0d)\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), line_addr, read_data, req_id);
        end 
        if (write && ~stall) begin
            dpi_trace("%d: cache%0d:%0d data-write: addr=0x%0h, byteen=%b, blk_addr=%0d, data=0x%0h (#%0d)\n", $time, CACHE_ID, BANK_ID, `LINE_TO_BYTE_ADDR(addr, BANK_ID), byteen, line_addr, write_data, req_id);
        end      
    end    
`endif

endmodule