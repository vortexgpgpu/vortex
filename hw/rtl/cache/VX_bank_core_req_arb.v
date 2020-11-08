`include "VX_cache_config.vh"

module VX_bank_core_req_arb #(
    // Size of a word in bytes
    parameter WORD_SIZE                     = 1,     
    // Number of Word requests per cycle
    parameter NUM_REQUESTS                  = 1, 
    // Core Request Queue Size
    parameter CREQ_SIZE                     = 1, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    input  wire clk,
    input  wire reset,

    // Enqueue Data
    input wire                                                  push,
    input wire [NUM_REQUESTS-1:0]                               valids_in,
    input wire [`CORE_REQ_TAG_COUNT-1:0]                        rw_in,  
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]                byteen_in,    
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]              writedata_in,
    input wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]         addr_in,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]    tag_in,    

    // Dequeue Data
    input  wire                             pop,
    output wire [`REQS_BITS-1:0]            tid_out,    
    output wire                             rw_out,  
    output wire [WORD_SIZE-1:0]             byteen_out,    
    output wire [`WORD_ADDR_WIDTH-1:0]      addr_out,
    output wire [`WORD_WIDTH-1:0]           writedata_out,
    output wire [CORE_TAG_WIDTH-1:0]        tag_out,    

    // State Data
    output wire                             empty,
    output wire                             full
);

    wire [NUM_REQUESTS-1:0]                             out_per_valids;    
    wire [`CORE_REQ_TAG_COUNT-1:0]                      out_per_rw;  
    wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              out_per_byteen;
    wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       out_per_addr;    
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            out_per_writedata;    
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  out_per_tag;

    reg [NUM_REQUESTS-1:0]                              use_per_valids;
    reg [`CORE_REQ_TAG_COUNT-1:0]                       use_per_rw;  
    reg [NUM_REQUESTS-1:0][WORD_SIZE-1:0]               use_per_byteen;
    reg [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]        use_per_addr;
    reg [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]             use_per_writedata;        
    reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]   use_per_tag;

    wire [NUM_REQUESTS-1:0]                             qual_valids;  
    wire [`CORE_REQ_TAG_COUNT-1:0]                      qual_rw;  
    wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              qual_byteen;
    wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       qual_addr;
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            qual_writedata;  
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  qual_tag;

    wire o_empty;

    wire use_empty = !(| use_per_valids);
    wire out_empty = !(| out_per_valids) || o_empty;

    wire push_qual = push && !full;
    wire pop_qual  = !out_empty && use_empty;

    VX_generic_queue #(
        .DATAW($bits(valids_in) + $bits(addr_in) + $bits(writedata_in) + $bits(tag_in) + $bits(rw_in) + $bits(byteen_in)), 
        .SIZE(CREQ_SIZE)
    ) reqq_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (push_qual),
        .data_in  ({valids_in,    rw_in,    byteen_in,    addr_in,    writedata_in,    tag_in}),
        .pop      (pop_qual),
        .data_out ({out_per_valids, out_per_rw, out_per_byteen, out_per_addr, out_per_writedata, out_per_tag}),
        .empty    (o_empty),
        .full     (full),
        `UNUSED_PIN (size)
    );

    wire[NUM_REQUESTS-1:0] real_out_per_valids = out_per_valids & {NUM_REQUESTS{~out_empty}};

    assign qual_valids     = use_per_valids; 
    assign qual_addr       = use_per_addr;
    assign qual_writedata  = use_per_writedata;
    assign qual_tag        = use_per_tag;
    assign qual_rw         = use_per_rw;
    assign qual_byteen     = use_per_byteen;

    wire sel_valid;
    wire[`REQS_BITS-1:0] sel_idx;
    
    VX_fixed_arbiter #(
        .N(NUM_REQUESTS)
    ) sel_bank (
        .clk         (clk),
        .reset       (reset),
        .requests    (qual_valids),
        .grant_valid (sel_valid),
        .grant_index (sel_idx),
        `UNUSED_PIN  (grant_onehot)
    );

    assign empty          = !sel_valid;
    assign tid_out        = sel_idx;    
    assign byteen_out     = qual_byteen[sel_idx];
    assign addr_out       = qual_addr[sel_idx];
    assign writedata_out  = qual_writedata[sel_idx];
    
    if (CORE_TAG_ID_BITS != 0) begin
        assign tag_out = qual_tag;
        assign rw_out  = qual_rw;
    end else begin
        assign tag_out = qual_tag[sel_idx];
        assign rw_out  = qual_rw[sel_idx];
    end

    always @(posedge clk) begin
        if (reset) begin
            use_per_valids <= 0;
        end else begin
            if (pop_qual) begin
                use_per_valids    <= real_out_per_valids;
                use_per_rw        <= out_per_rw;  
                use_per_byteen    <= out_per_byteen;
                use_per_addr      <= out_per_addr;
                use_per_writedata <= out_per_writedata;
                use_per_tag       <= out_per_tag;                
            end else if (pop) begin
                use_per_valids[sel_idx] <= 0;
            end
        end
    end

endmodule