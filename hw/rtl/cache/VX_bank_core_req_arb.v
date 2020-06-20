`include "VX_cache_config.vh"

module VX_bank_core_req_arb #(
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0,     
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 0, 
    // Core Request Queue Size
    parameter CREQ_SIZE                     = 0, 
    // core request tag size
    parameter CORE_TAG_WIDTH                = 0,
    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    input  wire clk,
    input  wire reset,

    // Enqueue Data
    input wire                                                  reqq_push,
    input wire [NUM_REQUESTS-1:0]                               bank_valids,
    input wire [NUM_REQUESTS-1:0]                               bank_rw,  
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]                bank_byteen,    
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]              bank_writedata,
    input wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]         bank_addr,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]    bank_tag,    

    // Dequeue Data
    input  wire                             reqq_pop,
    output wire                             reqq_req_st0,
    output wire [`REQS_BITS-1:0]            reqq_req_tid_st0,    
    output wire                             reqq_req_rw_st0,  
    output wire [WORD_SIZE-1:0]             reqq_req_byteen_st0,    
    output wire [`WORD_ADDR_WIDTH-1:0]      reqq_req_addr_st0,
    output wire [`WORD_WIDTH-1:0]           reqq_req_writedata_st0,
    output wire [CORE_TAG_WIDTH-1:0]        reqq_req_tag_st0,    

    // State Data
    output wire                             reqq_empty,
    output wire                             reqq_full
);

    wire [NUM_REQUESTS-1:0]                             out_per_valids;    
    wire [NUM_REQUESTS-1:0]                             out_per_rw;  
    wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              out_per_byteen;
    wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       out_per_addr;    
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            out_per_writedata;    
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  out_per_tag;

    reg [NUM_REQUESTS-1:0]                              use_per_valids;
    reg [NUM_REQUESTS-1:0]                              use_per_rw;  
    reg [NUM_REQUESTS-1:0][WORD_SIZE-1:0]               use_per_byteen;
    reg [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]        use_per_addr;
    reg [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]             use_per_writedata;        
    reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]   use_per_tag;

    wire [NUM_REQUESTS-1:0]                             qual_valids;  
    wire [NUM_REQUESTS-1:0]                             qual_rw;  
    wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              qual_byteen;
    wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       qual_addr;
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            qual_writedata;  
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  qual_tag;

    wire o_empty;

    wire use_empty = !(| use_per_valids);
    wire out_empty = !(| out_per_valids) || o_empty;

    wire push_qual = reqq_push && !reqq_full;
    wire pop_qual  = !out_empty && use_empty;

    VX_generic_queue #(
        .DATAW($bits(bank_valids) + $bits(bank_addr) + $bits(bank_writedata) + $bits(bank_tag) + $bits(bank_rw) + $bits(bank_byteen)), 
        .SIZE(CREQ_SIZE)
    ) reqq_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (push_qual),
        .data_in  ({bank_valids,    bank_rw,    bank_byteen,    bank_addr,    bank_writedata,    bank_tag}),
        .pop      (pop_qual),
        .data_out ({out_per_valids, out_per_rw, out_per_byteen, out_per_addr, out_per_writedata, out_per_tag}),
        .empty    (o_empty),
        .full     (reqq_full),
        `UNUSED_PIN (size)
    );

    wire[NUM_REQUESTS-1:0] real_out_per_valids = out_per_valids & {NUM_REQUESTS{~out_empty}};

    assign qual_valids     = use_per_valids; 
    assign qual_addr       = use_per_addr;
    assign qual_writedata  = use_per_writedata;
    assign qual_tag        = use_per_tag;
    assign qual_rw         = use_per_rw;
    assign qual_byteen     = use_per_byteen;

    wire[`REQS_BITS-1:0] qual_request_index;
    wire                 qual_has_request;

    VX_fixed_arbiter #(
        .N(NUM_REQUESTS)
    ) sel_bank (
        .clk         (clk),
        .reset       (reset),
        .requests    (qual_valids),
        .grant_index (qual_request_index),
        .grant_valid (qual_has_request),
        `UNUSED_PIN  (grant_onehot)
    );

    assign reqq_empty              = !qual_has_request;
    assign reqq_req_st0            = qual_has_request;
    assign reqq_req_tid_st0        = qual_request_index;
    assign reqq_req_rw_st0         = qual_rw[qual_request_index];
    assign reqq_req_byteen_st0     = qual_byteen[qual_request_index];
    assign reqq_req_addr_st0       = qual_addr[qual_request_index];
    assign reqq_req_writedata_st0  = qual_writedata[qual_request_index];
    
    if (CORE_TAG_ID_BITS != 0) begin
        assign reqq_req_tag_st0 = qual_tag;
    end else begin
        assign reqq_req_tag_st0  = qual_tag[qual_request_index];
    end    

`DEBUG_BLOCK(
    reg [NUM_REQUESTS-1:0] updated_valids;
    always @(*) begin
        updated_valids = qual_valids;
        if (qual_has_request) begin
            updated_valids[qual_request_index] = 0;
        end
    end
)

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
            end else if (reqq_pop) begin
                use_per_valids[qual_request_index] <= 0;
            end
        end
    end

endmodule