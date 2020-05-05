`include "VX_cache_config.vh"

module VX_cache_req_queue #(
    // Size of cache in bytes
    parameter CACHE_SIZE                    = 1024, 
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 16, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 8, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 4, 
    // Number of Word requests per cycle {1, 2, 4, 8, ...}
    parameter NUM_REQUESTS                  = 2, 
    // Number of cycles to complete stage 1 (read from memory)
    parameter STAGE_1_CYCLES                = 2, 

    // Queues feeding into banks Knobs {1, 2, 4, 8, ...}
    // Core Request Queue Size
    parameter REQQ_SIZE                     = 8, 
    // Miss Reserv Queue Knob
    parameter MRVQ_SIZE                     = 8, 
    // Dram Fill Rsp Queue Size
    parameter DFPQ_SIZE                     = 2, 
    // Snoop Req Queue
    parameter SNRQ_SIZE                     = 8, 

    // Queues for writebacks Knobs {1, 2, 4, 8, ...}
    // Core Writeback Queue Size
    parameter CWBQ_SIZE                     = 8, 
    // Dram Writeback Queue Size
    parameter DWBQ_SIZE                     = 4, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 8, 
    // Lower Level Cache Hit Queue Size
    parameter LLVQ_SIZE                     = 16, 

    // Fill Invalidator Size {Fill invalidator must be active}
    parameter FILL_INVALIDAOR_SIZE         = 16,

    // core request tag size
    parameter CORE_TAG_WIDTH                = 1,

    // size of tag id in core request tag
    parameter CORE_TAG_ID_BITS              = 0
) (
    input  wire clk,
    input  wire reset,

    // Enqueue Data
    input wire                                        reqq_push,
    input wire [NUM_REQUESTS-1:0]                     bank_valids,
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  bank_mem_read,  
    input wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  bank_mem_write,    
    input wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]    bank_writedata,
    input wire [NUM_REQUESTS-1:0][31:0]               bank_addr,
    input wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] bank_tag,    

    // Dequeue Data
    input  wire                             reqq_pop,
    output wire                             reqq_req_st0,
    output wire [`LOG2UP(NUM_REQUESTS)-1:0] reqq_req_tid_st0,    
    output wire [`BYTE_EN_BITS-1:0]         reqq_req_mem_read_st0,  
    output wire [`BYTE_EN_BITS-1:0]         reqq_req_mem_write_st0,
    output wire [`WORD_WIDTH-1:0]           reqq_req_writedata_st0,
    output wire [31:0]                      reqq_req_addr_st0,
    output wire [CORE_TAG_WIDTH-1:0]        reqq_req_tag_st0,    

    // State Data
    output wire                             reqq_empty,
    output wire                             reqq_full
);

    wire [NUM_REQUESTS-1:0]                     out_per_valids;
    wire [NUM_REQUESTS-1:0][31:0]               out_per_addr;
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]    out_per_writedata;    
    wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  out_per_mem_read;  
    wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  out_per_mem_write;
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] out_per_tag;

    reg [NUM_REQUESTS-1:0]                     use_per_valids;
    reg [NUM_REQUESTS-1:0][31:0]               use_per_addr;
    reg [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]    use_per_writedata;    
    reg [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  use_per_mem_read;  
    reg [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  use_per_mem_write;
    reg [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] use_per_tag;

    wire [NUM_REQUESTS-1:0]                     qual_valids;
    wire [NUM_REQUESTS-1:0][31:0]               qual_addr;
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]    qual_writedata;    
    wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  qual_mem_read;  
    wire [NUM_REQUESTS-1:0][`BYTE_EN_BITS-1:0]  qual_mem_write;
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] qual_tag;

`DEBUG_BEGIN
    reg [NUM_REQUESTS-1:0] updated_valids;
`DEBUG_END

    wire o_empty;

    wire use_empty = !(| use_per_valids);
    wire out_empty = !(| out_per_valids) || o_empty;

    wire push_qual = reqq_push && !reqq_full;
    wire pop_qual  = !out_empty && use_empty;

    VX_generic_queue #(
        .DATAW($bits(bank_valids) + $bits(bank_addr) + $bits(bank_writedata) + $bits(bank_tag) + $bits(bank_mem_read) + $bits(bank_mem_write)), 
        .SIZE(REQQ_SIZE)
    ) reqq_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (push_qual),
        .data_in  ({bank_valids,    bank_addr,    bank_writedata,    bank_tag,    bank_mem_read,    bank_mem_write}),
        .pop      (pop_qual),
        .data_out ({out_per_valids, out_per_addr, out_per_writedata, out_per_tag, out_per_mem_read, out_per_mem_write}),
        .empty    (o_empty),
        .full     (reqq_full)
    );

    wire[NUM_REQUESTS-1:0] real_out_per_valids = out_per_valids & {NUM_REQUESTS{~out_empty}};

    assign qual_valids     = use_per_valids; 
    assign qual_addr       = use_per_addr;
    assign qual_writedata  = use_per_writedata;
    assign qual_tag        = use_per_tag;
    assign qual_mem_read   = use_per_mem_read;
    assign qual_mem_write  = use_per_mem_write;

    wire[`LOG2UP(NUM_REQUESTS)-1:0] qual_request_index;
    wire                            qual_has_request;

    VX_generic_priority_encoder #(
        .N(NUM_REQUESTS)
    ) sel_bank (
        .valids(qual_valids),
        .index (qual_request_index),
        .found (qual_has_request)
    );

    assign reqq_empty              = !qual_has_request;
    assign reqq_req_st0            = qual_has_request;
    assign reqq_req_tid_st0        = qual_request_index;
    assign reqq_req_addr_st0       = qual_addr[qual_request_index];
    assign reqq_req_writedata_st0  = qual_writedata[qual_request_index];
    
    if (CORE_TAG_ID_BITS != 0) begin
        assign reqq_req_tag_st0 = qual_tag;
    end else begin
        assign reqq_req_tag_st0  = qual_tag[qual_request_index];
    end

    assign reqq_req_mem_read_st0  = qual_mem_read [qual_request_index];
    assign reqq_req_mem_write_st0 = qual_mem_write[qual_request_index];

    always @(*) begin
        updated_valids = qual_valids;
        if (qual_has_request) begin
            updated_valids[qual_request_index] = 0;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            use_per_valids <= 0;
        end else begin
            if (pop_qual) begin
                use_per_valids    <= real_out_per_valids;
                use_per_addr      <= out_per_addr;
                use_per_writedata <= out_per_writedata;
                use_per_tag       <= out_per_tag;
                use_per_mem_read  <= out_per_mem_read;  
                use_per_mem_write <= out_per_mem_write;
            end else if (reqq_pop) begin
                use_per_valids[qual_request_index] <= 0;
            end
        end
    end

endmodule