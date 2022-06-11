`include "VX_cache_define.vh"

module VX_cache_cluster #(
    parameter string INSTANCE_ID    = "",

    parameter NUM_UNITS             = 1,
    parameter NUM_INPUTS            = 1,
    parameter TAG_SEL_IDX           = 0,

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 16384, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64, 
    // Number of banks
    parameter NUM_BANKS             = 1,
    // Number of ports per banks
    parameter NUM_PORTS             = 1,
    // Number of associative ways
    parameter NUM_WAYS              = 1,
    // Size of a word in bytes
    parameter WORD_SIZE             = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE             = 0,
    // Core Response Queue Size
    parameter CRSQ_SIZE             = 2,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 8, 
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 0,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = UUID_WIDTH + 1,

    // enable bypass for non-cacheable addresses
    parameter NC_TAG_BIT            = 0,
    parameter NC_ENABLE             = 0
 ) (    
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if.master perf_cache_if [`UP(NUM_UNITS)],
`endif

    // Core request
    VX_cache_req_if.slave   core_req_if [NUM_INPUTS],

    // Core response
    VX_cache_rsp_if.master  core_rsp_if [NUM_INPUTS],

    // Memory request
    VX_mem_req_if.master    mem_req_if,
    
    // Memory response
    VX_mem_rsp_if.slave     mem_rsp_if
);

    `STATIC_ASSERT(NUM_INPUTS >= `UP(NUM_UNITS), ("invalid parameter"))

    localparam PASSTHRU  = (NUM_UNITS == 0);
    localparam WORD_ADDR = 32 - `CLOG2(WORD_SIZE);
    localparam ARB_TAG_WIDTH = TAG_WIDTH + `ARB_SEL_BITS(NUM_INPUTS, `UP(NUM_UNITS));    
    localparam MEM_TAG_WIDTH = PASSTHRU ? (NC_ENABLE ? `CACHE_NC_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH) : 
                                                       `CACHE_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH)) : 
                                          (NC_ENABLE ? `CACHE_NC_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH) :
                                                       `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS));

    VX_mem_req_if #(
        .DATA_WIDTH (`LINE_WIDTH),
        .TAG_WIDTH  (MEM_TAG_WIDTH)
    ) cache_mem_req_if[`UP(NUM_UNITS)]();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`LINE_WIDTH),
        .TAG_WIDTH  (MEM_TAG_WIDTH)
    ) cache_mem_rsp_if[`UP(NUM_UNITS)]();

    VX_cache_req_if #(
        .NUM_REQS  (NUM_REQS), 
        .WORD_SIZE (WORD_SIZE),
        .TAG_WIDTH (ARB_TAG_WIDTH)
    ) arb_core_req_if[`UP(NUM_UNITS)]();

    VX_cache_rsp_if #(
        .NUM_REQS  (NUM_REQS), 
        .WORD_SIZE (WORD_SIZE), 
        .TAG_WIDTH (ARB_TAG_WIDTH)
    ) arb_core_rsp_if[`UP(NUM_UNITS)]();

    VX_cache_arb #(
        .NUM_INPUTS   (NUM_INPUTS),
        .NUM_OUTPUTS  (`UP(NUM_UNITS)),
        .NUM_LANES    (NUM_REQS),
        .DATA_SIZE    (WORD_SIZE),
        .TAG_WIDTH    (TAG_WIDTH),
        .TAG_SEL_IDX  (TAG_SEL_IDX),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((NUM_INPUTS != `UP(NUM_UNITS)) ? 2 : 0),
        .BUFFERED_RSP ((NUM_INPUTS != `UP(NUM_UNITS)) ? 2 : 0)        
    ) cache_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (core_req_if),
        .rsp_in_if  (core_rsp_if),
        .req_out_if (arb_core_req_if),
        .rsp_out_if (arb_core_rsp_if)
    );

    for (genvar i = 0; i < `UP(NUM_UNITS); ++i) begin

        VX_mem_req_if #(
            .DATA_WIDTH (`WORD_WIDTH),
            .TAG_WIDTH  (ARB_TAG_WIDTH)
        ) arb_core_req_m_if[NUM_REQS]();

        VX_mem_rsp_if #(
            .DATA_WIDTH (`WORD_WIDTH), 
            .TAG_WIDTH (ARB_TAG_WIDTH)
        ) arb_core_rsp_m_if[NUM_REQS]();

        for (genvar j = 0; j < NUM_REQS; ++j) begin
            `CACHE_REQ_TO_MEM(arb_core_req_m_if, arb_core_req_if[i], j);
        end

        for (genvar j = 0; j < NUM_REQS; ++j) begin
            `CACHE_RSP_FROM_MEM(arb_core_rsp_if[i], arb_core_rsp_m_if, j);
        end

        `RESET_RELAY (cache_reset, reset);

        VX_cache_wrap #(
            .INSTANCE_ID  ($sformatf("%s%d", INSTANCE_ID, i)),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .NUM_PORTS    (NUM_PORTS),
            .WORD_SIZE    (WORD_SIZE),
            .NUM_REQS     (NUM_REQS),
            .CREQ_SIZE    (CREQ_SIZE),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MRSQ_SIZE    (MRSQ_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .WRITE_ENABLE (WRITE_ENABLE),
            .UUID_WIDTH   (UUID_WIDTH),
            .TAG_WIDTH    (ARB_TAG_WIDTH),
            .CORE_OUT_REG (3),
            .MEM_OUT_REG  (2),
            .NC_ENABLE    (NC_ENABLE),
            .PASSTHRU     (PASSTHRU)
        ) cache_wrap (
        `ifdef PERF_ENABLE
            .perf_cache_if (perf_cache_if[i]),
        `endif
            
            .clk         (clk),
            .reset       (cache_reset),

            .core_req_if (arb_core_req_m_if),
            .core_rsp_if (arb_core_rsp_m_if),
            .mem_req_if  (cache_mem_req_if[i]),
            .mem_rsp_if  (cache_mem_rsp_if[i])
        );
    end

    VX_mem_arb #(
        .NUM_REQS     (`UP(NUM_UNITS)),
        .DATA_WIDTH   (`LINE_WIDTH),
        .TAG_WIDTH    (MEM_TAG_WIDTH),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`UP(NUM_UNITS) > 1) ? 2 : 0),
        .BUFFERED_RSP ((`UP(NUM_UNITS) > 1) ? 2 : 0)        
    ) mem_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (cache_mem_req_if),        
        .rsp_in_if  (cache_mem_rsp_if),
        .req_out_if (mem_req_if),
        .rsp_out_if (mem_rsp_if)
    );

endmodule
