`include "VX_cache_define.vh"

module VX_cache_cluster #(
    parameter `STRING INSTANCE_ID    = "",

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
    parameter NC_ENABLE             = 0,

    // Core response output register
    parameter CORE_OUT_REG          = 0,

    // Memory request output register
    parameter MEM_OUT_REG           = 0
 ) (    
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_perf_cache_if.master perf_cache_if,
`endif

    VX_cache_bus_if.slave   core_bus_if [NUM_INPUTS],
    VX_mem_bus_if.master    mem_bus_if
);
    localparam NUM_CACHES = `UP(NUM_UNITS);
    localparam PASSTHRU   = (NUM_UNITS == 0);
    localparam ARB_TAG_WIDTH = TAG_WIDTH + `ARB_SEL_BITS(NUM_INPUTS, NUM_CACHES);    
    localparam MEM_TAG_WIDTH = PASSTHRU ? (NC_ENABLE ? `CACHE_NC_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH) : 
                                                       `CACHE_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH)) : 
                                          (NC_ENABLE ? `CACHE_NC_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH) :
                                                       `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS));

    `STATIC_ASSERT(NUM_INPUTS >= NUM_CACHES, ("invalid parameter"))

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_cache_unit_if[NUM_CACHES]();
    `PERF_CACHE_ADD (perf_cache_if, perf_cache_unit_if, NUM_CACHES);
`endif

    VX_mem_bus_if #(
        .DATA_WIDTH (`LINE_WIDTH),
        .TAG_WIDTH  (MEM_TAG_WIDTH)
    ) cache_mem_bus_if[NUM_CACHES]();

    VX_cache_bus_if #(
        .NUM_REQS  (NUM_REQS), 
        .WORD_SIZE (WORD_SIZE),
        .TAG_WIDTH (ARB_TAG_WIDTH)
    ) arb_core_bus_if[NUM_CACHES]();

    `RESET_RELAY (cache_arb_reset, reset);

    VX_cache_arb #(
        .NUM_INPUTS   (NUM_INPUTS),
        .NUM_OUTPUTS  (NUM_CACHES),
        .NUM_LANES    (NUM_REQS),
        .DATA_SIZE    (WORD_SIZE),
        .TAG_WIDTH    (TAG_WIDTH),
        .TAG_SEL_IDX  (TAG_SEL_IDX),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((NUM_INPUTS != NUM_CACHES) ? 2 : 0),
        .BUFFERED_RSP ((NUM_INPUTS != NUM_CACHES) ? ((CORE_OUT_REG + 1) / 2) : 0)
    ) cache_arb (
        .clk        (clk),
        .reset      (cache_arb_reset),
        .bus_in_if  (core_bus_if),
        .bus_out_if (arb_core_bus_if)
    );

    for (genvar i = 0; i < NUM_CACHES; ++i) begin

        VX_mem_bus_if #(
            .DATA_WIDTH (`WORD_WIDTH),
            .TAG_WIDTH  (ARB_TAG_WIDTH)
        ) arb_core_bus_m_if[NUM_REQS]();

        for (genvar j = 0; j < NUM_REQS; ++j) begin
            `CACHE_BUS_TO_MEM(arb_core_bus_m_if, arb_core_bus_if[i], j);
        end

        `RESET_RELAY (cache_reset, reset);

        VX_cache_wrap #(
            .INSTANCE_ID  ($sformatf("%s%0d", INSTANCE_ID, i)),
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
            .CORE_OUT_REG ((NUM_INPUTS != NUM_CACHES) ? 2 : CORE_OUT_REG),
            .MEM_OUT_REG  ((NUM_CACHES > 1) ? 2 : MEM_OUT_REG),
            .NC_ENABLE    (NC_ENABLE),
            .PASSTHRU     (PASSTHRU)
        ) cache_wrap (
        `ifdef PERF_ENABLE
            .perf_cache_if (perf_cache_unit_if[i]),
        `endif
            
            .clk         (clk),
            .reset       (cache_reset),

            .core_bus_if (arb_core_bus_m_if),
            .mem_bus_if  (cache_mem_bus_if[i])
        );
    end

    `RESET_RELAY (mem_arb_reset, reset);

    VX_mem_arb #(
        .NUM_REQS     (NUM_CACHES),
        .DATA_WIDTH   (`LINE_WIDTH),
        .TAG_WIDTH    (MEM_TAG_WIDTH),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .ARBITER      ("R"),
        .BUFFERED_REQ ((NUM_CACHES > 1) ? ((MEM_OUT_REG + 1) / 2) : 0),
        .BUFFERED_RSP ((NUM_CACHES > 1) ? 2 : 0)
    ) mem_arb (
        .clk        (clk),
        .reset      (mem_arb_reset),
        .bus_in_if  (cache_mem_bus_if),
        .bus_out_if (mem_bus_if)
    );

endmodule

///////////////////////////////////////////////////////////////////////////////

module VX_cache_cluster_top #(
    parameter `STRING INSTANCE_ID    = "",

    parameter NUM_UNITS             = 8,
    parameter NUM_INPUTS            = 16,
    parameter TAG_SEL_IDX           = 0,

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 16384, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 16, 
    // Number of banks
    parameter NUM_BANKS             = 4,
    // Number of ports per banks
    parameter NUM_PORTS             = 1,
    // Number of associative ways
    parameter NUM_WAYS              = 2,
    // Size of a word in bytes
    parameter WORD_SIZE             = 4, 

    // Core Request Queue Size
    parameter CREQ_SIZE             = 0,
    // Core Response Queue Size
    parameter CRSQ_SIZE             = 2,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 16, 
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 0,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = 16,

    // enable bypass for non-cacheable addresses
    parameter NC_ENABLE             = 1,

    // Core response output register
    parameter CORE_OUT_REG          = 3,

    // Memory request output register
    parameter MEM_OUT_REG           = 3,

    parameter NUM_CACHES = `UP(NUM_UNITS),
    parameter PASSTHRU   = (NUM_UNITS == 0),
    parameter ARB_TAG_WIDTH = TAG_WIDTH + `ARB_SEL_BITS(NUM_INPUTS, NUM_CACHES),
    parameter MEM_TAG_WIDTH = PASSTHRU ? (NC_ENABLE ? `CACHE_NC_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH) : 
                                                       `CACHE_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH)) : 
                                          (NC_ENABLE ? `CACHE_NC_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, NUM_REQS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH) :
                                                       `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS))
 ) (    
    input wire clk,
    input wire reset,

    // Core request
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0]                 core_req_valid,
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0]                 core_req_rw,
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0][WORD_SIZE-1:0]  core_req_byteen,
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0][`WORD_ADDR_WIDTH-1:0] core_req_addr,
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0][`WORD_WIDTH-1:0] core_req_data,
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0][TAG_WIDTH-1:0]  core_req_tag,
    output wire [NUM_INPUTS-1:0][NUM_REQS-1:0]                 core_req_ready,

    // Core response
    output wire [NUM_INPUTS-1:0][NUM_REQS-1:0]                 core_rsp_valid,
    output wire [NUM_INPUTS-1:0][NUM_REQS-1:0][`WORD_WIDTH-1:0] core_rsp_data,
    output wire [NUM_INPUTS-1:0][NUM_REQS-1:0][TAG_WIDTH-1:0]  core_rsp_tag,
    input  wire [NUM_INPUTS-1:0][NUM_REQS-1:0]                 core_rsp_ready,

    // Memory request
    output wire                    mem_req_valid,
    output wire                    mem_req_rw, 
    output wire [LINE_SIZE-1:0]    mem_req_byteen,
    output wire [`MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [`LINE_WIDTH-1:0]  mem_req_data,  
    output wire [MEM_TAG_WIDTH-1:0] mem_req_tag, 
    input  wire                    mem_req_ready,
    
    // Memory response
    input  wire                    mem_rsp_valid,    
    input  wire [`LINE_WIDTH-1:0]  mem_rsp_data,
    input  wire [MEM_TAG_WIDTH-1:0] mem_rsp_tag, 
    output wire                    mem_rsp_ready
);
    VX_cache_bus_if #(
        .NUM_REQS  (NUM_REQS), 
        .WORD_SIZE (WORD_SIZE), 
        .TAG_WIDTH (TAG_WIDTH)
    ) core_bus_if[NUM_INPUTS]();

    VX_mem_bus_if #(
        .DATA_WIDTH (`LINE_WIDTH),
        .TAG_WIDTH  (MEM_TAG_WIDTH)
    ) mem_bus_if();

    // Core request
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign core_bus_if[i].req_valid = core_req_valid[i];
        assign core_bus_if[i].req_rw = core_req_rw[i];
        assign core_bus_if[i].req_byteen = core_req_byteen[i];
        assign core_bus_if[i].req_addr = core_req_addr[i];
        assign core_bus_if[i].req_data = core_req_data[i];
        assign core_bus_if[i].req_tag = core_req_tag[i];
        assign core_req_ready[i] = core_bus_if[i].req_ready;
    end

    // Core response
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign core_rsp_valid[i] = core_bus_if[i].rsp_valid;
        assign core_rsp_data[i] = core_bus_if[i].rsp_data;
        assign core_rsp_tag[i] = core_bus_if[i].rsp_tag;
        assign core_bus_if[i].rsp_ready = core_rsp_ready[i];
    end

    // Memory request
    assign mem_req_valid = mem_bus_if.req_valid;
    assign mem_req_rw = mem_bus_if.req_rw; 
    assign mem_req_byteen = mem_bus_if.req_byteen;
    assign mem_req_addr = mem_bus_if.req_addr;
    assign mem_req_data = mem_bus_if.req_data;  
    assign mem_req_tag = mem_bus_if.req_tag; 
    assign mem_bus_if.req_ready = mem_req_ready;
    
    // Memory response
    assign mem_bus_if.rsp_valid = mem_rsp_valid;    
    assign mem_bus_if.rsp_data = mem_rsp_data;
    assign mem_bus_if.rsp_tag = mem_rsp_tag; 
    assign mem_rsp_ready = mem_bus_if.rsp_ready;

    VX_cache_cluster #(
        .INSTANCE_ID    (INSTANCE_ID),
        .NUM_UNITS      (NUM_UNITS),
        .NUM_INPUTS     (NUM_INPUTS),
        .TAG_SEL_IDX    (TAG_SEL_IDX),
        .CACHE_SIZE     (CACHE_SIZE),
        .LINE_SIZE      (LINE_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .NUM_WAYS       (NUM_WAYS),
        .WORD_SIZE      (WORD_SIZE),
        .NUM_REQS       (NUM_REQS),
        .CREQ_SIZE      (CREQ_SIZE),
        .CRSQ_SIZE      (CRSQ_SIZE),
        .MSHR_SIZE      (MSHR_SIZE),
        .MRSQ_SIZE      (MRSQ_SIZE),
        .MREQ_SIZE      (MREQ_SIZE),
        .TAG_WIDTH      (TAG_WIDTH),
        .UUID_WIDTH     (UUID_WIDTH),
        .WRITE_ENABLE   (WRITE_ENABLE),
        .CORE_OUT_REG   (CORE_OUT_REG),
        .MEM_OUT_REG    (MEM_OUT_REG)
    ) cache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_icache_if),
    `endif
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (core_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

 endmodule
