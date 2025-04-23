module VX_tlb_wrapper import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID   = "",

    // Number of requests per cycle
    // parameter NUM_REQS              = 4,

    // Number of memory ports
    parameter MEM_PORTS             = 1,

    // Address width
    parameter ADDR_WIDTH            = 32,

    // Tag width
    parameter TAG_WIDTH             = 32,

    // Flags width
    // parameter FLAGS_WIDTH           = 0,

    // TLB size in entries
    parameter TLB_SIZE              = 32,

    // Page size in bytes
    parameter PAGE_SIZE             = 4096,

    // PTW latency
    parameter PTW_LATENCY           = 10
) (
    input wire clk,
    input wire reset,

    // Input from arbiter
    VX_mem_bus_if.slave arb_bus_if [NUM_REQS],

    // Output to cache
    VX_mem_bus_if.master cache_bus_if [NUM_REQS],

    // Output to next-level memory on TLB miss
    VX_mem_bus_if.master ptw_mem_bus_if
);

    // TLB lookup results
    wire [NUM_REQS-1:0] tlb_hit;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] tlb_translated_addr;
    
    // PTW control signals
    wire ptw_busy;
    wire [NUM_REQS-1:0] ptw_req_valid;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] ptw_req_addr;
    wire [NUM_REQS-1:0] ptw_rsp_valid;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] ptw_rsp_addr;

        VX_tlb #(
        .INSTANCE_ID  (INSTANCE_ID),
        .TLB_SIZE    (TLB_SIZE),
        .PAGE_SIZE   (PAGE_SIZE),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .TAG_WIDTH   (TAG_WIDTH),
        .FLAGS_WIDTH (FLAGS_WIDTH)
    ) tlb (
        .clk            (clk),
        .reset          (reset),
        .req_valid      (arb_bus_if.req_valid),
        .req_addr       (arb_bus_if.req_data.addr),
        .req_rw         (arb_bus_if.req_data.rw),
        .req_tag        (arb_bus_if.req_data.tag),
        .req_flags      (arb_bus_if.req_data.flags),
        .req_ready      (arb_bus_if.req_ready),
        .tlb_hit        (tlb_hit),
        .translated_addr(tlb_translated_addr)
    );

        VX_ptw #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .PAGE_SIZE  (PAGE_SIZE),
        .PTW_LATENCY(PTW_LATENCY)
    ) ptw (
        .clk            (clk),
        .reset          (reset),
        .req_valid      (ptw_req_valid),
        .req_addr       (ptw_req_addr),
        .rsp_valid      (ptw_rsp_valid),
        .rsp_addr       (ptw_rsp_addr),
        .mem_bus_if     (ptw_mem_bus_if),
        .busy           (ptw_busy)
    );

        // Handle TLB hit and miss
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        // On TLB hit, forward the translated address to the cache
        assign cache_bus_if[i].req_valid = arb_bus_if[i].req_valid && tlb_hit[i];
        assign cache_bus_if[i].req_data.addr = tlb_translated_addr[i];
        assign cache_bus_if[i].req_data.rw = arb_bus_if[i].req_data.rw;
        assign cache_bus_if[i].req_data.tag = arb_bus_if[i].req_data.tag;
        assign cache_bus_if[i].req_data.flags = arb_bus_if[i].req_data.flags;
        assign arb_bus_if[i].req_ready = cache_bus_if[i].req_ready && tlb_hit[i];

        // On TLB miss, forward the request to the PTW
        assign ptw_req_valid[i] = arb_bus_if[i].req_valid && !tlb_hit[i];
        assign ptw_req_addr[i] = arb_bus_if[i].req_data.addr;
    end

        // Handle PTW response
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset logic
        end else begin
            for (int i = 0; i < NUM_REQS; ++i) begin
                if (ptw_rsp_valid[i]) begin
                    // Update TLB with the new translation
                    // (This logic depends on the TLB implementation)
                end
            end
        end
    end

endmodule