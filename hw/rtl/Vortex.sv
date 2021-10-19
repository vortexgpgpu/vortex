`include "VX_define.vh"

module Vortex (
    `SCOPE_IO_Vortex

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,    
    output wire [`VX_MEM_BYTEEN_WIDTH-1:0]  mem_req_byteen,    
    output wire [`VX_MEM_ADDR_WIDTH-1:0]    mem_req_addr,
    output wire [`VX_MEM_DATA_WIDTH-1:0]    mem_req_data,
    output wire [`VX_MEM_TAG_WIDTH-1:0]     mem_req_tag,
    input  wire                             mem_req_ready,

    // Memory response    
    input wire                              mem_rsp_valid,        
    input wire [`VX_MEM_DATA_WIDTH-1:0]     mem_rsp_data,
    input wire [`VX_MEM_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                             mem_rsp_ready,

    // Status
    output wire                             busy
);
    `STATIC_ASSERT((`L3_ENABLE == 0 || `NUM_CLUSTERS > 1), ("invalid parameter"))

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_mem_req_valid;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_mem_req_rw;
    wire [`NUM_CLUSTERS-1:0][`L2_MEM_BYTEEN_WIDTH-1:0] per_cluster_mem_req_byteen;
    wire [`NUM_CLUSTERS-1:0][`L2_MEM_ADDR_WIDTH-1:0] per_cluster_mem_req_addr;
    wire [`NUM_CLUSTERS-1:0][`L2_MEM_DATA_WIDTH-1:0] per_cluster_mem_req_data;
    wire [`NUM_CLUSTERS-1:0][`L2_MEM_TAG_WIDTH-1:0]  per_cluster_mem_req_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_mem_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_mem_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][`L2_MEM_DATA_WIDTH-1:0] per_cluster_mem_rsp_data;
    wire [`NUM_CLUSTERS-1:0][`L2_MEM_TAG_WIDTH-1:0]  per_cluster_mem_rsp_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_mem_rsp_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_busy;

    for (genvar i = 0; i < `NUM_CLUSTERS; i++) begin

        `RESET_RELAY (cluster_reset);

        VX_cluster #(
            .CLUSTER_ID(i)
        ) cluster (
            `SCOPE_BIND_Vortex_cluster(i)

            .clk            (clk),
            .reset          (cluster_reset),

            .mem_req_valid  (per_cluster_mem_req_valid [i]),
            .mem_req_rw     (per_cluster_mem_req_rw    [i]),
            .mem_req_byteen (per_cluster_mem_req_byteen[i]),
            .mem_req_addr   (per_cluster_mem_req_addr  [i]),
            .mem_req_data   (per_cluster_mem_req_data  [i]),
            .mem_req_tag    (per_cluster_mem_req_tag   [i]),
            .mem_req_ready  (per_cluster_mem_req_ready [i]),

            .mem_rsp_valid  (per_cluster_mem_rsp_valid [i]),
            .mem_rsp_data   (per_cluster_mem_rsp_data  [i]),
            .mem_rsp_tag    (per_cluster_mem_rsp_tag   [i]),
            .mem_rsp_ready  (per_cluster_mem_rsp_ready [i]),

            .busy           (per_cluster_busy           [i])
        );
    end

    assign busy   = (| per_cluster_busy);

    if (`L3_ENABLE) begin
    `ifdef PERF_ENABLE
        VX_perf_cache_if perf_l3cache_if();
    `endif

        `RESET_RELAY (l3_reset);

        VX_cache #(
            .CACHE_ID           (`L3_CACHE_ID),
            .CACHE_SIZE         (`L3_CACHE_SIZE),
            .CACHE_LINE_SIZE    (`L3_CACHE_LINE_SIZE),
            .NUM_BANKS          (`L3_NUM_BANKS),
            .NUM_PORTS          (`L3_NUM_PORTS),
            .WORD_SIZE          (`L3_WORD_SIZE),
            .NUM_REQS           (`L3_NUM_REQS),
            .CREQ_SIZE          (`L3_CREQ_SIZE),
            .CRSQ_SIZE          (`L3_CRSQ_SIZE),
            .MSHR_SIZE          (`L3_MSHR_SIZE),
            .MRSQ_SIZE          (`L3_MRSQ_SIZE),
            .MREQ_SIZE          (`L3_MREQ_SIZE),
            .WRITE_ENABLE       (1),
            .CORE_TAG_WIDTH     (`L2_MEM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .MEM_TAG_WIDTH      (`L3_MEM_TAG_WIDTH),
            .NC_ENABLE          (1)
        ) l3cache (
            `SCOPE_BIND_Vortex_l3cache
 
            .clk                (clk),
            .reset              (l3_reset),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_l3cache_if),
        `endif

            // Core request    
            .core_req_valid     (per_cluster_mem_req_valid),
            .core_req_rw        (per_cluster_mem_req_rw),
            .core_req_byteen    (per_cluster_mem_req_byteen),
            .core_req_addr      (per_cluster_mem_req_addr),
            .core_req_data      (per_cluster_mem_req_data),
            .core_req_tag       (per_cluster_mem_req_tag),
            .core_req_ready     (per_cluster_mem_req_ready),

            // Core response
            .core_rsp_valid     (per_cluster_mem_rsp_valid),
            .core_rsp_data      (per_cluster_mem_rsp_data),
            .core_rsp_tag       (per_cluster_mem_rsp_tag),              
            .core_rsp_ready     (per_cluster_mem_rsp_ready),
            `UNUSED_PIN (core_rsp_tmask),

            // Memory request
            .mem_req_valid      (mem_req_valid),
            .mem_req_rw         (mem_req_rw),
            .mem_req_byteen     (mem_req_byteen),
            .mem_req_addr       (mem_req_addr),
            .mem_req_data       (mem_req_data),
            .mem_req_tag        (mem_req_tag),
            .mem_req_ready      (mem_req_ready),

            // Memory response
            .mem_rsp_valid      (mem_rsp_valid),            
            .mem_rsp_data       (mem_rsp_data),
            .mem_rsp_tag        (mem_rsp_tag),
            .mem_rsp_ready      (mem_rsp_ready)
        );

    end else begin

        `RESET_RELAY (mem_arb_reset);

        VX_mem_arb #(
            .NUM_REQS     (`NUM_CLUSTERS),
            .DATA_WIDTH   (`L3_MEM_DATA_WIDTH),            
            .ADDR_WIDTH   (`L3_MEM_ADDR_WIDTH),
            .TAG_IN_WIDTH (`L2_MEM_TAG_WIDTH),
            .TYPE         ("R"),
            .BUFFERED_REQ (1),
            .BUFFERED_RSP (1)
        ) mem_arb (
            .clk            (clk),
            .reset          (mem_arb_reset),

            // Core request
            .req_valid_in   (per_cluster_mem_req_valid),
            .req_rw_in      (per_cluster_mem_req_rw),
            .req_byteen_in  (per_cluster_mem_req_byteen),
            .req_addr_in    (per_cluster_mem_req_addr),
            .req_data_in    (per_cluster_mem_req_data),  
            .req_tag_in     (per_cluster_mem_req_tag),  
            .req_ready_in   (per_cluster_mem_req_ready),

            // Memory request
            .req_valid_out  (mem_req_valid),
            .req_rw_out     (mem_req_rw),        
            .req_byteen_out (mem_req_byteen),        
            .req_addr_out   (mem_req_addr),
            .req_data_out   (mem_req_data),
            .req_tag_out    (mem_req_tag),
            .req_ready_out  (mem_req_ready),

            // Core response
            .rsp_valid_out  (per_cluster_mem_rsp_valid),
            .rsp_data_out   (per_cluster_mem_rsp_data),
            .rsp_tag_out    (per_cluster_mem_rsp_tag),
            .rsp_ready_out  (per_cluster_mem_rsp_ready),
            
            // Memory response
            .rsp_valid_in   (mem_rsp_valid),
            .rsp_tag_in     (mem_rsp_tag),
            .rsp_data_in    (mem_rsp_data),
            .rsp_ready_in   (mem_rsp_ready)
        );

    end

    `SCOPE_ASSIGN (reset, reset);
    `SCOPE_ASSIGN (mem_req_fire, mem_req_valid && mem_req_ready);
    `SCOPE_ASSIGN (mem_req_addr, `TO_FULL_ADDR(mem_req_addr));
    `SCOPE_ASSIGN (mem_req_rw,   mem_req_rw);
    `SCOPE_ASSIGN (mem_req_byteen, mem_req_byteen);
    `SCOPE_ASSIGN (mem_req_data, mem_req_data);
    `SCOPE_ASSIGN (mem_req_tag,  mem_req_tag);
    `SCOPE_ASSIGN (mem_rsp_fire, mem_rsp_valid && mem_rsp_ready);
    `SCOPE_ASSIGN (mem_rsp_data, mem_rsp_data);
    `SCOPE_ASSIGN (mem_rsp_tag,  mem_rsp_tag);
    `SCOPE_ASSIGN (busy, busy);

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (mem_req_valid && mem_req_ready) begin
            if (mem_req_rw)
                dpi_trace("%d: MEM Wr Req: addr=%0h, tag=%0h, byteen=%0h data=%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen, mem_req_data);
            else
                dpi_trace("%d: MEM Rd Req: addr=%0h, tag=%0h, byteen=%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen);
        end
        if (mem_rsp_valid && mem_rsp_ready) begin
            dpi_trace("%d: MEM Rsp: tag=%0h, data=%0h\n", $time, mem_rsp_tag, mem_rsp_data);
        end
    end
`endif


`ifndef NDEBUG
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule