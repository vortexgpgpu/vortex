`include "VX_define.vh"

module VX_cluster #(
    parameter CLUSTER_ID = 0
) ( 
    `SCOPE_IO_VX_cluster

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,    
    output wire [`L2MEM_BYTEEN_WIDTH-1:0]   mem_req_byteen,    
    output wire [`L2MEM_ADDR_WIDTH-1:0]     mem_req_addr,
    output wire [`L2MEM_LINE_WIDTH-1:0]     mem_req_data,
    output wire [`L2MEM_TAG_WIDTH-1:0]      mem_req_tag,
    input  wire                             mem_req_ready,

    // Memory response    
    input wire                              mem_rsp_valid,        
    input wire [`L2MEM_LINE_WIDTH-1:0]      mem_rsp_data,
    input wire [`L2MEM_TAG_WIDTH-1:0]       mem_rsp_tag,
    output wire                             mem_rsp_ready,

    // CSR Request
    input  wire                             csr_req_valid,
    input  wire [`NC_BITS-1:0]              csr_req_coreid,
    input  wire [11:0]                      csr_req_addr,
    input  wire                             csr_req_rw,
    input  wire [31:0]                      csr_req_data,
    output wire                             csr_req_ready,

    // CSR Response
    output wire                             csr_rsp_valid,
    output wire [31:0]                      csr_rsp_data,
    input wire                              csr_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
); 
    `STATIC_ASSERT((`L2_ENABLE == 0 || `NUM_CORES > 1), ("invalid parameter"))

    wire [`NUM_CORES-1:0]                       per_core_mem_req_valid;
    wire [`NUM_CORES-1:0]                       per_core_mem_req_rw;    
    wire [`NUM_CORES-1:0][`DMEM_BYTEEN_WIDTH-1:0] per_core_mem_req_byteen;    
    wire [`NUM_CORES-1:0][`DMEM_ADDR_WIDTH-1:0] per_core_mem_req_addr;
    wire [`NUM_CORES-1:0][`DMEM_LINE_WIDTH-1:0] per_core_mem_req_data;
    wire [`NUM_CORES-1:0][`XMEM_TAG_WIDTH-1:0]  per_core_mem_req_tag;
    wire [`NUM_CORES-1:0]                       per_core_mem_req_ready;

    wire [`NUM_CORES-1:0]                       per_core_mem_rsp_valid;            
    wire [`NUM_CORES-1:0][`DMEM_LINE_WIDTH-1:0] per_core_mem_rsp_data;
    wire [`NUM_CORES-1:0][`XMEM_TAG_WIDTH-1:0]  per_core_mem_rsp_tag;
    wire [`NUM_CORES-1:0]                       per_core_mem_rsp_ready;

    wire [`NUM_CORES-1:0]                       per_core_csr_req_valid;
    wire [`NUM_CORES-1:0][11:0]                 per_core_csr_req_addr;
    wire [`NUM_CORES-1:0]                       per_core_csr_req_rw;
    wire [`NUM_CORES-1:0][31:0]                 per_core_csr_req_data;
    wire [`NUM_CORES-1:0]                       per_core_csr_req_ready;

    wire [`NUM_CORES-1:0]                       per_core_csr_rsp_valid;
    wire [`NUM_CORES-1:0][31:0]                 per_core_csr_rsp_data;
    wire [`NUM_CORES-1:0]                       per_core_csr_rsp_ready;

    wire [`NUM_CORES-1:0]                       per_core_busy;
    wire [`NUM_CORES-1:0]                       per_core_ebreak;

    for (genvar i = 0; i < `NUM_CORES; i++) begin
        
        wire core_reset;
        VX_reset_relay #(
            .DEPTH (`NUM_CORES > 1)
        ) reset_relay (
            .clk     (clk),
            .reset   (reset),
            .reset_o (core_reset)
        );

        VX_core #(
            .CORE_ID(i + (CLUSTER_ID * `NUM_CORES))
        ) core (
            `SCOPE_BIND_VX_cluster_core(i)

            .clk            (clk),
            .reset          (core_reset),

            .mem_req_valid  (per_core_mem_req_valid[i]),
            .mem_req_rw     (per_core_mem_req_rw   [i]),                
            .mem_req_byteen (per_core_mem_req_byteen[i]),                
            .mem_req_addr   (per_core_mem_req_addr [i]),
            .mem_req_data   (per_core_mem_req_data [i]),
            .mem_req_tag    (per_core_mem_req_tag  [i]),
            .mem_req_ready  (per_core_mem_req_ready[i]),
                     
            .mem_rsp_valid  (per_core_mem_rsp_valid[i]),                
            .mem_rsp_data   (per_core_mem_rsp_data [i]),
            .mem_rsp_tag    (per_core_mem_rsp_tag  [i]),
            .mem_rsp_ready  (per_core_mem_rsp_ready[i]),

            .csr_req_valid  (per_core_csr_req_valid [i]),
            .csr_req_rw     (per_core_csr_req_rw    [i]),
            .csr_req_addr   (per_core_csr_req_addr  [i]),
            .csr_req_data   (per_core_csr_req_data  [i]),
            .csr_req_ready  (per_core_csr_req_ready [i]),

            .csr_rsp_valid  (per_core_csr_rsp_valid [i]),            
            .csr_rsp_data   (per_core_csr_rsp_data  [i]),
            .csr_rsp_ready  (per_core_csr_rsp_ready [i]),

            .busy           (per_core_busy          [i]),
            .ebreak         (per_core_ebreak        [i])
        );
    end

    VX_csr_arb #(
        .NUM_REQS     (`NUM_CORES),
        .DATA_WIDTH   (32),
        .ADDR_WIDTH   (12),
        .BUFFERED_REQ (1), 
        .BUFFERED_RSP (1)
    ) csr_arb (
        .clk            (clk),
        .reset          (reset),

        .request_id     (csr_req_coreid), 

        // input requests
        .req_valid_in   (csr_req_valid),     
        .req_addr_in    (csr_req_addr),
        .req_rw_in      (csr_req_rw),
        .req_data_in    (csr_req_data),
        .req_ready_in   (csr_req_ready),

        // output request
        .req_valid_out  (per_core_csr_req_valid),
        .req_addr_out   (per_core_csr_req_addr),            
        .req_rw_out     (per_core_csr_req_rw),
        .req_data_out   (per_core_csr_req_data),  
        .req_ready_out  (per_core_csr_req_ready),     

        // input responses
        .rsp_valid_in   (per_core_csr_rsp_valid),
        .rsp_data_in    (per_core_csr_rsp_data),
        .rsp_ready_in   (per_core_csr_rsp_ready),       
        
        // output response
        .rsp_valid_out  (csr_rsp_valid),
        .rsp_data_out   (csr_rsp_data),
        .rsp_ready_out  (csr_rsp_ready)
    );
    
    assign busy = (| per_core_busy);
    assign ebreak = (| per_core_ebreak);

    if (`L2_ENABLE) begin
    `ifdef PERF_ENABLE
        VX_perf_cache_if perf_l2cache_if();
    `endif

        VX_cache #(
            .CACHE_ID           (`L2CACHE_ID),
            .CACHE_SIZE         (`L2CACHE_SIZE),
            .CACHE_LINE_SIZE    (`L2CACHE_LINE_SIZE),
            .NUM_BANKS          (`L2NUM_BANKS),
            .WORD_SIZE          (`L2WORD_SIZE),
            .NUM_REQS           (`L2NUM_REQS),
            .CREQ_SIZE          (`L2CREQ_SIZE),
            .MSHR_SIZE          (`L2MSHR_SIZE),
            .MRSQ_SIZE          (`L2MRSQ_SIZE),
            .MREQ_SIZE          (`L2MREQ_SIZE),
            .WRITE_ENABLE       (1),          
            .CORE_TAG_WIDTH     (`XMEM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .MEM_TAG_WIDTH      (`L2MEM_TAG_WIDTH),
            .NC_ENABLE          (1)
        ) l2cache (
            `SCOPE_BIND_VX_cluster_l2cache
              
            .clk                (clk),
            .reset              (reset),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_l2cache_if),
        `endif

            // Core request
            .core_req_valid     (per_core_mem_req_valid),
            .core_req_rw        (per_core_mem_req_rw),
            .core_req_byteen    (per_core_mem_req_byteen),
            .core_req_addr      (per_core_mem_req_addr),
            .core_req_data      (per_core_mem_req_data),  
            .core_req_tag       (per_core_mem_req_tag),  
            .core_req_ready     (per_core_mem_req_ready),

            // Core response
            .core_rsp_valid     (per_core_mem_rsp_valid),
            .core_rsp_data      (per_core_mem_rsp_data),
            .core_rsp_tag       (per_core_mem_rsp_tag),
            .core_rsp_ready     (per_core_mem_rsp_ready),

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
            .mem_rsp_tag        (mem_rsp_tag),
            .mem_rsp_data       (mem_rsp_data),
            .mem_rsp_ready      (mem_rsp_ready)
        );

    end else begin

        VX_mem_arb #(
            .NUM_REQS       (`NUM_CORES),
            .DATA_WIDTH     (`L2MEM_LINE_WIDTH),            
            .TAG_IN_WIDTH   (`XMEM_TAG_WIDTH),
            .TAG_OUT_WIDTH  (`L2MEM_TAG_WIDTH),
            .BUFFERED_REQ   (1),
            .BUFFERED_RSP   (1)
        ) mem_arb (
            .clk            (clk),
            .reset          (reset),

            // Core request
            .req_valid_in   (per_core_mem_req_valid),
            .req_rw_in      (per_core_mem_req_rw),
            .req_byteen_in  (per_core_mem_req_byteen),
            .req_addr_in    (per_core_mem_req_addr),
            .req_data_in    (per_core_mem_req_data),  
            .req_tag_in     (per_core_mem_req_tag),  
            .req_ready_in   (per_core_mem_req_ready),

            // Memory request
            .req_valid_out  (mem_req_valid),
            .req_rw_out     (mem_req_rw),        
            .req_byteen_out (mem_req_byteen),        
            .req_addr_out   (mem_req_addr),
            .req_data_out   (mem_req_data),
            .req_tag_out    (mem_req_tag),
            .req_ready_out  (mem_req_ready),

            // Core response
            .rsp_valid_out  (per_core_mem_rsp_valid),
            .rsp_data_out   (per_core_mem_rsp_data),
            .rsp_tag_out    (per_core_mem_rsp_tag),
            .rsp_ready_out  (per_core_mem_rsp_ready),
            
            // Memory response
            .rsp_valid_in   (mem_rsp_valid),
            .rsp_tag_in     (mem_rsp_tag),
            .rsp_data_in    (mem_rsp_data),
            .rsp_ready_in   (mem_rsp_ready)
        );

    end

endmodule
