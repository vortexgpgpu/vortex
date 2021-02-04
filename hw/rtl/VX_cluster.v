`include "VX_define.vh"

module VX_cluster #(
    parameter CLUSTER_ID = 0
) ( 
    `SCOPE_IO_VX_cluster

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DRAM request
    output wire                             dram_req_valid,
    output wire                             dram_req_rw,    
    output wire [`L2DRAM_BYTEEN_WIDTH-1:0]  dram_req_byteen,    
    output wire [`L2DRAM_ADDR_WIDTH-1:0]    dram_req_addr,
    output wire [`L2DRAM_LINE_WIDTH-1:0]    dram_req_data,
    output wire [`L2DRAM_TAG_WIDTH-1:0]     dram_req_tag,
    input  wire                             dram_req_ready,

    // DRAM response    
    input wire                              dram_rsp_valid,        
    input wire [`L2DRAM_LINE_WIDTH-1:0]     dram_rsp_data,
    input wire [`L2DRAM_TAG_WIDTH-1:0]      dram_rsp_tag,
    output wire                             dram_rsp_ready,

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

    wire [`NUM_CORES-1:0]                        per_core_dram_req_valid;
    wire [`NUM_CORES-1:0]                        per_core_dram_req_rw;    
    wire [`NUM_CORES-1:0][`DDRAM_BYTEEN_WIDTH-1:0] per_core_dram_req_byteen;    
    wire [`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0] per_core_dram_req_addr;
    wire [`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_dram_req_data;
    wire [`NUM_CORES-1:0][`XDRAM_TAG_WIDTH-1:0]  per_core_dram_req_tag;
    wire [`NUM_CORES-1:0]                        per_core_dram_req_ready;

    wire [`NUM_CORES-1:0]                        per_core_dram_rsp_valid;            
    wire [`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_dram_rsp_data;
    wire [`NUM_CORES-1:0][`XDRAM_TAG_WIDTH-1:0]  per_core_dram_rsp_tag;
    wire [`NUM_CORES-1:0]                        per_core_dram_rsp_ready;

    wire [`NUM_CORES-1:0]                        per_core_csr_req_valid;
    wire [`NUM_CORES-1:0][11:0]                  per_core_csr_req_addr;
    wire [`NUM_CORES-1:0]                        per_core_csr_req_rw;
    wire [`NUM_CORES-1:0][31:0]                  per_core_csr_req_data;
    wire [`NUM_CORES-1:0]                        per_core_csr_req_ready;

    wire [`NUM_CORES-1:0]                        per_core_csr_rsp_valid;
    wire [`NUM_CORES-1:0][31:0]                  per_core_csr_rsp_data;
    wire [`NUM_CORES-1:0]                        per_core_csr_rsp_ready;

    wire [`NUM_CORES-1:0]                        per_core_busy;
    wire [`NUM_CORES-1:0]                        per_core_ebreak;

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

            .dram_req_valid (per_core_dram_req_valid[i]),
            .dram_req_rw    (per_core_dram_req_rw   [i]),                
            .dram_req_byteen(per_core_dram_req_byteen[i]),                
            .dram_req_addr  (per_core_dram_req_addr [i]),
            .dram_req_data  (per_core_dram_req_data [i]),
            .dram_req_tag   (per_core_dram_req_tag  [i]),
            .dram_req_ready (per_core_dram_req_ready[i]),
                     
            .dram_rsp_valid (per_core_dram_rsp_valid[i]),                
            .dram_rsp_data  (per_core_dram_rsp_data [i]),
            .dram_rsp_tag   (per_core_dram_rsp_tag  [i]),
            .dram_rsp_ready (per_core_dram_rsp_ready[i]),

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
        .BUFFERED_RSP (`NUM_CORES >= 4)
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
            .NUM_REQS           (`NUM_CORES),
            .CREQ_SIZE          (`L2CREQ_SIZE),
            .MSHR_SIZE          (`L2MSHR_SIZE),
            .DRSQ_SIZE          (`L2DRSQ_SIZE),
            .CRSQ_SIZE          (`L2CRSQ_SIZE),
            .DREQ_SIZE          (`L2DREQ_SIZE),
            .WRITE_ENABLE       (1),          
            .CORE_TAG_WIDTH     (`XDRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .DRAM_TAG_WIDTH     (`L2DRAM_TAG_WIDTH)
        ) l2cache (
            `SCOPE_BIND_VX_cluster_l2cache
              
            .clk                (clk),
            .reset              (reset),

            .flush              (1'b0),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_l2cache_if),
        `endif

            // Core request
            .core_req_valid     (per_core_dram_req_valid),
            .core_req_rw        (per_core_dram_req_rw),
            .core_req_byteen    (per_core_dram_req_byteen),
            .core_req_addr      (per_core_dram_req_addr),
            .core_req_data      (per_core_dram_req_data),  
            .core_req_tag       (per_core_dram_req_tag),  
            .core_req_ready     (per_core_dram_req_ready),

            // Core response
            .core_rsp_valid     (per_core_dram_rsp_valid),
            .core_rsp_data      (per_core_dram_rsp_data),
            .core_rsp_tag       (per_core_dram_rsp_tag),
            .core_rsp_ready     (per_core_dram_rsp_ready),

            // DRAM request
            .dram_req_valid     (dram_req_valid),
            .dram_req_rw        (dram_req_rw),        
            .dram_req_byteen    (dram_req_byteen),
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),
            
            // DRAM response
            .dram_rsp_valid     (dram_rsp_valid),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_ready     (dram_rsp_ready)
        );

    end else begin

        VX_mem_arb #(
            .NUM_REQS      (`NUM_CORES),
            .DATA_WIDTH    (`L2DRAM_LINE_WIDTH),            
            .TAG_IN_WIDTH  (`XDRAM_TAG_WIDTH),
            .TAG_OUT_WIDTH (`L2DRAM_TAG_WIDTH),
            .BUFFERED_REQ  (`NUM_CORES >= 4),
            .BUFFERED_RSP  (1)
        ) dram_arb (
            .clk            (clk),
            .reset          (reset),

            // Core request
            .req_valid_in   (per_core_dram_req_valid),
            .req_rw_in      (per_core_dram_req_rw),
            .req_byteen_in  (per_core_dram_req_byteen),
            .req_addr_in    (per_core_dram_req_addr),
            .req_data_in    (per_core_dram_req_data),  
            .req_tag_in     (per_core_dram_req_tag),  
            .req_ready_in   (per_core_dram_req_ready),

            // DRAM request
            .req_valid_out  (dram_req_valid),
            .req_rw_out     (dram_req_rw),        
            .req_byteen_out (dram_req_byteen),        
            .req_addr_out   (dram_req_addr),
            .req_data_out   (dram_req_data),
            .req_tag_out    (dram_req_tag),
            .req_ready_out  (dram_req_ready),

            // Core response
            .rsp_valid_out  (per_core_dram_rsp_valid),
            .rsp_data_out   (per_core_dram_rsp_data),
            .rsp_tag_out    (per_core_dram_rsp_tag),
            .rsp_ready_out  (per_core_dram_rsp_ready),
            
            // DRAM response
            .rsp_valid_in   (dram_rsp_valid),
            .rsp_tag_in     (dram_rsp_tag),
            .rsp_data_in    (dram_rsp_data),
            .rsp_ready_in   (dram_rsp_ready)
        );

    end

endmodule
