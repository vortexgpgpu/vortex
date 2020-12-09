`include "VX_define.vh"

module Vortex (
    `SCOPE_IO_Vortex

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DRAM request
    output wire                             dram_req_valid,
    output wire                             dram_req_rw,    
    output wire [`VX_DRAM_BYTEEN_WIDTH-1:0] dram_req_byteen,    
    output wire [`VX_DRAM_ADDR_WIDTH-1:0]   dram_req_addr,
    output wire [`VX_DRAM_LINE_WIDTH-1:0]   dram_req_data,
    output wire [`VX_DRAM_TAG_WIDTH-1:0]    dram_req_tag,
    input  wire                             dram_req_ready,

    // DRAM response    
    input wire                              dram_rsp_valid,        
    input wire [`VX_DRAM_LINE_WIDTH-1:0]    dram_rsp_data,
    input wire [`VX_DRAM_TAG_WIDTH-1:0]     dram_rsp_tag,
    output wire                             dram_rsp_ready,

    // Snoop request
    input wire                              snp_req_valid,
    input wire [`VX_DRAM_ADDR_WIDTH-1:0]    snp_req_addr,
    input wire                              snp_req_inv,
    input wire [`VX_SNP_TAG_WIDTH-1:0]      snp_req_tag,
    output wire                             snp_req_ready, 

    // Snoop response
    output wire                             snp_rsp_valid,
    output wire [`VX_SNP_TAG_WIDTH-1:0]     snp_rsp_tag,
    input wire                              snp_rsp_ready,     

    // I/O request
    output wire [`NUM_THREADS-1:0]          io_req_valid,
    output wire                             io_req_rw,  
    output wire [`NUM_THREADS-1:0][3:0]     io_req_byteen,  
    output wire [`NUM_THREADS-1:0][29:0]    io_req_addr,
    output wire [`NUM_THREADS-1:0][31:0]    io_req_data,    
    output wire [`VX_CORE_TAG_WIDTH-1:0]    io_req_tag,    
    input wire                              io_req_ready,

    // I/O response
    input wire                              io_rsp_valid,
    input wire [31:0]                       io_rsp_data,
    input wire [`VX_CORE_TAG_WIDTH-1:0]     io_rsp_tag,
    output wire                             io_rsp_ready,

    // CSR I/O Request
    input  wire                             csr_io_req_valid,
    input  wire [`VX_CSR_ID_WIDTH-1:0]      csr_io_req_coreid,
    input  wire [11:0]                      csr_io_req_addr,
    input  wire                             csr_io_req_rw,
    input  wire [31:0]                      csr_io_req_data,
    output wire                             csr_io_req_ready,

    // CSR I/O Response
    output wire                             csr_io_rsp_valid,
    output wire [31:0]                      csr_io_rsp_data,
    input wire                              csr_io_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_valid;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_rw;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] per_cluster_dram_req_byteen;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_dram_req_addr;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_req_data;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_req_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_rsp_data;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0]  per_cluster_dram_rsp_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_dram_rsp_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_valid;
    wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_snp_req_addr;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_inv;
    wire [`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]   per_cluster_snp_req_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][`L2SNP_TAG_WIDTH-1:0]   per_cluster_snp_rsp_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_snp_rsp_ready;

    wire [`NUM_CLUSTERS-1:0][`NUM_THREADS-1:0]       per_cluster_io_req_valid;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_req_rw;
    wire [`NUM_CLUSTERS-1:0][`NUM_THREADS-1:0][3:0]  per_cluster_io_req_byteen;
    wire [`NUM_CLUSTERS-1:0][`NUM_THREADS-1:0][29:0] per_cluster_io_req_addr;
    wire [`NUM_CLUSTERS-1:0][`NUM_THREADS-1:0][31:0] per_cluster_io_req_data;
    wire [`NUM_CLUSTERS-1:0][`L2CORE_TAG_WIDTH-1:0]  per_cluster_io_req_tag;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][`L2CORE_TAG_WIDTH-1:0]  per_cluster_io_rsp_tag;
    wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_io_rsp_data;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_io_rsp_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_req_valid;
    wire [`NUM_CLUSTERS-1:0][11:0]                   per_cluster_csr_io_req_addr;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_req_rw;
    wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_csr_io_req_data;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_req_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_rsp_valid;
    wire [`NUM_CLUSTERS-1:0][31:0]                   per_cluster_csr_io_rsp_data;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_csr_io_rsp_ready;

    wire [`NUM_CLUSTERS-1:0]                         per_cluster_busy;
    wire [`NUM_CLUSTERS-1:0]                         per_cluster_ebreak;

    wire [`LOG2UP(`NUM_CLUSTERS)-1:0] csr_io_cluster_id = `LOG2UP(`NUM_CLUSTERS)'(csr_io_req_coreid >> `CLOG2(`NUM_CORES));
    wire [`NC_BITS-1:0] csr_io_core_id = `NC_BITS'(csr_io_req_coreid);

    for (genvar i = 0; i < `NUM_CLUSTERS; i++) begin
        VX_cluster #(
            .CLUSTER_ID(i)
        ) cluster (
            `SCOPE_BIND_Vortex_cluster(i)

            .clk                (clk),
            .reset              (reset),

            .dram_req_valid     (per_cluster_dram_req_valid [i]),
            .dram_req_rw        (per_cluster_dram_req_rw    [i]),
            .dram_req_byteen    (per_cluster_dram_req_byteen[i]),
            .dram_req_addr      (per_cluster_dram_req_addr  [i]),
            .dram_req_data      (per_cluster_dram_req_data  [i]),
            .dram_req_tag       (per_cluster_dram_req_tag   [i]),
            .dram_req_ready     (per_cluster_dram_req_ready [i]),

            .dram_rsp_valid     (per_cluster_dram_rsp_valid [i]),
            .dram_rsp_data      (per_cluster_dram_rsp_data  [i]),
            .dram_rsp_tag       (per_cluster_dram_rsp_tag   [i]),
            .dram_rsp_ready     (per_cluster_dram_rsp_ready [i]),

            .snp_req_valid      (per_cluster_snp_req_valid  [i]),
            .snp_req_addr       (per_cluster_snp_req_addr   [i]),
            .snp_req_inv        (per_cluster_snp_req_inv    [i]),
            .snp_req_tag        (per_cluster_snp_req_tag    [i]),
            .snp_req_ready      (per_cluster_snp_req_ready  [i]),

            .snp_rsp_valid      (per_cluster_snp_rsp_valid  [i]),
            .snp_rsp_tag        (per_cluster_snp_rsp_tag    [i]),
            .snp_rsp_ready      (per_cluster_snp_rsp_ready  [i]),

            .io_req_valid       (per_cluster_io_req_valid   [i]),
            .io_req_rw          (per_cluster_io_req_rw      [i]),
            .io_req_byteen      (per_cluster_io_req_byteen  [i]),
            .io_req_addr        (per_cluster_io_req_addr    [i]),
            .io_req_data        (per_cluster_io_req_data    [i]),
            .io_req_tag         (per_cluster_io_req_tag     [i]),
            .io_req_ready       (per_cluster_io_req_ready   [i]),

            .io_rsp_valid       (per_cluster_io_rsp_valid   [i]),
            .io_rsp_data        (per_cluster_io_rsp_data    [i]),
            .io_rsp_tag         (per_cluster_io_rsp_tag     [i]),
            .io_rsp_ready       (per_cluster_io_rsp_ready   [i]),

            .csr_io_req_valid   (per_cluster_csr_io_req_valid[i]),
            .csr_io_req_coreid  (csr_io_core_id),
            .csr_io_req_rw      (per_cluster_csr_io_req_rw  [i]),
            .csr_io_req_addr    (per_cluster_csr_io_req_addr[i]),
            .csr_io_req_data    (per_cluster_csr_io_req_data[i]),
            .csr_io_req_ready   (per_cluster_csr_io_req_ready[i]),

            .csr_io_rsp_valid   (per_cluster_csr_io_rsp_valid[i]),
            .csr_io_rsp_data    (per_cluster_csr_io_rsp_data[i]),
            .csr_io_rsp_ready   (per_cluster_csr_io_rsp_ready[i]),

            .busy               (per_cluster_busy           [i]),
            .ebreak             (per_cluster_ebreak         [i])
        );
    end

    VX_databus_arb #(
        .NUM_REQS      (`NUM_CLUSTERS),
        .WORD_SIZE     (4),
        .TAG_IN_WIDTH  (`L2CORE_TAG_WIDTH),
        .TAG_OUT_WIDTH (`L3CORE_TAG_WIDTH)
    ) io_arb (
        .clk            (clk),
        .reset          (reset),

        // input requests
        .req_valid_in   (per_cluster_io_req_valid),
        .req_rw_in      (per_cluster_io_req_rw),
        .req_byteen_in  (per_cluster_io_req_byteen),
        .req_addr_in    (per_cluster_io_req_addr),
        .req_data_in    (per_cluster_io_req_data),
        .req_tag_in     (per_cluster_io_req_tag),
        .req_ready_in   (per_cluster_io_req_ready),

        // output request
        .req_valid_out  (io_req_valid),
        .req_rw_out     (io_req_rw),
        .req_byteen_out (io_req_byteen),
        .req_addr_out   (io_req_addr),
        .req_data_out   (io_req_data),
        .req_tag_out    (io_req_tag),
        .req_ready_out  (io_req_ready),

        // input response
        .rsp_valid_in   (io_rsp_valid),
        .rsp_tag_in     (io_rsp_tag),
        .rsp_data_in    (io_rsp_data),
        .rsp_ready_in   (io_rsp_ready),

        // output responses
        .rsp_valid_out  (per_cluster_io_rsp_valid),
        .rsp_data_out   (per_cluster_io_rsp_data),
        .rsp_tag_out    (per_cluster_io_rsp_tag),
        .rsp_ready_out  (per_cluster_io_rsp_ready)
    );

    VX_csr_io_arb #(
        .NUM_REQS   (`NUM_CLUSTERS),
        .DATA_WIDTH (32),
        .ADDR_WIDTH (12)
    ) csr_io_arb (
        .clk            (clk),
        .reset          (reset),

        .request_id     (csr_io_cluster_id),

        // input requests
        .req_valid_in   (csr_io_req_valid),
        .req_addr_in    (csr_io_req_addr),
        .req_rw_in      (csr_io_req_rw),
        .req_data_in    (csr_io_req_data),
        .req_ready_in   (csr_io_req_ready),

        // output request
        .req_valid_out  (per_cluster_csr_io_req_valid),
        .req_addr_out   (per_cluster_csr_io_req_addr),
        .req_rw_out     (per_cluster_csr_io_req_rw),
        .req_data_out   (per_cluster_csr_io_req_data),
        .req_ready_out  (per_cluster_csr_io_req_ready),

        // input responses
        .rsp_valid_in   (per_cluster_csr_io_rsp_valid),
        .rsp_data_in    (per_cluster_csr_io_rsp_data),
        .rsp_ready_in   (per_cluster_csr_io_rsp_ready),
        
        // output response
        .rsp_valid_out  (csr_io_rsp_valid),
        .rsp_data_out   (csr_io_rsp_data),
        .rsp_ready_out  (csr_io_rsp_ready)
    );

    assign busy   = (| per_cluster_busy);
    assign ebreak = (| per_cluster_ebreak);

    wire                            snp_fwd_rsp_valid;
    wire [`L3DRAM_ADDR_WIDTH-1:0]   snp_fwd_rsp_addr;
    wire                            snp_fwd_rsp_inv;
    wire [`L3SNP_TAG_WIDTH-1:0]     snp_fwd_rsp_tag;
    wire                            snp_fwd_rsp_ready;

    VX_snp_forwarder #(
        .CACHE_ID           (`L3CACHE_ID),            
        .NUM_REQS           (`NUM_CLUSTERS), 
        .SRC_ADDR_WIDTH     (`L3DRAM_ADDR_WIDTH), 
        .DST_ADDR_WIDTH     (`L2DRAM_ADDR_WIDTH),             
        .TAG_IN_WIDTH       (`L3SNP_TAG_WIDTH),
        .TAG_OUT_WIDTH      (`L2SNP_TAG_WIDTH),
        .SREQ_SIZE          (`L3SREQ_SIZE)
    ) snp_forwarder (
        .clk                (clk),
        .reset              (reset),

        .snp_req_valid      (snp_req_valid),
        .snp_req_addr       (snp_req_addr),
        .snp_req_inv        (snp_req_inv),
        .snp_req_tag        (snp_req_tag),
        .snp_req_ready      (snp_req_ready),

        .snp_rsp_valid      (snp_fwd_rsp_valid),       
        .snp_rsp_addr       (snp_fwd_rsp_addr),
        .snp_rsp_inv        (snp_fwd_rsp_inv),
        .snp_rsp_tag        (snp_fwd_rsp_tag),
        .snp_rsp_ready      (snp_fwd_rsp_ready),   

        .snp_fwdout_valid   (per_cluster_snp_req_valid),
        .snp_fwdout_addr    (per_cluster_snp_req_addr),
        .snp_fwdout_inv     (per_cluster_snp_req_inv),
        .snp_fwdout_tag     (per_cluster_snp_req_tag),
        .snp_fwdout_ready   (per_cluster_snp_req_ready),

        .snp_fwdin_valid    (per_cluster_snp_rsp_valid),
        .snp_fwdin_tag      (per_cluster_snp_rsp_tag),
        .snp_fwdin_ready    (per_cluster_snp_rsp_ready)      
    );

    if (`L3_ENABLE) begin
    `ifdef PERF_ENABLE
        VX_perf_cache_if perf_l3cache_if();
    `endif

        wire [`NUM_CLUSTERS-1:0]                        per_cluster_dram_req_valid_qual;
        wire [`NUM_CLUSTERS-1:0]                        per_cluster_dram_req_rw_qual;    
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_BYTEEN_WIDTH-1:0] per_cluster_dram_req_byteen_qual;    
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_ADDR_WIDTH-1:0] per_cluster_dram_req_addr_qual;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_req_data_qual;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0] per_cluster_dram_req_tag_qual;
        wire [`NUM_CLUSTERS-1:0]                        per_cluster_dram_req_ready_qual;

        wire [`NUM_CLUSTERS-1:0]                        per_cluster_dram_rsp_valid_unqual;            
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_LINE_WIDTH-1:0] per_cluster_dram_rsp_data_unqual;
        wire [`NUM_CLUSTERS-1:0][`L2DRAM_TAG_WIDTH-1:0] per_cluster_dram_rsp_tag_unqual;
        wire [`NUM_CLUSTERS-1:0]                        per_cluster_dram_rsp_ready_unqual;

        for (genvar i = 0; i < `NUM_CLUSTERS; i++) begin 
            VX_skid_buffer #(
                .DATAW    (1 + `L2DRAM_BYTEEN_WIDTH + `L2DRAM_ADDR_WIDTH + `L2DRAM_LINE_WIDTH + `L2DRAM_TAG_WIDTH),
                .PASSTHRU (`NUM_CLUSTERS < 4)
            ) dram_req_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (per_cluster_dram_req_valid[i]),        
                .data_in   ({per_cluster_dram_req_rw[i], per_cluster_dram_req_byteen[i], per_cluster_dram_req_addr[i], per_cluster_dram_req_data[i], per_cluster_dram_req_tag[i]}),
                .ready_in  (per_cluster_dram_req_ready[i]),        
                .valid_out (per_cluster_dram_req_valid_qual[i]),
                .data_out  ({per_cluster_dram_req_rw_qual[i], per_cluster_dram_req_byteen_qual[i], per_cluster_dram_req_addr_qual[i], per_cluster_dram_req_data_qual[i], per_cluster_dram_req_tag_qual[i]}),
                .ready_out (per_cluster_dram_req_ready_qual[i])
            );

            VX_skid_buffer #(
                .DATAW    (`L2DRAM_LINE_WIDTH + `L2DRAM_TAG_WIDTH),
                .PASSTHRU (1)
            ) core_rsp_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (per_cluster_dram_rsp_valid_unqual[i]),        
                .data_in   ({per_cluster_dram_rsp_data_unqual[i], per_cluster_dram_rsp_tag_unqual[i]}),
                .ready_in  (per_cluster_dram_rsp_ready_unqual[i]),        
                .valid_out (per_cluster_dram_rsp_valid[i]),
                .data_out  ({per_cluster_dram_rsp_data[i], per_cluster_dram_rsp_tag[i]}),
                .ready_out (per_cluster_dram_rsp_ready[i])
            );
        end

        VX_cache #(
            .CACHE_ID           (`L3CACHE_ID),
            .CACHE_SIZE         (`L3CACHE_SIZE),
            .BANK_LINE_SIZE     (`L3BANK_LINE_SIZE),
            .NUM_BANKS          (`L3NUM_BANKS),
            .WORD_SIZE          (`L3WORD_SIZE),
            .NUM_REQS           (`NUM_CLUSTERS),
            .CREQ_SIZE          (`L3CREQ_SIZE),
            .MSHR_SIZE          (`L3MSHR_SIZE),
            .DRSQ_SIZE          (`L3DRSQ_SIZE),
            .SREQ_SIZE          (`L3SREQ_SIZE),
            .CRSQ_SIZE          (`L3CRSQ_SIZE),
            .DREQ_SIZE          (`L3DREQ_SIZE),
            .SRSQ_SIZE          (`L3SRSQ_SIZE),
            .DRAM_ENABLE        (1),
            .FLUSH_ENABLE       (1), 
            .WRITE_ENABLE       (1),
            .CORE_TAG_WIDTH     (`L2DRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .DRAM_TAG_WIDTH     (`L3DRAM_TAG_WIDTH),
            .SNP_TAG_WIDTH      (`L3SNP_TAG_WIDTH)
        ) l3cache (
            `SCOPE_BIND_Vortex_l3cache
 
            .clk                (clk),
            .reset              (reset),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_l3cache_if),
        `endif

            // Core request    
            .core_req_valid     (per_cluster_dram_req_valid_qual),
            .core_req_rw        (per_cluster_dram_req_rw_qual),
            .core_req_byteen    (per_cluster_dram_req_byteen_qual),
            .core_req_addr      (per_cluster_dram_req_addr_qual),
            .core_req_data      (per_cluster_dram_req_data_qual),
            .core_req_tag       (per_cluster_dram_req_tag_qual),
            .core_req_ready     (per_cluster_dram_req_ready_qual),

            // Core response
            .core_rsp_valid     (per_cluster_dram_rsp_valid_unqual),
            .core_rsp_data      (per_cluster_dram_rsp_data_unqual),
            .core_rsp_tag       (per_cluster_dram_rsp_tag_unqual),              
            .core_rsp_ready     (per_cluster_dram_rsp_ready_unqual),

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
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_ready     (dram_rsp_ready),

            // Snoop request
            .snp_req_valid      (snp_fwd_rsp_valid),
            .snp_req_addr       (snp_fwd_rsp_addr),
            .snp_req_inv        (snp_fwd_rsp_inv),
            .snp_req_tag        (snp_fwd_rsp_tag),
            .snp_req_ready      (snp_fwd_rsp_ready),

            // Snoop response
            .snp_rsp_valid      (snp_rsp_valid),
            .snp_rsp_tag        (snp_rsp_tag),
            .snp_rsp_ready      (snp_rsp_ready),

            // Miss status
            `UNUSED_PIN (miss_vec)
        );

    end else begin

        VX_mem_arb #(
            .NUM_REQS      (`NUM_CLUSTERS),
            .DATA_WIDTH    (`L3DRAM_LINE_WIDTH),            
            .TAG_IN_WIDTH  (`L2DRAM_TAG_WIDTH),
            .TAG_OUT_WIDTH (`L3DRAM_TAG_WIDTH)
        ) dram_arb (
            .clk            (clk),
            .reset          (reset),

            // Core request
            .req_valid_in   (per_cluster_dram_req_valid),
            .req_rw_in      (per_cluster_dram_req_rw),
            .req_byteen_in  (per_cluster_dram_req_byteen),
            .req_addr_in    (per_cluster_dram_req_addr),
            .req_data_in    (per_cluster_dram_req_data),  
            .req_tag_in     (per_cluster_dram_req_tag),  
            .req_ready_in   (per_cluster_dram_req_ready),

            // DRAM request
            .req_valid_out  (dram_req_valid),
            .req_rw_out     (dram_req_rw),        
            .req_byteen_out (dram_req_byteen),        
            .req_addr_out   (dram_req_addr),
            .req_data_out   (dram_req_data),
            .req_tag_out    (dram_req_tag),
            .req_ready_out  (dram_req_ready),

            // Core response
            .rsp_valid_out  (per_cluster_dram_rsp_valid),
            .rsp_data_out   (per_cluster_dram_rsp_data),
            .rsp_tag_out    (per_cluster_dram_rsp_tag),
            .rsp_ready_out  (per_cluster_dram_rsp_ready),
            
            // DRAM response
            .rsp_valid_in   (dram_rsp_valid),
            .rsp_tag_in     (dram_rsp_tag),
            .rsp_data_in    (dram_rsp_data),
            .rsp_ready_in   (dram_rsp_ready)
        );

        `UNUSED_VAR (snp_fwd_rsp_addr)
        `UNUSED_VAR (snp_fwd_rsp_inv)
        
        assign snp_rsp_valid = snp_fwd_rsp_valid;
        assign snp_rsp_tag   = snp_fwd_rsp_tag;
        assign snp_fwd_rsp_ready = snp_rsp_ready;

    end

    `SCOPE_ASSIGN (reset, reset);

    `SCOPE_ASSIGN (dram_req_fire,  dram_req_valid && dram_req_ready);
    `SCOPE_ASSIGN (dram_req_addr,  `TO_FULL_ADDR(dram_req_addr));
    `SCOPE_ASSIGN (dram_req_rw,    dram_req_rw);
    `SCOPE_ASSIGN (dram_req_byteen,dram_req_byteen);
    `SCOPE_ASSIGN (dram_req_data,  dram_req_data);
    `SCOPE_ASSIGN (dram_req_tag,   dram_req_tag);

    `SCOPE_ASSIGN (dram_rsp_fire,  dram_rsp_valid && dram_rsp_ready);
    `SCOPE_ASSIGN (dram_rsp_data,  dram_rsp_data);
    `SCOPE_ASSIGN (dram_rsp_tag,   dram_rsp_tag);

    `SCOPE_ASSIGN (snp_req_fire,  snp_req_valid && snp_req_ready);
    `SCOPE_ASSIGN (snp_req_addr,  `TO_FULL_ADDR(snp_req_addr));
    `SCOPE_ASSIGN (snp_req_inv,   snp_req_inv);
    `SCOPE_ASSIGN (snp_req_tag,   snp_req_tag);

    `SCOPE_ASSIGN (snp_rsp_fire,  snp_rsp_valid && snp_rsp_ready);
    `SCOPE_ASSIGN (snp_rsp_tag,   snp_rsp_tag);

    `SCOPE_ASSIGN (snp_rsp_fire,  snp_rsp_valid && snp_rsp_ready);
    `SCOPE_ASSIGN (snp_rsp_tag,   snp_rsp_tag);

    `SCOPE_ASSIGN (busy, busy);

`ifdef DBG_PRINT_DRAM
    always @(posedge clk) begin
        if (dram_req_valid && dram_req_ready) begin
            if (dram_req_rw)
                $display("%t: DRAM Wr Req: addr=%0h, tag=%0h, byteen=%0h data=%0h", $time, `TO_FULL_ADDR(dram_req_addr), dram_req_tag, dram_req_byteen, dram_req_data);
            else
                $display("%t: DRAM Rd Req: addr=%0h, tag=%0h, byteen=%0h", $time, `TO_FULL_ADDR(dram_req_addr), dram_req_tag, dram_req_byteen);
        end
        if (dram_rsp_valid && dram_rsp_ready) begin
            $display("%t: DRAM Rsp: tag=%0h, data=%0h", $time, dram_rsp_tag, dram_rsp_data);
        end
    end
`endif


`ifndef NDEBUG
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule