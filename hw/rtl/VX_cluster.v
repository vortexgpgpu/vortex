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

    // Snoop request
    input wire                              snp_req_valid,
    input wire [`L2DRAM_ADDR_WIDTH-1:0]     snp_req_addr,
    input wire                              snp_req_invalidate,
    input wire [`L2SNP_TAG_WIDTH-1:0]       snp_req_tag,
    output wire                             snp_req_ready, 

    // Snoop response
    output wire                             snp_rsp_valid,
    output wire [`L2SNP_TAG_WIDTH-1:0]      snp_rsp_tag,
    input wire                              snp_rsp_ready,     

    // I/O request
    output wire [`NUM_THREADS-1:0]          io_req_valid,
    output wire                             io_req_rw,  
    output wire [`NUM_THREADS-1:0][3:0]     io_req_byteen,  
    output wire [`NUM_THREADS-1:0][29:0]    io_req_addr,
    output wire [`NUM_THREADS-1:0][31:0]    io_req_data,    
    output wire [`L2CORE_TAG_WIDTH-1:0]     io_req_tag,    
    input wire                              io_req_ready,

    // I/O response
    input wire                              io_rsp_valid,
    input wire [31:0]                       io_rsp_data,
    input wire [`L2CORE_TAG_WIDTH-1:0]      io_rsp_tag,
    output wire                             io_rsp_ready,

    // CSR I/O Request
    input  wire                             csr_io_req_valid,
    input  wire [`NC_BITS-1:0]              csr_io_req_coreid,
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
    wire [`NUM_CORES-1:0]                        per_core_D_dram_req_valid;
    wire [`NUM_CORES-1:0]                        per_core_D_dram_req_rw;    
    wire [`NUM_CORES-1:0][`DDRAM_BYTEEN_WIDTH-1:0] per_core_D_dram_req_byteen;    
    wire [`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0] per_core_D_dram_req_addr;
    wire [`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_D_dram_req_data;
    wire [`NUM_CORES-1:0][`DDRAM_TAG_WIDTH-1:0]  per_core_D_dram_req_tag;
    wire [`NUM_CORES-1:0]                        per_core_D_dram_req_ready;

    wire [`NUM_CORES-1:0]                        per_core_D_dram_rsp_valid;            
    wire [`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_D_dram_rsp_data;
    wire [`NUM_CORES-1:0][`DDRAM_TAG_WIDTH-1:0]  per_core_D_dram_rsp_tag;
    wire [`NUM_CORES-1:0]                        per_core_D_dram_rsp_ready;

    wire [`NUM_CORES-1:0]                        per_core_I_dram_req_valid;  
    wire [`NUM_CORES-1:0]                        per_core_I_dram_req_rw; 
    wire [`NUM_CORES-1:0][`IDRAM_BYTEEN_WIDTH-1:0] per_core_I_dram_req_byteen;    
    wire [`NUM_CORES-1:0][`IDRAM_ADDR_WIDTH-1:0] per_core_I_dram_req_addr;
    wire [`NUM_CORES-1:0][`IDRAM_LINE_WIDTH-1:0] per_core_I_dram_req_data;
    wire [`NUM_CORES-1:0][`IDRAM_TAG_WIDTH-1:0]  per_core_I_dram_req_tag;
    wire [`NUM_CORES-1:0]                        per_core_I_dram_req_ready;
   
    wire [`NUM_CORES-1:0]                        per_core_I_dram_rsp_valid;        
    wire [`NUM_CORES-1:0][`IDRAM_LINE_WIDTH-1:0] per_core_I_dram_rsp_data;        
    wire [`NUM_CORES-1:0][`IDRAM_TAG_WIDTH-1:0]  per_core_I_dram_rsp_tag;
    wire [`NUM_CORES-1:0]                        per_core_I_dram_rsp_ready;

    wire [`NUM_CORES-1:0]                        per_core_snp_req_valid;    
    wire [`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0] per_core_snp_req_addr;
    wire [`NUM_CORES-1:0]                        per_core_snp_req_invalidate;
    wire [`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]   per_core_snp_req_tag;
    wire [`NUM_CORES-1:0]                        per_core_snp_req_ready;
    
    wire [`NUM_CORES-1:0]                        per_core_snp_rsp_valid;
    wire [`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]   per_core_snp_rsp_tag;
    wire [`NUM_CORES-1:0]                        per_core_snp_rsp_ready;

    wire [`NUM_CORES-1:0][`NUM_THREADS-1:0]      per_core_io_req_valid;
    wire [`NUM_CORES-1:0]                        per_core_io_req_rw;
    wire [`NUM_CORES-1:0][`NUM_THREADS-1:0][3:0] per_core_io_req_byteen;
    wire [`NUM_CORES-1:0][`NUM_THREADS-1:0][29:0] per_core_io_req_addr;
    wire [`NUM_CORES-1:0][`NUM_THREADS-1:0][31:0] per_core_io_req_data;    
    wire [`NUM_CORES-1:0][`DCORE_TAG_WIDTH-1:0]  per_core_io_req_tag;
    wire [`NUM_CORES-1:0]                        per_core_io_req_ready;
    
    wire [`NUM_CORES-1:0]                        per_core_io_rsp_valid;
    wire [`NUM_CORES-1:0][`DCORE_TAG_WIDTH-1:0]  per_core_io_rsp_tag;
    wire [`NUM_CORES-1:0][31:0]                  per_core_io_rsp_data;
    wire [`NUM_CORES-1:0]                        per_core_io_rsp_ready;

    wire [`NUM_CORES-1:0]                        per_core_csr_io_req_valid;
    wire [`NUM_CORES-1:0][11:0]                  per_core_csr_io_req_addr;
    wire [`NUM_CORES-1:0]                        per_core_csr_io_req_rw;
    wire [`NUM_CORES-1:0][31:0]                  per_core_csr_io_req_data;
    wire [`NUM_CORES-1:0]                        per_core_csr_io_req_ready;

    wire [`NUM_CORES-1:0]                        per_core_csr_io_rsp_valid;
    wire [`NUM_CORES-1:0][31:0]                  per_core_csr_io_rsp_data;
    wire [`NUM_CORES-1:0]                        per_core_csr_io_rsp_ready;

    wire [`NUM_CORES-1:0]                        per_core_busy;
    wire [`NUM_CORES-1:0]                        per_core_ebreak;

    for (genvar i = 0; i < `NUM_CORES; i++) begin    
        VX_core #(
            .CORE_ID(i + (CLUSTER_ID * `NUM_CORES))
        ) core (
            `SCOPE_BIND_VX_cluster_core(i)

            .clk                (clk),
            .reset              (reset),

            .D_dram_req_valid   (per_core_D_dram_req_valid  [i]),
            .D_dram_req_rw      (per_core_D_dram_req_rw     [i]),                
            .D_dram_req_byteen  (per_core_D_dram_req_byteen [i]),                
            .D_dram_req_addr    (per_core_D_dram_req_addr   [i]),
            .D_dram_req_data    (per_core_D_dram_req_data   [i]),
            .D_dram_req_tag     (per_core_D_dram_req_tag    [i]),
            .D_dram_req_ready   (per_core_D_dram_req_ready  [i]),         
            .D_dram_rsp_valid   (per_core_D_dram_rsp_valid  [i]),                
            .D_dram_rsp_data    (per_core_D_dram_rsp_data   [i]),
            .D_dram_rsp_tag     (per_core_D_dram_rsp_tag    [i]),
            .D_dram_rsp_ready   (per_core_D_dram_rsp_ready  [i]),

            .I_dram_req_valid   (per_core_I_dram_req_valid  [i]),
            .I_dram_req_rw      (per_core_I_dram_req_rw     [i]),
            .I_dram_req_byteen  (per_core_I_dram_req_byteen [i]),
            .I_dram_req_addr    (per_core_I_dram_req_addr   [i]),                
            .I_dram_req_data    (per_core_I_dram_req_data   [i]),
            .I_dram_req_tag     (per_core_I_dram_req_tag    [i]),                
            .I_dram_req_ready   (per_core_I_dram_req_ready  [i]),          
            .I_dram_rsp_valid   (per_core_I_dram_rsp_valid  [i]),
            .I_dram_rsp_tag     (per_core_I_dram_rsp_tag    [i]),
            .I_dram_rsp_data    (per_core_I_dram_rsp_data   [i]),
            .I_dram_rsp_ready   (per_core_I_dram_rsp_ready  [i]),   

            .snp_req_valid      (per_core_snp_req_valid     [i]),
            .snp_req_addr       (per_core_snp_req_addr      [i]),
            .snp_req_invalidate (per_core_snp_req_invalidate[i]),
            .snp_req_tag        (per_core_snp_req_tag       [i]),
            .snp_req_ready      (per_core_snp_req_ready     [i]),

            .snp_rsp_valid      (per_core_snp_rsp_valid     [i]),
            .snp_rsp_tag        (per_core_snp_rsp_tag       [i]),
            .snp_rsp_ready      (per_core_snp_rsp_ready     [i]),

            .io_req_valid       (per_core_io_req_valid      [i]),
            .io_req_rw          (per_core_io_req_rw         [i]),
            .io_req_byteen      (per_core_io_req_byteen     [i]),
            .io_req_addr        (per_core_io_req_addr       [i]),
            .io_req_data        (per_core_io_req_data       [i]),            
            .io_req_tag         (per_core_io_req_tag        [i]),
            .io_req_ready       (per_core_io_req_ready      [i]),

            .io_rsp_valid       (per_core_io_rsp_valid      [i]),            
            .io_rsp_data        (per_core_io_rsp_data       [i]),
            .io_rsp_tag         (per_core_io_rsp_tag        [i]),
            .io_rsp_ready       (per_core_io_rsp_ready      [i]),

            .csr_io_req_valid   (per_core_csr_io_req_valid  [i]),
            .csr_io_req_rw      (per_core_csr_io_req_rw     [i]),
            .csr_io_req_addr    (per_core_csr_io_req_addr   [i]),
            .csr_io_req_data    (per_core_csr_io_req_data   [i]),
            .csr_io_req_ready   (per_core_csr_io_req_ready  [i]),

            .csr_io_rsp_valid   (per_core_csr_io_rsp_valid  [i]),            
            .csr_io_rsp_data    (per_core_csr_io_rsp_data   [i]),
            .csr_io_rsp_ready   (per_core_csr_io_rsp_ready  [i]),

            .busy               (per_core_busy              [i]),
            .ebreak             (per_core_ebreak            [i])
        );
    end     

    VX_io_arb #(
        .NUM_REQUESTS  (`NUM_CORES),
        .WORD_SIZE     (4),
        .TAG_IN_WIDTH  (`DCORE_TAG_WIDTH),
        .TAG_OUT_WIDTH (`L2CORE_TAG_WIDTH)
    ) io_arb (
        .clk                   (clk),
        .reset                 (reset),

        // input requests
        .io_req_valid_in       (per_core_io_req_valid),
        .io_req_rw_in          (per_core_io_req_rw),
        .io_req_byteen_in      (per_core_io_req_byteen),
        .io_req_addr_in        (per_core_io_req_addr),
        .io_req_data_in        (per_core_io_req_data),  
        .io_req_tag_in         (per_core_io_req_tag),  
        .io_req_ready_in       (per_core_io_req_ready),

        // input responses
        .io_rsp_valid_in       (per_core_io_rsp_valid),
        .io_rsp_data_in        (per_core_io_rsp_data),
        .io_rsp_tag_in         (per_core_io_rsp_tag),
        .io_rsp_ready_in       (per_core_io_rsp_ready),

        // output request
        .io_req_valid_out      (io_req_valid),
        .io_req_rw_out         (io_req_rw),        
        .io_req_byteen_out     (io_req_byteen),        
        .io_req_addr_out       (io_req_addr),
        .io_req_data_out       (io_req_data),
        .io_req_tag_out        (io_req_tag),
        .io_req_ready_out      (io_req_ready),
         
        // output response
        .io_rsp_valid_out      (io_rsp_valid),
        .io_rsp_tag_out        (io_rsp_tag),
        .io_rsp_data_out       (io_rsp_data),
        .io_rsp_ready_out      (io_rsp_ready)
    );   

    VX_csr_io_arb #(
        .NUM_REQUESTS (`NUM_CORES)
    ) csr_io_arb (
        .clk                    (clk),
        .reset                  (reset),

        .request_id             (csr_io_req_coreid), 

        // input requests
        .csr_io_req_valid_in    (csr_io_req_valid),     
        .csr_io_req_addr_in     (csr_io_req_addr),
        .csr_io_req_rw_in       (csr_io_req_rw),
        .csr_io_req_data_in     (csr_io_req_data),
        .csr_io_req_ready_in    (csr_io_req_ready),

        // input responses
        .csr_io_rsp_valid_in    (per_core_csr_io_rsp_valid),
        .csr_io_rsp_data_in     (per_core_csr_io_rsp_data),
        .csr_io_rsp_ready_in    (per_core_csr_io_rsp_ready),

        // output request
        .csr_io_req_valid_out   (per_core_csr_io_req_valid),
        .csr_io_req_addr_out    (per_core_csr_io_req_addr),            
        .csr_io_req_rw_out      (per_core_csr_io_req_rw),
        .csr_io_req_data_out    (per_core_csr_io_req_data),  
        .csr_io_req_ready_out   (per_core_csr_io_req_ready),            
        
        // output response
        .csr_io_rsp_valid_out   (csr_io_rsp_valid),
        .csr_io_rsp_data_out    (csr_io_rsp_data),
        .csr_io_rsp_ready_out   (csr_io_rsp_ready)
    );
    
    assign busy = (| per_core_busy);
    assign ebreak = (| per_core_ebreak);

    if (`L2_ENABLE) begin

        // L2 Cache ///////////////////////////////////////////////////////////

        wire[`L2NUM_REQUESTS-1:0]                           core_dram_req_valid;
        wire[`L2NUM_REQUESTS-1:0]                           core_dram_req_rw;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_BYTEEN_WIDTH-1:0]  core_dram_req_byteen;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_ADDR_WIDTH-1:0]    core_dram_req_addr;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]     core_dram_req_tag;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0]    core_dram_req_data;
        wire                                                core_dram_req_ready;

        wire[`L2NUM_REQUESTS-1:0]                           core_dram_rsp_valid;        
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0]    core_dram_rsp_data;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]     core_dram_rsp_tag;
        wire                                                core_dram_rsp_ready;

        wire[`NUM_CORES-1:0]                                core_snp_fwdout_valid;
        wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0]         core_snp_fwdout_addr;
        wire[`NUM_CORES-1:0]                                core_snp_fwdout_invalidate;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]           core_snp_fwdout_tag;
        wire[`NUM_CORES-1:0]                                core_snp_fwdout_ready;    

        wire[`NUM_CORES-1:0]                                core_snp_fwdin_valid;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]           core_snp_fwdin_tag;
        wire[`NUM_CORES-1:0]                                core_snp_fwdin_ready;

        wire                                                snp_fwd_rsp_valid;
        wire [`L2DRAM_ADDR_WIDTH-1:0]                       snp_fwd_rsp_addr;
        wire                                                snp_fwd_rsp_invalidate;
        wire [`L2SNP_TAG_WIDTH-1:0]                         snp_fwd_rsp_tag;
        wire                                                snp_fwd_rsp_ready;

        reg [`L2NUM_REQUESTS-1:0] core_dram_rsp_ready_other;
        reg core_dram_rsp_ready_all;

        always @(*) begin            
            core_dram_rsp_ready_other = {`L2NUM_REQUESTS{1'b1}};
            core_dram_rsp_ready_all = 1'b1;

            for (integer i = 0; i < `L2NUM_REQUESTS; i++) begin                
                for (integer j = 0; j < `L2NUM_REQUESTS; j++) begin
                    if (i != j) begin
                        if (0 == (j & 1))
                            core_dram_rsp_ready_other[i] &= (per_core_D_dram_rsp_ready [(j/2)] | !core_dram_rsp_valid [j]);
                        else
                            core_dram_rsp_ready_other[i] &= (per_core_I_dram_rsp_ready [(j/2)] | !core_dram_rsp_valid [j]);
                    end
                end
                
                if (0 == (i & 1))
                    core_dram_rsp_ready_all &= (per_core_D_dram_rsp_ready [(i/2)] | !core_dram_rsp_valid [i]);
                else
                    core_dram_rsp_ready_all &= (per_core_I_dram_rsp_ready [(i/2)] | !core_dram_rsp_valid [i]);
            end
        end

        for (genvar i = 0; i < `L2NUM_REQUESTS; i = i + 2) begin
            assign core_dram_req_valid [i]   = per_core_D_dram_req_valid [(i/2)];
            assign core_dram_req_valid [i+1] = per_core_I_dram_req_valid [(i/2)];

            assign core_dram_req_rw  [i]   = per_core_D_dram_req_rw [(i/2)];
            assign core_dram_req_rw  [i+1] = per_core_I_dram_req_rw [(i/2)];
            
            assign core_dram_req_byteen [i]   = per_core_D_dram_req_byteen [(i/2)];
            assign core_dram_req_byteen [i+1] = per_core_I_dram_req_byteen [(i/2)];

            assign core_dram_req_addr  [i]   = per_core_D_dram_req_addr [(i/2)];
            assign core_dram_req_addr  [i+1] = per_core_I_dram_req_addr [(i/2)];

            assign core_dram_req_data  [i]   = per_core_D_dram_req_data [(i/2)];
            assign core_dram_req_data  [i+1] = per_core_I_dram_req_data [(i/2)];

            assign core_dram_req_tag   [i]   = per_core_D_dram_req_tag [(i/2)];
            assign core_dram_req_tag   [i+1] = per_core_I_dram_req_tag [(i/2)];

            assign per_core_D_dram_req_ready [(i/2)] = core_dram_req_ready;
            assign per_core_I_dram_req_ready [(i/2)] = core_dram_req_ready;

            assign per_core_D_dram_rsp_valid [(i/2)] = core_dram_rsp_valid[i] & core_dram_rsp_ready_other [i];
            assign per_core_I_dram_rsp_valid [(i/2)] = core_dram_rsp_valid[i+1] & core_dram_rsp_ready_other [i+1];

            assign per_core_D_dram_rsp_data  [(i/2)] = core_dram_rsp_data[i];
            assign per_core_I_dram_rsp_data  [(i/2)] = core_dram_rsp_data[i+1];

            assign per_core_D_dram_rsp_tag   [(i/2)] = core_dram_rsp_tag[i];
            assign per_core_I_dram_rsp_tag   [(i/2)] = core_dram_rsp_tag[i+1];  

            assign per_core_snp_req_valid      [(i/2)] = core_snp_fwdout_valid [(i/2)];
            assign per_core_snp_req_addr       [(i/2)] = core_snp_fwdout_addr [(i/2)];    
            assign per_core_snp_req_invalidate [(i/2)] = core_snp_fwdout_invalidate [(i/2)];    
            assign per_core_snp_req_tag        [(i/2)] = core_snp_fwdout_tag [(i/2)];
            assign core_snp_fwdout_ready       [(i/2)] = per_core_snp_req_ready[(i/2)];

            assign core_snp_fwdin_valid   [(i/2)] = per_core_snp_rsp_valid [(i/2)];
            assign core_snp_fwdin_tag     [(i/2)] = per_core_snp_rsp_tag [(i/2)];
            assign per_core_snp_rsp_ready [(i/2)] = core_snp_fwdin_ready [(i/2)];
        end

        assign core_dram_rsp_ready = core_dram_rsp_ready_all;

        VX_snp_forwarder #(
            .CACHE_ID           (`L2CACHE_ID),            
            .NUM_REQUESTS       (`NUM_CORES), 
            .SRC_ADDR_WIDTH     (`L2DRAM_ADDR_WIDTH), 
            .DST_ADDR_WIDTH     (`DDRAM_ADDR_WIDTH),             
            .SNP_TAG_WIDTH      (`L2SNP_TAG_WIDTH),
            .SNRQ_SIZE          (`L2SNRQ_SIZE)
        ) snp_forwarder (
            .clk                (clk),
            .reset              (reset),

            .snp_req_valid      (snp_req_valid),
            .snp_req_addr       (snp_req_addr),
            .snp_req_invalidate (snp_req_invalidate),
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            .snp_rsp_valid      (snp_fwd_rsp_valid),       
            .snp_rsp_addr       (snp_fwd_rsp_addr),
            .snp_rsp_invalidate (snp_fwd_rsp_invalidate),
            .snp_rsp_tag        (snp_fwd_rsp_tag),
            .snp_rsp_ready      (snp_fwd_rsp_ready),   

            .snp_fwdout_valid   (core_snp_fwdout_valid),
            .snp_fwdout_addr    (core_snp_fwdout_addr),
            .snp_fwdout_invalidate(core_snp_fwdout_invalidate),
            .snp_fwdout_tag     (core_snp_fwdout_tag),
            .snp_fwdout_ready   (core_snp_fwdout_ready),

            .snp_fwdin_valid    (core_snp_fwdin_valid),
            .snp_fwdin_tag      (core_snp_fwdin_tag),
            .snp_fwdin_ready    (core_snp_fwdin_ready)      
        );

        VX_cache #(
            .CACHE_ID           (`L2CACHE_ID),
            .CACHE_SIZE         (`L2CACHE_SIZE),
            .BANK_LINE_SIZE     (`L2BANK_LINE_SIZE),
            .NUM_BANKS          (`L2NUM_BANKS),
            .WORD_SIZE          (`L2WORD_SIZE),
            .NUM_REQUESTS       (`L2NUM_REQUESTS),
            .CREQ_SIZE          (`L2CREQ_SIZE),
            .MRVQ_SIZE          (`L2MRVQ_SIZE),
            .DRFQ_SIZE          (`L2DRFQ_SIZE),
            .SNRQ_SIZE          (`L2SNRQ_SIZE),
            .CWBQ_SIZE          (`L2CWBQ_SIZE),
            .DREQ_SIZE          (`L2DREQ_SIZE),
            .SNPQ_SIZE          (`L2SNPQ_SIZE),
            .DRAM_ENABLE        (1),
            .FLUSH_ENABLE       (1),
            .WRITE_ENABLE       (1),          
            .CORE_TAG_WIDTH     (`DDRAM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .DRAM_TAG_WIDTH     (`L2DRAM_TAG_WIDTH),
            .SNP_TAG_WIDTH      (`L2SNP_TAG_WIDTH)
        ) l2cache (
            `SCOPE_BIND_VX_cluster_l2cache
            
            .clk                (clk),
            .reset              (reset),

            // Core request
            .core_req_valid     (core_dram_req_valid),
            .core_req_rw        (core_dram_req_rw),
            .core_req_byteen    (core_dram_req_byteen),
            .core_req_addr      (core_dram_req_addr),
            .core_req_data      (core_dram_req_data),  
            .core_req_tag       (core_dram_req_tag),  
            .core_req_ready     (core_dram_req_ready),

            // Core response
            .core_rsp_valid     (core_dram_rsp_valid),
            .core_rsp_data      (core_dram_rsp_data),
            .core_rsp_tag       (core_dram_rsp_tag),
            .core_rsp_ready     (core_dram_rsp_ready),

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
            .dram_rsp_ready     (dram_rsp_ready),   

            // Snoop request
            .snp_req_valid      (snp_fwd_rsp_valid),
            .snp_req_addr       (snp_fwd_rsp_addr),
            .snp_req_invalidate (snp_fwd_rsp_invalidate),
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
    
        wire[`L2NUM_REQUESTS-1:0]                        core_dram_req_valid;
        wire[`L2NUM_REQUESTS-1:0]                        core_dram_req_rw;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_BYTEEN_WIDTH-1:0] core_dram_req_byteen;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_ADDR_WIDTH-1:0] core_dram_req_addr;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]  core_dram_req_tag;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0] core_dram_req_data;
        wire[`L2NUM_REQUESTS-1:0]                        core_dram_req_ready;

        wire[`L2NUM_REQUESTS-1:0]                        core_dram_rsp_valid;        
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0] core_dram_rsp_data;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]  core_dram_rsp_tag;
        wire[`L2NUM_REQUESTS-1:0]                        core_dram_rsp_ready;

        wire[`NUM_CORES-1:0]                             core_snp_fwdout_valid;
        wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0]      core_snp_fwdout_addr;
        wire[`NUM_CORES-1:0]                             core_snp_fwdout_invalidate;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]        core_snp_fwdout_tag;
        wire[`NUM_CORES-1:0]                             core_snp_fwdout_ready;    

        wire[`NUM_CORES-1:0]                             core_snp_fwdin_valid;
        wire[`NUM_CORES-1:0][`DSNP_TAG_WIDTH-1:0]        core_snp_fwdin_tag;
        wire[`NUM_CORES-1:0]                             core_snp_fwdin_ready;

        for (genvar i = 0; i < `L2NUM_REQUESTS; i = i + 2) begin            
            assign core_dram_req_valid [i]   = per_core_D_dram_req_valid[(i/2)];
            assign core_dram_req_valid [i+1] = per_core_I_dram_req_valid[(i/2)];

            assign core_dram_req_rw    [i]   = per_core_D_dram_req_rw[(i/2)];
            assign core_dram_req_rw    [i+1] = per_core_I_dram_req_rw[(i/2)];

            assign core_dram_req_byteen[i]   = per_core_D_dram_req_byteen[(i/2)];
            assign core_dram_req_byteen[i+1] = per_core_I_dram_req_byteen[(i/2)];

            assign core_dram_req_addr  [i]   = per_core_D_dram_req_addr[(i/2)];
            assign core_dram_req_addr  [i+1] = per_core_I_dram_req_addr[(i/2)];

            assign core_dram_req_data  [i]   = per_core_D_dram_req_data[(i/2)];
            assign core_dram_req_data  [i+1] = per_core_I_dram_req_data[(i/2)];

            assign core_dram_req_tag   [i]   = per_core_D_dram_req_tag[(i/2)];
            assign core_dram_req_tag   [i+1] = per_core_I_dram_req_tag[(i/2)];

            assign per_core_D_dram_req_ready [(i/2)] = core_dram_req_ready[i];
            assign per_core_I_dram_req_ready [(i/2)] = core_dram_req_ready[i+1];

            assign per_core_D_dram_rsp_valid [(i/2)] = core_dram_rsp_valid[i];
            assign per_core_I_dram_rsp_valid [(i/2)] = core_dram_rsp_valid[i+1];

            assign per_core_D_dram_rsp_data  [(i/2)] = core_dram_rsp_data[i];
            assign per_core_I_dram_rsp_data  [(i/2)] = core_dram_rsp_data[i+1];

            assign per_core_D_dram_rsp_tag   [(i/2)] = core_dram_rsp_tag[i];
            assign per_core_I_dram_rsp_tag   [(i/2)] = core_dram_rsp_tag[i+1];

            assign core_dram_rsp_ready [i]   = per_core_D_dram_rsp_ready[(i/2)];
            assign core_dram_rsp_ready [i+1] = per_core_I_dram_rsp_ready[(i/2)];  

            assign per_core_snp_req_valid      [(i/2)] = core_snp_fwdout_valid [(i/2)];
            assign per_core_snp_req_addr       [(i/2)] = core_snp_fwdout_addr [(i/2)];    
            assign per_core_snp_req_invalidate [(i/2)] = core_snp_fwdout_invalidate [(i/2)];   
            assign per_core_snp_req_tag        [(i/2)] = core_snp_fwdout_tag [(i/2)];
            assign core_snp_fwdout_ready       [(i/2)] = per_core_snp_req_ready[(i/2)];

            assign core_snp_fwdin_valid   [(i/2)] = per_core_snp_rsp_valid [(i/2)];
            assign core_snp_fwdin_tag     [(i/2)] = per_core_snp_rsp_tag [(i/2)];
            assign per_core_snp_rsp_ready [(i/2)] = core_snp_fwdin_ready [(i/2)];
        end

    if (`NUM_CORES > 1) begin
        VX_snp_forwarder #(
            .CACHE_ID       (`L2CACHE_ID),            
            .NUM_REQUESTS   (`NUM_CORES), 
            .SRC_ADDR_WIDTH (`L2DRAM_ADDR_WIDTH), 
            .DST_ADDR_WIDTH (`DDRAM_ADDR_WIDTH),             
            .SNP_TAG_WIDTH  (`L2SNP_TAG_WIDTH),
            .SNRQ_SIZE      (`L2SNRQ_SIZE)
        ) snp_forwarder (
            .clk                (clk),
            .reset              (reset),

            .snp_req_valid      (snp_req_valid),
            .snp_req_addr       (snp_req_addr),
            .snp_req_invalidate (snp_req_invalidate),
            .snp_req_tag        (snp_req_tag),
            .snp_req_ready      (snp_req_ready),

            .snp_rsp_valid      (snp_rsp_valid),       
            `UNUSED_PIN         (snp_rsp_addr),
            `UNUSED_PIN         (snp_rsp_invalidate),
            .snp_rsp_tag        (snp_rsp_tag),
            .snp_rsp_ready      (snp_rsp_ready),   

            .snp_fwdout_valid   (core_snp_fwdout_valid),
            .snp_fwdout_addr    (core_snp_fwdout_addr),
            .snp_fwdout_invalidate(core_snp_fwdout_invalidate),
            .snp_fwdout_tag     (core_snp_fwdout_tag),
            .snp_fwdout_ready   (core_snp_fwdout_ready),

            .snp_fwdin_valid    (core_snp_fwdin_valid),
            .snp_fwdin_tag      (core_snp_fwdin_tag),
            .snp_fwdin_ready    (core_snp_fwdin_ready)      
        );
    end else begin
        assign core_snp_fwdout_valid      = snp_req_valid;
        assign core_snp_fwdout_addr       = snp_req_addr;
        assign core_snp_fwdout_invalidate = snp_req_invalidate;
        assign core_snp_fwdout_tag        = snp_req_tag;
        assign snp_req_ready              = core_snp_fwdout_ready;
 
        assign snp_rsp_valid        = core_snp_fwdin_valid;
        assign snp_rsp_tag          = core_snp_fwdin_tag;
        assign core_snp_fwdin_ready = snp_rsp_ready;
    end

        VX_mem_arb #(
            .NUM_REQUESTS  (`L2NUM_REQUESTS),
            .DATA_WIDTH    (`L2DRAM_LINE_WIDTH),            
            .TAG_IN_WIDTH  (`DDRAM_TAG_WIDTH),
            .TAG_OUT_WIDTH (`L2DRAM_TAG_WIDTH)
        ) dram_arb (
            .clk            (clk),
            .reset          (reset),

            // Core request
            .req_valid_in   (core_dram_req_valid),
            .req_rw_in      (core_dram_req_rw),
            .req_byteen_in  (core_dram_req_byteen),
            .req_addr_in    (core_dram_req_addr),
            .req_data_in    (core_dram_req_data),  
            .req_tag_in     (core_dram_req_tag),  
            .req_ready_in   (core_dram_req_ready),

            // Core response
            .rsp_valid_out  (core_dram_rsp_valid),
            .rsp_data_out   (core_dram_rsp_data),
            .rsp_tag_out    (core_dram_rsp_tag),
            .rsp_ready_out  (core_dram_rsp_ready),

            // DRAM request
            .req_valid_out  (dram_req_valid),
            .req_rw_out     (dram_req_rw),        
            .req_byteen_out (dram_req_byteen),        
            .req_addr_out   (dram_req_addr),
            .req_data_out   (dram_req_data),
            .req_tag_out    (dram_req_tag),
            .req_ready_out  (dram_req_ready),
            
            // DRAM response
            .rsp_valid_in   (dram_rsp_valid),
            .rsp_tag_in     (dram_rsp_tag),
            .rsp_data_in    (dram_rsp_data),
            .rsp_ready_in   (dram_rsp_ready)
        );

    end

endmodule
