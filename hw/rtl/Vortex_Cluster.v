`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex_Cluster #(
    parameter CLUSTER_ID = 0
) ( 
    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // IO
    output wire[`NUM_CORES-1:0]         io_valid,
    output wire[`NUM_CORES-1:0][31:0]   io_data,

    // DRAM Req
    output wire                         dram_req_read,
    output wire                         dram_req_write,    
    output wire[`L2DRAM_ADDR_WIDTH-1:0] dram_req_addr,
    output wire[`L2DRAM_LINE_WIDTH-1:0] dram_req_data,
    output wire[`L2DRAM_TAG_WIDTH-1:0]  dram_req_tag,
    input  wire                         dram_req_ready,

    // DRAM Rsp    
    input  wire                         dram_rsp_valid,
    input  wire[`L2DRAM_LINE_WIDTH-1:0] dram_rsp_data,
    input  wire[`L2DRAM_TAG_WIDTH-1:0]  dram_rsp_tag,
    output wire                         dram_rsp_ready,

    // LLC Snooping
    input  wire                         llc_snp_req_valid,
    input  wire[`L2DRAM_ADDR_WIDTH-1:0] llc_snp_req_addr,
    output wire                         llc_snp_req_ready,

    output wire                         ebreak
);
    if (`NUM_CORES == 1) begin

        VX_cache_dram_req_if #(
            .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
            .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
            .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
        ) dcache_dram_req_if();

        VX_cache_dram_rsp_if #(
            .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
            .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
        ) dcache_dram_rsp_if();

        VX_cache_dram_req_if #(
            .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
            .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
            .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
        ) icache_dram_req_if();

        VX_cache_dram_rsp_if #(
            .DRAM_LINE_WIDTH(`IDRAM_LINE_WIDTH),
            .DRAM_TAG_WIDTH(`IDRAM_TAG_WIDTH)
        ) icache_dram_rsp_if();

        VX_cache_dram_req_if #(
            .DRAM_LINE_WIDTH(`L2DRAM_LINE_WIDTH),
            .DRAM_ADDR_WIDTH(`L2DRAM_ADDR_WIDTH),
            .DRAM_TAG_WIDTH(`L2DRAM_TAG_WIDTH)
        ) dram_req_if();

        VX_cache_dram_rsp_if #(
            .DRAM_LINE_WIDTH(`L2DRAM_LINE_WIDTH),
            .DRAM_TAG_WIDTH(`L2DRAM_TAG_WIDTH)
        ) dram_rsp_if();

        assign dram_req_read  = dram_req_if.dram_req_read;
        assign dram_req_write = dram_req_if.dram_req_write;
        assign dram_req_addr  = dram_req_if.dram_req_addr;
        assign dram_req_data  = dram_req_if.dram_req_data;
        assign dram_req_tag   = dram_req_if.dram_req_tag;
        assign dram_req_if.dram_req_ready = dram_req_ready;

        assign dram_rsp_if.dram_rsp_valid = dram_rsp_valid;        
        assign dram_rsp_if.dram_rsp_data  = dram_rsp_data;
        assign dram_rsp_if.dram_rsp_tag   = dram_rsp_tag;
        assign dram_rsp_ready = dram_rsp_if.dram_rsp_ready;

        VX_l1c_to_dram_arb #(
            .REQQ_SIZE(`L2REQQ_SIZE)
        ) l1c_to_dram_arb (
            .clk                (clk),
            .reset              (reset),
            .dcache_dram_req_if (dcache_dram_req_if),
            .dcache_dram_rsp_if (dcache_dram_rsp_if),
            .icache_dram_req_if (icache_dram_req_if),
            .icache_dram_rsp_if (icache_dram_rsp_if),
            .dram_req_if        (dram_req_if),
            .dram_rsp_if        (dram_rsp_if)
        );

        Vortex #(
            .CORE_ID(0)
        ) vortex_core (
            .clk                (clk),
            .reset              (reset),

            .io_valid           (io_valid[0]),
            .io_data            (io_data[0]),
            
            .D_dram_req_read    (dcache_dram_req_if.dram_req_read),
            .D_dram_req_write   (dcache_dram_req_if.dram_req_write),                
            .D_dram_req_addr    (dcache_dram_req_if.dram_req_addr),
            .D_dram_req_data    (dcache_dram_req_if.dram_req_data),
            .D_dram_req_tag     (dcache_dram_req_if.dram_req_tag),
            .D_dram_req_ready   (dcache_dram_req_if.dram_req_ready),                

            .D_dram_rsp_valid   (dcache_dram_rsp_if.dram_rsp_valid),            
            .D_dram_rsp_data    (dcache_dram_rsp_if.dram_rsp_data),
            .D_dram_rsp_tag     (dcache_dram_rsp_if.dram_rsp_tag),
            .D_dram_rsp_ready   (dcache_dram_rsp_if.dram_rsp_ready),
            
            .I_dram_req_read    (icache_dram_req_if.dram_req_read),
            .I_dram_req_write   (icache_dram_req_if.dram_req_write),                
            .I_dram_req_addr    (icache_dram_req_if.dram_req_addr),
            .I_dram_req_data    (icache_dram_req_if.dram_req_data),
            .I_dram_req_tag     (icache_dram_req_if.dram_req_tag),
            .I_dram_req_ready   (icache_dram_req_if.dram_req_ready),                

            .I_dram_rsp_valid   (icache_dram_rsp_if.dram_rsp_valid),            
            .I_dram_rsp_data    (icache_dram_rsp_if.dram_rsp_data),
            .I_dram_rsp_ready   (icache_dram_rsp_if.dram_rsp_ready),
            .I_dram_rsp_tag     (icache_dram_rsp_if.dram_rsp_tag),                    
            
            .llc_snp_req_valid  (llc_snp_req_valid),
            .llc_snp_req_addr   (llc_snp_req_addr),
            .llc_snp_req_ready  (llc_snp_req_ready),

            .ebreak             (ebreak)
        );

    end else begin

        // DRAM Dcache Req
        wire[`NUM_CORES-1:0]                        per_core_D_dram_req_read;
        wire[`NUM_CORES-1:0]                        per_core_D_dram_req_write;    
        wire[`NUM_CORES-1:0][`DDRAM_ADDR_WIDTH-1:0] per_core_D_dram_req_addr;
        wire[`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_D_dram_req_data;
        wire[`NUM_CORES-1:0][`DDRAM_TAG_WIDTH-1:0]  per_core_D_dram_req_tag;

        // DRAM Dcache Rsp
        wire[`NUM_CORES-1:0]                        per_core_D_dram_rsp_valid;            
        wire[`NUM_CORES-1:0][`DDRAM_LINE_WIDTH-1:0] per_core_D_dram_rsp_data;
        wire[`NUM_CORES-1:0][`DDRAM_TAG_WIDTH-1:0]  per_core_D_dram_rsp_tag;
        wire[`NUM_CORES-1:0]                        per_core_D_dram_rsp_ready;

        // DRAM Icache Req
        wire[`NUM_CORES-1:0]                        per_core_I_dram_req_read;
        wire[`NUM_CORES-1:0]                        per_core_I_dram_req_write;    
        wire[`NUM_CORES-1:0][`IDRAM_ADDR_WIDTH-1:0] per_core_I_dram_req_addr;
        wire[`NUM_CORES-1:0][`IDRAM_LINE_WIDTH-1:0] per_core_I_dram_req_data;
        wire[`NUM_CORES-1:0][`IDRAM_TAG_WIDTH-1:0]  per_core_I_dram_req_tag;

        // DRAM Icache Rsp    
        wire[`NUM_CORES-1:0]                        per_core_I_dram_rsp_valid;        
        wire[`NUM_CORES-1:0][`IDRAM_LINE_WIDTH-1:0] per_core_I_dram_rsp_data;        
        wire[`NUM_CORES-1:0][`IDRAM_TAG_WIDTH-1:0]  per_core_I_dram_rsp_tag;
        wire[`NUM_CORES-1:0]                        per_core_I_dram_rsp_ready;

        // Out ebreak
        wire[`NUM_CORES-1:0]                        per_core_ebreak;

        wire[`NUM_CORES-1:0]                        per_core_io_valid;
        wire[`NUM_CORES-1:0][31:0]                  per_core_io_data;

        wire                                        l2_core_req_ready;

        wire                                        snp_fwd_valid;
        wire[`DDRAM_ADDR_WIDTH-1:0]                 snp_fwd_addr;
        wire[`NUM_CORES-1:0]                        per_core_snp_fwd_ready;

        assign ebreak = (& per_core_ebreak);

        genvar i;
        for (i = 0; i < `NUM_CORES; i = i + 1) begin

            wire [`IDRAM_LINE_WIDTH-1:0] curr_core_D_dram_req_data;
            wire [`DDRAM_LINE_WIDTH-1:0] curr_core_I_dram_req_data;

            assign io_valid[i] = per_core_io_valid[i];
            assign io_data[i] = per_core_io_data[i];

            Vortex #(
                .CORE_ID(i + (CLUSTER_ID * `NUM_CORES))
            ) vortex_core (
                .clk                (clk),
                .reset              (reset),
                .io_valid           (per_core_io_valid            [i]),
                .io_data            (per_core_io_data             [i]),
                .D_dram_req_read    (per_core_D_dram_req_read     [i]),
                .D_dram_req_write   (per_core_D_dram_req_write    [i]),                
                .D_dram_req_addr    (per_core_D_dram_req_addr     [i]),
                .D_dram_req_data    (curr_core_D_dram_req_data       ),
                .D_dram_req_tag     (per_core_D_dram_req_tag      [i]),
                .D_dram_req_ready   (l2_core_req_ready               ),                
                .D_dram_rsp_valid   (per_core_D_dram_rsp_valid    [i]),                
                .D_dram_rsp_data    (per_core_D_dram_rsp_data     [i]),
                .D_dram_rsp_tag     (per_core_D_dram_rsp_tag      [i]),
                .D_dram_rsp_ready   (per_core_D_dram_rsp_ready    [i]),                
                .I_dram_req_read    (per_core_I_dram_req_read     [i]),
                .I_dram_req_write   (per_core_I_dram_req_write    [i]),                
                .I_dram_req_addr    (per_core_I_dram_req_addr     [i]),                
                .I_dram_req_data    (curr_core_I_dram_req_data       ),
                .I_dram_req_tag     (per_core_I_dram_req_tag      [i]),                
                .I_dram_req_ready   (l2_core_req_ready               ),                
                .I_dram_rsp_valid   (per_core_I_dram_rsp_valid    [i]),
                .I_dram_rsp_tag     (per_core_I_dram_rsp_tag      [i]),
                .I_dram_rsp_data    (per_core_I_dram_rsp_data     [i]),
                .I_dram_rsp_ready   (per_core_I_dram_rsp_ready    [i]),                                
                .llc_snp_req_valid  (snp_fwd_valid),
                .llc_snp_req_addr   (snp_fwd_addr),
                .llc_snp_req_ready  (per_core_snp_fwd_ready       [i]),
                .ebreak             (per_core_ebreak              [i])
            );

            assign per_core_D_dram_req_data [i] = curr_core_D_dram_req_data;
            assign per_core_I_dram_req_data [i] = curr_core_I_dram_req_data;
        end

        // L2 Cache ///////////////////////////////////////////////////////////
        
        wire[`L2NUM_REQUESTS-1:0]                           l2_core_req_valid;
        wire[`L2NUM_REQUESTS-1:0][`WORD_SEL_BITS-1:0]       l2_core_req_mem_write;
        wire[`L2NUM_REQUESTS-1:0][`WORD_SEL_BITS-1:0]       l2_core_req_mem_read;
        wire[`L2NUM_REQUESTS-1:0][31:0]                     l2_core_req_addr;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]     l2_core_req_tag;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0]    l2_core_req_data;

        wire[`L2NUM_REQUESTS-1:0]                           l2_core_rsp_valid;        
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_LINE_WIDTH-1:0]    l2_core_rsp_data;
        wire[`L2NUM_REQUESTS-1:0][`DDRAM_TAG_WIDTH-1:0]     l2_core_rsp_tag;
        wire[`L2NUM_REQUESTS-1:0]                           l2_core_rsp_ready;

        wire[`DDRAM_LINE_WIDTH-1:0]                         l2_dram_req_data;
        wire[`DDRAM_LINE_WIDTH-1:0]                         l2_dram_rsp_data;

        assign dram_req_data = l2_dram_req_data;
        assign l2_dram_rsp_data = dram_rsp_data;

        for (i = 0; i < `L2NUM_REQUESTS; i = i + 2) begin
            // Core Request
            assign l2_core_req_valid     [i]   = (per_core_D_dram_req_read[(i/2)] | per_core_D_dram_req_write[(i/2)]);
            assign l2_core_req_valid     [i+1] = (per_core_I_dram_req_read[(i/2)] | per_core_I_dram_req_write[(i/2)]);
            
            assign l2_core_req_mem_write [i]   = per_core_D_dram_req_write[(i/2)] ? `WORD_SEL_LW : `WORD_SEL_NO;
            assign l2_core_req_mem_write [i+1] = `WORD_SEL_NO;

            assign l2_core_req_mem_read  [i]   = per_core_D_dram_req_read[(i/2)] ? `WORD_SEL_LW : `WORD_SEL_NO;
            assign l2_core_req_mem_read  [i+1] = `WORD_SEL_NO;

            assign l2_core_req_addr      [i]   = {per_core_D_dram_req_addr[(i/2)], {`LOG2UP(`DBANK_LINE_SIZE){1'b0}}};
            assign l2_core_req_addr      [i+1] = {per_core_I_dram_req_addr[(i/2)], {`LOG2UP(`IBANK_LINE_SIZE){1'b0}}};

            assign l2_core_req_data      [i]   = per_core_D_dram_req_data[(i/2)];
            assign l2_core_req_data      [i+1] = per_core_I_dram_req_data[(i/2)];

            assign l2_core_req_tag       [i]   = per_core_D_dram_req_tag[(i/2)];
            assign l2_core_req_tag       [i+1] = per_core_I_dram_req_tag[(i/2)];

            assign per_core_D_dram_rsp_valid [(i/2)] = l2_core_rsp_valid[i];
            assign per_core_I_dram_rsp_valid [(i/2)] = l2_core_rsp_valid[i+1];

            assign per_core_D_dram_rsp_data  [(i/2)] = l2_core_rsp_data[i];
            assign per_core_I_dram_rsp_data  [(i/2)] = l2_core_rsp_data[i+1];

            assign per_core_D_dram_rsp_tag   [(i/2)] = l2_core_rsp_tag[i];
            assign per_core_I_dram_rsp_tag   [(i/2)] = l2_core_rsp_tag[i+1];

            assign l2_core_rsp_ready         [i]   = per_core_D_dram_rsp_ready  [(i/2)];
            assign l2_core_rsp_ready         [i+1] = per_core_I_dram_rsp_ready[(i/2)];       
        end

        VX_cache #(
            .CACHE_SIZE             (`L2CACHE_SIZE),
            .BANK_LINE_SIZE         (`L2BANK_LINE_SIZE),
            .NUM_BANKS              (`L2NUM_BANKS),
            .WORD_SIZE              (`L2WORD_SIZE),
            .NUM_REQUESTS           (`L2NUM_REQUESTS),
            .STAGE_1_CYCLES         (`L2STAGE_1_CYCLES),
            .FUNC_ID                (`L2FUNC_ID),
            .REQQ_SIZE              (`L2REQQ_SIZE),
            .MRVQ_SIZE              (`L2MRVQ_SIZE),
            .DFPQ_SIZE              (`L2DFPQ_SIZE),
            .SNRQ_SIZE              (`L2SNRQ_SIZE),
            .CWBQ_SIZE              (`L2CWBQ_SIZE),
            .DWBQ_SIZE              (`L2DWBQ_SIZE),
            .DFQQ_SIZE              (`L2DFQQ_SIZE),
            .LLVQ_SIZE              (`L2LLVQ_SIZE),
            .FFSQ_SIZE              (`L2FFSQ_SIZE),
            .PRFQ_SIZE              (`L2PRFQ_SIZE),
            .PRFQ_STRIDE            (`L2PRFQ_STRIDE),
            .FILL_INVALIDAOR_SIZE   (`L2FILL_INVALIDAOR_SIZE),
            .CORE_TAG_WIDTH         (`DDRAM_TAG_WIDTH),
            .DRAM_TAG_WIDTH         (`L2DRAM_TAG_WIDTH)
        ) gpu_l2cache (
            .clk                (clk),
            .reset              (reset),

            // Core request
            .core_req_valid     (l2_core_req_valid),
            .core_req_read      (l2_core_req_mem_read),
            .core_req_write     (l2_core_req_mem_write),
            .core_req_addr      (l2_core_req_addr),
            .core_req_data      (l2_core_req_data),  
            .core_req_tag       (l2_core_req_tag),  
            .core_req_ready     (l2_core_req_ready),

            // Core response
            .core_rsp_valid     (l2_core_rsp_valid),
            .core_rsp_data      (l2_core_rsp_data),
            .core_rsp_tag       (l2_core_rsp_tag),
            .core_rsp_ready     (|l2_core_rsp_ready),

            // DRAM request
            .dram_req_read      (dram_req_read),
            .dram_req_write     (dram_req_write),        
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (l2_dram_req_data),
            .dram_req_tag       (dram_req_tag),
            .dram_req_ready     (dram_req_ready),
            
            // L2 Cache DRAM Fill response
            .dram_rsp_valid     (dram_rsp_valid),
            .dram_rsp_tag       (dram_rsp_tag),
            .dram_rsp_data      (l2_dram_rsp_data),
            .dram_rsp_ready     (dram_rsp_ready),           

            // Snoop request
            .snp_req_valid      (llc_snp_req_valid),
            .snp_req_addr       (llc_snp_req_addr),
            .snp_req_ready      (llc_snp_req_ready),

            // Snoop forwarding    
            .snp_fwd_valid      (snp_fwd_valid),
            .snp_fwd_addr       (snp_fwd_addr),
            .snp_fwd_ready      (& per_core_snp_fwd_ready)
        );
    end

endmodule