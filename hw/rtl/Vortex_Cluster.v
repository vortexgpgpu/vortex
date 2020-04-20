`include "VX_define.v"
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
    output wire [31:0]                  dram_req_addr,
    output wire [`DBANK_LINE_SIZE-1:0]  dram_req_data,
    input  wire                         dram_req_ready,

    // DRAM Rsp    
    input  wire                         dram_rsp_valid,
    input  wire [31:0]                  dram_rsp_addr,
    input  wire [`DBANK_LINE_SIZE-1:0]  dram_rsp_data,
    output wire                         dram_rsp_ready,

    // LLC Snooping
    input  wire                         llc_snp_req_valid,
    input  wire[31:0]                   llc_snp_req_addr,
    output wire                         llc_snp_req_ready,

    output wire                         ebreak
);
    // DRAM Dcache Req
    wire[`NUM_CORES-1:0]                              per_core_dram_req_read;
    wire[`NUM_CORES-1:0]                              per_core_dram_req_write;    
    wire[`NUM_CORES-1:0] [31:0]                       per_core_dram_req_addr;
    wire[`NUM_CORES-1:0][`DBANK_LINE_WORDS-1:0][31:0] per_core_dram_req_data;

    // DRAM Dcache Rsp
    wire[`NUM_CORES-1:0]                              per_core_dram_rsp_valid;    
    wire[`NUM_CORES-1:0] [31:0]                       per_core_dram_rsp_addr;
    wire[`NUM_CORES-1:0][`DBANK_LINE_WORDS-1:0][31:0] per_core_dram_rsp_data;
    wire[`NUM_CORES-1:0]                              per_core_dram_rsp_ready;

    // DRAM Icache Req
    wire[`NUM_CORES-1:0]                              per_core_I_dram_req_read;
    wire[`NUM_CORES-1:0]                              per_core_I_dram_req_write;    
    wire[`NUM_CORES-1:0] [31:0]                       per_core_I_dram_req_addr;
    wire[`NUM_CORES-1:0][`IBANK_LINE_WORDS-1:0][31:0] per_core_I_dram_req_data;

    // DRAM Icache Rsp    
    wire[`NUM_CORES-1:0]                              per_core_I_dram_rsp_valid;
    wire[`NUM_CORES-1:0] [31:0]                       per_core_I_dram_rsp_addr;
    wire[`NUM_CORES-1:0][`IBANK_LINE_WORDS-1:0][31:0] per_core_I_dram_rsp_data;
    wire[`NUM_CORES-1:0]                              per_core_I_dram_rsp_ready;

    // Out ebreak
    wire[`NUM_CORES-1:0]                              per_core_ebreak;

    wire[`NUM_CORES-1:0]                              per_core_io_valid;
    wire[`NUM_CORES-1:0][31:0]                        per_core_io_data;

    wire                                              l2c_core_req_ready;

    wire                                              snp_fwd_valid;
    wire[31:0]                                        snp_fwd_addr;
    wire[`NUM_CORES-1:0]                              snp_fwd_ready;

    assign ebreak = (&per_core_ebreak);

    genvar curr_core;
    generate

        for (curr_core = 0; curr_core < `NUM_CORES; curr_core=curr_core+1) begin

            wire [`IBANK_LINE_WORDS-1:0][31:0] curr_core_I_dram_req_data;
            wire [`DBANK_LINE_WORDS-1:0][31:0] curr_core_dram_req_data ;

            assign io_valid[curr_core]  = per_core_io_valid[curr_core];
            assign io_data [curr_core]  = per_core_io_data [curr_core];

            Vortex #(
                .CORE_ID(curr_core + (CLUSTER_ID * `NUM_CORES))
            ) vortex_core (
                .clk                        (clk),
                .reset                      (reset),
                .io_valid                   (per_core_io_valid            [curr_core]),
                .io_data                    (per_core_io_data             [curr_core]),
                .dram_req_read              (per_core_dram_req_read       [curr_core]),
                .dram_req_write             (per_core_dram_req_write      [curr_core]),                
                .dram_req_addr              (per_core_dram_req_addr       [curr_core]),
                .dram_req_data              (curr_core_dram_req_data                 ),
                .dram_req_ready             (l2c_core_req_ready                      ),                
                .dram_rsp_valid             (per_core_dram_rsp_valid      [curr_core]),
                .dram_rsp_addr              (per_core_dram_rsp_addr       [curr_core]),
                .dram_rsp_data              (per_core_dram_rsp_data       [curr_core]),
                .dram_rsp_ready             (per_core_dram_rsp_ready      [curr_core]),
                .I_dram_req_read            (per_core_I_dram_req_read     [curr_core]),
                .I_dram_req_write           (per_core_I_dram_req_write    [curr_core]),                
                .I_dram_req_addr            (per_core_I_dram_req_addr     [curr_core]),                
                .I_dram_req_data            (curr_core_I_dram_req_data               ),
                .I_dram_req_ready           (l2c_core_req_ready                      ),                
                .I_dram_rsp_valid           (per_core_I_dram_rsp_valid    [curr_core]),
                .I_dram_rsp_addr            (per_core_I_dram_rsp_addr     [curr_core]),
                .I_dram_rsp_data            (per_core_I_dram_rsp_data     [curr_core]),
                .I_dram_rsp_ready           (per_core_I_dram_rsp_ready    [curr_core]),                                
                .llc_snp_req_valid          (snp_fwd_valid),
                .llc_snp_req_addr           (snp_fwd_addr),
                .llc_snp_req_ready          (snp_fwd_ready                [curr_core]),
                .ebreak                     (per_core_ebreak              [curr_core])
            );

            assign per_core_dram_req_data  [curr_core] = curr_core_dram_req_data;
            assign per_core_I_dram_req_data[curr_core] = curr_core_I_dram_req_data;
        end
    endgenerate

    //////////////////// L2 Cache ////////////////////
    
    wire[`L2NUM_REQUESTS-1:0]                               l2c_core_req_valid;
    wire[`L2NUM_REQUESTS-1:0][2:0]                          l2c_core_req_mem_write;
    wire[`L2NUM_REQUESTS-1:0][2:0]                          l2c_core_req_mem_read;
    wire[`L2NUM_REQUESTS-1:0][31:0]                         l2c_core_req_addr;
    wire[`L2NUM_REQUESTS-1:0][`IBANK_LINE_WORDS-1:0][31:0]  l2c_core_req_data;
    wire[`L2NUM_REQUESTS-1:0][1:0]                          l2c_core_req_wb;

    wire[`L2NUM_REQUESTS-1:0]                               l2c_core_rsp_ready;

    wire[`L2NUM_REQUESTS-1:0]                               l2c_wb;
    wire[`L2NUM_REQUESTS-1:0] [31:0]                        l2c_wb_addr;
    wire[`L2NUM_REQUESTS-1:0][`IBANK_LINE_WORDS-1:0][31:0]  l2c_wb_data;

    wire[`DBANK_LINE_WORDS-1:0][31:0]                       dram_req_data_port;
    wire[`DBANK_LINE_WORDS-1:0][31:0]                       dram_rsp_data_port;

    genvar llb_index;
    generate
        for (llb_index = 0; llb_index < `DBANK_LINE_WORDS; llb_index=llb_index+1) begin
            assign dram_req_data      [llb_index * `DWORD_SIZE_BITS +: `DWORD_SIZE_BITS] = dram_req_data_port[llb_index];
            assign dram_rsp_data_port [llb_index] = dram_rsp_data[llb_index * `DWORD_SIZE_BITS +: `DWORD_SIZE_BITS];
        end
    endgenerate

    genvar l2c_curr_core;
    generate
        for (l2c_curr_core = 0; l2c_curr_core < `L2NUM_REQUESTS; l2c_curr_core=l2c_curr_core+2) begin
            // Core Request
            assign l2c_core_req_valid     [l2c_curr_core]   = (per_core_dram_req_read[(l2c_curr_core/2)] | per_core_dram_req_write[(l2c_curr_core/2)]);
            assign l2c_core_req_valid     [l2c_curr_core+1] = (per_core_I_dram_req_read[(l2c_curr_core/2)] | per_core_I_dram_req_write[(l2c_curr_core/2)]);
            
            assign l2c_core_req_mem_write [l2c_curr_core]   = per_core_dram_req_write[(l2c_curr_core/2)] ? `SW_MEM_WRITE : `NO_MEM_WRITE;
            assign l2c_core_req_mem_write [l2c_curr_core+1] = `NO_MEM_WRITE; // I caches don't write

            assign l2c_core_req_mem_read  [l2c_curr_core]   = per_core_dram_req_read[(l2c_curr_core/2)] ? `LW_MEM_READ : `NO_MEM_READ;
            assign l2c_core_req_mem_read  [l2c_curr_core+1] = `LW_MEM_READ; // I caches don't write

            assign l2c_core_req_wb        [l2c_curr_core]   = per_core_dram_req_read[(l2c_curr_core/2)] ? 1 : 0;
            assign l2c_core_req_wb        [l2c_curr_core+1] = 1; // I caches don't write

            assign l2c_core_req_addr      [l2c_curr_core]   = per_core_dram_req_addr  [(l2c_curr_core/2)];
            assign l2c_core_req_addr      [l2c_curr_core+1] = per_core_I_dram_req_addr[(l2c_curr_core/2)];

            assign l2c_core_req_data      [l2c_curr_core]   = per_core_dram_req_data  [(l2c_curr_core/2)];
            assign l2c_core_req_data      [l2c_curr_core+1] = per_core_I_dram_req_data[(l2c_curr_core/2)];

            // Core can't accept Response
            assign l2c_core_rsp_ready     [l2c_curr_core]   = per_core_dram_rsp_ready  [(l2c_curr_core/2)];
            assign l2c_core_rsp_ready     [l2c_curr_core+1] = per_core_I_dram_rsp_ready[(l2c_curr_core/2)];       

            // Cache Fill Response
            assign per_core_dram_rsp_valid      [(l2c_curr_core/2)] = l2c_wb[l2c_curr_core];
            assign per_core_I_dram_rsp_valid    [(l2c_curr_core/2)] = l2c_wb[l2c_curr_core+1];

            assign per_core_dram_rsp_data       [(l2c_curr_core/2)] = l2c_wb_data[l2c_curr_core];
            assign per_core_I_dram_rsp_data     [(l2c_curr_core/2)] = l2c_wb_data[l2c_curr_core+1];

            assign per_core_dram_rsp_addr       [(l2c_curr_core/2)] = l2c_wb_addr[l2c_curr_core];
            assign per_core_I_dram_rsp_addr     [(l2c_curr_core/2)] = l2c_wb_addr[l2c_curr_core+1];
        end
    endgenerate

    VX_cache #(
        .CACHE_SIZE_BYTES             (`L2CACHE_SIZE_BYTES),
        .BANK_LINE_SIZE_BYTES         (`L2BANK_LINE_SIZE_BYTES),
        .NUM_BANKS                    (`L2NUM_BANKS),
        .WORD_SIZE_BYTES              (`L2WORD_SIZE_BYTES),
        .NUM_REQUESTS                 (`L2NUM_REQUESTS),
        .STAGE_1_CYCLES               (`L2STAGE_1_CYCLES),
        .FUNC_ID                      (`L2FUNC_ID),
        .REQQ_SIZE                    (`L2REQQ_SIZE),
        .MRVQ_SIZE                    (`L2MRVQ_SIZE),
        .DFPQ_SIZE                    (`L2DFPQ_SIZE),
        .SNRQ_SIZE                    (`L2SNRQ_SIZE),
        .CWBQ_SIZE                    (`L2CWBQ_SIZE),
        .DWBQ_SIZE                    (`L2DWBQ_SIZE),
        .DFQQ_SIZE                    (`L2DFQQ_SIZE),
        .LLVQ_SIZE                    (`L2LLVQ_SIZE),
        .FFSQ_SIZE                    (`L2FFSQ_SIZE),
        .PRFQ_SIZE                    (`L2PRFQ_SIZE),
        .PRFQ_STRIDE                  (`L2PRFQ_STRIDE),
        .FILL_INVALIDAOR_SIZE         (`L2FILL_INVALIDAOR_SIZE),
        .SIMULATED_DRAM_LATENCY_CYCLES(`L2SIMULATED_DRAM_LATENCY_CYCLES)
    ) gpu_l2cache (
        .clk                (clk),
        .reset              (reset),

        // Core Req (DRAM Fills/WB) To L2 Request
        .core_req_valid     (l2c_core_req_valid),
        .core_req_read      (l2c_core_req_mem_read),
        .core_req_write     (l2c_core_req_mem_write),
        .core_req_addr      (l2c_core_req_addr),
        .core_req_data      ({l2c_core_req_data}),        
        .core_req_rd        (0),
        .core_req_wb        (l2c_core_req_wb),
        .core_req_warp_num  (0),
        .core_req_pc        (0),

        // L2 can't accept Core Request
        .core_req_ready     (l2c_core_req_ready),

        // Core can't accept L2 Request
        .core_rsp_ready     (|l2c_core_rsp_ready),

        // Core Writeback
        .core_rsp_valid     (l2c_wb),
    `IGNORE_WARNINGS_BEGIN
        .core_rsp_read      (),
        .core_rsp_write     (),
        .core_rsp_warp_num  (),
        .core_rsp_pc        (),
    `IGNORE_WARNINGS_END
        .core_rsp_data      ({l2c_wb_data}),
        .core_rsp_addr      (l2c_wb_addr),
        
        // L2 Cache DRAM Fill response
        .dram_rsp_valid     (dram_rsp_valid),
        .dram_rsp_addr      (dram_rsp_addr),
        .dram_rsp_data      ({dram_rsp_data_port}),

        // L2 Cache can't accept Fill Response
        .dram_rsp_ready     (dram_rsp_ready),

        // L2 Cache DRAM Fill Request
        .dram_req_read      (dram_req_read),
        .dram_req_write     (dram_req_write),        
        .dram_req_addr      (dram_req_addr),
        .dram_req_data      ({dram_req_data_port}),
        .dram_req_ready     (dram_req_ready),

        // Snoop Request
        .snp_req_valid      (llc_snp_req_valid),
        .snp_req_addr       (llc_snp_req_addr),
        .snp_req_ready      (llc_snp_req_ready),

        .snp_fwd_valid      (snp_fwd_valid),
        .snp_fwd_addr       (snp_fwd_addr),
        .snp_fwd_ready      (& snp_fwd_ready)
    );

endmodule