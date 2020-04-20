`include "VX_define.v"
`include "VX_cache_config.vh"

module Vortex_Socket (
    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // IO
    output wire                         io_valid[(`NUM_CORES * `NUM_CLUSTERS)-1:0],
    output wire[31:0]                   io_data [(`NUM_CORES * `NUM_CLUSTERS)-1:0],

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
    if (`NUM_CLUSTERS == 1) begin

        wire[`NUM_CORES-1:0]       cluster_io_valid;
        wire[`NUM_CORES-1:0][31:0] cluster_io_data;


        genvar curr_c;
        for (curr_c = 0; curr_c < `NUM_CORES; curr_c=curr_c+1) begin
            assign io_valid[curr_c] = cluster_io_valid[curr_c];
            assign io_data [curr_c] = cluster_io_data [curr_c];
        end

        Vortex_Cluster #(.CLUSTER_ID(0)) Vortex_Cluster(
            .clk                (clk),
            .reset              (reset),
            .io_valid           (cluster_io_valid),
            .io_data            (cluster_io_data),
            
            .dram_req_read      (dram_req_read),
            .dram_req_write     (dram_req_write),
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      (dram_req_data),
            .dram_req_ready     (dram_req_ready),

            .dram_rsp_valid     (dram_rsp_valid),
            .dram_rsp_addr      (dram_rsp_addr),
            .dram_rsp_data      (dram_rsp_data),
            .dram_rsp_ready     (dram_rsp_ready),

            .llc_snp_req_valid  (llc_snp_req_valid),
            .llc_snp_req_addr   (llc_snp_req_addr),
            .llc_snp_req_ready  (llc_snp_req_ready),

            .ebreak             (ebreak)
        );

    end else begin

        wire                    snp_fwd_valid;
        wire[31:0]              snp_fwd_addr;
        wire[`NUM_CLUSTERS-1:0] snp_fwd_ready;

        wire[`NUM_CLUSTERS-1:0] per_cluster_ebreak;

        assign ebreak = (&per_cluster_ebreak);

        // // DRAM Dcache Req
        wire[`NUM_CLUSTERS-1:0]                              per_cluster_dram_req_valid;
        wire[`NUM_CLUSTERS-1:0]                              per_cluster_dram_req_write;
        wire[`NUM_CLUSTERS-1:0]                              per_cluster_dram_req_read;
        wire[`NUM_CLUSTERS-1:0] [31:0]                       per_cluster_dram_req_addr;
        wire[`NUM_CLUSTERS-1:0][`DBANK_LINE_WORDS-1:0][31:0] per_cluster_dram_req_data;
        wire[31:0]                                           per_cluster_dram_req_data_up[`NUM_CLUSTERS-1:0][`DBANK_LINE_WORDS-1:0];

        wire                                                 l3c_core_req_ready;

        // // DRAM Dcache Rsp
        wire[`NUM_CLUSTERS-1:0]                              per_cluster_dram_rsp_ready;
        wire[`NUM_CLUSTERS-1:0]                              per_cluster_dram_rsp_valid;
        wire[`NUM_CLUSTERS-1:0] [31:0]                       per_cluster_dram_rsp_addr;
        wire[`NUM_CLUSTERS-1:0][`DBANK_LINE_WORDS-1:0][31:0] per_cluster_dram_rsp_data;
        wire[31:0]                                           per_cluster_dram_rsp_data_up[`NUM_CLUSTERS-1:0][`DBANK_LINE_WORDS-1:0];

        wire[`NUM_CLUSTERS-1:0][`NUM_CORES-1:0]        per_cluster_io_valid;
        wire[`NUM_CLUSTERS-1:0][`NUM_CORES-1:0][31:0]  per_cluster_io_data;

        genvar curr_c, curr_cc, curr_word;
        for (curr_c = 0; curr_c < `NUM_CLUSTERS; curr_c =curr_c+1) begin
            for (curr_cc = 0; curr_cc < `NUM_CORES; curr_cc=curr_cc+1) begin
                assign io_valid[curr_cc+(curr_c*`NUM_CORES)] = per_cluster_io_valid[curr_c][curr_cc];
                assign io_data [curr_cc+(curr_c*`NUM_CORES)] = per_cluster_io_data [curr_c][curr_cc];
            end

            for (curr_word = 0; curr_word < `DBANK_LINE_WORDS; curr_word = curr_word+1) begin
                assign per_cluster_dram_req_data        [curr_c][curr_word] = per_cluster_dram_req_data_up  [curr_c][curr_word];
                assign per_cluster_dram_rsp_data_up[curr_c][curr_word] = per_cluster_dram_rsp_data[curr_c][curr_word];
            end
        end

        genvar curr_cluster;
        for (curr_cluster = 0; curr_cluster < `NUM_CLUSTERS; curr_cluster=curr_cluster+1) begin

            Vortex_Cluster #(
                .CLUSTER_ID(curr_cluster)
            ) Vortex_Cluster (
                .clk                    (clk),
                .reset                  (reset),
                .io_valid               (per_cluster_io_valid           [curr_cluster]),
                .io_data                (per_cluster_io_data            [curr_cluster]),

                .dram_req_write         (per_cluster_dram_req_write     [curr_cluster]),
                .dram_req_read          (per_cluster_dram_req_read      [curr_cluster]),
                .dram_req_addr          (per_cluster_dram_req_addr      [curr_cluster]),
                .dram_req_data          (per_cluster_dram_req_data_up   [curr_cluster]),
                .dram_req_ready         (l3c_core_req_ready),

                .dram_rsp_valid         (per_cluster_dram_rsp_valid     [curr_cluster]),
                .dram_rsp_addr          (per_cluster_dram_rsp_addr      [curr_cluster]),
                .dram_rsp_data          (per_cluster_dram_rsp_data_up   [curr_cluster]),
                .dram_rsp_ready         (per_cluster_dram_rsp_ready     [curr_cluster]),

                .llc_snp_req_valid      (snp_fwd_valid),
                .llc_snp_req_addr       (snp_fwd_addr),
                .llc_snp_req_ready      (snp_fwd_ready                  [curr_cluster]),

                .ebreak                 (per_cluster_ebreak             [curr_cluster])
            );
        end

        //////////////////// L3 Cache ////////////////////

        wire[`L3NUM_REQUESTS-1:0]                               l3c_core_req_valid;
        wire[`L3NUM_REQUESTS-1:0][2:0]                          l3c_core_req_mem_write;
        wire[`L3NUM_REQUESTS-1:0][2:0]                          l3c_core_req_mem_read;
        wire[`L3NUM_REQUESTS-1:0][31:0]                         l3c_core_req_addr;
        wire[`L3NUM_REQUESTS-1:0][`IBANK_LINE_WORDS-1:0][31:0]  l3c_core_req_data;
        wire[`L3NUM_REQUESTS-1:0][1:0]                          l3c_core_req_wb;

        wire[`L3NUM_REQUESTS-1:0]                               l3c_core_rsp_ready;

        wire[`L3NUM_REQUESTS-1:0]                               l3c_wb;
        wire[`L3NUM_REQUESTS-1:0] [31:0]                        l3c_wb_addr;
        wire[`L3NUM_REQUESTS-1:0][`IBANK_LINE_WORDS-1:0][31:0]  l3c_wb_data;

        wire[`DBANK_LINE_WORDS-1:0][31:0]                       dram_req_data_port;
        wire[`DBANK_LINE_WORDS-1:0][31:0]                       dram_rsp_data_port;

        genvar llb_index;
        for (llb_index = 0; llb_index < `DBANK_LINE_WORDS; llb_index=llb_index+1) begin
            assign dram_req_data          [llb_index] = dram_req_data_port[llb_index];
            assign dram_rsp_data_port[llb_index] = dram_rsp_data[llb_index];
        end

        genvar l3c_curr_cluster;
        for (l3c_curr_cluster = 0; l3c_curr_cluster < `L3NUM_REQUESTS; l3c_curr_cluster=l3c_curr_cluster+1) begin
            // Core Request
            assign l3c_core_req_valid     [l3c_curr_cluster]   = per_cluster_dram_req_valid[l3c_curr_cluster];
            assign l3c_core_req_mem_read  [l3c_curr_cluster]   = per_cluster_dram_req_read [l3c_curr_cluster] ? `LW_MEM_READ : `NO_MEM_READ;
            assign l3c_core_req_mem_write [l3c_curr_cluster]   = per_cluster_dram_req_write[l3c_curr_cluster] ? `SW_MEM_WRITE : `NO_MEM_WRITE;
            assign l3c_core_req_wb        [l3c_curr_cluster]   = per_cluster_dram_req_read [l3c_curr_cluster] ? 1 : 0;
            assign l3c_core_req_addr      [l3c_curr_cluster]   = per_cluster_dram_req_addr [l3c_curr_cluster];
            assign l3c_core_req_data      [l3c_curr_cluster]   = per_cluster_dram_req_data [l3c_curr_cluster];

            // Core can't accept Response
            assign l3c_core_rsp_ready    [l3c_curr_cluster]   = per_cluster_dram_rsp_ready[l3c_curr_cluster];  

            // Cache Fill Response
            assign per_cluster_dram_rsp_valid [l3c_curr_cluster] = l3c_wb      [l3c_curr_cluster];
            assign per_cluster_dram_rsp_data  [l3c_curr_cluster] = l3c_wb_data [l3c_curr_cluster];
            assign per_cluster_dram_rsp_addr  [l3c_curr_cluster] = l3c_wb_addr [l3c_curr_cluster];
        end

        VX_cache #(
            .CACHE_SIZE_BYTES             (`L3CACHE_SIZE_BYTES),
            .BANK_LINE_SIZE_BYTES         (`L3BANK_LINE_SIZE_BYTES),
            .NUM_BANKS                    (`L3NUM_BANKS),
            .WORD_SIZE_BYTES              (`L3WORD_SIZE_BYTES),
            .NUM_REQUESTS                 (`L3NUM_REQUESTS),
            .STAGE_1_CYCLES               (`L3STAGE_1_CYCLES),
            .FUNC_ID                      (`L2FUNC_ID),
            .REQQ_SIZE                    (`L3REQQ_SIZE),
            .MRVQ_SIZE                    (`L3MRVQ_SIZE),
            .DFPQ_SIZE                    (`L3DFPQ_SIZE),
            .SNRQ_SIZE                    (`L3SNRQ_SIZE),
            .CWBQ_SIZE                    (`L3CWBQ_SIZE),
            .DWBQ_SIZE                    (`L3DWBQ_SIZE),
            .DFQQ_SIZE                    (`L3DFQQ_SIZE),
            .LLVQ_SIZE                    (`L3LLVQ_SIZE),
            .FFSQ_SIZE                    (`L3FFSQ_SIZE),
            .PRFQ_SIZE                    (`L3PRFQ_SIZE),
            .PRFQ_STRIDE                  (`L3PRFQ_STRIDE),
            .FILL_INVALIDAOR_SIZE         (`L3FILL_INVALIDAOR_SIZE),
            .SIMULATED_DRAM_LATENCY_CYCLES(`L3SIMULATED_DRAM_LATENCY_CYCLES)
        ) gpu_l3cache (
            .clk                (clk),
            .reset              (reset),

            // Core Req (DRAM Fills/WB) To L2 Request            
            .core_req_valid     (l3c_core_req_valid),
            .core_req_read      (l3c_core_req_mem_read),
            .core_req_write     (l3c_core_req_mem_write),
            .core_req_addr      (l3c_core_req_addr),
            .core_req_data      ({l3c_core_req_data}),
            .core_req_rd        (0),
            .core_req_wb        (l3c_core_req_wb),
            .core_req_warp_num  (0),
            .core_req_pc        (0),

            // L2 can't accept Core Request
            .core_req_ready     (l3c_core_req_ready),

            // Core can't accept L2 Request
            .core_rsp_ready     (|l3c_core_rsp_ready),

            // Core Writeback
            .core_rsp_valid     (l3c_wb),
        `IGNORE_WARNINGS_BEGIN
            .core_rsp_read      (),
            .core_rsp_write     (),
            .core_rsp_warp_num  (),
            .core_rsp_pc        (),
        `IGNORE_WARNINGS_END
            .core_rsp_data      ({l3c_wb_data}),
            .core_rsp_addr      (l3c_wb_addr),            

            // L2 Cache DRAM Fill response
            .dram_rsp_valid     (dram_rsp_valid),
            .dram_rsp_addr      (dram_rsp_addr),
            .dram_rsp_data      ({dram_rsp_data_port}),

            // L2 Cache can't accept Fill Response
            .dram_rsp_ready     (dram_rsp_ready),

            // L2 Cache DRAM Fill Request
            .dram_req_write     (dram_req_write),
            .dram_req_read      (dram_req_read),
            .dram_req_addr      (dram_req_addr),
            .dram_req_data      ({dram_req_data_port}),
            .dram_req_ready     (dram_req_ready),

            // Snoop Request
            .snp_req_valid      (llc_snp_req_valid),
            .snp_req_addr       (llc_snp_req_addr),
            .snp_req_ready      (llc_snp_req_ready),

            // Snoop Forward
            .snp_fwd_valid      (snp_fwd_valid),
            .snp_fwd_addr       (snp_fwd_addr),
            .snp_fwd_ready      (& snp_fwd_ready)
        );

    end

endmodule