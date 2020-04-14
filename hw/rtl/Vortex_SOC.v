`include "VX_define.v"
`include "VX_cache_config.v"

module Vortex_SOC (

    // Clock
    input  wire             clk,
    input  wire             reset,

    // IO
    output wire             io_valid[`NUMBER_CORES-1:0],
    output wire[31:0]       io_data [`NUMBER_CORES-1:0],

    output wire[31:0]       number_cores,

    // DRAM Req
    output wire             out_dram_req,
    output wire             out_dram_req_write,
    output wire             out_dram_req_read,
    output wire [31:0]      out_dram_req_addr,
    output wire [31:0]      out_dram_req_size,
    output wire [31:0]      out_dram_req_data[`DBANK_LINE_SIZE_RNG],
    output wire [31:0]      out_dram_expected_lat,
    input  wire             out_dram_req_delay,

    // DRAM Res
    output wire             out_dram_fill_accept,
    input  wire             out_dram_fill_rsp,
    input  wire [31:0]      out_dram_fill_rsp_addr,
    input  wire [31:0]      out_dram_fill_rsp_data[`DBANK_LINE_SIZE_RNG],

    // LLC Snooping
    input  wire             llc_snp_req,
    input  wire[31:0]       llc_snp_req_addr,
    output wire             llc_snp_req_delay,

    output wire             out_ebreak
);

    assign number_cores = `NUMBER_CORES;


    if (`NUMBER_CLUSTERS == 1) begin

        wire[`NUMBER_CORES-1:0]       cluster_io_valid;
        wire[`NUMBER_CORES-1:0][31:0] cluster_io_data;


        genvar curr_c;
        for (curr_c = 0; curr_c < `NUMBER_CORES; curr_c=curr_c+1) begin
            assign io_valid[curr_c] = cluster_io_valid[curr_c];
            assign io_data [curr_c] = cluster_io_data [curr_c];
        end

        Vortex_Cluster #(.CLUSTER_ID(0)) Vortex_Cluster(
            .clk                   (clk),
            .reset                 (reset),
            .io_valid              (cluster_io_valid),
            .io_data               (cluster_io_data),
            
            .out_dram_req          (out_dram_req),
            .out_dram_req_write    (out_dram_req_write),
            .out_dram_req_read     (out_dram_req_read),
            .out_dram_req_addr     (out_dram_req_addr),
            .out_dram_req_size     (out_dram_req_size),
            .out_dram_req_data     (out_dram_req_data),
            .out_dram_expected_lat (out_dram_expected_lat),
            .out_dram_req_delay    (out_dram_req_delay),

            .out_dram_fill_accept  (out_dram_fill_accept),
            .out_dram_fill_rsp     (out_dram_fill_rsp),
            .out_dram_fill_rsp_addr(out_dram_fill_rsp_addr),
            .out_dram_fill_rsp_data(out_dram_fill_rsp_data),

            .llc_snp_req           (llc_snp_req),
            .llc_snp_req_addr      (llc_snp_req_addr),
            .llc_snp_req_delay     (llc_snp_req_delay),
            .out_ebreak            (out_ebreak)
            );
    end else begin

        wire                       snp_fwd;
        wire[31:0]                 snp_fwd_addr;
        wire[`NUMBER_CLUSTERS-1:0] snp_fwd_delay;

        wire[`NUMBER_CLUSTERS-1:0] per_cluster_out_ebreak;

        assign out_ebreak = (&per_cluster_out_ebreak);


        // // DRAM Dcache Req
        wire[`NUMBER_CLUSTERS-1:0]                              per_cluster_dram_req;
        wire[`NUMBER_CLUSTERS-1:0]                              per_cluster_dram_req_write;
        wire[`NUMBER_CLUSTERS-1:0]                              per_cluster_dram_req_read;
        wire[`NUMBER_CLUSTERS-1:0] [31:0]                       per_cluster_dram_req_addr;
        wire[`NUMBER_CLUSTERS-1:0] [31:0]                       per_cluster_dram_req_size;
        wire[`NUMBER_CLUSTERS-1:0] [31:0]                       per_cluster_dram_expected_lat;
        wire[`NUMBER_CLUSTERS-1:0][`DBANK_LINE_SIZE_RNG][31:0]  per_cluster_dram_req_data;
        wire[31:0]                                              per_cluster_dram_req_data_up[`NUMBER_CLUSTERS-1:0][`DBANK_LINE_SIZE_RNG];

        wire                                                    l3c_core_accept;

        // // DRAM Dcache Res
        wire[`NUMBER_CLUSTERS-1:0]                              per_cluster_dram_fill_accept;
        wire[`NUMBER_CLUSTERS-1:0]                              per_cluster_dram_fill_rsp;
        wire[`NUMBER_CLUSTERS-1:0] [31:0]                       per_cluster_dram_fill_rsp_addr;
        wire[`NUMBER_CLUSTERS-1:0][`DBANK_LINE_SIZE_RNG][31:0]  per_cluster_dram_fill_rsp_data;
        wire[31:0]                                              per_cluster_dram_fill_rsp_data_up[`NUMBER_CLUSTERS-1:0][`DBANK_LINE_SIZE_RNG];

        wire[`NUMBER_CLUSTERS-1:0][`NUMBER_CORES_PER_CLUSTER-1:0]        per_cluster_io_valid;
        wire[`NUMBER_CLUSTERS-1:0][`NUMBER_CORES_PER_CLUSTER-1:0][31:0]  per_cluster_io_data;

        genvar curr_c;
        genvar curr_cc;
        genvar curr_word;
        for (curr_c = 0; curr_c < `NUMBER_CLUSTERS; curr_c =curr_c+1) begin
            for (curr_cc = 0; curr_cc < `NUMBER_CORES_PER_CLUSTER; curr_cc=curr_cc+1) begin
                assign io_valid[curr_cc+(curr_c*`NUMBER_CORES_PER_CLUSTER)] = per_cluster_io_valid[curr_c][curr_cc];
                assign io_data [curr_cc+(curr_c*`NUMBER_CORES_PER_CLUSTER)] = per_cluster_io_data [curr_c][curr_cc];
            end


            for (curr_word = 0; curr_word < `DBANK_LINE_SIZE_WORDS; curr_word = curr_word+1) begin
                assign per_cluster_dram_req_data        [curr_c][curr_word] = per_cluster_dram_req_data_up  [curr_c][curr_word];
                assign per_cluster_dram_fill_rsp_data_up[curr_c][curr_word] = per_cluster_dram_fill_rsp_data[curr_c][curr_word];
            end

        end



        genvar curr_cluster;
        for (curr_cluster = 0; curr_cluster < `NUMBER_CLUSTERS; curr_cluster=curr_cluster+1) begin


            Vortex_Cluster #(.CLUSTER_ID(curr_cluster)) Vortex_Cluster(
                .clk                   (clk),
                .reset                 (reset),
                .io_valid              (per_cluster_io_valid             [curr_cluster]),
                .io_data               (per_cluster_io_data              [curr_cluster]),

                .out_dram_req          (per_cluster_dram_req             [curr_cluster]),
                .out_dram_req_write    (per_cluster_dram_req_write       [curr_cluster]),
                .out_dram_req_read     (per_cluster_dram_req_read        [curr_cluster]),
                .out_dram_req_addr     (per_cluster_dram_req_addr        [curr_cluster]),
                .out_dram_req_size     (per_cluster_dram_req_size        [curr_cluster]),
                .out_dram_req_data     (per_cluster_dram_req_data_up     [curr_cluster]),
                .out_dram_expected_lat (per_cluster_dram_expected_lat    [curr_cluster]),
                .out_dram_req_delay    (l3c_core_accept),

                .out_dram_fill_accept  (per_cluster_dram_fill_accept     [curr_cluster]),
                .out_dram_fill_rsp     (per_cluster_dram_fill_rsp        [curr_cluster]),
                .out_dram_fill_rsp_addr(per_cluster_dram_fill_rsp_addr   [curr_cluster]),
                .out_dram_fill_rsp_data(per_cluster_dram_fill_rsp_data_up[curr_cluster]),

                .llc_snp_req           (snp_fwd),
                .llc_snp_req_addr      (snp_fwd_addr),
                .llc_snp_req_delay     (snp_fwd_delay[curr_cluster]),

                .out_ebreak            (per_cluster_out_ebreak           [curr_cluster])
                );
        end


        //////////////////// L3 Cache ////////////////////
        wire[`L3NUMBER_REQUESTS-1:0]                             l3c_core_req;
        wire[`L3NUMBER_REQUESTS-1:0][2:0]                        l3c_core_req_mem_write;
        wire[`L3NUMBER_REQUESTS-1:0][2:0]                        l3c_core_req_mem_read;
        wire[`L3NUMBER_REQUESTS-1:0][31:0]                       l3c_core_req_addr;
        wire[`L3NUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0] l3c_core_req_data;
        wire[`L3NUMBER_REQUESTS-1:0][1:0]                        l3c_core_req_wb;

        wire[`L3NUMBER_REQUESTS-1:0]                             l3c_core_no_wb_slot;



        wire[`L3NUMBER_REQUESTS-1:0]                                  l3c_wb;
        wire[`L3NUMBER_REQUESTS-1:0] [31:0]                           l3c_wb_addr;
        wire[`L3NUMBER_REQUESTS-1:0][`IBANK_LINE_SIZE_RNG][31:0]      l3c_wb_data;


        wire[`DBANK_LINE_SIZE_RNG][31:0]                             dram_req_data_port;
        wire[`DBANK_LINE_SIZE_RNG][31:0]                             dram_fill_rsp_data_port;

        genvar llb_index;
            for (llb_index = 0; llb_index < `DBANK_LINE_SIZE_WORDS; llb_index=llb_index+1) begin
                assign out_dram_req_data          [llb_index] = dram_req_data_port[llb_index];
                assign dram_fill_rsp_data_port[llb_index] = out_dram_fill_rsp_data[llb_index];
            end


        // 
        genvar l3c_curr_cluster;
            for (l3c_curr_cluster = 0; l3c_curr_cluster < `L3NUMBER_REQUESTS; l3c_curr_cluster=l3c_curr_cluster+1) begin
                // Core Request
                assign l3c_core_req           [l3c_curr_cluster]   = per_cluster_dram_req      [l3c_curr_cluster];

                assign l3c_core_req_mem_write [l3c_curr_cluster]   = per_cluster_dram_req_write[l3c_curr_cluster] ? `SW_MEM_WRITE : `NO_MEM_WRITE;

                assign l3c_core_req_mem_read  [l3c_curr_cluster]   = per_cluster_dram_req_read [l3c_curr_cluster] ? `LW_MEM_READ : `NO_MEM_READ;

                assign l3c_core_req_wb        [l3c_curr_cluster]   = per_cluster_dram_req_read [l3c_curr_cluster] ? 1 : 0;

                assign l3c_core_req_addr      [l3c_curr_cluster]   = per_cluster_dram_req_addr [l3c_curr_cluster];

                assign l3c_core_req_data      [l3c_curr_cluster]   = per_cluster_dram_req_data [l3c_curr_cluster];

                // Core can't accept Response
                assign l3c_core_no_wb_slot    [l3c_curr_cluster]   = ~per_cluster_dram_fill_accept[l3c_curr_cluster];  

                // Cache Fill Response
                assign per_cluster_dram_fill_rsp     [l3c_curr_cluster] = l3c_wb     [l3c_curr_cluster];
                assign per_cluster_dram_fill_rsp_data[l3c_curr_cluster] = l3c_wb_data[l3c_curr_cluster];
                assign per_cluster_dram_fill_rsp_addr[l3c_curr_cluster] = l3c_wb_addr[l3c_curr_cluster];

            end

        wire dram_snp_full;
        wire dram_req_because_of_wb;
        VX_cache #(
            .CACHE_SIZE_BYTES             (`L3CACHE_SIZE_BYTES),
            .BANK_LINE_SIZE_BYTES         (`L3BANK_LINE_SIZE_BYTES),
            .NUMBER_BANKS                 (`L3NUMBER_BANKS),
            .WORD_SIZE_BYTES              (`L3WORD_SIZE_BYTES),
            .NUMBER_REQUESTS              (`L3NUMBER_REQUESTS),
            .STAGE_1_CYCLES               (`L3STAGE_1_CYCLES),
            .FUNC_ID                      (`LLFUNC_ID),
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
            )
            gpu_l3cache
            (
            .clk               (clk),
            .reset             (reset),

            // Core Req (DRAM Fills/WB) To L2 Request
            .core_req_valid    (l3c_core_req),
            .core_req_addr     (l3c_core_req_addr),
            .core_req_writedata({l3c_core_req_data}),
            .core_req_mem_read (l3c_core_req_mem_read),
            .core_req_mem_write(l3c_core_req_mem_write),
            .core_req_rd       (0),
            .core_req_wb       (l3c_core_req_wb),
            .core_req_warp_num (0),
            .core_req_pc       (0),

            // L2 can't accept Core Request
            .delay_req         (l3c_core_accept),

            // Core can't accept L2 Request
            .core_no_wb_slot   (|l3c_core_no_wb_slot),

            // Core Writeback
            .core_wb_valid     (l3c_wb),
            .core_wb_req_rd    (),
            .core_wb_req_wb    (),
            .core_wb_warp_num  (),
            .core_wb_readdata  ({l3c_wb_data}),
            .core_wb_address   (l3c_wb_addr),
            .core_wb_pc        (),

            // L2 Cache DRAM Fill response
            .dram_fill_rsp     (out_dram_fill_rsp),
            .dram_fill_rsp_addr(out_dram_fill_rsp_addr),
            .dram_fill_rsp_data({dram_fill_rsp_data_port}),

            // L2 Cache can't accept Fill Response
            .dram_fill_accept  (out_dram_fill_accept),

            // L2 Cache DRAM Fill Request
            .dram_req          (out_dram_req),
            .dram_req_write    (out_dram_req_write),
            .dram_req_read     (out_dram_req_read),
            .dram_req_addr     (out_dram_req_addr),
            .dram_req_size     (out_dram_req_size),
            .dram_req_data     ({dram_req_data_port}),
            .dram_req_delay    (out_dram_req_delay),

            // Snoop Response
            .dram_req_because_of_wb(dram_req_because_of_wb),
            .dram_snp_full         (dram_snp_full),

            // Snoop Request
            .snp_req               (llc_snp_req),
            .snp_req_addr          (llc_snp_req_addr),
            .snp_req_delay         (llc_snp_req_delay),

            // Snoop Forward
            .snp_fwd               (snp_fwd),
            .snp_fwd_addr          (snp_fwd_addr),
            .snp_fwd_delay         (|snp_fwd_delay)
            );

    end



endmodule